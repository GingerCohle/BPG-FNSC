import numpy as np
import scipy.io as sio 
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import faiss
except ImportError:
    faiss = None

def run_kmeans(x, args,gpu_id,device):
    """
    Args:
        x: data to be clustered
    """
    if faiss is None:
        raise ImportError('run_kmeans requires faiss, but faiss is not installed in this environment')
    
    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[]}
    
    for seed, num_cluster in enumerate(args.num_cluster):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        
        clus = faiss.Clustering(d, k)

        # 关闭输出日志
        clus.verbose = False  # 关闭详细输出日志

        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 200
        clus.min_points_per_centroid = 2

        res = faiss.StandardGpuResources()
        cfg_cluster = faiss.GpuIndexFlatConfig()
        cfg_cluster.useFloat16 = False
        cfg_cluster.device = gpu_id  
        index = faiss.GpuIndexFlatL2(res, d, cfg_cluster)  #存疑，这里是利用l2距离查询

        clus.train(x, index)   

        D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I]
        
        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
        
        # sample-to-centroid distances for each cluster 
        Dcluster = [[] for c in range(k)]          
        for im,i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])
        
        # concentration estimation (phi)        
        density = np.zeros(k)
        for i,dist in enumerate(Dcluster):
            if len(dist)>1:
                d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
                density[i] = d     
                
        #if cluster only has one point, use the max to estimate its concentration        
        dmax = density.max()
        for i,dist in enumerate(Dcluster):
            if len(dist)<=1:
                density[i] = dmax 

        density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
        density = args.kmeans_temperature*density/density.mean()  #scale the mean to temperature 
        
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).to(device)
        centroids = nn.functional.normalize(centroids, p=2, dim=1)    

        im2cluster = torch.LongTensor(im2cluster).to(device)               
        density = torch.Tensor(density).to(device)
        
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)    
        
    return results

def _as_numpy_float32(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    return np.ascontiguousarray(x)


def _l2_normalize_np(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def _contiguous_labels(labels):
    _, labels = np.unique(labels, return_inverse=True)
    return labels.astype(np.int64)


def _connected_components_from_first_neighbors(nn_idx):
    parent = np.arange(len(nn_idx), dtype=np.int64)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, j in enumerate(nn_idx):
        union(i, int(j))

    roots = np.array([find(i) for i in range(len(nn_idx))], dtype=np.int64)
    return _contiguous_labels(roots)


def _first_neighbor_partition(x_norm):
    n = x_norm.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.int64)

    if faiss is not None:
        try:
            index = faiss.IndexFlatIP(x_norm.shape[1])
            index.add(x_norm)
            _, indices = index.search(x_norm, min(2, n))
            nn_idx = indices[:, 1] if indices.shape[1] > 1 else indices[:, 0]
            return _connected_components_from_first_neighbors(nn_idx)
        except Exception as exc:
            print('[FINCH] FAISS nearest-neighbor search failed, using torch fallback: {}'.format(exc))

    chunk_size = 1024
    x_tensor = torch.from_numpy(x_norm)
    nn_idx = np.empty(n, dtype=np.int64)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sim = torch.mm(x_tensor[start:end], x_tensor.t())
        rows = torch.arange(end - start)
        sim[rows, start + rows] = -float('inf')
        nn_idx[start:end] = sim.argmax(dim=1).cpu().numpy()
    return _connected_components_from_first_neighbors(nn_idx)


def _partition_centroids(x_norm, labels):
    labels = _contiguous_labels(labels)
    num_clusters = int(labels.max()) + 1
    centroids = np.zeros((num_clusters, x_norm.shape[1]), dtype=np.float32)
    counts = np.bincount(labels, minlength=num_clusters).astype(np.float32)
    np.add.at(centroids, labels, x_norm)
    centroids /= np.maximum(counts[:, None], 1.0)
    return _l2_normalize_np(centroids)


def _valid_finch_partition_count(count, min_clusters, max_clusters):
    if count < min_clusters:
        return False
    if max_clusters is not None and count > max_clusters:
        return False
    return True


def _filter_finch_partitions(partitions, max_partitions=10, min_clusters=10, max_clusters=None):
    filtered = []
    for labels in partitions[:max_partitions]:
        labels = _contiguous_labels(labels)
        count = int(labels.max()) + 1
        if _valid_finch_partition_count(count, min_clusters, max_clusters):
            filtered.append(labels)

    if filtered:
        return filtered
    return [_contiguous_labels(partitions[0])]


def _build_finch_like_partitions(x_norm, max_partitions=10, min_clusters=10, max_clusters=None):
    partitions = []
    current_features = x_norm
    current_to_original = np.arange(x_norm.shape[0], dtype=np.int64)
    previous_count = x_norm.shape[0] + 1

    for _ in range(max_partitions):
        level_labels = _first_neighbor_partition(current_features)
        level_count = int(level_labels.max()) + 1
        original_labels = level_labels[current_to_original]
        original_labels = _contiguous_labels(original_labels)
        original_count = int(original_labels.max()) + 1

        if _valid_finch_partition_count(original_count, min_clusters, max_clusters):
            partitions.append(original_labels)

        if level_count <= 1 or level_count >= previous_count or original_count < min_clusters:
            break

        previous_count = level_count
        current_features = _partition_centroids(current_features, level_labels)
        current_to_original = level_labels[current_to_original]

    if not partitions:
        partitions.append(np.zeros(x_norm.shape[0], dtype=np.int64))
    return partitions


def _try_external_finch(x_norm):
    try:
        from finch import FINCH
    except ImportError:
        return None

    try:
        partitions, _, _ = FINCH(x_norm, initial_rank=None, req_clust=None, distance='cosine', verbose=False)
    except Exception as exc:
        print('[FINCH] external FINCH failed, using FINCH-like fallback: {}'.format(exc))
        return None

    if partitions.ndim == 1:
        partitions = partitions[:, None]
    return [_contiguous_labels(partitions[:, i]) for i in range(partitions.shape[1])]


def _select_finch_partitions(partitions, targets, args):
    mode = getattr(args, 'finch_select_mode', 'match_k')
    counts = [int(labels.max()) + 1 for labels in partitions]

    if mode == 'match_k':
        selected = []
        used_levels = []
        for target in targets:
            level = min(range(len(partitions)), key=lambda i: abs(counts[i] - int(target)))
            selected.append((int(target), level, partitions[level]))
            used_levels.append(level)
        if len(set(used_levels)) < len(used_levels):
            print('[FINCH] warning: duplicate FINCH levels selected for target num_cluster {}'.format(list(targets)))
        return selected

    if mode == 'middle':
        level = len(partitions) // 2
        return [(counts[level], level, partitions[level])]

    if mode == 'all_valid':
        return [(counts[i], i, labels) for i, labels in enumerate(partitions)]

    raise ValueError('Unknown finch_select_mode: {}'.format(mode))


def _density_from_partition(x_norm, labels, centroids, args):
    labels = _contiguous_labels(labels)
    num_clusters = int(labels.max()) + 1
    sims = np.sum(x_norm * centroids[labels], axis=1)
    dists = np.sqrt(np.maximum(2.0 - 2.0 * sims, 0.0))
    density = np.zeros(num_clusters, dtype=np.float32)

    for i in range(num_clusters):
        cluster_dist = dists[labels == i]
        if len(cluster_dist) > 1:
            density[i] = cluster_dist.mean() / np.log(len(cluster_dist) + 10)

    positive = density[density > 0]
    fill_value = positive.max() if len(positive) else 1.0
    density[density <= 0] = fill_value

    density = density.clip(np.percentile(density, 10), np.percentile(density, 90))
    temperature = getattr(args, 'kmeans_temperature', getattr(args, 'temperature', 0.2))
    density = temperature * density / max(density.mean(), 1e-12)
    return density.astype(np.float32)


def run_finch(x, args, gpu_id=None, device=None):
    """
    FINCH-based clustering result generator.

    Returns the same cluster_result structure as run_kmeans:
    {'im2cluster': list[LongTensor[N]], 'centroids': list[Tensor[K, D]], 'density': list[Tensor[K]]}.
    """
    x_np = _as_numpy_float32(x)
    x_norm = _l2_normalize_np(x_np)
    if device is None:
        device = x.device if torch.is_tensor(x) else torch.device('cpu')

    max_partitions = int(getattr(args, 'finch_max_partitions', 10))
    min_clusters = int(getattr(args, 'finch_min_clusters', 10))
    max_clusters = getattr(args, 'finch_max_clusters', None)
    targets = list(getattr(args, 'num_cluster', []))

    print('[FINCH] total samples: {}, dim: {}'.format(x_norm.shape[0], x_norm.shape[1]))
    partitions = _try_external_finch(x_norm)
    if partitions is None:
        partitions = _build_finch_like_partitions(
            x_norm,
            max_partitions=max_partitions,
            min_clusters=min_clusters,
            max_clusters=max_clusters)
    else:
        partitions = _filter_finch_partitions(
            partitions,
            max_partitions=max_partitions,
            min_clusters=min_clusters,
            max_clusters=max_clusters)

    partition_summary = [(i, int(labels.max()) + 1) for i, labels in enumerate(partitions)]
    print('[FINCH] generated partitions: {}'.format(partition_summary))
    print('[FINCH] target num_cluster: {}'.format(targets))

    selected = _select_finch_partitions(partitions, targets, args)
    print('[FINCH] selected partitions: {}'.format(
        [(target, level, int(labels.max()) + 1) for target, level, labels in selected]))

    results = {'im2cluster': [], 'centroids': [], 'density': []}
    for _, _, labels in selected:
        labels = _contiguous_labels(labels)
        centroids_np = _partition_centroids(x_norm, labels)
        density_np = _density_from_partition(x_norm, labels, centroids_np, args)

        results['im2cluster'].append(torch.LongTensor(labels).to(device))
        results['centroids'].append(torch.Tensor(centroids_np).to(device))
        results['density'].append(torch.Tensor(density_np).to(device))

    print('[FINCH] centroid shapes: {}'.format([tuple(t.shape) for t in results['centroids']]))
    print('[FINCH] density shapes: {}'.format([tuple(t.shape) for t in results['density']]))
    return results

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


def mAP(cateTrainTest, IX, num_return_NN=None):
    numTrain, numTest = IX.shape

    num_return_NN = numTrain if not num_return_NN else num_return_NN

    apall = np.zeros((numTest, 1))
    yescnt_all = np.zeros((numTest, 1))
    for qid in range(numTest):
        query = IX[:, qid]
        x, p = 0, 0

        for rid in range(num_return_NN):
            if cateTrainTest[query[rid], qid]:
                x += 1
                p += x/(rid*1.0 + 1.0)
        yescnt_all[qid] = x
        if not p: apall[qid] = 0.0
        else: apall[qid] = p/(num_return_NN*1.0)

    return np.mean(apall),apall,yescnt_all  


def topK(cateTrainTest, HammingRank, k=500):
    numTest = cateTrainTest.shape[1]

    precision = np.zeros((numTest, 1))
    recall = np.zeros((numTest, 1))

    topk = HammingRank[:k, :]

    for qid in range(numTest):
        retrieved = topk[:, qid]
        rel = cateTrainTest[retrieved, qid]
        retrieved_relevant_num = np.sum(rel)
        real_relevant_num = np.sum(cateTrainTest[:, qid])

        precision[qid] = retrieved_relevant_num/(k*1.0)
        recall[qid] = retrieved_relevant_num/(real_relevant_num*1.0)

    return precision.mean(), recall.mean()

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import sample
import sys
sys.path.append('..')
from torch.autograd import Variable


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def dcl(out_1, out_2, batch_size, temperature=0.5, tau_plus=0.1,device=None):
    out_1 = F.normalize(out_1, dim=1)
    out_2 = F.normalize(out_2, dim=1)

    out = torch.cat([out_1, out_2], dim=0)
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = get_negative_mask(batch_size).to(device)
    neg = neg.masked_select(mask).view(2 * batch_size, -1)

    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)

    if True:
        N = batch_size * 2 - 2
        Ng = (-tau_plus * N * pos + neg.sum(dim = -1)) / (1 - tau_plus)
        Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
    else:
        Ng = neg.sum(dim=-1)

    loss = (- torch.log(pos / (pos + Ng) )).mean()
    return loss


def _fns_contrast_one_direction(
    anchor,
    target,
    temperature=0.2,
    threshold=0.7,
    min_weight=0.2,
    suppress_scale=0.05,
    detach_weight=True,
):
    if anchor is None or target is None:
        return None
    if (not torch.is_tensor(anchor)) or (not torch.is_tensor(target)):
        return None
    if anchor.dim() != 2 or target.dim() != 2:
        return anchor.new_tensor(0.0)
    if anchor.size(1) != target.size(1):
        return anchor.new_tensor(0.0)

    if anchor.size(0) != target.size(0):
        batch_size = min(anchor.size(0), target.size(0))
        anchor = anchor[:batch_size]
        target = target[:batch_size]

    if anchor.size(0) <= 1:
        return anchor.new_tensor(0.0)

    anchor = F.normalize(anchor, dim=1)
    target = F.normalize(target, dim=1)

    tau = max(float(temperature), 1e-6)
    suppress_scale = max(float(suppress_scale), 1e-6)
    min_weight = max(0.0, min(1.0, float(min_weight)))

    sim = torch.matmul(anchor, target.t())
    logits = sim / tau

    batch_size = logits.size(0)
    eye = torch.eye(batch_size, device=logits.device, dtype=torch.bool)
    pos_logits = logits.diag()

    weight_source = sim.detach() if detach_weight else sim
    weights = 1.0 - torch.sigmoid((weight_source - float(threshold)) / suppress_scale) * (1.0 - min_weight)
    weights = weights.clamp(min=min_weight, max=1.0)
    weights = weights.masked_fill(eye, 1.0)

    weighted_logits = logits + torch.log(weights.clamp_min(1e-8))
    log_denom = torch.logsumexp(weighted_logits, dim=1)
    loss = -(pos_logits - log_denom).mean()
    if not torch.isfinite(loss):
        return anchor.new_tensor(0.0)
    return loss


def dcl_fns(
    z1,
    z2,
    temperature=0.2,
    threshold=0.7,
    min_weight=0.2,
    suppress_scale=0.05,
    detach_weight=True,
    symmetric=True,
):
    loss_12 = _fns_contrast_one_direction(
        z1,
        z2,
        temperature=temperature,
        threshold=threshold,
        min_weight=min_weight,
        suppress_scale=suppress_scale,
        detach_weight=detach_weight,
    )
    if loss_12 is None:
        ref = z1 if torch.is_tensor(z1) else z2
        return ref.new_tensor(0.0) if torch.is_tensor(ref) else torch.tensor(0.0)

    if symmetric:
        loss_21 = _fns_contrast_one_direction(
            z2,
            z1,
            temperature=temperature,
            threshold=threshold,
            min_weight=min_weight,
            suppress_scale=suppress_scale,
            detach_weight=detach_weight,
        )
        loss = 0.5 * (loss_12 + loss_21)
    else:
        loss = loss_12

    if not torch.isfinite(loss):
        return z1.new_tensor(0.0)
    return loss


# Paper-facing alias: False-Negative Suppressed Contrast (FNSC).
fnsc_contrastive_loss = dcl_fns


def Loss_CVH(all_hashcode_norm,index,criterion,cluster_result,batchsize,cfg,device):
        proto_labels = []
        proto_logits = []
        for n, (im2cluster,prototypes,density) in enumerate(zip(cluster_result['im2cluster'],cluster_result['centroids'],cluster_result['density'])):
            # get positive prototypes
            pos_proto_id = im2cluster[index]
            pos_prototypes = prototypes[pos_proto_id]    
            
            # sample negative prototypes
            all_proto_id = [i for i in range(im2cluster.max()+1)]       
            neg_proto_id = list(set(all_proto_id)-set(pos_proto_id.tolist()))
            sample_size = min(batchsize,len(neg_proto_id))
            #neg_proto_id = sample(neg_proto_id,batchsize) #sample r negative prototypes 
            neg_proto_id = sample(neg_proto_id,sample_size) 
            neg_prototypes = prototypes[neg_proto_id]    

            proto_selected = torch.cat([pos_prototypes,neg_prototypes],dim=0)
            # compute prototypical logits
            logits_proto = torch.mm(all_hashcode_norm,proto_selected.t())

            # targets for prototype assignment
            labels_proto = torch.linspace(0, all_hashcode_norm.size(0)-1, steps=all_hashcode_norm.size(0)).long().to(device)
            
            # scaling temperatures for the selected prototypes
            temp_proto = density[torch.cat([pos_proto_id,torch.LongTensor(neg_proto_id).to(device)],dim=0)]  
            logits_proto /= temp_proto
            
            proto_labels.append(labels_proto)
            proto_logits.append(logits_proto)
        loss_proto = 0.

        for proto_out,proto_target in zip(proto_logits, proto_labels):
            loss_proto += criterion(proto_out, proto_target)  

        # average loss across all sets of prototypes
        loss_proto /= len(cfg.num_cluster) 
        return loss_proto


def build_pgafs_mask_and_labels(
    visual_word,
    index,
    cluster_result,
    original_mask,
    cfg,
    device,
    model,
):
    """
    Build a prototype-guided mask and matching reconstruction labels.

    Args:
        visual_word: Tensor [B, T, D].
        index: Tensor [B].
        cluster_result: dict with im2cluster / centroids / density.
        original_mask: Tensor [B, T], bool, True means masked.
        cfg: config.
        device: torch device.
        model: AutoSSVH model, must provide get_frame_hash_tokens(...).

    Returns:
        pgafs_mask: Tensor [B, T], bool.
        pgafs_labels: Tensor [B, M, D].
    """
    index = index.view(-1).long().to(device)
    visual_word = visual_word.to(device)
    original_mask = original_mask.to(device).bool()

    level = getattr(cfg, "pgafs_cluster_level", -1)
    im2cluster = cluster_result["im2cluster"][level].to(device)
    centroids = cluster_result["centroids"][level].to(device)

    pos_proto_id = im2cluster[index].long().to(device)
    pos_proto = centroids[pos_proto_id]

    model_for_tokens = model.module if hasattr(model, "module") else model
    if not hasattr(model_for_tokens, "get_frame_hash_tokens"):
        raise AttributeError("PGAFS requires model.get_frame_hash_tokens(...)")

    with torch.no_grad():
        frame_tokens = model_for_tokens.get_frame_hash_tokens(visual_word)

    if frame_tokens.dim() != 3:
        raise ValueError("PGAFS frame tokens must have shape [B, T, nbits]")
    if frame_tokens.shape[0] != visual_word.shape[0] or frame_tokens.shape[1] != visual_word.shape[1]:
        raise ValueError("PGAFS frame token shape must match visual_word batch/time dimensions")
    if frame_tokens.shape[2] != pos_proto.shape[1]:
        raise ValueError(
            "PGAFS frame token dim must match prototype dim: {} vs {}".format(
                frame_tokens.shape[2], pos_proto.shape[1]
            )
        )

    score_type = getattr(cfg, "pgafs_score_type", "proto_sim")
    frame_norm = F.normalize(frame_tokens, dim=-1)
    centroids = centroids.to(frame_tokens.device).float()
    all_proto = F.normalize(centroids, dim=-1)
    pos_proto_id = pos_proto_id.to(frame_tokens.device)
    T = frame_tokens.size(1)
    balanced_selection = False
    pos_sim = None
    margin_scores = None

    if score_type == "proto_sim":
        pos_proto = centroids[pos_proto_id]
        pos_proto_norm = F.normalize(pos_proto, dim=-1)
        scores = torch.sum(frame_norm * pos_proto_norm[:, None, :], dim=-1)
        if getattr(cfg, "pgafs_use_abs_score", False):
            scores = scores.abs()
    elif score_type == "proto_margin":
        if all_proto.size(0) <= 1:
            pos_proto = centroids[pos_proto_id]
            pos_proto_norm = F.normalize(pos_proto, dim=-1)
            scores = torch.sum(frame_norm * pos_proto_norm[:, None, :], dim=-1)
        else:
            all_sim = torch.matmul(frame_norm, all_proto.t())
            gather_idx = pos_proto_id[:, None, None].expand(-1, T, 1)
            pos_sim = all_sim.gather(dim=2, index=gather_idx).squeeze(-1)
            neg_sim = all_sim.clone()
            neg_sim.scatter_(dim=2, index=gather_idx, value=-1e4)
            hard_neg_sim = neg_sim.max(dim=2).values
            scores = pos_sim - hard_neg_sim
    elif score_type == "proto_margin_balanced":
        pos_proto = centroids[pos_proto_id]
        pos_proto_norm = F.normalize(pos_proto, dim=-1)
        pos_sim = torch.sum(frame_norm * pos_proto_norm[:, None, :], dim=-1)
        if all_proto.size(0) <= 1:
            scores = pos_sim
        else:
            all_sim = torch.matmul(frame_norm, all_proto.t())
            gather_idx = pos_proto_id[:, None, None].expand(-1, T, 1)
            pos_sim = all_sim.gather(dim=2, index=gather_idx).squeeze(-1)
            neg_sim = all_sim.clone()
            neg_sim.scatter_(dim=2, index=gather_idx, value=-1e4)
            hard_neg_sim = neg_sim.max(dim=2).values
            margin_scores = pos_sim - hard_neg_sim
            balanced_selection = True
    else:
        raise ValueError("Unsupported pgafs_score_type: {}".format(score_type))

    mask_counts = original_mask.sum(dim=1)
    if not torch.all(mask_counts == mask_counts[0]):
        raise ValueError("PGAFS-v1 requires equal mask counts per sample")

    B, T, D = visual_word.shape
    M = int(mask_counts[0].item())
    if getattr(cfg, "pgafs_topk", None) is not None:
        M = min(int(cfg.pgafs_topk), T)
    if M <= 0:
        pgafs_mask = torch.zeros_like(original_mask, dtype=torch.bool)
        pgafs_labels = visual_word.new_empty((B, 0, D))
        return pgafs_mask, pgafs_labels

    if balanced_selection:
        margin_ratio = float(getattr(cfg, "pgafs_margin_ratio", 0.7))
        M_margin = int(round(M * margin_ratio))
        M_margin = max(0, min(M, M_margin))
        M_pos = M - M_margin
        selected_parts = []
        margin_idx = None

        if M_margin > 0:
            margin_idx = torch.topk(margin_scores, k=M_margin, dim=1, largest=True).indices
            selected_parts.append(margin_idx)
        if M_pos > 0:
            pos_scores_for_select = pos_sim.clone()
            if margin_idx is not None:
                pos_scores_for_select.scatter_(1, margin_idx, value=-1e4)
            pos_idx = torch.topk(pos_scores_for_select, k=M_pos, dim=1, largest=True).indices
            selected_parts.append(pos_idx)
        topk_idx = torch.cat(selected_parts, dim=1)
        if topk_idx.shape[1] != M:
            raise ValueError("Balanced PGAFS selected {} frames, expected {}".format(topk_idx.shape[1], M))
    else:
        topk_idx = torch.topk(scores, k=M, dim=1, largest=True).indices

    pgafs_mask = torch.zeros_like(original_mask, dtype=torch.bool)
    pgafs_mask.scatter_(1, topk_idx, True)
    if not torch.all(pgafs_mask.sum(dim=1) == M):
        raise ValueError("PGAFS mask construction produced an invalid mask count")

    pgafs_labels = visual_word[pgafs_mask].reshape(B, M, D)
    return pgafs_mask, pgafs_labels


# Paper-facing alias: Balanced Prototype-Margin Sampling (BPMS).
build_bpms_mask_and_labels = build_pgafs_mask_and_labels


def compute_shvd_loss(
    hash_code_teacher,
    hash_code_student,
    cluster_result,
    cfg,
    device,
):
    """
    Semantic-Hard View Distillation.

    The random-mask view is the teacher and the PGAFS semantic-hard view is the
    student. The objective distills prototype-level semantic distributions.
    """
    level = getattr(cfg, "shvd_cluster_level", getattr(cfg, "pgafs_cluster_level", -1))
    prototypes = cluster_result["centroids"][level].to(device).float()

    tau = float(getattr(cfg, "shvd_tau", 0.2))
    tau = max(tau, 1e-6)

    hash_code_teacher = F.normalize(hash_code_teacher, dim=1)
    hash_code_student = F.normalize(hash_code_student, dim=1)
    prototypes = F.normalize(prototypes, dim=1)

    teacher_logits = torch.mm(hash_code_teacher, prototypes.t()) / tau
    student_logits = torch.mm(hash_code_student, prototypes.t()) / tau

    if getattr(cfg, "shvd_stopgrad_teacher", True):
        q_teacher = F.softmax(teacher_logits.detach(), dim=1)
    else:
        q_teacher = F.softmax(teacher_logits, dim=1)

    log_q_student = F.log_softmax(student_logits, dim=1)
    return F.kl_div(log_q_student, q_teacher, reduction="batchmean")


# Paper-facing alias: PA-HVL prototype-distribution alignment.
compute_pahvl_distillation_loss = compute_shvd_loss


def get_positive_prototype(index, cluster_result, cfg, device, level=None):
    if level is None:
        level = getattr(cfg, "pc_decoder_cluster_level", getattr(cfg, "pgafs_cluster_level", -1))
    im2cluster = cluster_result["im2cluster"][level].to(device)
    centroids = cluster_result["centroids"][level].to(device).float()
    index = index.view(-1).long().to(device)
    pos_proto_id = im2cluster[index]
    return centroids[pos_proto_id]




def AutoSSVH_criterion(cfg, data, model, epoch, i, total_len, logger,cluster_result=None,criterion=None,device=None):
    data = {key: value.to(device) for key, value in data.items()}
    batchsize = data["visual_word"].size(0)
    device = data["visual_word"].device
    index = data["index"].squeeze()

    bool_masked_pos_1 = data["mask"][:,0,:].to(device, non_blocking=True).flatten(1).to(torch.bool)
    bool_masked_pos_2 = data["mask"][:,1,:].to(device, non_blocking=True).flatten(1).to(torch.bool)

    labels_1 = data["visual_word"][bool_masked_pos_1].reshape(batchsize, -1, cfg.feature_size)
    pgafs_triggered = False
    use_pgafs = (
        getattr(cfg, "pgafs_enable", False)
        and cluster_result is not None
        and epoch >= cfg.warmup_epoch
        and getattr(cfg, "pgafs_view", "view2") in ["view2", "both"]
    )
    if use_pgafs and torch.rand((), device=device).item() < getattr(cfg, "pgafs_prob", 0.5):
        bool_masked_pos_2, labels_2 = build_pgafs_mask_and_labels(
            visual_word=data["visual_word"],
            index=data["index"].view(-1),
            cluster_result=cluster_result,
            original_mask=bool_masked_pos_2,
            cfg=cfg,
            device=device,
            model=model,
        )
        pgafs_triggered = True
    else:
        labels_2 = data["visual_word"][bool_masked_pos_2].reshape(batchsize, -1, cfg.feature_size)

    use_pc_decoder_view2 = (
        getattr(cfg, "pc_decoder_enable", False)
        and cluster_result is not None
        and epoch >= cfg.warmup_epoch
        and getattr(cfg, "pc_decoder_view", "view2") in ["view2", "both"]
    )
    if getattr(cfg, "pc_decoder_only_pgafs_triggered", True):
        use_pc_decoder_view2 = use_pc_decoder_view2 and pgafs_triggered

    proto_cond_2 = None
    if use_pc_decoder_view2:
        proto_cond_2 = get_positive_prototype(
            index=data["index"].view(-1),
            cluster_result=cluster_result,
            cfg=cfg,
            device=device,
            level=getattr(cfg, "pc_decoder_cluster_level", getattr(cfg, "pgafs_cluster_level", -1)),
        )


    frame_1, hash_code_1,_ = model.forward(data["visual_word"], bool_masked_pos_1)
    frame_2, hash_code_2,_ = model.forward(
        data["visual_word"],
        bool_masked_pos_2,
        proto_cond=proto_cond_2,
        pc_decoder_enable=use_pc_decoder_view2,
        pc_decoder_gamma=getattr(cfg, "pc_decoder_gamma", 0.1),
        pc_decoder_detach_proto=getattr(cfg, "pc_decoder_detach_proto", True),
    )

    hash_code_1 = torch.mean(hash_code_1, 1)
    hash_code_2 = torch.mean(hash_code_2, 1)

    # recon_loss
    if frame_2.shape != labels_2.shape:
        raise ValueError("PGAFS label/mask mismatch: frame_2 shape {} vs labels_2 shape {}".format(
            tuple(frame_2.shape), tuple(labels_2.shape)))
    recon_loss = F.mse_loss(frame_1, labels_1) + F.mse_loss(frame_2, labels_2)

    # contra_loss(loss_vc)
    if getattr(cfg, "fns_dcl_enable", False):
        loss_vc = dcl_fns(
            hash_code_1,
            hash_code_2,
            temperature=getattr(cfg, "fns_dcl_temperature", getattr(cfg, "temperature", 0.2)),
            threshold=getattr(cfg, "fns_dcl_threshold", 0.7),
            min_weight=getattr(cfg, "fns_dcl_min_weight", 0.2),
            suppress_scale=getattr(cfg, "fns_dcl_suppress_scale", 0.05),
            detach_weight=getattr(cfg, "fns_dcl_detach_weight", True),
            symmetric=getattr(cfg, "fns_dcl_symmetric", True),
        )
    else:
        loss_vc = dcl(hash_code_1, hash_code_2, batchsize, temperature=cfg.temperature, tau_plus=cfg.tau_plus,device=device)
    
    # loss_cvh
    loss_cvh = 0.0
    if cfg.CVH and cluster_result is not None:
        hash_code_norm_1 = F.normalize(hash_code_1, dim=1)
        hash_code_norm_2 = F.normalize(hash_code_2, dim=1)
        loss_cvh = Loss_CVH(hash_code_norm_1, index, criterion, cluster_result, batchsize, cfg, device)+Loss_CVH(hash_code_norm_2, index, criterion, cluster_result, batchsize, cfg, device)
        loss_cvh = loss_cvh/2

    loss_shvd = torch.tensor(0.0, device=device)
    use_shvd = (
        getattr(cfg, "shvd_enable", False)
        and getattr(cfg, "pgafs_enable", False)
        and cluster_result is not None
        and epoch >= cfg.warmup_epoch
    )
    if getattr(cfg, "shvd_only_pgafs_triggered", True):
        use_shvd = use_shvd and pgafs_triggered
    if use_shvd:
        loss_shvd = compute_shvd_loss(
            hash_code_teacher=hash_code_1,
            hash_code_student=hash_code_2,
            cluster_result=cluster_result,
            cfg=cfg,
            device=device,
        )

    loss = recon_loss + cfg.a * loss_vc + cfg.b * loss_cvh + getattr(cfg, "shvd_weight", 0.03) * loss_shvd
    if i % 10 == 0 or batchsize < cfg.batch_size:  
        if getattr(cfg, "shvd_enable", False):
            logger.info('Epoch:[%d/%d] Step:[%d/%d] reconstruction_loss: %.2f loss_vc: %.2f loss_cvh: %.2f loss_shvd: %.4f' \
                % (epoch+1, cfg.num_epochs, i, total_len,\
                recon_loss.data.cpu().numpy(), loss_vc.data.cpu().numpy(),\
                loss_cvh, loss_shvd.data.cpu().numpy()))
        else:
            logger.info('Epoch:[%d/%d] Step:[%d/%d] reconstruction_loss: %.2f loss_vc: %.2f loss_cvh: %.2f' \
                % (epoch+1, cfg.num_epochs, i, total_len,\
                recon_loss.data.cpu().numpy(), loss_vc.data.cpu().numpy(),\
                loss_cvh))

    return loss

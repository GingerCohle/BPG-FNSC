# model
model_name = 'AutoSSVH'
use_checkpoint = None
feature_size = 4096
hidden_size = 256
max_frames = 25
nbits = 64
AutoSSVH_type = 'small'

# dataset
dataset = 'fcv'
workers = 1
batch_size = 512
mask_prob = 0.75

# train
seed = 1
num_epochs = 805
a = 1.0
temperature = 0.5
tau_plus = 0.1
train_num_sample = 45585

# Component Voting Hash Learning(CVH)
CVH=True
cluster_method = "kmeans"
finch_select_mode = "match_k"
finch_max_partitions = 10
finch_min_clusters = 10
finch_max_clusters = None
num_cluster = [250,500,1000]#[250,400,600]#
warmup_epoch = 100  #40 60- 80 
kmeans_temperature = 0.2
b = 0.01
data_drop_rate = 0.

# Prototype-Guided Adaptive Frame Sampling
pgafs_enable = False
pgafs_view = "view2"
pgafs_prob = 0.5
pgafs_cluster_level = -1
pgafs_topk = None
pgafs_score_type = "proto_sim"
pgafs_margin_ratio = 0.5
pgafs_use_abs_score = False
pgafs_random_warmup = True

# Semantic-Hard View Distillation
shvd_enable = False
shvd_weight = 0.03
shvd_tau = 0.2
shvd_cluster_level = -1
shvd_stopgrad_teacher = True
shvd_only_pgafs_triggered = True

# Prototype-Conditioned Decoder
pc_decoder_enable = False
pc_decoder_gamma = 0.1
pc_decoder_cluster_level = -1
pc_decoder_detach_proto = True
pc_decoder_view = "view2"
pc_decoder_only_pgafs_triggered = True

# False-Negative Suppressed DCL
fns_dcl_enable = False
fns_dcl_temperature = 0.2
fns_dcl_threshold = 0.7
fns_dcl_min_weight = 0.2
fns_dcl_suppress_scale = 0.05
fns_dcl_detach_weight = True
fns_dcl_symmetric = True

# test
test_batch_size = 128
test_num_sample = 45600

# optimizer
optimizer_name = 'Adam'
schedule = 'StepLR'
lr = 1e-4
min_lr = 1e-5
lr_decay_rate = 20
lr_decay_gamma = 0.9
weight_decay = 0.0

# path
data_root = f"data/{dataset}/"
home_root = './'

# path:train
train_feat_path = [data_root + 'fcv_train_feats.h5']

# path:test
test_feat_path = [data_root + 'fcv_test_feats.h5'] # database+query
label_path = [data_root + 'fcv_test_labels.mat']

# path:save
save_dir = home_root + "checkpoint/" + dataset
file_path = f"{save_dir}/{model_name}_{nbits}bit"
log_path = f"{home_root}logs/{dataset}S5VH_{nbits}bit"+"cluster"+"wohashvoting"

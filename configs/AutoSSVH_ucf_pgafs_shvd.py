from AutoSSVH_ucf import *

# SHVD: semantic-hard view distillation on top of PGAFS-v1.
cluster_method = "kmeans"

pgafs_enable = True
pgafs_view = "view2"
pgafs_prob = 0.5
pgafs_cluster_level = -1
pgafs_topk = None
pgafs_score_type = "proto_sim"
pgafs_use_abs_score = False

shvd_enable = True
shvd_weight = 0.03
shvd_tau = 0.2
shvd_cluster_level = -1
shvd_stopgrad_teacher = True
shvd_only_pgafs_triggered = True

file_path = f"{save_dir}/{model_name}_{nbits}bit_pgafs_shvd_w003_tau02"
log_path = f"{home_root}logs/{dataset}_{nbits}bit_pgafs_shvd_w003_tau02"

from AutoSSVH_fcv import *

# PGAFS-v1: hash-space prototype-guided second-view masking.
cluster_method = "kmeans"
pgafs_enable = True
pgafs_view = "view2"
pgafs_prob = 0.5
pgafs_cluster_level = -1
pgafs_score_type = "proto_sim"
pgafs_use_abs_score = False

file_path = f"{save_dir}/{model_name}_{nbits}bit_pgafs_v1_p05"
log_path = f"{home_root}logs/{dataset}_{nbits}bit_pgafs_v1_p05"

from AutoSSVH_hmdb import *

# FINCH-CVH experiment config. Baseline configs keep cluster_method="kmeans".
cluster_method = "finch"
finch_select_mode = "match_k"
finch_max_partitions = 10
finch_min_clusters = 10
finch_max_clusters = None

file_path = f"{save_dir}/{model_name}_{nbits}bit_finch"
log_path = f"{home_root}logs/{dataset}_{nbits}bit_finch"

from AutoSSVH_hmdb_pgafs_shvd_pcdecoder import *

# BPM-PGAFS: balanced prototype-margin guided semantic-hard sampling.
cluster_method = "kmeans"

pgafs_enable = True
pgafs_view = "view2"
pgafs_prob = 0.35
pgafs_cluster_level = -1
pgafs_topk = None
pgafs_score_type = "proto_margin_balanced"
pgafs_margin_ratio = 0.5
pgafs_use_abs_score = False

shvd_enable = True
shvd_weight = 0.02
shvd_tau = 0.2
shvd_cluster_level = -1
shvd_stopgrad_teacher = True
shvd_only_pgafs_triggered = True

pc_decoder_enable = True
pc_decoder_gamma = 0.05
pc_decoder_cluster_level = -1
pc_decoder_detach_proto = True
pc_decoder_view = "view2"
pc_decoder_only_pgafs_triggered = True

file_path = f"{save_dir}/{model_name}_{nbits}bit_pgafs_margin_balanced_r05_shvd_pcdec_g005_sw002_p035"
log_path = f"{home_root}logs/{dataset}_{nbits}bit_pgafs_margin_balanced_r05_shvd_pcdec_g005_sw002_p035"

from AutoSSVH_fcv_pgafs_shvd_pcdecoder import *

# PM-PGAFS: prototype-margin guided semantic-hard sampling.
pgafs_score_type = "proto_margin"
pgafs_use_abs_score = False

file_path = f"{save_dir}/{model_name}_{nbits}bit_pgafs_margin_shvd_pcdec_g005_sw002_p035"
log_path = f"{home_root}logs/{dataset}_{nbits}bit_pgafs_margin_shvd_pcdec_g005_sw002_p035"

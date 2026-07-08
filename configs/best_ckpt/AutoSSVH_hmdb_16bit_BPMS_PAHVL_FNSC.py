import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from AutoSSVH_hmdb_bpm_pgafs_shvd_pcdec_fnsdcl import *

paper_method = "BPMS+PA-HVL+FNSC"
nbits = 16
pgafs_margin_ratio = 0.6
pgafs_prob = 0.45
best_ckpt_name = "hmdb_16bit_r06_p045_metric_majority.pth"
file_path = f"{save_dir}/{model_name}_{nbits}bit_BPMS_PAHVL_FNSC_hmdb_r06_p045"
log_path = f"{home_root}logs/{dataset}_{nbits}bit_BPMS_PAHVL_FNSC_hmdb_r06_p045"

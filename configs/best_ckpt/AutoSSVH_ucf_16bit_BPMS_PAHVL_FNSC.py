import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from AutoSSVH_ucf_bpm_pgafs_shvd_pcdec_fnsdcl import *

paper_method = "BPMS+PA-HVL+FNSC"
nbits = 16
pgafs_margin_ratio = 0.6
pgafs_prob = 0.55
best_ckpt_name = "ucf_16bit_r06_p055_metric_majority.pth"
file_path = f"{save_dir}/{model_name}_{nbits}bit_BPMS_PAHVL_FNSC_ucf_r06_p055"
log_path = f"{home_root}logs/{dataset}_{nbits}bit_BPMS_PAHVL_FNSC_ucf_r06_p055"

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from AutoSSVH_act_bpm_pgafs_shvd_pcdec_fnsdcl import *

paper_method = "BPMS+PA-HVL+FNSC"
nbits = 64
pgafs_margin_ratio = 0.3
pgafs_prob = 0.25
best_ckpt_name = "activitynet_64bit_r03_p025_metric_majority.pth"
file_path = f"{save_dir}/{model_name}_{nbits}bit_BPMS_PAHVL_FNSC_activitynet_r03_p025"
log_path = f"{home_root}logs/{dataset}_{nbits}bit_BPMS_PAHVL_FNSC_activitynet_r03_p025"

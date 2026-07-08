import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from AutoSSVH_act_bpm_pgafs_shvd_pcdec_fnsdcl import *

paper_method = "BPMS+PA-HVL+FNSC"
nbits = 32
pgafs_margin_ratio = 0.7
pgafs_prob = 0.35
best_ckpt_name = "activitynet_32bit_r07_p035_metric_majority.pth"
file_path = f"{save_dir}/{model_name}_{nbits}bit_BPMS_PAHVL_FNSC_activitynet_r07_p035"
log_path = f"{home_root}logs/{dataset}_{nbits}bit_BPMS_PAHVL_FNSC_activitynet_r07_p035"

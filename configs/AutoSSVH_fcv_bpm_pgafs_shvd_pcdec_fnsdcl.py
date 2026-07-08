from AutoSSVH_fcv_pgafs_shvd_pcmr_margin_balanced import *

# BPM-PGAFS r05 + SHVD + PCMR + FNS-DCL.
fns_dcl_enable = True
fns_dcl_temperature = 0.2
fns_dcl_threshold = 0.7
fns_dcl_min_weight = 0.2
fns_dcl_suppress_scale = 0.05
fns_dcl_detach_weight = True
fns_dcl_symmetric = True

file_path = f"{save_dir}/{model_name}_{nbits}bit_bpm_pgafs_shvd_pcdec_fnsdcl_r05_g005_sw002"
log_path = f"{home_root}logs/{dataset}_{nbits}bit_bpm_pgafs_shvd_pcdec_fnsdcl_r05_g005_sw002"

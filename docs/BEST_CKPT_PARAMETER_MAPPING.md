# Best Checkpoint Parameter Mapping

All checkpoints in `best_ckpt/` are metric-majority selections. The selection
prioritizes how many of mAP@5/20/40/60/80/100 are led by a setting, rather than
only mAP@5.

Fixed method settings:

```python
pgafs_score_type = "proto_margin_balanced"  # BPMS
shvd_enable = True                          # PA-HVL distillation
shvd_weight = 0.02
pc_decoder_enable = True                    # PA-HVL reconstruction
pc_decoder_gamma = 0.05
fns_dcl_enable = True                       # FNSC
fns_dcl_threshold = 0.7
fns_dcl_min_weight = 0.2
fns_dcl_suppress_scale = 0.05
```

Variable settings per checkpoint:

| Dataset | Bit | Checkpoint | rho | pgafs_prob | Selected combo | Metric wins |
|---|---:|---|---:|---:|---|---|
| ActivityNet | 16 | `activitynet_16bit_r06_p045_metric_majority.pth` | 0.6 | 0.45 | r06_p045 | 5/6 |
| ActivityNet | 32 | `activitynet_32bit_r07_p035_metric_majority.pth` | 0.7 | 0.35 | r07_p035 | 3/6 |
| ActivityNet | 64 | `activitynet_64bit_r03_p025_metric_majority.pth` | 0.3 | 0.25 | r03_p025 | 5/6 |
| FCVID | 16 | `fcv_16bit_r05_p035_metric_majority.pth` | 0.5 | 0.35 | r05_p035 | 6/6 |
| FCVID | 32 | `fcv_32bit_r05_p035_metric_majority.pth` | 0.5 | 0.35 | r05_p035 | 6/6 |
| FCVID | 64 | `fcv_64bit_r05_p035_metric_majority.pth` | 0.5 | 0.35 | r05_p035 | 6/6 |
| HMDB51 | 16 | `hmdb_16bit_r06_p045_metric_majority.pth` | 0.6 | 0.45 | r06_p045 | 4/6 |
| HMDB51 | 32 | `hmdb_32bit_r05_p045_metric_majority.pth` | 0.5 | 0.45 | r05_p045 | 5/6 |
| HMDB51 | 64 | `hmdb_64bit_r05_p055_metric_majority.pth` | 0.5 | 0.55 | r05_p055 | 5/6 |
| UCF101 | 16 | `ucf_16bit_r06_p055_metric_majority.pth` | 0.6 | 0.55 | r06_p055 | 4/6 |
| UCF101 | 32 | `ucf_32bit_r06_p025_metric_majority.pth` | 0.6 | 0.25 | r06_p025 | 2/6 |
| UCF101 | 64 | `ucf_64bit_r06_p055_metric_majority.pth` | 0.6 | 0.55 | r06_p055 | 6/6 |

Full metrics are stored in:

```text
best_ckpt/metric_majority_best_ckpt_manifest_20260625.csv
```


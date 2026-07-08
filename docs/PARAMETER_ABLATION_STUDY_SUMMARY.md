# Parameter Ablation Study Summary

当前代码库：

```text
/media/kejunjie_coco/autohash/autossh_newwek/BPM-PGAFS+SHVD+PCMR
```

当前主方法：

```text
BPM-PGAFS + SHVD + PCMR + FNS-DCL
```

当前主 tag：

```text
bpm_pgafs_shvd_pcdec_fnsdcl_r05_g005_sw002
```

本文档总结当前已经完成的参数消融实验、对应取值、结果文件、checkpoint 路径，以及后续论文中建议呈现的参数敏感性实验。

## 1. Executive Summary

目前已经完成得最系统的是 **BPMS / BPM-PGAFS 的 `pgafs_margin_ratio` 与 `pgafs_prob` 二维参数搜索**。

已经完成：

- ActivityNet / HMDB51 / UCF101 上的 20 组 `ratio x prob` 搜索。
- ActivityNet 上的早期 `pgafs_margin_ratio` 单参数搜索。
- ActivityNet 上的 PA-HVL 相关 5 组联合参数搜索，包括 `pc_decoder_gamma`、`shvd_weight`、`pgafs_prob`。
- FNS-DCL 的 on/off 主线对比已有结果，但 FNS-DCL 内部参数还没有系统 sweep。

尚未系统完成：

- FCVID 的 20 组 `ratio x prob` 参数搜索。
- FNS-DCL 的 `threshold`、`w_min`、`suppress_scale`、`temperature` 参数敏感性。
- PA-HVL 中 `pc_decoder_gamma` 和 `shvd_weight` 的严格单因素消融。
- 在最终 FNS-DCL 主线上完整的 PCMR on/off、SHVD on/off 阶梯式消融。

## 2. Ablation Inventory

| 模块 | 参数 | 已跑取值 | 数据集 / bit | 是否有 mAP | 主要结果文件 |
|---|---:|---|---|---|---|
| BPMS / BPM-PGAFS | `pgafs_margin_ratio` | `0.5, 0.6, 0.7, 0.8, 0.9` | ActivityNet 16/32/64 | yes | `eval_results/activitynet_bpm_pgafs_ratio_sweep_5gpus_20260611_224958_summary.csv` |
| BPMS + FNS-DCL | `pgafs_margin_ratio` | `0.3, 0.4, 0.5, 0.6, 0.7` | ActivityNet/HMDB/UCF 16/32/64 | yes | `eval_results/non_fcv_sweep20x_all_results_20260625.csv` |
| BPMS + FNS-DCL | `pgafs_prob` | `0.25, 0.35, 0.45, 0.55` | ActivityNet/HMDB/UCF 16/32/64 | yes | `eval_results/non_fcv_sweep20x_all_results_20260625.csv` |
| PA-HVL / PCMR | `pc_decoder_gamma` | `0.03, 0.05, 0.10` | ActivityNet 16/32/64 | yes | `eval_results/activitynet_pcmr_5param_summary.csv` |
| PA-HVL / SHVD | `shvd_weight` | `0.01, 0.02, 0.03` | ActivityNet 16/32/64 | yes | `eval_results/activitynet_pcmr_5param_summary.csv` |
| BPM-PGAFS trigger | `pgafs_prob` | `0.35, 0.50` | ActivityNet 16/32/64 | yes | `eval_results/activitynet_pcmr_5param_summary.csv` |
| FNS-DCL / FNSC | `fns_dcl_enable` | off vs on | ActivityNet/HMDB/UCF/FCVID | yes | `eval_results/bpm_pgafs_shvd_pcdec_fnsdcl_*` |
| FNS-DCL / FNSC | `threshold`, `w_min`, `suppress_scale` | not swept | none | no | not available |

## 3. BPMS / BPM-PGAFS Parameter Studies

### 3.1 Early ActivityNet Ratio-Only Sweep

结果文件：

```text
eval_results/activitynet_bpm_pgafs_ratio_sweep_5gpus_20260611_224958_summary.csv
eval_results/activitynet_bpm_pgafs_ratio_sweep_5gpus_20260611_224958_maps.csv
```

脚本：

```text
scripts/run_activitynet_bpm_pgafs_ratio_sweep_5gpus.sh
scripts/eval_activitynet_bpm_pgafs_ratio_sweep_5gpus.sh
```

已跑参数：

```text
pgafs_margin_ratio = 0.5, 0.6, 0.7, 0.8, 0.9
pgafs_prob = 0.35
shvd_weight = 0.02
pc_decoder_gamma = 0.05
```

对应 tag：

```text
pgafs_margin_balanced_r05_shvd_pcdec_g005_sw002_p035
pgafs_margin_balanced_r06_shvd_pcdec_g005_sw002_p035
pgafs_margin_balanced_r07_shvd_pcdec_g005_sw002_p035
pgafs_margin_balanced_r08_shvd_pcdec_g005_sw002_p035
pgafs_margin_balanced_r09_shvd_pcdec_g005_sw002_p035
```

该实验可以用于说明：在 ActivityNet 上，`pgafs_margin_ratio` 过高时会更接近 margin-only sampling；`r05` 在较深检索指标上更稳定。

### 3.2 Current 20-Combination Ratio-Probability Sweep

主结果文件：

```text
eval_results/non_fcv_sweep20x_all_results_20260625.csv
eval_results/non_fcv_sweep20x_metric_majority_ranking_20260625.csv
eval_results/non_fcv_sweep20x_metric_majority_summary_20260625.md
eval_results/non_fcv_sweep20x_metric_majority_summary_20260625.xlsx
```

已跑参数网格：

```text
pgafs_margin_ratio = 0.3, 0.4, 0.5, 0.6, 0.7
pgafs_prob = 0.25, 0.35, 0.45, 0.55
```

共 20 组组合：

```text
r03_p025, r03_p035, r03_p045, r03_p055
r04_p025, r04_p035, r04_p045, r04_p055
r05_p025, r05_p035, r05_p045, r05_p055
r06_p025, r06_p035, r06_p045, r06_p055
r07_p025, r07_p035, r07_p045, r07_p055
```

比例含义：

| `pgafs_margin_ratio` | margin-hard frames | positive-representative frames |
|---:|---:|---:|
| 0.3 | 30% | 70% |
| 0.4 | 40% | 60% |
| 0.5 | 50% | 50% |
| 0.6 | 60% | 40% |
| 0.7 | 70% | 30% |

训练 / 测试脚本：

```text
scripts/run_activitynet16_bpm_pgafs_ratio_prob_sweep_20x_2gpus_4batches.sh
scripts/run_activitynet32_64_bpm_pgafs_ratio_prob_sweep_20x_2gpus_4batches.sh
scripts/run_activitynet64_bpm_pgafs_ratio_prob_sweep_20x_gpu0.sh
scripts/run_activitynet64_bpm_pgafs_ratio_prob_sweep_20x_gpus0_1_10pergpu.sh
scripts/run_hmdb16_bpm_pgafs_ratio_prob_sweep_gpu0_20x_two_batches.sh
scripts/run_hmdb32_64_bpm_pgafs_ratio_prob_sweep_20x_2gpus_4batches.sh
scripts/run_ucf_bpm_pgafs_ratio_prob_sweep_20x_gpus3_4.sh
scripts/run_ucf_remaining_bpm_pgafs_ratio_prob_sweep_gpu1_5perbatch.sh
scripts/run_ucf64_remaining_batches3_4_gpus2_3_5pergpu.sh
scripts/eval_activitynet16_bpm_pgafs_ratio_prob_sweep_20x.sh
scripts/eval_activitynet64_bpm_pgafs_ratio_prob_sweep_20x.sh
scripts/eval_hmdb16_bpm_pgafs_ratio_prob_sweep_20x.sh
scripts/eval_hmdb32_64_bpm_pgafs_ratio_prob_sweep_20x.sh
scripts/eval_ucf_bpm_pgafs_ratio_prob_sweep_60x.sh
scripts/eval_ucf_complete_sweep_merge_previous_gpus2_3_4.sh
```

覆盖情况：

| dataset | bit | evaluated combos | note |
|---|---:|---:|---|
| ActivityNet | 16 | 20/20 | complete |
| ActivityNet | 32 | 20/20 | complete |
| ActivityNet | 64 | 20/20 | complete |
| HMDB51 | 16 | 20/20 | complete |
| HMDB51 | 32 | 20/20 | complete |
| HMDB51 | 64 | 20/20 | complete |
| UCF101 | 16 | 20/20 | complete |
| UCF101 | 32 | 17/20 ok | 3 incomplete in current CSV |
| UCF101 | 64 | 20/20 | complete |
| FCVID | 16/32/64 | not swept | only fixed r05_p035 FNS-DCL results found |

## 4. Metric-Majority Selected Best Settings

选择标准：不是只看 `mAP@5`，而是看 `mAP@5,20,40,60,80,100` 中领先指标数量和综合排名。

来源：

```text
eval_results/non_fcv_sweep20x_metric_majority_ranking_20260625.csv
best_ckpt/metric_majority_best_ckpt_manifest_20260625.csv
```

| dataset | bit | selected combo | ratio | prob | leading metrics | mAP@5 | mAP@20 | mAP@40 | mAP@60 | mAP@80 | mAP@100 |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| ActivityNet | 16 | `r06_p045` | 0.6 | 0.45 | 5 metrics: 20/40/60/80/100 | 0.18347 | 0.09902 | 0.06052 | 0.04342 | 0.03362 | 0.02747 |
| ActivityNet | 32 | `r07_p035` | 0.7 | 0.35 | 3 metrics: 60/80/100 | 0.25119 | 0.13558 | 0.08089 | 0.05720 | 0.04407 | 0.03591 |
| ActivityNet | 64 | `r03_p025` | 0.3 | 0.25 | 5 metrics: 20/40/60/80/100 | 0.30624 | 0.16767 | 0.10058 | 0.07079 | 0.05434 | 0.04410 |
| HMDB51 | 16 | `r06_p045` | 0.6 | 0.45 | 4 metrics: 5/20/40/60 | 0.17989 | 0.11048 | 0.07741 | 0.05884 | 0.04750 | 0.04078 |
| HMDB51 | 32 | `r05_p045` | 0.5 | 0.45 | 5 metrics: 20/40/60/80/100 | 0.22589 | 0.15134 | 0.10805 | 0.08323 | 0.06741 | 0.05661 |
| HMDB51 | 64 | `r05_p055` | 0.5 | 0.55 | 5 metrics: 20/40/60/80/100 | 0.25971 | 0.18166 | 0.13691 | 0.10875 | 0.08943 | 0.07547 |
| UCF101 | 16 | `r06_p055` | 0.6 | 0.55 | 4 metrics: 40/60/80/100 | 0.43856 | 0.36687 | 0.31311 | 0.26673 | 0.22793 | 0.19728 |
| UCF101 | 32 | `r06_p025` | 0.6 | 0.25 | 2 metrics: 60/100 | 0.54975 | 0.47539 | 0.41625 | 0.36660 | 0.31650 | 0.27230 |
| UCF101 | 64 | `r06_p055` | 0.6 | 0.55 | 6 metrics: all | 0.59089 | 0.51753 | 0.46095 | 0.41235 | 0.36318 | 0.31654 |

## 5. Best Checkpoint Collection

当前最佳权重已经集中复制到：

```text
best_ckpt/
```

清单文件：

```text
best_ckpt/metric_majority_best_ckpt_manifest_20260625.csv
best_ckpt/metric_majority_best_ckpt_manifest_20260625.md
```

已汇聚权重：

```text
best_ckpt/activitynet_16bit_r06_p045_metric_majority.pth
best_ckpt/activitynet_32bit_r07_p035_metric_majority.pth
best_ckpt/activitynet_64bit_r03_p025_metric_majority.pth
best_ckpt/fcv_16bit_r05_p035_metric_majority.pth
best_ckpt/fcv_32bit_r05_p035_metric_majority.pth
best_ckpt/fcv_64bit_r05_p035_metric_majority.pth
best_ckpt/hmdb_16bit_r06_p045_metric_majority.pth
best_ckpt/hmdb_32bit_r05_p045_metric_majority.pth
best_ckpt/hmdb_64bit_r05_p055_metric_majority.pth
best_ckpt/ucf_16bit_r06_p055_metric_majority.pth
best_ckpt/ucf_32bit_r06_p025_metric_majority.pth
best_ckpt/ucf_64bit_r06_p055_metric_majority.pth
```

FCVID 当前不是 20 组 sweep 选出的 best，而是固定 `r05_p035` 的 FNS-DCL 主设置结果。

FCVID source checkpoint：

```text
checkpoint/fcv/AutoSSVH_16bit_bpm_pgafs_shvd_pcdec_fnsdcl_r05_g005_sw002/run_20260615_114544_243599/fcv_16.pth
checkpoint/fcv/AutoSSVH_32bit_bpm_pgafs_shvd_pcdec_fnsdcl_r05_g005_sw002/run_20260615_114544_243730/fcv_32.pth
checkpoint/fcv/AutoSSVH_64bit_bpm_pgafs_shvd_pcdec_fnsdcl_r05_g005_sw002/run_20260615_114544_243474/fcv_64.pth
```

## 6. PA-HVL / SHVD / PCMR Parameter Study

PA-HVL 相关参数主要通过 ActivityNet 5 组联合实验完成。

结果文件：

```text
eval_results/activitynet_pcmr_5param_summary.csv
eval_results/activitynet_pcmr_5param_maps.csv
```

脚本：

```text
scripts/run_activitynet_pcmr_5param_5gpus.sh
scripts/eval_activitynet_pcmr_5param_5gpus.sh
```

参数组：

| group/tag | `pgafs_prob` | `shvd_weight` | `pc_decoder_gamma` | note |
|---|---:|---:|---:|---|
| `g1_gamma005_sw003_p050` | 0.50 | 0.03 | 0.05 | higher SHVD weight |
| `g2_gamma003_sw003_p050` | 0.50 | 0.03 | 0.03 | lower PCMR gamma |
| `g3_gamma010_sw001_p050` | 0.50 | 0.01 | 0.10 | higher gamma, lower SHVD |
| `g4_gamma005_sw002_p050` | 0.50 | 0.02 | 0.05 | middle SHVD, high PGAFS probability |
| `g5_gamma005_sw002_p035` | 0.35 | 0.02 | 0.05 | selected main base setting |

Important note:

```text
This is a coupled parameter screening, not a strict one-factor-at-a-time ablation.
```

覆盖取值：

```text
pc_decoder_gamma = 0.03, 0.05, 0.10
shvd_weight = 0.01, 0.02, 0.03
pgafs_prob = 0.35, 0.50
```

`g5_gamma005_sw002_p035` 在 ActivityNet 16/32/64 上整体最稳定，因此后续主线使用：

```text
pgafs_prob = 0.35
shvd_weight = 0.02
pc_decoder_gamma = 0.05
```

## 7. FNS-DCL / FNSC Parameter Status

当前 FNS-DCL 主配置：

```text
fns_dcl_enable = True
fns_dcl_temperature = 0.2
fns_dcl_threshold = 0.7
fns_dcl_min_weight = 0.2
fns_dcl_suppress_scale = 0.05
fns_dcl_detach_weight = True
fns_dcl_symmetric = True
```

主配置文件：

```text
configs/AutoSSVH_act_bpm_pgafs_shvd_pcdec_fnsdcl.py
configs/AutoSSVH_fcv_bpm_pgafs_shvd_pcdec_fnsdcl.py
configs/AutoSSVH_hmdb_bpm_pgafs_shvd_pcdec_fnsdcl.py
configs/AutoSSVH_ucf_bpm_pgafs_shvd_pcdec_fnsdcl.py
```

FNS-DCL 已有结果：

```text
eval_results/bpm_pgafs_shvd_pcdec_fnsdcl_activitynet_3bits_20260615_102147_fnsdcl_summary.csv
eval_results/bpm_pgafs_shvd_pcdec_fnsdcl_activitynet_hmdb_ucf_gpu2_3bits_20260615_212452.csv
eval_results/bpm_pgafs_shvd_pcdec_fnsdcl_fcv_3bits_20260616_025520.csv
```

已经支持的结论：

- `fns_dcl_enable=True` 已作为当前主方法运行。
- 可以和无 FNS-DCL 的 BPM-PGAFS + SHVD + PCMR baseline 做 on/off 对比。
- FNS-DCL 是替换原始 `L_vc = dcl(...)`，不是额外增加一个 loss term。

还没有系统 sweep 的参数：

```text
fns_dcl_threshold
fns_dcl_min_weight
fns_dcl_suppress_scale
fns_dcl_temperature
fns_dcl_detach_weight
fns_dcl_symmetric
```

命名注意：

```text
当前 tag 中的 sw002 指 shvd_weight = 0.02，不是 FNS-DCL suppression weight。
```

## 8. Parameters Not Yet Systematically Ablated

| 模块 | 缺失项 | 当前状态 |
|---|---|---|
| FCVID BPMS | `pgafs_margin_ratio x pgafs_prob` 20 组搜索 | not found |
| FNS-DCL | `threshold` sensitivity | not found |
| FNS-DCL | `w_min` sensitivity | not found |
| FNS-DCL | `suppress_scale` / smoothness sensitivity | not found |
| FNS-DCL | `temperature` sensitivity | not found |
| PA-HVL / SHVD | strict one-factor `shvd_weight` sweep | only coupled 5-param screening |
| PA-HVL / PCMR | strict one-factor `pc_decoder_gamma` sweep | only coupled 5-param screening |
| PA-HVL / PCMR | `pc_decoder_enable` on/off under final FNS-DCL setting | not complete |
| Full modules | BPMS-only -> BPMS+SHVD -> BPMS+SHVD+PCMR -> +FNS-DCL | partially supported, needs organized table |

## 9. Recommended Paper Sensitivity Experiments

### 9.1 Main Sensitivity: BPMS Balance Ratio

Recommended to write:

```text
We vary rho in BPMS to control the proportion of margin-hard frames and positive-representative frames.
```

Use:

```text
rho = 0.3, 0.4, 0.5, 0.6, 0.7
```

Report:

- ActivityNet 16/32/64
- HMDB51 16/32/64
- UCF101 16/32/64

Selection should use metric-majority across:

```text
mAP@5, mAP@20, mAP@40, mAP@60, mAP@80, mAP@100
```

Do not select only by `mAP@5`.

### 9.2 Main Sensitivity: BPMS Trigger Probability

Recommended to write:

```text
We vary pgafs_prob to control how often the second view is reconstructed as a semantic-hard view.
```

Use:

```text
pgafs_prob = 0.25, 0.35, 0.45, 0.55
```

Claim boundary:

- Moderate probability is often more stable.
- Too frequent semantic-hard construction may over-strengthen prototype pressure.

### 9.3 PA-HVL Hyperparameter Screening

Can write as:

```text
We conduct a small hyperparameter screening for the SHVD weight and the prototype-conditioned decoder strength on ActivityNet.
```

Use:

```text
eval_results/activitynet_pcmr_5param_summary.csv
```

Avoid saying this is a strict independent sensitivity curve because multiple parameters change together.

### 9.4 FNS-DCL On/Off Ablation

Recommended:

```text
Compare BPM-PGAFS + SHVD + PCMR with and without FNS-DCL.
```

Useful for module contribution:

```text
FNS-DCL suppresses likely false negatives in the view contrastive denominator and improves selected settings, especially in top-ranked retrieval or specific code lengths.
```

Claim boundary:

- Do not claim universal all-depth improvement unless the specific table supports it.
- State clearly that FNS-DCL replaces DCL in `L_vc`, rather than adding an independent loss.

## 10. Safe Paper Claims

Safe claims:

1. BPMS has been systematically evaluated with multiple `pgafs_margin_ratio` and `pgafs_prob` values on ActivityNet, HMDB51, and UCF101.
2. The best `ratio/prob` setting is dataset- and bit-dependent.
3. Selecting by metric-majority across retrieval depths is more stable than selecting only by `mAP@5`.
4. `pgafs_margin_ratio` controls the balance between margin-hard frames and positive-representative frames.
5. The current main setting uses `shvd_weight=0.02` and `pc_decoder_gamma=0.05`, supported by an ActivityNet hyperparameter screening.
6. FNS-DCL has been evaluated as an on/off module, but its internal parameters have not yet been systematically swept.

Unsafe or not yet supported claims:

1. FNS-DCL threshold is optimal at `0.7`.
2. `w_min=0.2` is globally optimal.
3. `suppress_scale=0.05` is globally optimal.
4. `pc_decoder_gamma=0.05` is globally optimal across all datasets.
5. `shvd_weight=0.02` is globally optimal across all datasets.
6. FCVID has completed the same 20-combination BPMS sweep as the other datasets.
7. FNS-DCL improves every retrieval depth on every dataset and bit.

## 11. Recommended Next Experiments

If more ablation is needed for IEEE Transactions, priority order:

1. FNS-DCL threshold sweep on ActivityNet and UCF101:

```text
fns_dcl_threshold = 0.65, 0.70, 0.75, 0.80
```

2. FNS-DCL minimum negative weight sweep:

```text
fns_dcl_min_weight = 0.1, 0.2, 0.4, 0.6
```

3. FNS-DCL smoothness sweep:

```text
fns_dcl_suppress_scale = 0.03, 0.05, 0.08, 0.10
```

4. PCMR gamma strict single-factor sweep:

```text
pc_decoder_gamma = 0.03, 0.05, 0.07, 0.10
```

5. SHVD weight strict single-factor sweep:

```text
shvd_weight = 0.01, 0.02, 0.03, 0.05
```

6. FCVID 20-combination BPMS sweep if compute allows:

```text
pgafs_margin_ratio = 0.3, 0.4, 0.5, 0.6, 0.7
pgafs_prob = 0.25, 0.35, 0.45, 0.55
```

## 12. Key Files

Result files:

```text
eval_results/non_fcv_sweep20x_all_results_20260625.csv
eval_results/non_fcv_sweep20x_metric_majority_ranking_20260625.csv
eval_results/non_fcv_sweep20x_metric_majority_summary_20260625.md
eval_results/activitynet_bpm_pgafs_ratio_sweep_5gpus_20260611_224958_summary.csv
eval_results/activitynet_pcmr_5param_summary.csv
eval_results/bpm_pgafs_shvd_pcdec_fnsdcl_activitynet_3bits_20260615_102147_fnsdcl_summary.csv
eval_results/bpm_pgafs_shvd_pcdec_fnsdcl_activitynet_hmdb_ucf_gpu2_3bits_20260615_212452.csv
eval_results/bpm_pgafs_shvd_pcdec_fnsdcl_fcv_3bits_20260616_025520.csv
```

Best checkpoint manifest:

```text
best_ckpt/metric_majority_best_ckpt_manifest_20260625.csv
best_ckpt/metric_majority_best_ckpt_manifest_20260625.md
```

Main sweep scripts:

```text
scripts/run_activitynet16_bpm_pgafs_ratio_prob_sweep_20x_2gpus_4batches.sh
scripts/run_activitynet32_64_bpm_pgafs_ratio_prob_sweep_20x_2gpus_4batches.sh
scripts/run_hmdb16_bpm_pgafs_ratio_prob_sweep_gpu0_20x_two_batches.sh
scripts/run_hmdb32_64_bpm_pgafs_ratio_prob_sweep_20x_2gpus_4batches.sh
scripts/run_ucf_bpm_pgafs_ratio_prob_sweep_20x_gpus3_4.sh
scripts/run_ucf_remaining_bpm_pgafs_ratio_prob_sweep_gpu1_5perbatch.sh
scripts/eval_ucf_complete_sweep_merge_previous_gpus2_3_4.sh
scripts/run_activitynet_pcmr_5param_5gpus.sh
scripts/eval_activitynet_pcmr_5param_5gpus.sh
```

## 13. WandB / TensorBoard

No obvious WandB or TensorBoard experiment directory was found from the project root during inspection.

Checked patterns:

```text
wandb/
runs/
tensorboard/
tb_logs/
```


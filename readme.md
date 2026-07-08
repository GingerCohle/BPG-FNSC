# PABSHVL-FNSC for Self-Supervised Video Hashing

This is the cleaned release repository for:

**Prototype-Aware Balanced Semantic-Hard View Learning with False-Negative Suppressed Contrast**

Paper module names:

- **BPMS**: Balanced Prototype-Margin Sampling
- **PA-HVL**: Prototype-Aware Hard-View Learning
- **FNSC**: False-Negative Suppressed Contrast

The implementation is checkpoint-compatible with the internal development names:

- `BPM-PGAFS` / `proto_margin_balanced` -> **BPMS**
- `SHVD + PCMR` -> **PA-HVL**
- `FNS-DCL` / `dcl_fns` -> **FNSC**

## 1. Environment

The code was tested with Python 3.11 and CUDA 11.8 PyTorch wheels.

```bash
conda create -n hash python=3.11 -y
conda activate hash

pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 2. Dataset Layout

Datasets are not stored in this repository. Link the four feature folders into `data/`:

```bash
bash scripts/link_datasets.sh /media/kejunjie_coco/autohash
```

Expected layout:

```text
data/activitynet/train_feats.h5
data/activitynet/test_feats.h5
data/activitynet/query_feats.h5
data/activitynet/re_label.mat
data/activitynet/q_label.mat

data/fcv/fcv_train_feats.h5
data/fcv/fcv_test_feats.h5
data/fcv/fcv_test_labels.mat

data/ucf/ucf_train_feats.h5
data/ucf/ucf_test_feats.h5
data/ucf/ucf_train_labels.mat
data/ucf/ucf_test_labels.mat

data/hmdb/hmdb_train_feats.h5
data/hmdb/hmdb_test_feats.h5
data/hmdb/hmdb_train_labels.mat
data/hmdb/hmdb_test_labels.mat
```

## 3. Best Checkpoints

The `best_ckpt/` folder contains 12 metric-majority selected checkpoints:

- ActivityNet: 16, 32, 64 bits
- FCVID: 16, 32, 64 bits
- UCF101: 16, 32, 64 bits
- HMDB51: 16, 32, 64 bits

See `docs/BEST_CKPT_PARAMETER_MAPPING.md` and
`best_ckpt/metric_majority_best_ckpt_manifest_20260625.csv` for the exact
`rho`, `pgafs_prob`, checkpoint path, and mAP metrics.

## 4. Evaluate / Inference with Best Checkpoints

Dry run:

```bash
DRY_RUN=1 GPU=0 bash scripts/evaluate_best_ckpt_all.sh
```

Run all 12 checkpoints on GPU 0:

```bash
GPU=0 bash scripts/evaluate_best_ckpt_all.sh
```

Run a subset:

```bash
DATASETS="activitynet ucf" BITS="32 64" GPU=0 bash scripts/evaluate_best_ckpt_all.sh
```

The output CSV is written to:

```text
eval_results/best_ckpt_BPMS_PAHVL_FNSC_<RUN_ID>.csv
```

This evaluation path performs inference with the encoder-hash branch, binarizes
the continuous representation with `sign`, computes Hamming distance, and reports
mAP@5, mAP@20, mAP@40, mAP@60, mAP@80, and mAP@100.

## 5. Train One Setting

The clean release also keeps `train.py`. To train one paper-named setting:

```bash
DRY_RUN=1 bash scripts/train_one_setting.sh configs/best_ckpt/AutoSSVH_activitynet_16bit_BPMS_PAHVL_FNSC.py 0
bash scripts/train_one_setting.sh configs/best_ckpt/AutoSSVH_activitynet_16bit_BPMS_PAHVL_FNSC.py 0
```

The configs under `configs/best_ckpt/` are the metric-majority selected settings.
They can be used for either training or checkpoint evaluation.

## 6. Notes for GitHub Upload

The checkpoint files are around 46-50 MB each. They are below GitHub's single-file
100 MB limit, but the total repository size is large. Git LFS is recommended if
you plan to keep all checkpoints in the public repository.

```bash
git lfs install
git lfs track "best_ckpt/*.pth"
```

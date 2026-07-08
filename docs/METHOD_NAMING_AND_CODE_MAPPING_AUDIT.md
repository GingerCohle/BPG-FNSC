# METHOD_NAMING_AND_CODE_MAPPING_AUDIT

## 1. 执行摘要

本次审计对象为 `/media/kejunjie_coco/autohash/autossh_newwek/BPM-PGAFS+SHVD+PCMR`。当前活动方法代码 tag 是 `bpm_pgafs_shvd_pcdec_fnsdcl_r05_g005_sw002`，对应实现为：

```text
BPM-PGAFS + SHVD + PCMR + FNS-DCL
```

建议论文中组织为三个更高层的方法贡献：

1. `BPMS`，Balanced Prototype-Margin Sampling：对应代码中的 `BPM-PGAFS` / `pgafs_score_type="proto_margin_balanced"`。
2. `PA-HVL`，Prototype-Aware Hard-View Learning：对应代码中的 `SHVD + PCMR` 协同机制。
3. `FNSC`，False-Negative Suppressed Contrast：对应代码中的 `FNS-DCL` / `dcl_fns(...)`。

代码事实与论文命名基本一致。当前方法没有修改 backbone、dataset loading、mAP evaluation、`Loss_CVH(...)`、`run_kmeans(...)`、`run_finch(...)`。BPMS、SHVD、PCMR 和 FNSC 都发生在训练阶段；推理阶段仍使用 `model.inference(...)` 生成二值哈希码，并按原始 Hamming retrieval 协议评估。

需要注意的安全表述边界：FNSC 在 ActivityNet 32-bit 和 64-bit 上提升了 mAP@5、mAP@20，但 mAP@40/60/80/100 相比 r05 baseline 略降；不能宣称“所有检索深度均提升”。

## 2. 论文命名与代码命名对应表

| Paper term | Short name | Code term | Main file/function | Config fields | Paper role |
|---|---|---|---|---|---|
| Balanced Prototype-Margin Sampling | BPMS | BPM-PGAFS / `proto_margin_balanced` | `Loss/AutoSSVH_loss.py:177-320` `build_pgafs_mask_and_labels(...)`; `model/AutoSSVH.py:639-649` `get_frame_hash_tokens(...)` | `pgafs_score_type`, `pgafs_margin_ratio`, `pgafs_prob`, `pgafs_view` | 构造平衡的 semantic-hard view2 mask |
| Prototype-Aware Hard-View Learning | PA-HVL | SHVD + PCMR | SHVD: `Loss/AutoSSVH_loss.py:323-355`; PCMR: `model/AutoSSVH.py:545-552`, `611-624`; criterion: `Loss/AutoSSVH_loss.py:401-429`, `463-481` | `shvd_*`, `pc_decoder_*` | 用 prototype distribution distillation 和 prototype-conditioned decoder 稳定 hard view 学习 |
| False-Negative Suppressed Contrast | FNSC | FNS-DCL / `dcl_fns` | `Loss/AutoSSVH_loss.py:44-136`; criterion gate: `Loss/AutoSSVH_loss.py:440-453` | `fns_dcl_*` | 替换原 `L_vc=dcl(...)`，抑制潜在伪负样本 |

旧代码名仍会出现在脚本和配置中：`BPM-PGAFS`、`SHVD`、`PCMR` / `pcdecoder`、`FNS-DCL`、`dcl_fns`。论文名是更高层的组织方式，代码名是实现名。

## 3. 当前完整训练流程

训练入口为 `train.py`。每个 epoch 中，warmup 后先用训练集特征聚类：

- `train.py:108-122`：`compute_features(...)` 后根据 `cluster_method` 调用 `run_kmeans(...)` 或 `run_finch(...)`。
- 当前 FNSC 主配置继承 balanced r05 config，`cluster_method="kmeans"`。

核心 batch 级训练流程在 `Loss/AutoSSVH_loss.py:370-494` 的 `AutoSSVH_criterion(...)`：

1. 输入 `data["visual_word"]`、`data["mask"]`、`data["index"]`，全部移动到 GPU。
2. `bool_masked_pos_1 = data["mask"][:,0,:]`，`bool_masked_pos_2 = data["mask"][:,1,:]`。
3. `labels_1 = visual_word[bool_masked_pos_1]`，shape 为 `[B, M, feature_size]`。
4. BPMS gate：
   `pgafs_enable and cluster_result is not None and epoch >= warmup_epoch and pgafs_view in ["view2","both"]`，再以 `pgafs_prob` 随机触发。
5. 若 BPMS 触发，调用 `build_pgafs_mask_and_labels(...)` 替换 view2 mask，并重新生成 `labels_2`；否则使用原始 random view2 mask。
6. PCMR gate：
   `pc_decoder_enable and cluster_result is not None and epoch >= warmup_epoch and pc_decoder_view in ["view2","both"]`。若 `pc_decoder_only_pgafs_triggered=True`，还必须 `pgafs_triggered=True`。
7. 若 PCMR gate 通过，通过 `get_positive_prototype(...)` 得到 `proto_cond_2`。
8. Forward view1：`model.forward(visual_word, bool_masked_pos_1)`，不使用 PCMR。
9. Forward view2：`model.forward(..., proto_cond=proto_cond_2, pc_decoder_enable=use_pc_decoder_view2, pc_decoder_gamma=..., pc_decoder_detach_proto=...)`。
10. 对两路 frame-level hash tokens 做 mean pooling：`hash_code_1 = mean(hash_code_1, 1)`，`hash_code_2 = mean(hash_code_2, 1)`。
11. Reconstruction loss：`MSE(frame_1, labels_1) + MSE(frame_2, labels_2)`。
12. View contrast loss：若 `fns_dcl_enable=True`，`loss_vc=dcl_fns(...)`；否则保留原 `dcl(...)`。
13. CVH loss：若 `cfg.CVH and cluster_result is not None`，对两路 hash code 分别调用原 `Loss_CVH(...)` 后取平均。
14. SHVD gate：
    `shvd_enable and pgafs_enable and cluster_result is not None and epoch >= warmup_epoch`；若 `shvd_only_pgafs_triggered=True`，还要求 `pgafs_triggered=True`。
15. 总损失：
    `loss = recon_loss + cfg.a * loss_vc + cfg.b * loss_cvh + shvd_weight * loss_shvd`。

## 4. BPMS 对应代码与公式

代码位置：

- `Loss/AutoSSVH_loss.py:177-320`，`build_pgafs_mask_and_labels(...)`
- `model/AutoSSVH.py:639-649`，`get_frame_hash_tokens(...)`

函数签名：

```python
def build_pgafs_mask_and_labels(
    visual_word,
    index,
    cluster_result,
    original_mask,
    cfg,
    device,
    model,
):
```

主要 shape：

- `visual_word`: `[B, T, feature_size]`
- `original_mask`: `[B, T]`，bool，`True` 表示 masked frame
- `frame_tokens`: `[B, T, nbits]`
- `centroids`: `[K, nbits]`
- 输出 `pgafs_mask`: `[B, T]`
- 输出 `pgafs_labels`: `[B, M, feature_size]`

实现细节：

- `frame_tokens` 通过 `model.get_frame_hash_tokens(visual_word)` 得到，并且在 `torch.no_grad()` 下调用。
- `get_frame_hash_tokens(...)` 内部直接调用 `self.inference(x)`，即无 mask 的 encoder/hash-token 路径。
- `pos_proto_id = cluster_result["im2cluster"][level][index]`。
- `centroids = cluster_result["centroids"][level]`。
- `pos_sim` 是 frame token 与 positive prototype 的 cosine similarity。
- `hard_neg_sim` 是 frame token 与除 positive prototype 外所有 prototypes 的最大 cosine similarity。

论文公式可写为：

```text
s_pos(i,t) = cos(h_{i,t}, p_i)
s_margin(i,t) = cos(h_{i,t}, p_i) - max_{k != y_i} cos(h_{i,t}, p_k)

M_margin = round(M * rho)
M_pos = M - M_margin

S_margin = Top-M_margin(s_margin)
S_pos = Top-M_pos(s_pos | t not in S_margin)
S_mask = S_margin union S_pos
```

其中 `rho = pgafs_margin_ratio`，当前主配置为 `rho=0.5`。

与旧模式的区别：

- `proto_sim`：只用 `s_pos` 选 Top-M。
- `proto_margin`：只用 `s_margin` 选 Top-M。
- `proto_margin_balanced`：把 M 个 mask budget 拆成 `M_margin` 和 `M_pos`，先选 margin-hard frames，再在剩余位置中选 positive-representative frames。

边界处理：

- 若 `K <= 1`，`proto_margin_balanced` fallback 到 positive-only score，即只使用 `pos_sim`。
- `M` 来自原始 mask 的 masked count；若 batch 内 mask 数不一致，直接报错。
- margin 选择后，代码用 `scatter_(1, margin_idx, -1e4)` 屏蔽已选位置，再做 positive top-k，避免 overlap。
- 构造 mask 后检查 `pgafs_mask.sum(dim=1) == M`，保证每个样本 exactly M masked frames。
- 当前 `AutoSSVH_criterion(...)` 只替换 view2 mask，并重新 gather `labels_2`。

启用配置：

- `configs/AutoSSVH_act_pgafs_shvd_pcmr_margin_balanced.py`
- `configs/AutoSSVH_fcv_pgafs_shvd_pcmr_margin_balanced.py`
- `configs/AutoSSVH_hmdb_pgafs_shvd_pcmr_margin_balanced.py`
- `configs/AutoSSVH_ucf_pgafs_shvd_pcmr_margin_balanced.py`
- FNSC 版本继承这些 balanced r05 configs。

## 5. PA-HVL 对应代码与公式

PA-HVL 建议在论文中作为 `SHVD + PCMR` 的协同机制来写：SHVD 从 prototype distribution 层面对齐 random view 和 semantic-hard view；PCMR 从 decoder reconstruction 路径把 positive prototype 注入 view2 的 masked token。二者都围绕 BPMS 构造出的 semantic-hard view2 生效。

### 5.1 SHVD

代码位置：

- `Loss/AutoSSVH_loss.py:323-355`，`compute_shvd_loss(...)`
- gate 和 loss assembly：`Loss/AutoSSVH_loss.py:463-481`

函数签名：

```python
def compute_shvd_loss(
    hash_code_teacher,
    hash_code_student,
    cluster_result,
    cfg,
    device,
):
```

代码行为：

- teacher 是 view1 random-mask video hash：`hash_code_1`。
- student 是 view2 BPMS semantic-hard video hash：`hash_code_2`。
- `hash_code_teacher`、`hash_code_student` 和 prototypes 都会 `F.normalize(..., dim=1)`。
- `teacher_logits = hash_code_teacher @ prototypes.T / shvd_tau`。
- `student_logits = hash_code_student @ prototypes.T / shvd_tau`。
- 若 `shvd_stopgrad_teacher=True`，代码对 `teacher_logits.detach()` 做 softmax。
- KL 方向是 `KL(q_teacher || q_student)`，代码形式为 `F.kl_div(log_q_student, q_teacher, reduction="batchmean")`。

论文公式：

```text
q_t = softmax(h_1 P^T / tau)
q_s = softmax(h_2 P^T / tau)
L_SHVD = KL(stopgrad(q_t) || q_s)
```

gate：

```text
shvd_enable
and pgafs_enable
and cluster_result is not None
and epoch >= warmup_epoch
and (not shvd_only_pgafs_triggered or pgafs_triggered)
```

当前配置：

```text
shvd_enable = True
shvd_weight = 0.02
shvd_tau = 0.2
shvd_cluster_level = -1
shvd_stopgrad_teacher = True
shvd_only_pgafs_triggered = True
```

结论：SHVD 增加一个训练 loss term，不改模型结构，不影响推理。

### 5.2 PCMR

代码位置：

- `model/AutoSSVH.py:545-552`，`pc_decoder_proto_proj`
- `model/AutoSSVH.py:583-631`，`forward(...)`
- `Loss/AutoSSVH_loss.py:358-365`，`get_positive_prototype(...)`
- gate 与 view2 forward：`Loss/AutoSSVH_loss.py:401-429`

`pc_decoder_proto_proj` 结构：

```python
nn.Sequential(
    nn.Linear(cfg.nbits, decoder_embed_dim),
    nn.GELU(),
    nn.LayerNorm(decoder_embed_dim),
    nn.Linear(decoder_embed_dim, decoder_embed_dim),
)
```

因此 PCMR 会增加训练模型中的可学习参数。若评估 checkpoint 来自启用 PCMR 的训练，评估 config 也需要 `pc_decoder_enable=True` 来初始化相同结构，否则 `state_dict` 不匹配。

forward 相关参数：

```python
proto_cond=None
pc_decoder_enable=False
pc_decoder_gamma=0.1
pc_decoder_detach_proto=True
```

PCMR 公式：

```text
m_i^p = m + gamma * W_p p_i
```

代码对应：

```python
proto_embed = self.pc_decoder_proto_proj(proto_cond)
mask_token = self.mask_token + float(pc_decoder_gamma) * proto_embed[:, None, :]
```

当前 gate：

```text
pc_decoder_enable
and cluster_result is not None
and epoch >= warmup_epoch
and pc_decoder_view in ["view2", "both"]
and (not pc_decoder_only_pgafs_triggered or pgafs_triggered)
```

当前配置：

```text
pc_decoder_enable = True
pc_decoder_gamma = 0.05
pc_decoder_cluster_level = -1
pc_decoder_detach_proto = True
pc_decoder_view = "view2"
pc_decoder_only_pgafs_triggered = True
```

结论：

- PCMR 只修改 decoder mask token。
- PCMR 不新增独立 loss。
- 当前 criterion 中只有 view2 使用 `proto_cond_2`；view1 不使用 PCMR。
- 推理阶段调用 `model.inference(...)`，不走 decoder mask token conditioning，因此 PCMR 不改变 Hamming retrieval 推理流程。

## 6. FNSC 对应代码与公式

代码位置：

- 原始 DCL：`Loss/AutoSSVH_loss.py:21-41`
- FNS one-direction：`Loss/AutoSSVH_loss.py:44-94`
- FNS symmetric wrapper：`Loss/AutoSSVH_loss.py:97-136`
- criterion gate：`Loss/AutoSSVH_loss.py:440-453`

原始 DCL 签名：

```python
def dcl(out_1, out_2, batch_size, temperature=0.5, tau_plus=0.1, device=None):
```

原始 DCL 行为：

- 先 normalize `out_1` 和 `out_2`。
- 拼成 `[2B, D]` 后计算 full similarity matrix。
- 通过 `get_negative_mask(batch_size)` 去除 self 和 paired positive。
- positive 是 `out_1[i]` 与 `out_2[i]`，并复制成两个方向。
- 使用 `tau_plus` 做 debiased negative correction：
  `Ng = (-tau_plus * N * pos + neg.sum) / (1 - tau_plus)`。
- 输出是 symmetric batch loss，因为同时包含 view1->view2 和 view2->view1。

FNSC/FNS-DCL 签名：

```python
def _fns_contrast_one_direction(
    anchor,
    target,
    temperature=0.2,
    threshold=0.7,
    min_weight=0.2,
    suppress_scale=0.05,
    detach_weight=True,
)

def dcl_fns(
    z1,
    z2,
    temperature=0.2,
    threshold=0.7,
    min_weight=0.2,
    suppress_scale=0.05,
    detach_weight=True,
    symmetric=True,
)
```

FNSC 公式：

```text
sim_ij = cos(z_i, z_j)
w_ij = 1 - sigmoid((sim_ij - theta) / s) * (1 - w_min)

L_i = -log(
    exp(sim_ii / tau)
    /
    sum_j w_ij exp(sim_ij / tau)
)

L_FNSC = 0.5 * (L_{z1 -> z2} + L_{z2 -> z1})
```

代码行为：

- `z1/z2` 会 normalize。
- `sim = anchor @ target.T`。
- `logits = sim / tau`。
- 若 `detach_weight=True`，权重由 `sim.detach()` 计算，避免模型直接利用权重函数。
- diagonal positive weight 被强制设为 1。
- denominator 通过 `weighted_logits = logits + log(weights)`，再 `torch.logsumexp(...)` 实现，数值稳定。
- `B <= 1`、feature dim mismatch、non-2D tensor 会安全返回 zero loss。
- `dcl_fns(..., symmetric=True)` 默认对两个方向取平均。
- FNS-DCL 不使用原始 DCL 的 `tau_plus`。

当前配置：

```text
fns_dcl_enable = True
fns_dcl_temperature = 0.2
fns_dcl_threshold = 0.7
fns_dcl_min_weight = 0.2
fns_dcl_suppress_scale = 0.05
fns_dcl_detach_weight = True
fns_dcl_symmetric = True
```

在 `AutoSSVH_criterion(...)` 中：

```text
if fns_dcl_enable:
    L_vc = dcl_fns(...)
else:
    L_vc = dcl(...)
```

结论：FNSC 是对 video-level view contrastive loss 的替换增强，不是额外增加一个 loss term。

## 7. 总损失与推理流程

当前代码中的总目标：

```text
L = L_rec + a * L_vc + b * L_CVH + shvd_weight * L_SHVD
```

对应代码：`Loss/AutoSSVH_loss.py:481`。

各项含义：

- `L_rec`：`MSE(frame_1, labels_1) + MSE(frame_2, labels_2)`。
- `L_vc`：若 `fns_dcl_enable=True`，为 FNSC/FNS-DCL；否则为原始 `dcl(...)`。
- `L_CVH`：原始 prototype contrastive hashing loss，代码未改。
- `L_SHVD`：PA-HVL 中的 semantic-hard view distillation loss。

不额外加 loss 的模块：

- BPMS 只改变 view2 mask sampling。
- PCMR 只改变 view2 decoder mask token。
- FNSC 替换 `L_vc`，不是新增 loss。

推理流程：

- `scripts/eval_checkpoint_maps.py` 加载 config 和 checkpoint，然后调用 `eval.evaluate(..., return_all=True)`。
- `eval.py:79-208` 中 `evaluate(...)` 对 test/query 数据调用 `get_inference(...)`。
- `inference/AutoSSVH_inference.py:4-10` 调用 `model.inference(data["visual_word"])`，mean pooling 后 `torch.sign(...)` 得到 binary code。
- Hamming distance 仍为 `0.5 * (-dot(test_hashcode, query_hashcode.T) + nbits)`。
- 指标仍为 mAP@5/20/40/60/80/100。

推理阶段不使用：

- BPMS mask selection。
- SHVD loss。
- PCMR decoder conditioning。
- FNSC/FNS-DCL。

因此当前方法不改变 Hamming retrieval protocol。

## 8. Config / script / tag 对应关系

### 8.1 Config 表

| Dataset | FNSC config | Base balanced config | Tag path suffix | BPMS params | PA-HVL params | FNSC params |
|---|---|---|---|---|---|---|
| ActivityNet | `configs/AutoSSVH_act_bpm_pgafs_shvd_pcdec_fnsdcl.py` | `AutoSSVH_act_pgafs_shvd_pcmr_margin_balanced.py` | `bpm_pgafs_shvd_pcdec_fnsdcl_r05_g005_sw002` | `proto_margin_balanced`, ratio `0.5`, prob `0.35` | `shvd_weight=0.02`, `shvd_tau=0.2`, `pc_decoder_gamma=0.05` | enable, tau `0.2`, threshold `0.7`, min weight `0.2`, scale `0.05`, symmetric |
| FCVID/FCV | `configs/AutoSSVH_fcv_bpm_pgafs_shvd_pcdec_fnsdcl.py` | `AutoSSVH_fcv_pgafs_shvd_pcmr_margin_balanced.py` | same | same | same | same |
| HMDB | `configs/AutoSSVH_hmdb_bpm_pgafs_shvd_pcdec_fnsdcl.py` | `AutoSSVH_hmdb_pgafs_shvd_pcmr_margin_balanced.py` | same | same | same | same |
| UCF101 | `configs/AutoSSVH_ucf_bpm_pgafs_shvd_pcdec_fnsdcl.py` | `AutoSSVH_ucf_pgafs_shvd_pcmr_margin_balanced.py` | same | same | same | same |

这些基础 config 继承各自 dataset config 的默认 `nbits`，训练/评估脚本会在 `configs/generated/` 中生成 bit-specific config。

### 8.2 Script 表

| Script | 作用 | Dataset / bits | GPU 默认 | 输出 |
|---|---|---|---|---|
| `scripts/run_bpm_pgafs_shvd_pcdec_fnsdcl_activitynet_3bits.sh` | 训练 ActivityNet FNSC 三个 bit | ActivityNet 16/32/64 | 3/4/5 并发 | `checkpoint/activitynet/AutoSSVH_{bit}bit_bpm_pgafs_shvd_pcdec_fnsdcl_r05_g005_sw002/...`，日志在 `run_logs/bpm_pgafs_shvd_pcdec_fnsdcl_activitynet_3bits/<RUN_ID>/` |
| `scripts/run_fnsdcl_fcv_hmdb_ucf_selected_gpus.sh` | 训练 FCV/HMDB/UCF | FCV 16/32/64；HMDB 16/32/64；UCF 16/32/64 | FCV: 3/4/5；HMDB: 2 顺序；UCF: 1 并发 | `checkpoint/{dataset}/AutoSSVH_{bit}bit_bpm_pgafs_shvd_pcdec_fnsdcl_r05_g005_sw002/...` |
| `scripts/eval_bpm_pgafs_shvd_pcdec_fnsdcl_activitynet_3bits.sh` | 测 ActivityNet 三个 bit | ActivityNet 16/32/64 | 默认 physical GPU 0 | `eval_results/bpm_pgafs_shvd_pcdec_fnsdcl_activitynet_3bits_<RUN_ID>_maps.csv` 和 summary |
| `scripts/eval_fnsdcl_activitynet_hmdb_ucf_gpu2_3bits.sh` | 测 ActivityNet/HMDB/UCF 三个 bit 并合并 | ActivityNet/HMDB/UCF 16/32/64 | 默认 physical GPU 2 | `eval_results/bpm_pgafs_shvd_pcdec_fnsdcl_activitynet_hmdb_ucf_gpu2_3bits_<RUN_ID>.csv` |

上述脚本均支持 `DRY_RUN=1`。非 dry-run 时会生成 configs、检查数据软链接/依赖/GPU，并拒绝覆盖已存在输出，除非显式设置覆盖变量。

## 9. 当前已有结果与安全表述

### 9.1 FNSC/FNS-DCL 当前主结果

文件：`eval_results/bpm_pgafs_shvd_pcdec_fnsdcl_activitynet_hmdb_ucf_gpu2_3bits_20260615_212452.csv`

| Dataset | bit | mAP@5 | mAP@20 | mAP@40 | mAP@60 | mAP@80 | mAP@100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ActivityNet | 16 | 0.179477 | 0.093965 | 0.057561 | 0.041356 | 0.031918 | 0.026087 |
| ActivityNet | 32 | 0.252910 | 0.136280 | 0.080598 | 0.056822 | 0.043809 | 0.035641 |
| ActivityNet | 64 | 0.301417 | 0.165650 | 0.098655 | 0.069556 | 0.053463 | 0.043382 |
| HMDB | 16 | 0.164682 | 0.099804 | 0.071223 | 0.054952 | 0.044716 | 0.038800 |
| HMDB | 32 | 0.212553 | 0.141571 | 0.099742 | 0.076988 | 0.062422 | 0.052752 |
| HMDB | 64 | 0.258076 | 0.177072 | 0.132990 | 0.106335 | 0.087400 | 0.074048 |
| UCF101 | 16 | 0.433764 | 0.355738 | 0.300278 | 0.253612 | 0.215790 | 0.186862 |
| UCF101 | 32 | 0.550330 | 0.475433 | 0.414260 | 0.363232 | 0.313416 | 0.269011 |
| UCF101 | 64 | 0.582041 | 0.509272 | 0.449665 | 0.401928 | 0.355553 | 0.309636 |

FCV 单独结果文件：`eval_results/bpm_pgafs_shvd_pcdec_fnsdcl_fcv_3bits_20260616_025520.csv`

| Dataset | bit | mAP@5 | mAP@20 | mAP@40 | mAP@60 | mAP@80 | mAP@100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FCV | 16 | 0.378807 | 0.268343 | 0.231222 | 0.210029 | 0.194298 | 0.181256 |
| FCV | 32 | 0.505501 | 0.351828 | 0.303271 | 0.276665 | 0.256572 | 0.239852 |
| FCV | 64 | 0.544756 | 0.392097 | 0.342343 | 0.314552 | 0.292908 | 0.274387 |

### 9.2 ActivityNet r05 baseline vs FNSC

Baseline 文件：`eval_results/activitynet_bpm_pgafs_ratio_sweep_5gpus_20260611_224958_summary.csv`，取 `ratio_name=r05`。FNSC 文件：`eval_results/bpm_pgafs_shvd_pcdec_fnsdcl_activitynet_hmdb_ucf_gpu2_3bits_20260615_212452.csv`。

| bit | baseline mAP@5 | FNSC mAP@5 | delta | baseline mAP@20 | FNSC mAP@20 | delta | baseline mAP@40 | FNSC mAP@40 | delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.187280 | 0.179477 | -0.007803 | 0.095916 | 0.093965 | -0.001951 | 0.058789 | 0.057561 | -0.001229 |
| 32 | 0.235430 | 0.252910 | +0.017480 | 0.132635 | 0.136280 | +0.003646 | 0.082366 | 0.080598 | -0.001768 |
| 64 | 0.280613 | 0.301417 | +0.020803 | 0.161512 | 0.165650 | +0.004139 | 0.099058 | 0.098655 | -0.000403 |

| bit | baseline mAP@60 | FNSC mAP@60 | delta | baseline mAP@80 | FNSC mAP@80 | delta | baseline mAP@100 | FNSC mAP@100 | delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 0.042653 | 0.041356 | -0.001297 | 0.033367 | 0.031918 | -0.001448 | 0.027396 | 0.026087 | -0.001309 |
| 32 | 0.059159 | 0.056822 | -0.002337 | 0.046163 | 0.043809 | -0.002355 | 0.037654 | 0.035641 | -0.002013 |
| 64 | 0.070863 | 0.069556 | -0.001306 | 0.055154 | 0.053463 | -0.001691 | 0.045000 | 0.043382 | -0.001619 |

安全表述：

- 可以说 FNSC 在 ActivityNet 32/64-bit 的 top-ranked retrieval 上有明显收益，尤其 mAP@5。
- 可以说 FNSC 对高 bit top-ranked retrieval 更有帮助。
- 不应说 FNSC 在 ActivityNet 所有 bit、所有 topK 都提升。
- 不应说 FNSC 是无条件提升模块；它更像是 top-rank precision oriented 的 contrastive adjustment。

## 10. Method 章节写作提纲

### 3.1 Overview

写法建议：

- 先说明基线 AutoSSVH 的 masked reconstruction + view contrast + CVH。
- 再说明本文围绕 semantic-hard view learning 做三点增强：
  BPMS 构造 hard view，PA-HVL 稳定 hard view 学习，FNSC 修正 view-level contrast 中的潜在伪负样本。
- 总损失写为：
  `L = L_rec + a L_vc + b L_CVH + lambda_shvd L_SHVD`。

不要过度表述：

- 不要说推理结构变复杂；当前推理仍是原 encoder/hash extraction。

### 3.2 Balanced Prototype-Margin Sampling (BPMS)

应包含：

- `s_pos`、`s_margin`、`M_margin/M_pos`、`S_margin/S_pos/S_mask` 公式。
- 解释 positive-representative 和 margin-hard 的互补性。
- 说明只替换 semantic-hard view2，view1 保持 random mask。

代码对应：

- `build_pgafs_mask_and_labels(...)`
- `get_frame_hash_tokens(...)`

不要过度表述：

- 不要说 BPMS 改变了 loss；它只改变 view2 mask sampling。

### 3.3 Prototype-Aware Hard-View Learning (PA-HVL)

#### 3.3.1 Semantic-Hard View Distillation

应包含：

- `q_t = softmax(h_1 P^T/tau)`。
- `q_s = softmax(h_2 P^T/tau)`。
- `KL(stopgrad(q_t)||q_s)`。
- 说明 teacher 是 random view，student 是 BPMS hard view。

代码对应：

- `compute_shvd_loss(...)`

#### 3.3.2 Prototype-Conditioned Masked Reconstruction

应包含：

- `m_i^p = m + gamma W_p p_i`。
- 说明 decoder 在重建被 BPMS mask 的 semantic key frames 时获得 positive prototype condition。

代码对应：

- `pc_decoder_proto_proj`
- `model.forward(..., proto_cond=...)`
- `get_positive_prototype(...)`

不要过度表述：

- SHVD 是 loss term；PCMR 是 decoder conditioning，不是额外 loss。
- PCMR 增加训练模型参数，但不改变 inference hashing path。

### 3.4 False-Negative Suppressed Contrast (FNSC)

应包含：

- 原 DCL 把 batch 内所有非匹配 pair 都当 negatives。
- FNSC 根据跨 view cosine similarity 对高相似 negatives 降权。
- 权重公式：
  `w_ij = 1 - sigmoid((sim_ij - theta)/s)(1-w_min)`。
- loss 公式和 symmetric 版本。

代码对应：

- `_fns_contrast_one_direction(...)`
- `dcl_fns(...)`
- `AutoSSVH_criterion(...)` 中 `loss_vc` gate。

不要过度表述：

- FNSC 不使用原 DCL 的 `tau_plus`。
- FNSC 替换 `L_vc`，不是额外 loss。

### 3.5 Overall Objective and Inference

应包含：

- 总损失。
- 推理阶段只用 `model.inference(...)` 提取 hash code。
- Hamming distance 和 mAP 评估协议未变。

## 11. 需要避免的过度表述

1. 不要说 FNSC 全 topK、全 bit 都提升。ActivityNet 16-bit 下降，32/64-bit 的 mAP@40 以后也略降。
2. 不要说 PCMR 不增加参数。它增加 `pc_decoder_proto_proj`。
3. 不要说 PCMR 影响推理计算图。评估用 `model.inference(...)`，不使用 decoder conditioning。
4. 不要说 BPMS 直接优化新 loss。BPMS 是 sampling/mask construction。
5. 不要说 SHVD 改变 backbone。SHVD 是训练 loss。
6. 不要说 FNSC 仍使用 `tau_plus`。`dcl_fns(...)` 没有 tau_plus debias term。
7. 不要说 `run_kmeans(...)` 或 `run_finch(...)` 是本文创新；当前方法只是复用 cluster_result。
8. 不要说 dataset loading 或 evaluation metric 被改进；这些逻辑未改。

## 12. 仍需补充的信息或实验

1. 需要把当前 FNSC 与更强的 dataset-specific sweep 最优结果区分开。例如 HMDB 后续已有 `r06_p045` 参数结果，不能混作 `r05` FNSC 主配置。
2. 需要补齐 ablation 表：
   - AutoSSVH 原始 baseline；
   - BPMS only；
   - BPMS + SHVD；
   - BPMS + SHVD + PCMR；
   - BPMS + SHVD + PCMR + FNSC；
   - FNSC off vs on；
   - `pgafs_margin_ratio` sweep；
   - `pgafs_prob` sweep；
   - FNSC threshold/min_weight/suppress_scale sensitivity。
3. 若论文要强调“无推理开销”，需要在文中明确限定：训练时 PCMR 增加 decoder projection 参数，但推理阶段不走 decoder。
4. 若论文主打 ActivityNet top-ranked retrieval，建议单独报告 mAP@5/mAP@20，并将 deeper topK 作为完整表格呈现。
5. 建议在方法图中区分三条路径：
   - BPMS: view2 mask sampling path；
   - PA-HVL: SHVD loss path + PCMR decoder path；
   - FNSC: video-level contrastive loss path。


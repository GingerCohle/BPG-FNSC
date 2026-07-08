# EXACT_HASH_NETWORK_ARCHITECTURE_AUDIT

## 1. 执行摘要

本报告只审计当前代码库中的网络结构和数据流，不涉及训练、评估或代码改动。

工作目录：

```text
/media/kejunjie_coco/autohash/autossh_newwek/BPM-PGAFS+SHVD+PCMR
```

当前方法代码名：

```text
BPM-PGAFS + SHVD + PCMR + FNS-DCL
```

本报告关注的网络部分是 AutoSSVH 的 hashing network：

- 主模型类：`model/AutoSSVH.py:472` 的 `AutoSSVH_model`
- 模型构造入口：`model/AutoSSVH.py:667` 的 `AutoSSVH(cfg)`
- 当前配置均使用 `AutoSSVH_type = "small"`，即 12 层 encoder、2 层 decoder。
- 输入不是原始 RGB 帧，而是 HDF5 中预提取的 frame-level CNN features，键名为 `feats`。
- 代码中没有 CNN 特征提取器，也没有说明具体 CNN backbone；论文中不能从代码事实直接写死 VGG/ResNet/I3D，除非另有外部数据说明。
- frame-level hash tokens 来自 `model.inference(...)` / `model.get_frame_hash_tokens(...)` 的 encoder + hash projection 分支。
- video-level continuous representation 由 frame-level hash tokens 沿时间维均值池化得到。
- binary hash code 在 evaluation 中由 `torch.sign(mean(frame_hash_tokens))` 得到。
- decoder 只用于训练阶段的 masked feature reconstruction；inference / evaluation 不使用 decoder、BPMS、SHVD、PCMR、FNS-DCL。

关键代码位置：

| 作用 | 文件/函数 | 行号 |
|---|---|---:|
| 数据集读取预提取特征 | `dataset/AutoSSVH_dataset.py::TrainDataset` | 32-61 |
| 随机双视角 mask | `dataset/AutoSSVH_dataset.py::RandomMaskingGenerator` | 9-29 |
| Encoder | `model/AutoSSVH.py::Encoder` | 346-420 |
| Decoder | `model/AutoSSVH.py::Decoder` | 423-469 |
| 主模型 | `model/AutoSSVH.py::AutoSSVH_model` | 472-664 |
| small 配置模型构造 | `model/AutoSSVH.py::AutoSSVH` | 667-684 |
| training forward | `model/AutoSSVH.py::AutoSSVH_model.forward` | 583-631 |
| inference | `model/AutoSSVH.py::AutoSSVH_model.inference` | 650-662 |
| feature extraction | `model/AutoSSVH.py::AutoSSVH_model.get_features` | 632-638 |
| BPMS frame tokens | `model/AutoSSVH.py::AutoSSVH_model.get_frame_hash_tokens` | 639-649 |
| loss 中训练数据流 | `Loss/AutoSSVH_loss.py::AutoSSVH_criterion` | 370-494 |
| eval binarization | `inference/AutoSSVH_inference.py::AutoSSVH_inference` | 4-11 |
| Hamming retrieval | `eval.py::evaluate` | 79-208 |

## 2. 输入 CNN 特征来源与尺寸

### 2.1 数据来源

代码使用预提取视频帧特征，不读取原始 RGB 帧。

训练集读取路径：

- `dataset/AutoSSVH_dataset.py:32-61`
- `h5py.File(cfg.train_feat_path[0], "r")`
- 读取键：`h5_file["feats"][:]`
- 输出字典字段：
  - `visual_word`: 单个视频的帧特征序列
  - `mask`: 两个视角的 mask
  - `index`: 当前样本索引

测试集读取路径：

- `dataset/AutoSSVH_dataset.py:64-105`
- 同样读取 HDF5 中的 `feats`
- evaluation 不生成 mask，只返回 `visual_word`

代码没有包含 CNN backbone 或 RGB-to-feature 过程。因此，论文中可写：

> We use pre-extracted frame-level visual features as input. The feature extractor is outside the released training code.

不能仅凭本代码写：

> We extract features using ResNet/VGG/I3D.

除非另有 README、论文或数据说明作为依据。

### 2.2 各数据集输入维度

| Dataset | Config | `feature_size` D | `max_frames` T | `mask_prob` | 每视角 masked 数 M = int(T * mask_prob) | 每视角 visible 数 V = T - M |
|---|---|---:|---:|---:|---:|---:|
| ActivityNet | `configs/AutoSSVH_act.py:4-14` | 2048 | 30 | 0.5 | 15 | 15 |
| FCVID | `configs/AutoSSVH_fcv.py:4-14` | 4096 | 25 | 0.75 | 18 | 7 |
| UCF101 | `configs/AutoSSVH_ucf.py:4-14` | 4096 | 25 | 0.5 | 12 | 13 |
| HMDB51 | `configs/AutoSSVH_hmdb.py:4-14` | 4096 | 25 | 0.7 | 17 | 8 |

说明：

- `RandomMaskingGenerator` 在 `dataset/AutoSSVH_dataset.py:9-29` 中用 `int(mask_ratio * num_patches)` 计算 masked frame 数，因此小数会向下取整。
- `mask=True` 表示该位置被 masked；`mask=False` 表示该位置作为 visible token 进入 encoder。
- 每个样本生成两个 view masks：`mask[0]` 和 `mask[1]`。
- 每个 view 的 mask 数相同，均为 M。

### 2.3 输入张量

在模型入口处，视频特征名为：

```text
visual_word
```

形状为：

```text
visual_word: [B, T, D]
```

其中：

- B 是 batch size；
- T 是 `cfg.max_frames`；
- D 是 `cfg.feature_size`；
- `nbits` 记为 K。

代码中没有在 dataset 读取后对 `visual_word` 做显式归一化。训练时 ActivityNet 配置 `data_drop_rate=0.2`，因此 `AutoSSVH_model.forward(...)` 在训练模式下会对输入 features 做 Dropout；其他数据集默认 `data_drop_rate=0`。

相关代码：

- `model/AutoSSVH.py:500`
- `model/AutoSSVH.py:594-595`

## 3. 当前模型类与主要配置字段

### 3.1 主模型类

主模型类是：

```python
class AutoSSVH_model(nn.Module)
```

位置：

```text
model/AutoSSVH.py:472-664
```

构造入口：

```python
def AutoSSVH(cfg)
```

位置：

```text
model/AutoSSVH.py:667-733
```

当前配置均使用：

```python
AutoSSVH_type = "small"
```

因此实际构造参数来自 `model/AutoSSVH.py:669-684`：

| Hyperparameter | 实际值 / 来源 | 说明 |
|---|---:|---|
| `feature_num` | `cfg.feature_size` | 输入帧特征维度 D |
| `encoder_embed_dim` | `cfg.hidden_size = 256` | encoder hidden dim |
| `max_frame` | `cfg.max_frames` | 最大帧数 T |
| `nbits` | `cfg.nbits` | hash bit length K |
| `encoder_depth` | 12 | encoder Transformer blocks |
| `encoder_num_heads` | 6 | encoder attention heads |
| `decoder_embed_dim` | 192 | decoder hidden dim |
| `decoder_depth` | 2 | decoder Transformer blocks |
| `decoder_num_heads` | 3 | decoder attention heads |
| `mlp_ratio` | 4 | MLP hidden ratio |
| `qkv_bias` | True | q/v bias params enabled |
| `norm_layer` | `LayerNorm(eps=1e-6)` | normalization |
| `mask_prob` | `cfg.mask_prob` | mask ratio |

### 3.2 主要模块表

| Module name in code | Layer / structure | Input dim | Output dim | Purpose |
|---|---|---:|---:|---|
| `encoder.patch_embed` | `Linear(feature_num, 256)` | D | 256 | frame feature projection |
| `encoder.pos_embed` | sinusoidal table `[1,T,256]` by default | 256 | 256 | temporal position encoding |
| `encoder.blocks` | 12 x Transformer `Block` | 256 | 256 | encode visible frame tokens |
| `encoder.norm` | `LayerNorm(256)` | 256 | 256 | final encoder norm |
| `encoder.head` | `Linear(256,256)` | 256 | 256 | post-encoder projection |
| `binary` | `Linear(256,K)` | 256 | K | hash projection |
| `ln` | `LayerNorm(K)` | K | K | hash-space normalization |
| `activation` | `binary_tanh_unit` | K | K | hard binary-like activation |
| `encoder_to_decoder` | `Linear(K,192,bias=False)` | K | 192 | project hash tokens to decoder dim |
| `mask_token` | learnable `[1,1,192]` | - | 192 | decoder masked token |
| `pos_embed` | learnable `[1,T,192]` | 192 | 192 | decoder positional embedding |
| `decoder.blocks` | 2 x Transformer `Block` | 192 | 192 | reconstruct masked features |
| `decoder.norm` | `LayerNorm(192)` | 192 | 192 | decoder norm |
| `decoder.head` | `Linear(192,D)` | 192 | D | predict original frame feature |
| `pc_decoder_proto_proj` | `Linear(K,192) -> GELU -> LayerNorm(192) -> Linear(192,192)` | K | 192 | PCMR prototype-conditioned mask token |

## 4. Encoder 具体结构

### 4.1 Encoder 类

位置：

```text
model/AutoSSVH.py:346-420
```

初始化签名：

```python
Encoder(
    feature_num=4096,
    embed_dim=256,
    max_frame=25,
    nbits=64,
    depth=12,
    num_heads=12,
    mlp_ratio=4.,
    qkv_bias=False,
    ...
)
```

small 模型实际传入：

```text
feature_num = cfg.feature_size
embed_dim = cfg.hidden_size = 256
depth = 12
num_heads = 6
mlp_ratio = 4
qkv_bias = True
```

### 4.2 Frame projection

代码：

```text
model/AutoSSVH.py:358
self.patch_embed = nn.Linear(feature_num, embed_dim)
```

输入输出：

```text
visual_word: [B,T,D]
patch_embed(visual_word): [B,T,256]
```

这里没有卷积层，也没有 CNN backbone。

### 4.3 Positional embedding

代码：

```text
model/AutoSSVH.py:362-366
```

默认 `use_learnable_pos_emb=False`，因此 encoder 使用 sinusoidal positional embedding：

```python
self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)
```

forward 中加入方式：

```text
model/AutoSSVH.py:400-402
x = self.patch_embed(x)
x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
```

因此 encoder positional embedding 在当前构造下不是 learnable parameter。

### 4.4 Visible-token encoder

训练时 `AutoSSVH_criterion(...)` 总是传入 mask：

```text
Loss/AutoSSVH_loss.py:421-429
```

Encoder 的显式 mask 路径：

```text
model/AutoSSVH.py:403-405
B, _, C = x.shape
x_vis = x[~mask].reshape(B, -1, C)
```

即：

```text
mask=False 的位置作为 visible tokens 进入 encoder
mask=True 的位置不进入 encoder
```

visible token 数为：

```text
V = T - M
```

输出形状：

```text
x_vis: [B,V,256]
```

注意：`SampleNet` 存在于 `model/AutoSSVH.py:276-343`，且 encoder 在 `mask is None` 时会调用它；但当前训练 criterion 和 inference 都显式传入 mask，因此当前主路径不使用 `SampleNet` 进行采样。

### 4.5 Transformer block

Block 位置：

```text
model/AutoSSVH.py:148-176
```

每个 block 是 pre-norm Transformer block：

```text
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

Attention 位置：

```text
model/AutoSSVH.py:99-145
```

small encoder 中：

- dim = 256
- heads = 6
- `head_dim = 256 // 6 = 42`
- `all_head_dim = 42 * 6 = 252`
- `qkv = Linear(256, 756, bias=False)`
- 因 `qkv_bias=True`，额外有 `q_bias` 和 `v_bias`，每个长度 252
- attention output 先为 252 维，再经 `proj = Linear(252,256)` 投回 256 维

MLP 位置：

```text
model/AutoSSVH.py:81-96
```

small encoder MLP：

```text
Linear(256,1024) -> GELU -> Linear(1024,256) -> Dropout
```

默认 `drop_rate=0`、`attn_drop_rate=0`、`drop_path_rate=0`，因此这些 Dropout / DropPath 在当前构造下不引入随机丢弃。

### 4.6 Encoder output

代码：

```text
model/AutoSSVH.py:411-419
for blk in self.blocks:
    x_vis = blk(x_vis)
x_vis = self.norm(x_vis)
x = self.head(x)
```

输出：

```text
encoder output: [B,V,256]
```

encoder 不产生 class token，时间信息来自 positional embedding。

## 5. Hash projection / binary activation 具体结构

### 5.1 Hash projection

定义位置：

```text
model/AutoSSVH.py:542-544
```

结构：

```python
self.binary = nn.Linear(self.encoder.num_features, self.encoder.nbits)
self.ln = nn.LayerNorm(self.encoder.nbits)
self.activation = self.binary_tanh_unit
```

数据流：

```text
encoder output [B,V,256]
-> Linear(256,K)
-> LayerNorm(K)
-> binary_tanh_unit
-> hash_code [B,V,K]
```

其中 K = `cfg.nbits`，即 16 / 32 / 64。

### 5.2 Binary activation

代码：

```text
model/AutoSSVH.py:568-577
```

实现：

```python
y = hard_sigmoid(x)
out = 2. * Round3.apply(y) - 1.
```

`hard_sigmoid`：

```python
y = (x + 1.) / 2.
y[y > 1] = 1
y[y < 0] = 0
```

`Round3`：

```text
model/AutoSSVH.py:51-63
```

forward 用 `torch.round(input)`，backward 中对非零输入位置传递梯度。它不是 `torch.sign`，而是一个自定义 rounding function，可理解为硬二值化的近似反传实现。

因此，forward / inference 中的 frame-level hash token 已经是近似二值值：

```text
h_{i,t} in {-1,+1}^K
```

### 5.3 Video-level representation and binary code

训练 criterion 中均值池化位置：

```text
Loss/AutoSSVH_loss.py:431-432
hash_code_1 = torch.mean(hash_code_1, 1)
hash_code_2 = torch.mean(hash_code_2, 1)
```

形状：

```text
hash_code_1 / hash_code_2 before mean: [B,V,K]
z_1 / z_2 after mean: [B,K]
```

evaluation 中：

```text
inference/AutoSSVH_inference.py:7-10
my_H = model.inference(data["visual_word"])
my_H = torch.mean(my_H, 1)
BinaryCode = torch.sign(my_H)
```

因此：

```text
H_i = model.inference(X_i)              # [T,K]
z_i = mean_t H_i[t]                     # [K]
b_i = sign(z_i)                         # [K]
```

`torch.sign` 只在最终 evaluation hash code 生成中使用。

### 5.4 推荐符号定义

| Paper symbol | Code variable / function | Shape | Meaning |
|---|---|---|---|
| `X_i` | `visual_word` | `[T,D]` | video i 的预提取帧特征序列 |
| `H_i` | `model.inference(visual_word)` / `get_frame_hash_tokens` | `[T,K]` | frame-level hash-space tokens |
| `h_{i,t}` | `H_i[t]` | `[K]` | 第 t 帧 hash token |
| `z_i` | `mean(H_i, dim=time)` | `[K]` | continuous video hash representation |
| `b_i` | `torch.sign(z_i)` | `[K]` | final binary hash code |

## 6. Decoder 具体结构

### 6.1 Decoder 类

位置：

```text
model/AutoSSVH.py:423-469
```

small 模型 decoder 参数：

| Item | Value |
|---|---:|
| decoder embed dim | 192 |
| decoder depth | 2 |
| decoder heads | 3 |
| MLP ratio | 4 |
| MLP hidden dim | 768 |
| prediction head | `Linear(192,D)` |

### 6.2 Encoder-to-decoder bridge

定义位置：

```text
model/AutoSSVH.py:535
self.encoder_to_decoder = nn.Linear(self.encoder.nbits, decoder_embed_dim, bias=False)
```

输入输出：

```text
hash_code [B,V,K] -> x_vis [B,V,192]
```

### 6.3 Mask token and decoder positional embedding

定义位置：

```text
model/AutoSSVH.py:537-540
```

结构：

```python
self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
self.pos_embed = nn.Parameter(torch.zeros(1, self.encoder.num_patches, decoder_embed_dim))
```

这里 decoder positional embedding 是 learnable parameter：

```text
decoder pos_embed: [1,T,192]
```

### 6.4 Decoder input construction

forward 中相关代码：

```text
model/AutoSSVH.py:602-626
```

步骤：

1. 将 visible hash tokens 投影到 decoder hidden dim：

```text
x_vis = encoder_to_decoder(hash_code)  # [B,V,192]
```

2. 按原始 mask 拆出 visible 和 masked positional embeddings：

```text
pos_emd_vis  = expand_pos_embed[~mask].reshape(B,V,192)
pos_emd_mask = expand_pos_embed[ mask].reshape(B,M,192)
```

3. 构造 mask tokens：

```text
mask_token: [1,1,192]
mask_token + pos_emd_mask: [B,M,192]
```

4. 拼接 decoder 输入：

```text
x_full = cat([x_vis + pos_emd_vis, mask_token + pos_emd_mask], dim=1)
x_full: [B,V+M,192] = [B,T,192]
```

注意：visible tokens 和 mask tokens 是按两段拼接，但每个 token 加了对应的原始 temporal positional embedding，因此 decoder 能区分它们来自原视频中的哪些位置。

### 6.5 Decoder output

Decoder forward：

```text
model/AutoSSVH.py:461-469
```

当 `return_token_num=M>0` 时：

```python
x = self.head(self.norm(x[:, -return_token_num:]))
```

因此 decoder 只返回最后 M 个 masked tokens 的预测：

```text
frame_1 / frame_2: [B,M,D]
```

`frame_1` 和 `frame_2` 是重建出的 masked frame features，不是 hash tokens。

重建目标在 loss 中由原始 `visual_word` gather：

```text
Loss/AutoSSVH_loss.py:379
labels_1 = visual_word[bool_masked_pos_1].reshape(B,M,D)
```

view2 若 BPMS 触发，则重新 gather：

```text
Loss/AutoSSVH_loss.py:388-399
```

reconstruction loss：

```text
Loss/AutoSSVH_loss.py:438
recon_loss = mse(frame_1, labels_1) + mse(frame_2, labels_2)
```

## 7. PA-HVL 中 prototype-conditioned decoder 具体结构

当前论文命名中，PA-HVL 的 decoder 部分对应代码中的 PCMR / Prototype-Conditioned Decoder。

### 7.1 PCMR projection

定义位置：

```text
model/AutoSSVH.py:545-552
```

只有当 `cfg.pc_decoder_enable=True` 时才初始化：

```python
self.pc_decoder_proto_proj = nn.Sequential(
    nn.Linear(cfg.nbits, decoder_embed_dim),
    nn.GELU(),
    nn.LayerNorm(decoder_embed_dim),
    nn.Linear(decoder_embed_dim, decoder_embed_dim),
)
```

small 模型中：

```text
pc_decoder_proto_proj: Linear(K,192) -> GELU -> LayerNorm(192) -> Linear(192,192)
```

它有 learnable parameters，且 checkpoint 必须与启用该模块的模型结构匹配。

### 7.2 Positive prototype 来源

positive prototype 在 loss 中取得：

```text
Loss/AutoSSVH_loss.py:358-365
```

逻辑：

```text
level = cfg.pc_decoder_cluster_level or cfg.pgafs_cluster_level
im2cluster = cluster_result["im2cluster"][level]
centroids = cluster_result["centroids"][level]
pos_proto_id = im2cluster[index]
proto_cond = centroids[pos_proto_id]
```

形状：

```text
proto_cond: [B,K]
```

### 7.3 Mask token conditioning

forward 中：

```text
model/AutoSSVH.py:611-624
```

当 `pc_decoder_enable=True` 且 `proto_cond is not None`：

```text
proto_cond = proto_cond.detach()   # if pc_decoder_detach_proto=True
proto_embed = pc_decoder_proto_proj(proto_cond)  # [B,192]
mask_token = mask_token + pc_decoder_gamma * proto_embed[:,None,:]
```

公式：

```text
m_i^p = m + gamma * W_p p_i
```

其中：

- `m` 是 learnable decoder mask token；
- `p_i` 是样本 i 的 positive prototype；
- `W_p` 对应 `pc_decoder_proto_proj`；
- `gamma` 对应 `cfg.pc_decoder_gamma`，当前为 0.05。

### 7.4 Gate condition

PCMR 在 criterion 中只给 view2 使用：

```text
Loss/AutoSSVH_loss.py:401-418
Loss/AutoSSVH_loss.py:421-429
```

gate：

```text
cfg.pc_decoder_enable
cluster_result is not None
epoch >= cfg.warmup_epoch
cfg.pc_decoder_view in ["view2", "both"]
if cfg.pc_decoder_only_pgafs_triggered:
    require pgafs_triggered=True
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

配置位置：

```text
configs/AutoSSVH_*_pgafs_shvd_pcmr_margin_balanced.py:22-27
```

PCMR 不添加单独 loss，只改变 view2 decoder 的 mask token。

PCMR 不参与 inference / evaluation，因为 `model.inference(...)` 不调用 decoder。

## 8. forward(...) 训练数据流

forward 入口：

```text
model/AutoSSVH.py:583-631
```

签名：

```python
def forward(
    self,
    x,
    mask=None,
    adv=False,
    epoch=1,
    proto_cond=None,
    pc_decoder_enable=False,
    pc_decoder_gamma=0.1,
    pc_decoder_detach_proto=True,
):
```

### 8.1 单次 view forward shape trace

| Step | Code variable | Shape | Operation | Used for |
|---|---|---|---|---|
| Input | `x` | `[B,T,D]` | pre-extracted frame features | model input |
| Mask | `mask` | `[B,T]` | bool, True=masked | view construction |
| Data dropout | `x` | `[B,T,D]` | optional Dropout in training | feature augmentation |
| Projection | `patch_embed(x)` | `[B,T,256]` | `Linear(D,256)` | frame embedding |
| Encoder position | `x + encoder.pos_embed` | `[B,T,256]` | sinusoidal pos emb | temporal order |
| Visible selection | `x[~mask].reshape` | `[B,V,256]` | remove masked tokens | encoder input |
| Encoder blocks | `x_vis` | `[B,V,256]` | 12 Transformer blocks | contextual visible tokens |
| Encoder head | `x_feat` | `[B,V,256]` | `LayerNorm + Linear(256,256)` | hash projection input |
| Hash projection | `hash_code` | `[B,V,K]` | `Linear(256,K)+LN+binary_tanh_unit` | frame hash tokens for visible frames |
| Decoder bridge | `x_vis` | `[B,V,192]` | `Linear(K,192)` | decoder visible tokens |
| Decoder visible pos | `pos_emd_vis` | `[B,V,192]` | gather decoder pos emb at visible positions | decoder input |
| Decoder mask pos | `pos_emd_mask` | `[B,M,192]` | gather decoder pos emb at masked positions | decoder input |
| Mask token | `mask_token` | `[B,M,192]` after broadcast | learnable token, optionally PCMR-conditioned | masked token reconstruction |
| Decoder input | `x_full` | `[B,T,192]` | concat visible + masked tokens | decoder |
| Decoder output | `x` | `[B,M,D]` | predict masked frame features | reconstruction loss |
| Return | `(x, hash_code, first_k_indices)` | `[B,M,D]`, `[B,V,K]`, optional | loss computation | training |

### 8.2 Loss-side two-view pipeline

Criterion 入口：

```text
Loss/AutoSSVH_loss.py:370-494
```

主要步骤：

1. 从 batch 中取得：

```text
visual_word: [B,T,D]
mask: [B,2,T]
index: [B]
```

2. 得到两个 view masks：

```text
bool_masked_pos_1 = mask[:,0,:]  # [B,T]
bool_masked_pos_2 = mask[:,1,:]  # [B,T]
```

3. gather view1 reconstruction labels：

```text
labels_1: [B,M,D]
```

4. 若 BPMS 触发，替换 view2 mask 并重新 gather `labels_2`：

```text
bool_masked_pos_2, labels_2 = build_pgafs_mask_and_labels(...)
```

5. 若 PCMR gate 触发，取得 `proto_cond_2: [B,K]`。

6. view1 forward：

```text
frame_1, hash_code_1, _ = model.forward(visual_word, bool_masked_pos_1)
```

7. view2 forward：

```text
frame_2, hash_code_2, _ = model.forward(
    visual_word,
    bool_masked_pos_2,
    proto_cond=proto_cond_2,
    pc_decoder_enable=use_pc_decoder_view2,
    pc_decoder_gamma=cfg.pc_decoder_gamma,
    pc_decoder_detach_proto=cfg.pc_decoder_detach_proto,
)
```

8. mean pooling：

```text
hash_code_1 = mean(hash_code_1, dim=1)  # [B,K]
hash_code_2 = mean(hash_code_2, dim=1)  # [B,K]
```

9. reconstruction loss：

```text
L_rec = MSE(frame_1, labels_1) + MSE(frame_2, labels_2)
```

10. view contrastive loss:

```text
if fns_dcl_enable:
    L_vc = dcl_fns(hash_code_1, hash_code_2)
else:
    L_vc = dcl(hash_code_1, hash_code_2)
```

11. CVH loss and SHVD loss are added if their gates are active.

12. Final loss:

```text
L = L_rec + cfg.a * L_vc + cfg.b * L_CVH + shvd_weight * L_SHVD
```

代码位置：

```text
Loss/AutoSSVH_loss.py:481
```

## 9. inference(...) / get_features(...) / get_frame_hash_tokens(...) 数据流

### 9.1 inference(...)

位置：

```text
model/AutoSSVH.py:650-662
```

签名：

```python
def inference(self, x):
```

数据流：

```text
x: [B,T,D]
mask = zeros([B,T], dtype=bool)
encoder(x, mask)       # all frames visible
binary + LayerNorm + binary_tanh_unit
return [B,T,K]
```

`inference(...)` 不使用 decoder，不使用 reconstruction head，不使用 PCMR。

### 9.2 get_features(...)

位置：

```text
model/AutoSSVH.py:632-638
```

数据流：

```text
H = self.inference(x)          # [B,T,K]
z = mean(H, dim=1)             # [B,K]
z_norm = normalize(z, dim=1)   # [B,K]
return z, z_norm
```

`eval.py::compute_features(...)` 用 `z_norm` 做训练集聚类特征：

```text
eval.py:212-222
```

### 9.3 get_frame_hash_tokens(...)

位置：

```text
model/AutoSSVH.py:639-649
```

实现：

```python
return self.inference(x)
```

因此：

```text
get_frame_hash_tokens(visual_word) -> [B,T,K]
```

它使用 all-zero mask，即所有帧作为 visible frames 进入 encoder，不使用 decoder。

BPMS 中调用位置：

```text
Loss/AutoSSVH_loss.py:217-218
with torch.no_grad():
    frame_tokens = model_for_tokens.get_frame_hash_tokens(visual_word)
```

所以 BPMS 用于打分的 `h_{i,t}` 是无梯度的 all-frame hash-space tokens。

### 9.4 Evaluation binary code

位置：

```text
inference/AutoSSVH_inference.py:4-11
```

流程：

```text
H = model.inference(visual_word)  # [B,T,K]
z = mean(H, dim=1)                # [B,K]
b = torch.sign(z)                 # [B,K]
```

Hamming retrieval：

```text
eval.py:178
Hamming_distance = 0.5 * (-dot(test_hashcode, query_hashcode.T) + cfg.nbits)
```

因此当前方法所有新增模块都是 training-time mechanisms；inference retrieval protocol 不变。

## 10. 参数量统计

统计命令只实例化模型并计算参数量，没有训练或评估。

由于提示中的 `~/miniconda3/etc/profile.d/conda.sh` 在本机不存在，本次使用历史训练日志中的解释器：

```text
/home/kejunjie/anaconda3/envs/hash/bin/python
```

### 10.1 ActivityNet, D=2048, T=30

| nbits | total | encoder | decoder | encoder_to_decoder | binary | LayerNorm(K) | mask_token | decoder pos_embed | PCMR proj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 11,486,866 | 10,148,002 | 1,284,992 | 3,072 | 4,112 | 32 | 192 | 5,760 | 40,704 |
| 32 | 11,497,154 | 10,148,002 | 1,284,992 | 6,144 | 8,224 | 64 | 192 | 5,760 | 43,776 |
| 64 | 11,517,730 | 10,148,002 | 1,284,992 | 12,288 | 16,448 | 128 | 192 | 5,760 | 49,920 |

### 10.2 FCVID / UCF101 / HMDB51, D=4096, T=25

FCVID、UCF101、HMDB51 的 `feature_size=4096`，`max_frames=25`，模型参数量相同；mask ratio 不改变模型参数量。

| nbits | total | encoder | decoder | encoder_to_decoder | binary | LayerNorm(K) | mask_token | decoder pos_embed | PCMR proj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 12,405,458 | 10,672,290 | 1,680,256 | 3,072 | 4,112 | 32 | 192 | 4,800 | 40,704 |
| 32 | 12,415,746 | 10,672,290 | 1,680,256 | 6,144 | 8,224 | 64 | 192 | 4,800 | 43,776 |
| 64 | 12,436,322 | 10,672,290 | 1,680,256 | 12,288 | 16,448 | 128 | 192 | 4,800 | 49,920 |

### 10.3 参数量变化来源

`nbits` 改变时，主要影响：

- `binary: Linear(256,K)`
- `ln: LayerNorm(K)`
- `encoder_to_decoder: Linear(K,192,bias=False)`
- `pc_decoder_proto_proj` 第一层 `Linear(K,192)`

encoder 的 Transformer block 主体不随 K 改变，但 encoder 对象内部包含 `SampleNet` 参数；这些参数计入 encoder 参数量，虽然当前 explicit mask 主路径不使用 `SampleNet`。

PCMR projection 参数量：

```text
K=16: 40,704
K=32: 43,776
K=64: 49,920
```

## 11. 可直接写入论文的网络结构描述

下面文字可作为 Method 中网络结构部分的草稿，需根据论文整体符号统一后再放入正文。

> For each video, we use a sequence of pre-extracted frame-level visual features as input. Let \(X_i \in \mathbb{R}^{T \times D}\) denote the feature sequence of the \(i\)-th video, where \(T\) is the number of sampled frames and \(D\) is the dimension of the pre-extracted frame feature. The feature extractor that produces \(X_i\) is outside the training code; the hashing network takes \(X_i\) as its direct input.
>
> The hashing network first maps each frame feature into a 256-dimensional token by a linear projection. A temporal positional embedding is then added to preserve frame order. During masked reconstruction training, only visible frame tokens are passed to a Transformer encoder. In the current small configuration, the encoder contains 12 pre-normalization Transformer blocks. Each block consists of multi-head self-attention and a feed-forward MLP with GELU activation. The encoder output is further projected by a linear head, producing contextualized visible-frame representations.
>
> A hash projection layer maps the encoder hidden dimension to \(K\) hash dimensions. Specifically, the model applies a linear layer from 256 to \(K\), followed by layer normalization and a hard binary activation implemented by a clipped hard sigmoid and a custom rounding function. This produces frame-level hash-space tokens. We denote the all-frame hash tokens by
> \[
> H_i = g_\Theta(X_i) \in \mathbb{R}^{T \times K},
> \]
> where \(g_\Theta\) denotes the encoder-hash branch of the network. The continuous video representation is obtained by temporal average pooling,
> \[
> z_i = \frac{1}{T}\sum_{t=1}^{T} h_{i,t},
> \]
> where \(h_{i,t}\) is the \(t\)-th row of \(H_i\). At evaluation time, the binary hash code is generated by
> \[
> b_i = \operatorname{sgn}(z_i).
> \]
>
> The decoder is used only during training for masked feature reconstruction. The visible hash tokens are projected to the decoder dimension and concatenated with learnable mask tokens at the masked positions. A two-layer Transformer decoder predicts the original frame features of the masked positions, and the reconstruction loss is computed between the decoder outputs and the corresponding input frame features. The decoder, the mask construction modules, SHVD, PCMR, and FNS-DCL are not used during inference; retrieval uses only the encoder-hash branch followed by temporal pooling and binarization.
>
> In PA-HVL, the prototype-conditioned decoder modifies only the decoder mask token for the semantic-hard second view. Given the positive prototype \(p_i\) assigned to the current video, a small projection network maps \(p_i\) to the decoder dimension and injects it into the mask token:
> \[
> m_i^p = m + \gamma W_p p_i.
> \]
> This conditioning changes the reconstruction path during training but does not introduce an additional loss term and does not affect inference.

## 12. 仍不确定的信息

1. 代码没有实现或调用 CNN / video backbone，无法从代码确认预提取特征来自 VGG、ResNet、I3D 或其他 backbone。
2. 代码只显示 HDF5 文件中读取 `feats`，没有存储这些 features 的生成脚本或归一化细节。
3. 当前 `Encoder` 内部保留了 `SampleNet` 自动采样分支，但当前 `AutoSSVH_criterion(...)` 和 `inference(...)` 都显式传入 mask，因此主实验路径不使用 `mask is None` 的 `SampleNet` 分支。若论文要讨论原始 AutoSSVH 的自动采样网络，需要另行审计未改动 baseline 的实际训练入口。
4. small encoder 使用 `embed_dim=256` 和 `num_heads=6`，代码中 attention 的 `head_dim=256//6=42`，实际 attention 内部维度为 252，再投影回 256。这是代码事实；写论文时可避免展开该实现细节，除非需要复现实验细节。
5. 参数量统计包含当前模型对象中所有参数，包括当前主路径未用到的 `SampleNet` 参数。


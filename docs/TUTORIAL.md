# 手把手教程：在 H200 上用 Qwen3.8-27B (bf16) 复现 jlens-qwen36（含 Web App）

本教程带你从零开始，在 HPC 集群的 NVIDIA H200 上完整复现
[WeZZard/jlens-qwen36](https://github.com/WeZZard/jlens-qwen36) 的全部功能：

1. **拟合（fit）** 一个全深度（L0–L62）、1000 条 prompt 的 Jacobian lens；
2. **读出（readout）**：看模型在每一层、每个 token 位置“正在想什么词”；
3. **干预（intervention）**：往 workspace 里“写”概念（Replace / Add / Remove / Erase），以及 “Make it say…” 反向搜索；
4. **Web App**：原项目的可视化调试器，原样跑在 GPU 节点上，通过 SSH 隧道在你笔记本的浏览器里用。

原项目跑在 Apple Silicon + MLX + 4-bit 量化模型上；本仓库（`jlens-cuda`）是它的 CUDA/PyTorch 移植版，目标模型是 **`Qwen/Qwen3.8-27B` 原生 bf16 权重**。

> 约定：`$` 开头的命令在 HPC 上执行；标 **[登录节点]** 的步骤需要联网；标 **[GPU 节点]** 的步骤需要 H200。

---

## 目录

- [第 0 部分：先搞懂原理（15 分钟）](#第-0-部分先搞懂原理)
- [第 1 部分：代码地图——原项目的每一块去哪了](#第-1-部分代码地图)
- [第 2 部分：逐步复现（Step 1–12）](#第-2-部分逐步复现)
- [第 3 部分：Web App 使用导览](#第-3-部分web-app-使用导览)
- [第 4 部分：故障排查](#第-4-部分故障排查)
- [附录：成本估算、与原项目的差异、已验证/待验证清单](#附录)

---

## 第 0 部分：先搞懂原理

### 0.1 Jacobian lens 是什么

语言模型的残差流 `h_ℓ`（第 ℓ 层输出，维度 d=5120）里编码着“模型此刻在想的东西”，但中间层的 `h_ℓ` 和最后一层不在同一个“坐标系”里，直接接 unembedding（logit lens）只在最后几层有意义。

Jacobian lens 的做法：用**整个语料上平均的输入→输出雅可比矩阵**把 `h_ℓ` 线性搬运到最后一层的坐标系，再用模型自己的 unembedding 解码：

```
lens_ℓ(h) = W_U · RMSNorm_final( J_ℓ · h )          J_ℓ ∈ R^{5120×5120}

J_ℓ = E_prompt [ (1/|S|) Σ_{s∈S}  Σ_{t∈S, t≥s}  ∂h_63[t] / ∂h_ℓ[s] ]
```

- `h_63` 是最后一层（第 64 层，索引 63）的残差输出（final norm **之前**）；
- `S` 是“有效位置”：跳过前 16 个位置（attention sink），也不要最后一个位置；
- 对目标位置 `t` **求和**（源位置 s 的活动对“当前和未来所有位置”的影响——论文的“模型在未来某处更可能说这个词”），对源位置 `s` **求平均**；
- 最后对 1000 条 prompt 求平均。

**写（干预）** 是读的对偶：词 t 在第 ℓ 层的 J-lens 向量是 `v_t = J_ℓᵀ W_U[t]`，因为 `⟨v_t, h⟩ = ⟨W_U[t], J_ℓ h⟩`。往 `h_ℓ` 加 `α·v_t` 就让模型“更想说 t”。

### 0.2 怎么算 J_ℓ（这是 GPU 时间的大头）

直接算 5120×5120 的雅可比需要 5120 次反向传播（每次得到一行）。Anthropic 参考实现（`jlens.fitting.jacobian_for_prompt`）的技巧：

1. 把同一条 prompt **复制 B 份（B = `dim_batch`）** 组成一个 batch，做**一次**前向，保留计算图；
2. 第 b 个样本的 cotangent（反向的“种子”）是：在输出维度 `k+b`、所有有效目标位置上置 1；
3. **一次反向**就同时得到 **所有 63 层** J_ℓ 的第 `k..k+B` 行（对源位置取平均即可）；
4. 重复 `⌈5120/B⌉` 次反向（B=16 → 320 次；B=32 → 160 次）。

总 FLOPs 与 B 无关，B 只影响显存（保留的计算图 ∝ B × 128 tokens × 64 层）和 GPU 利用率。

### 0.3 为什么 CUDA 版反而比 MLX 版简单

Qwen3.x-27B 是混合架构：64 层 = **48 层 Gated DeltaNet（线性注意力）+ 16 层全注意力**（每 4 层一个）。

| | 原项目（MLX） | 本项目（CUDA） |
|---|---|---|
| GDN 反向传播 | MLX 的 fused GDN kernel **没有反向**，ops 回退慢 22 倍 → 作者手写 Metal 反向 kernel | [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) 的 Triton kernel **自带反向**，transformers 自动调用 |
| 拟合估计器 | 逐层解析组装 `M_ℓ`，再链式相乘 `J_{ℓ-1} = J_ℓ · M_ℓ`（“平均的乘积”≈“乘积的平均”，是近似） | 直接用 Anthropic 的 `jlens.fit`：端到端精确的论文估计器（Neuronpedia 公开的 lens 也是这么拟合的） |
| prompt 设置 | 32 tokens，跳过前 4 个位置（为了在笔记本上跑得动） | 论文设置：128 tokens，跳过前 16 个位置 |
| 权重 | 4-bit 量化 | 原生 bf16 |
| 单条 prompt 耗时 | M4 Pro 437 s；M4 Mac mini 1161 s | 同形状的 Qwen3.5-27B 在 B200 上实测约 57 s（Neuronpedia 数据）；H200 请用 Step 5 实测 |

所以 CUDA 版的“复现逻辑”=**拟合用官方参考估计器 + 读出/干预/Web App 全部沿用原项目**。原项目里为 Metal 写的 `custom_gdn_vjp.py`、`analytic_*.py`、`fit_analytic.py`、`patch_gdn.py` 在 CUDA 上都不需要了。

### 0.4 Web App 的架构

```
浏览器 (web/index.html, 原样)
   │  HTTP + SSE (Server-Sent Events)
   ▼
FastAPI (jlens_qwen/serve.py, 仅改了 MLX 调用点)
   │  asyncio.to_thread + _gpu_lock（GPU 调用串行化，事件循环不被阻塞）
   ▼
TorchLensModel / StreamSession (jlens_qwen/model.py, 新写)
   │  forward hook 挂在 64 个 decoder layer 上：捕获残差 / 就地改写残差
   │  HF 的混合 DynamicCache：KV cache(16层) + conv/recurrent state(48层) → O(T) 增量解码
   ▼
HF transformers Qwen3_5ForConditionalGeneration (bf16, H200)
   + JacobianLens (jlens_qwen/lens.py)：63 个 J_ℓ 常驻显存 (fp32, 6.6 GB)
   + interventions.py：steer / swap / ablate 的 J-lens 向量运算
```

聊天时每生成一个 token：`StreamSession.extend([tok])` 跑一步增量前向，hook 抓到 63 层在这个新位置的残差 → `J_ℓ h` → final norm → unembed → top-k，作为一帧 SSE 推给浏览器，网格里多出一行。

---

## 第 1 部分：代码地图

### 1.1 原项目 → 本项目

| 原项目文件 | 作用 | 本项目 |
|---|---|---|
| `web/index.html` | 前端（单文件，~340 KB） | **原样复制**（只改了页面标题） |
| `jlens_qwen/serve.py` | FastAPI 服务，所有 API | **复制 + 改 MLX 调用点**，完整 diff 见 [`serve-port.diff`](serve-port.diff) |
| `jlens_qwen/model.py` | MLX 模型适配、`StreamSession` | **重写**为 PyTorch：`TorchLensModel` + hook 版 `StreamSession`，接口不变 |
| `jlens_qwen/lens.py` | `JacobianLens`（npz） | **重写**：同时读写 `.npz`（原项目格式）和 `.pt`（jlens / Neuronpedia 格式） |
| `jlens_qwen/interventions.py` | steer/swap/ablate | **移植**到 torch（数学完全相同；W_U 是 bf16 稠密矩阵，不需要反量化） |
| `jlens_qwen/perf.py` | 性能计时 | 移植（`mx.eval` → `torch.cuda.synchronize`） |
| `jlens_qwen/prompts.py` | 语料加载 | 复制 + 修 bug（见下） + 加 `source` 选项 |
| `fit_qwen38_n1000.py`、`fit*.py`、`analytic*.py`、`*gdn*.py` | Metal/MLX 拟合管线 | **由 `scripts/fit_lens.py` + Anthropic `jlens` 取代** |
| `scripts/readout_smoke.py`、`scripts/measure_bands.py` | 读出冒烟测试、功能分带 | 移植 |
| `tests/test_intervention_*.py`、`test_planner_probe.py` … | serve 逻辑的契约测试 | 复制，把 `mx` 桩换成 `torch.argmax` 桩；**104 个全部通过** |
| — | — | 新增：`check_env.py`、`prepare_corpus.py`、`verify_fit.py`、`merge_lens.py`、`compare_lens.py`、`slurm/*`、`tests/test_tiny_qwen35.py`、`tests/test_gpu_real_model.py` |

> **顺手修掉的原项目 bug**：`prompts.py` 用 `load_dataset("wikitext", ...)` 加载 WikiText，新版 `datasets` 要求 `Salesforce/wikitext`，于是原项目**静默回退到了 c4**（这也是为什么原项目 Qwen3.8 lens 的 provenance 写的是 c4）。本项目默认用 WikiText-103（与 Neuronpedia 一致），`--source c4` 可复现原项目的语料选择。

### 1.2 关键代码讲解

#### (a) 用 hook 代替重写 forward — `jlens_qwen/model.py`

MLX 版为了抓残差把整个 `Qwen3_5TextModel.__call__` 重写了一遍。PyTorch 里用 `register_forward_hook` 就够了，HF 的 decoder layer 直接返回残差张量：

```python
def hook(module, args, output):
    h = output                          # 第 i 层之后的残差 [1, n, 5120]
    if ed: h = _apply_edits(h, ed, start)   # 干预：先改写
    if i in capture: acts[i] = h            # 读出：再捕获（所以网格显示的是“被写过的”值）
    if ed: return h                         # 返回值替换该层输出 → 下游层和 cache 都吃到改写后的流
```

`start = n_consumed` 把“全局位置”（聊天模板后整段序列里的坐标，也就是网格的行号）映射到当前 chunk 的局部行号——和原项目 `_chunk_local_indices` 一模一样。

#### (b) 增量解码 — `StreamSession`

```python
self.cache = DynamicCache(config=text_config)   # 混合 cache：KV + conv_state + recurrent_state
out = text_model(input_ids=chunk, past_key_values=self.cache, use_cache=True)
```

每次 `extend(ids)` 只跑新 token；返回最后位置的 fp32 logits 和各层残差。`tests/test_tiny_qwen35.py` 验证了“分块增量解码 ≡ 一次性全序列前向”（逐层残差和 logits 都对得上）。

#### (c) 读出 — `serve._readout_at_positions`

```python
h = acts[layer][0][pos_idx].float()          # [P, 5120]
h = _lens.transport(h, layer)                # J_ℓ h（fp32，TF32 tensor core）
logits = _model.unembed(_model.final_norm(torch.stack(hs)))   # [L, P, 248320]
ids, vals = torch.topk(logits, top_n)        # 原项目为 MLX 手写的分块 top-k，CUDA 上 torch.topk 即可
```

unembed 用了一份 fp32 的 `lm_head.weight`（5 GB），避免读出分数被 bf16 截断导致排名抖动。

#### (d) 干预 — `jlens_qwen/interventions.py`

- **Add / Remove**（`steer`）：`h ← h + α·v_t`
- **Replace**（`swap`）：`V=[v_s; v_t]`，坐标 `c=(VVᵀ)⁻¹Vh`，交换两个坐标再写回，正交分量不动——自校准，最稳；
- **Erase**（`ablate`）：减去 `span{v_t}` 上的最小二乘投影。

`compile_edits()` 在请求开始时把所有向量算好，返回的闭包只做几次小矩阵乘，每步解码开销可忽略。

#### (e) 拟合 — `scripts/fit_lens.py`

核心只有几行：

```python
hf_model, tokenizer = load_hf("Qwen/Qwen3.8-27B", device="cuda")   # bf16
lm = jlens.from_hf(hf_model, tokenizer)          # 自动找到 model.language_model.layers
lens = jlens.fit(lm, shard_prompts, source_layers=range(63),
                 dim_batch=16, max_seq_len=128, skip_first=16,
                 checkpoint_path=..., checkpoint_every=10)
```

1000 条 prompt 按 `prompts[shard::num_shards]` 切给多块 GPU，各自独立拟合，最后 `merge_lens.py` 按 prompt 数加权平均（与一次性拟合数学上等价）。

---

## 第 2 部分：逐步复现

### Step 0. 规划资源

| 资源 | 需求 |
|---|---|
| GPU | 拟合：1–8 块 H200（141 GB）；Web App：1 块 H200 |
| 共享存储 | 模型 ~54 GB；Python 环境 ~10 GB；lens 分片 8×6.6 GB（合并后可删）；最终 lens 3.3 GB(.npz)+3.3 GB(.pt) |
| 内存 | 每个拟合进程 ~20–30 GB 主机内存（两份 63×5120² fp32 累加矩阵 + 模型加载）；sbatch 里给了 96 GB |
| 时间 | 拟合：见 Step 5 实测；1000 prompts / 8×H200 预计数小时量级 |
| 网络 | 只有**登录节点**需要联网（装包、下模型、下语料） |

### Step 1. 把代码放到集群上 [登录节点]

本地仓库在 `~/Workspace/jlens-cuda`。任选一种：

```bash
# 方式 A：rsync（在你的 Mac 上执行）
rsync -av --exclude .venv --exclude 'data/lens/*' --exclude 'data/corpus/*.jsonl' \
  ~/Workspace/jlens-cuda/ <user>@<login-host>:~/jlens-cuda/
```

```bash
# 方式 B：推到你自己的私有 Git 仓库，再在集群上 clone
```

之后所有命令都在集群上的项目根目录执行：

```bash
cd ~/jlens-cuda
```

> 如果 `$HOME` 配额小，把项目放到 `/scratch/$USER/jlens-cuda` 之类的大容量共享盘。

### Step 2. 安装 Python 环境 [登录节点]

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
export UV_CACHE_DIR=/scratch/$USER/uv-cache   # 可选：避免撑爆 $HOME（torch+CUDA 库约 5 GB）
uv sync
```

`uv sync` 会按 `uv.lock` 装好：PyTorch（PyPI 的 Linux wheel 自带 CUDA 运行时，节点上只需要 NVIDIA 驱动）、transformers ≥ 5.8、`flash-linear-attention`（GDN 的 Triton kernel）、Anthropic 的 `jlens`（固定到 commit `581d398`）、FastAPI/uvicorn 等。

先在登录节点跑一遍不需要 GPU 的测试（1–2 分钟）：

```bash
uv run pytest -q
```

应看到 `108 passed, 3 skipped`（跳过的是 `tests/test_gpu_real_model.py` 里需要真模型和 GPU 的 3 个测试，Step 11 再跑）。

> 可选：`uv sync --extra conv1d` 安装 `causal-conv1d` CUDA kernel。它需要本地 CUDA toolkit 编译，而且 GDN 前面那个 depthwise 卷积很便宜，不装也几乎不影响速度。

### Step 3. 下载模型 [登录节点]

```bash
export HF_HOME=/scratch/$USER/hf        # 与 slurm/common.sh 里的 HF_HOME 保持一致！
uv run hf download Qwen/Qwen3.8-27B
```

约 54 GB（18 个 safetensors 分片）。架构参数：64 层（48 GDN + 16 全注意力）、d_model 5120、词表 248320、`Qwen3_5ForConditionalGeneration`（带一个小视觉塔，我们只用语言部分）。

然后**编辑 `slurm/common.sh`**：把 `HF_HOME` 改成上面的路径；每个 `slurm/*.sbatch` 里的 `--partition=CHANGE_ME` 改成你们的 H200 分区（有的集群还要 `--account=...`、`--gres=gpu:h200:1` 这种写法）。

### Step 4. 准备拟合语料 [登录节点]

```bash
uv run python scripts/prepare_corpus.py --n 1000 --min-chars 600 --tokenizer Qwen/Qwen3.8-27B
```

期望输出（本地实测）：

```
Loading WikiText-103 (Salesforce/wikitext)...
  got 1000 prompts from WikiText-103
1000 prompts (1000 unique), cached under data/corpus/ (source=wikitext)
token lengths: min 105, median 199, max 359; 975/1000 reach 128 tokens
```

生成 `data/corpus/prompts_1000_600.jsonl`（约 1 MB）。计算节点离线时，所有脚本都从这个缓存读。`--min-chars 600` 是为了让绝大多数段落够 128 个 token（论文设置）；每条截断到 1000 字符。

### Step 5. GPU 冒烟测试：环境 + 拟合正确性 + 成本 [GPU 节点]

先申请一块 H200 交互式调试（分区名换成你们的）：

```bash
salloc -p <h200-partition> --gres=gpu:1 --cpus-per-task=8 --mem=96G -t 2:00:00
```

```bash
srun --pty bash
```

```bash
cd ~/jlens-cuda && source slurm/common.sh
```

**5a. 环境检查**

```bash
uv run python scripts/check_env.py
```

重点看这一行必须指向 `fla` 的模块（具体模块路径随 fla 版本略有不同），不能是 `transformers torch fallback`：

```
torch_chunk_gated_delta_rule       -> fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule
[OK] fla Triton kernels will be used for Gated DeltaNet.
```

**5b. 拟合路径的正确性 + 单条 prompt 成本**

```bash
uv run python scripts/verify_fit.py --dim-batch 16
```

它做两件事：

1. 用拟合估计器（复制 batch + 保留计算图 + fla 反向重复用几百次）完整算一条 128-token prompt 的全部 63 层 J，**计时并报告显存峰值**；
2. 抽 J_0 / J_31 / J_62 的几行，与“全新 batch=1 前向 + 单次反向”独立算出的行对比。bf16 kernel 在不同 batch 下不逐位相同，所以看余弦相似度：

```
full depth (63 layers), seq_len=..., dim_batch=16: XXX s/prompt, peak GPU memory YY GB (320 backward passes)
  J_0  row 0     cos=0.99xxx  rel_err=x.x%
  ...
correctness: worst cosine 0.99xxx -> OK
projection: 1000 prompts on 8 GPUs ~ Z h wall
```

**判读与调参：**

- `correctness ... OK`（最差余弦 > 0.99）才能继续。不 OK 说明 fla 反向在 retain_graph 下有问题，别往下跑，见[第 4 部分](#第-4-部分故障排查)。
- 看 `peak GPU memory`：H200 有 141 GB，模型占 ~54 GB。显存有富余就试 `--dim-batch 32`（Neuronpedia 在 179 GB 的 B200 上用的是 64），每 prompt 通常会更快；OOM 就退回 16。
- 可再试 `--compile`（每层 `torch.compile`，Neuronpedia 就开了）：首次编译要几分钟，之后反向更快。**只有 verify 在 `--compile` 下也 OK，拟合才加 `--compile`**。
- 记下 `s/prompt`，它决定 Step 7 的时长。参考：同形状的 Qwen3.5-27B 在 B200 上 dim_batch 64 + compile ≈ 57 s/prompt；H200 的 bf16 算力约为 B200 的一半左右，粗估 1.5–2.5 分钟/prompt，以你的实测为准。

也可以把 5a+5b 一起作为批处理作业提交：

```bash
DIM_BATCH=32 sbatch slurm/verify.sbatch
```

### Step 6.（强烈推荐）用小模型对照 Neuronpedia，验证整条拟合管线

在花几十 GPU·小时之前，先用 **Qwen3.5-0.8B**（同一 `qwen3_5` 架构，d=1024，24 层）跑一遍完整的“拟合 → 合并 → 对比”，参照物是 Neuronpedia 用同一个 `jlens` 库、同一 WikiText 语料拟合的官方 lens。

**[登录节点]** 下载模型和参考 lens（48 MB）：

```bash
uv run hf download Qwen/Qwen3.5-0.8B
```

```bash
uv run hf download neuronpedia/jacobian-lens qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt --local-dir data/lens/neuronpedia
```

**[GPU 节点]** 拟合 200 条（1000 条语料切 5 片，只跑第 0 片；0.8B 模型很小，dim_batch 可以开到 128，和 Neuronpedia 一致）：

```bash
uv run python scripts/fit_lens.py --model Qwen/Qwen3.5-0.8B --n-prompts 1000 --shard 0 --num-shards 5 --dim-batch 128 --tag qwen35_08b
```

```bash
uv run python scripts/merge_lens.py data/lens/shards/qwen35_08b_n1000_shard00of05.pt --out data/lens/qwen35_08b_ours
```

```bash
uv run python scripts/compare_lens.py data/lens/qwen35_08b_ours.pt data/lens/neuronpedia/qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt --model Qwen/Qwen3.5-0.8B
```

**怎么看结果：** 两次独立拟合（不同的 prompt 子集和数量）不会逐位相同。健康的结果是：各层余弦相似度高（通常 0.9 以上）、越靠后的层越一致、读出 top-1 一致率高。如果余弦接近 0、范数差几个数量级、或读出几乎完全不一致，说明管线有问题（模型版本不对、语料异常等），先排查再上 27B。

### Step 7. 正式拟合 Qwen3.8-27B（1000 prompts，全深度 63 层）[GPU 节点]

两种提交方式，任选其一（都在项目根目录执行）。

**方式 A：作业数组，每个任务 1 块 GPU（排队最灵活）**

```bash
mkdir -p logs && DIM_BATCH=32 sbatch slurm/fit_array.sbatch
```

默认 `--array=0-7` 即 8 个分片、每片 125 条 prompt。想要 16 片：`sbatch --array=0-15 slurm/fit_array.sbatch`（分片数自动取 `SLURM_ARRAY_TASK_COUNT`）。`DIM_BATCH` 用 Step 5 选定的值；若 Step 5 验证过 `--compile`，加上 `FIT_EXTRA_ARGS=--compile`。

**方式 B：独占一个 8 卡节点**

```bash
mkdir -p logs && DIM_BATCH=32 sbatch slurm/fit_node.sbatch
```

**监控：**

```bash
squeue -u $USER
```

```bash
tail -f logs/fit_<jobid>_0.out
```

日志每条 prompt 一行（来自 `jlens.fit`）：

```
prompt 17/125  seq_len=128 n_valid=111  98s  max||J||/sqrt(d)=0.412  max_d_mean=4.1e-03
```

- `98s`：这条 prompt 的耗时；
- `max||J||/sqrt(d)`：本条 prompt 的 J 范数，突然大一个数量级说明遇到离群 prompt（会被平均掉，一般无需处理）；
- `max_d_mean`：running mean 的相对变化，随 n 约按 1/n 下降——这就是“收敛曲线”。Neuronpedia 在 Qwen3.5-27B 上 ~672 条时 < 0.002 就停了；我们按论文/原项目跑满 1000。

**断点续跑：** 每 10 条 prompt 写一次 checkpoint（`data/lens/shards/*.ckpt.pt`，每个 6.6 GB）。任务被抢占/超时后，**原样重新提交**即可从断点继续；已完成的分片会直接跳过。

### Step 8. 合并分片 [任意节点]

```bash
uv run python scripts/merge_lens.py data/lens/shards/qwen38_27b_n1000_shard*.pt --out data/lens/qwen38_27b_n1000 --expect-n 1000
```

生成：

- `data/lens/qwen38_27b_n1000.npz` —— Web App 加载的格式（fp16，3.3 GB）；
- `data/lens/qwen38_27b_n1000.pt` —— Anthropic `jlens` / Neuronpedia 格式，可直接给官方可视化工具用；
- `data/lens/qwen38_27b_n1000.provenance.json` —— 记录模型 id、语料、估计器、各分片耗时。

> ⚠️ **lens 只对拟合它的那份权重有效。** Qwen3.6-27B 和 Qwen3.8-27B 形状完全相同，拿错 lens 不会报错，只会给出“自信的胡话”。用之前看一眼 provenance 的 `model_id`。

确认无误后可以删掉分片（`data/lens/shards/*.pt`，约 53 GB）。

### Step 9. 读出冒烟测试 [GPU 节点]

```bash
uv run python scripts/readout_smoke.py --lens data/lens/qwen38_27b_n1000.npz
```

对 4 个补全 prompt（法国首都、蜘蛛几条腿、星期序列、水的化学式），打印若干层 J-lens top-1、logit-lens top-1 和模型真实的下一个 token。**健康的 lens**：答案概念（Paris / eight / Friday / H₂O）在中后层就浮现，比 logit lens 早得多；它经常以另一种语言/文字出现（例如 `八`、`周五`），所以“与模型输出逐字相同”的比例本来就不高——看表，不要只看最后的计数。

### Step 10. 测量功能分带（可选，让 UI 的色带更准）[GPU 节点]

```bash
uv run python scripts/measure_bands.py --lens data/lens/qwen38_27b_n1000.npz --prompts 60
```

按论文的四个逐层信号（下一 token 准确率、top-1 持续性、峰度、有效维度）检测 sensory / workspace / motor 三个带，写到 `data/bands/qwen38_27b_n1000.json`。Web App 启动时按 lens 文件名自动加载；没有这个文件时 UI 用百分比猜测（`bands_are_fallback: true`）。

### Step 11. 真模型因果测试 [GPU 节点]

原项目的“因果闸门”测试：把 ⟨France⟩ 换成 ⟨China⟩（第 30/40/48 层），问法国首都应当回答北京。

```bash
JLENS_SLOW_TESTS=1 JLENS_PATH=data/lens/qwen38_27b_n1000.npz uv run pytest tests/test_gpu_real_model.py -v -s
```

三个测试：增量流式解码 ≡ 全序列前向（bf16 容差）、空干预不改变输出、France→China 交换让答案变成 Beijing/China。

> 层号 30/40/48 是原项目 lens 的经验值。新 lens 的最佳干预层可能不同；若第三个测试没翻转，用 `JLENS_TEST_LAYERS=36,44,52` 之类换几组层试试，或者在 Web App 里用 **scan** 功能扫层 × 强度（第 3 部分）。

### Step 12. 启动 Web App [GPU 节点] + SSH 隧道 [你的电脑]

**提交服务作业：**

```bash
mkdir -p logs && JLENS_PATH=data/lens/qwen38_27b_n1000.npz sbatch slurm/serve.sbatch
```

```bash
cat logs/serve_<jobid>.out
```

日志开头会打印计算节点名和现成的 SSH 命令，等看到 `Uvicorn running on http://127.0.0.1:8765`（模型加载约 1–3 分钟）后，在**你的 Mac 上**执行（`LOGIN` 换成登录节点地址，`gpu-node-xx` 换成日志里的节点名）：

```bash
ssh -N -L 8765:localhost:8765 -J <user>@LOGIN <user>@gpu-node-xx
```

然后浏览器打开 <http://localhost:8765/>。

**先确认加载的是对的 lens：**

```bash
curl -s localhost:8765/api/lens | head -c 200
```

应看到 `"n_prompts": 1000` 和 `source_layers` 0..62。

> **如果集群不允许 SSH 进计算节点：** 把 `serve.sbatch` 最后一行的 `--host 127.0.0.1` 改成 `--host 0.0.0.0`，隧道改为 `ssh -N -L 8765:gpu-node-xx:8765 <user>@LOGIN`。注意这样集群内网的其他人也能访问你的服务（它没有鉴权）。有 Open OnDemand 的集群也可以用它的端口转发。

**交互式启动（调试用）：** 在 Step 5 的 `salloc` 会话里直接：

```bash
JLENS_PATH=data/lens/qwen38_27b_n1000.npz uv run uvicorn jlens_qwen.serve:app --host 127.0.0.1 --port 8765
```

**只读展示模式：** `JLENS_MODE=presentation` 启动后，UI 隐藏所有编辑功能、只能浏览已保存的会话（原项目 jlens.wezzard.com 用的就是这个模式）。

**不想等拟合、先体验 Web App？** 用 Neuronpedia 的 Qwen3.6-27B n=1000 lens 搭配 Qwen3.6-27B（注意必须配对！），`.pt` 可以直接加载：

```bash
uv run hf download Qwen/Qwen3.6-27B
```

```bash
uv run hf download neuronpedia/jacobian-lens qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt --local-dir data/lens/neuronpedia
```

```bash
JLENS_MODEL=Qwen/Qwen3.6-27B JLENS_PATH=data/lens/neuronpedia/qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt sbatch slurm/serve.sbatch
```

这也是验证 Web App 移植在 H200 上工作正常的最快方式（该 lens 本身是公认可靠的）。

---

## 第 3 部分：Web App 使用导览

界面和原项目完全一致（原项目 README 的截图可以对照看）。

### 3.1 读：看 workspace

1. 底部输入框输入问题，回车。聊天默认**关闭 thinking**（`enable_thinking=False`），让模型在潜空间里算而不是写成可见的 `<think>`——这正是 lens 要暴露的东西。
2. 左侧聊天，右侧是 **位置 × 层** 网格：每个格子是该位置、该层的 J-lens top-1 词；生成时一行一行实时流进来。
3. 顶部色带标出 **sensory / workspace / motor** 三个功能带（Step 10 测量的，或百分比猜测）。workspace 带是最值得看的地方：原项目首页那张图里，模型嘴上回答得很顺从，workspace 带里却亮着 *blackmail / murder / threatening*。
4. **点一个格子**：钉住它的 top-10 读出（分数就是 lens logit）。

### 3.2 写：手动干预

在格子详情卡里点 **Intervene** 打开编辑器（或悬停读出行用 ＋ / － / ⇄ 快捷操作）：

| UI 名 | 含义 | 底层操作 |
|---|---|---|
| **Replace** | “凡是在想 A 的地方，改成同样强度地想 B” | `swap`（自校准，最可靠，默认） |
| **Add** | “再多想一点 t” | `steer`，α > 0 |
| **Remove** | “少想一点 t” | `steer`，α < 0 |
| **Erase all** | “把这个格子在想的都抹掉” | `ablate`（自动排除该位置最终层 top-k，不会抹掉模型本来要说的词） |

作用范围：**层**（这一层 / 一个层带 / 自选，Pick… 模式下点列头切换）× **位置**（Just this one / This + after / The reply）。编辑器打开时受影响的格子会发光，其余变暗。

规格收集在左上角栏里，点 **Re-run ⊕N** 重新生成；右上角 **Baseline / Intervened** 胶囊按钮在两次运行之间整体切换网格和聊天，做 A/B 对比。

经典演示：问 “What is the capital of France?”，在 workspace 带用 Replace 把 ⟨France⟩ 换成 ⟨China⟩，Re-run，回答变成北京。

> Add/Remove 是**绝对强度**：lens 越好、J-lens 向量越大，需要的 α 越小。原项目 20-prompt lens 需要 α 50–800；1000-prompt lens 通常小得多。Replace 不需要调 α。

### 3.3 反向搜索：“Make it say…”

点回复里的一个词，输入你想让它说成什么（例如 Paris → Beijing）。服务端会：

1. 在 workspace 的若干 (层, 位置) 上生成候选“配方”，逐个**真实重放**（贪心解码到 EOS）来验证；
2. 通过验证的配方标**绿点**（回复里恰好出现一次目标词、不再出现原词、无重复崩坏）；
3. 如果直接编辑不起作用，常驻模型自己读对话，提出**潜在前提**（如 ⟨France⟩→⟨China⟩ 让 ⟨Paris⟩→⟨Beijing⟩），这类配方标**紫点**。

各接口的细节（`/api/intervention_scan`、`/api/intervention_search(_adaptive)`、`/api/planner_probe`）见原项目文档的副本 [`interventions-upstream.md`](interventions-upstream.md)。

### 3.4 会话

**Saved Sessions** 保存/加载对话及其全部快照（存在 `data/sessions/`）；**Copy Link** 复制可分享链接。

### 3.5 无界面地调用 API（调试用）

```bash
curl -s -X POST localhost:8765/api/slice -H 'content-type: application/json' -d '{"prompt":"The capital of France is","top_n":5}' | head -c 400
```

```bash
curl -sN -X POST localhost:8765/api/chat_stream -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":16,"interventions":[{"mode":"swap","layers":[30,40,48],"token":" France","target":" China","alpha":1.0,"from_position":0}]}'
```

---

## 第 4 部分：故障排查

| 现象 | 原因 / 处理 |
|---|---|
| `check_env` 显示 GDN 是 `transformers torch fallback` | `fla` 没装上或导入失败：`uv run python -c "import fla"` 看报错；通常是 triton/torch 版本不匹配，`uv sync` 重装。Web App 在 fallback 下也能跑，但拟合会慢一个数量级以上。 |
| 拟合 `CUDA out of memory` | 降 `DIM_BATCH`（32→16→8）；确认没有别的进程占卡；`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 已在 `common.sh` 里设置。 |
| `verify_fit` 余弦不 OK | 先去掉 `--compile` 重试；再在 CPU/fallback 下对比（`uv pip uninstall flash-linear-attention` 临时卸载后重跑 verify）。若只有 fla 路径不对，把 fla 版本和现象反馈给 fla 项目，暂时用较小的 `dim_batch` 或 fallback 跑。 |
| 计算节点报 `We couldn't connect to 'https://huggingface.co'` | 模型/语料没在登录节点预下载，或 `HF_HOME` 在登录节点和 `common.sh` 里不一致。 |
| `uv run` 在计算节点试图联网 | `common.sh` 已设 `UV_OFFLINE=1 UV_FROZEN=1`；确保在登录节点先 `uv sync` 过。 |
| 拟合报 `corpus is not 1000 unique prompts` | 没跑 Step 4，或语料下载失败回退到了内置的重复语料。回登录节点重跑 `prepare_corpus.py`。 |
| Web App 启动报 `LENS SELECTION FAILED` | 服务拒绝静默挑 lens：显式设 `JLENS_PATH=...`（`.npz` 或 `.pt`），或 `JLENS_PATH=none` 用纯 logit lens。 |
| 读出全是乱码/无意义 | 90% 是 lens 和模型不配对（看 provenance 的 `model_id`）。 |
| 浏览器连不上 | `squeue` 确认作业在跑、日志里 uvicorn 已启动；隧道命令里的节点名要和日志一致；本地 8765 端口被占就换 `PORT=8766`。 |
| 聊天很慢 | 每个 token 的读出本身很便宜（63 层 × 248k 词表 unembed ≈ 0.16 TFLOP）；慢多半是 HF 逐层 Python 开销或 GDN 走了 fallback。`JLENS_PERF=1` 打开分阶段计时。 |

---

## 附录

### A. 成本估算

| 项 | 数值 |
|---|---|
| 拟合 FLOPs | 每 prompt ≈ 5120 次输入梯度反向 × 128 tokens × 2 × ~25B 参数 ≈ 3×10¹⁶（与 dim_batch 无关） |
| 参考实测 | Neuronpedia：Qwen3.5-27B（同形状），B200，dim_batch 64，compile，bf16 → 中位数 57 s/prompt |
| H200 估计 | 以 `verify_fit.py` 实测为准；按算力比粗估 ~2 分钟/prompt → 1000 prompts ÷ 8 卡 ≈ 4–5 小时 + 模型加载 |
| Web App 显存 | 模型 54 GB + J(fp32) 6.6 GB + W_U(fp32) 5 GB + cache ≈ 70 GB |

### B. 与原项目的差异（读结果时要知道）

- **估计器不同**：原项目是逐层链式近似（32 tokens、跳过 4 个位置、并且它的 `J_63` 含 final norm 的雅可比）；本项目是论文/Anthropic 的端到端估计器（128 tokens、跳过 16 个位置、目标是 final norm 之前的残差）。原项目文档自己也提到，Neuronpedia（端到端）lens 与它的 lens 在“概念在第几层浮现”上有差别——这是拟合管线的性质。
- **权重不同**：bf16 vs 4-bit。
- 原项目 `v0.3-qwen38-n1000` release 的 `.npz` 也能被本项目加载（格式兼容，都是 `J @ h`），想对比可以下载后用 `compare_lens.py --model Qwen/Qwen3.8-27B` 看读出层面的一致率（由于上面两点，矩阵层面的差异会比较大，读出层面更有参考意义）。

### C. 已验证 / 待你在 H200 上验证

**已在本地（CPU、无 GPU）验证：**

- 原项目 serve 逻辑的 104 个契约测试全部通过（SSE 事件顺序、干预规格解析、扫描/搜索/前提搜索/planner 流程等）；
- 在一个随机初始化的迷你 `Qwen3_5ForConditionalGeneration`（真实 Qwen3.8 tokenizer）上：
  - 分块增量解码 ≡ 全序列前向（逐层残差、logits）；
  - 流式干预 ≡ 全序列干预；
  - `jlens` 估计器的行 ≡ 独立 autograd 逐行计算；
  - `.npz` / `.pt` 读写与 `J @ h` 方向；
  - `fit_lens → merge_lens → verify_fit → readout_smoke → measure_bands → compare_lens` 全流程跑通；
  - Web App 在浏览器里实际打开，聊天流式网格、`/api/slice`、流式 swap/steer 干预（steer ⟨Beijing⟩ 确实改变了输出）、intervention scan、planner probe、legacy `/api/intervene` 全部工作。

**只能在 GPU 上验证（按教程顺序会依次覆盖）：**

- fla Triton kernel 路径（Step 5a）、在 retain_graph 下的反向正确性（Step 5b）、实际速度与显存（Step 5b）；
- 与 Neuronpedia 参考 lens 的一致性（Step 6）；
- 27B 真模型上的流式 ≡ 全前向、France→China 因果翻转（Step 11）。

### D. 环境变量速查

| 变量 | 默认 | 作用 |
|---|---|---|
| `JLENS_MODEL` | `Qwen/Qwen3.8-27B` | 模型 id 或本地路径 |
| `JLENS_PATH` | 自动选择（有歧义则拒绝启动） | lens 文件（`.npz` / `.pt`），`none` = logit lens |
| `JLENS_DEVICE` | `cuda` | 设备 |
| `JLENS_J_DTYPE` | `float32` | 显存里 J 的精度（`float16` 省一半显存） |
| `JLENS_MODE` | `active` | `presentation` = 只读展示 |
| `JLENS_READOUT_CHUNK` | `8` | 读出时每批位置数 |
| `JLENS_PERF` | 关 | `1` = 分阶段计时 |
| `JLENS_SLOW_TESTS` | 关 | `1` = 运行真模型 GPU 测试 |

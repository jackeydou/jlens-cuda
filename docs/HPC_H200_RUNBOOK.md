# 在 HPC H200 上验证 Qwen3.8-27B BF16 的 J-lens

检查日期：2026-10-08。依赖以当前 `pyproject.toml` 与 `uv.lock` 为准。本文适用于使用 SLURM 调度的 HPC；登录地址、partition、GRES、存储路径和资源限制均以实际 HPC 配置为准。

目标顺序：环境 → 一条 prompt 的 Jacobian 数值检查 → 8 条 prompt 的全流程试跑 → 1000 条正式拟合 → 读出与因果干预 → Web UI。

如果已有 **同一模型 checkpoint** 拟合出的 lens，可跳过拟合，直接做读出和干预验证。Qwen3.5、3.6、3.8 即使形状相同，也不能据此认定 lens 可通用。BF16 与上游 4-bit 版本也不应要求结果逐 token 相同。

## 0. 开始前要知道的限制

- 下文以 `gpu:h200:1` 申请一张 H200，使用 `gpu`、`gpu-short`、`gpu-interactive` 和 `short` 作为 partition 示例。运行前查询实际 HPC 的 GRES、partition、时限和账户/QOS，并替换命令中的示例值。
- 仓库 `slurm/*.sbatch` 是通用模板，含 `CHANGE_ME`；`fit_array.sbatch` 默认 24 小时。因此下面通过 `sbatch` 命令行覆盖这些字段。
- 不需要 `fit_node.sbatch`。普通账户用单卡作业数组即可，数组任务数不等于同时占用的 GPU 数。
- Linux 使用官方 torch **2.11.0+cu128 / CUDA 12.8**，具体配套依赖见下表。此前 HPC 日志中的旧环境 torch 2.14.0+cu130 因驱动过旧而无法初始化。
- `scripts/check_env.py` 的慢速 GDN fallback 警告不会返回非零；必须查看日志中选择的实现。
- H200 + Triton 3.6.0 的 gated GDN 反向有已知错误；FLA 会拒绝执行。当前环境通过 TileLang 替代该反向 op，并需要系统 CUDA 12.8 Toolkit 中的 `nvcc`。`check_env.py` 会在缺少替代路径时返回非零。
- `scripts/verify_fit.py` 当前在 Jacobian 数值不匹配时只打印 `MISMATCH`，仍返回 0。SLURM `COMPLETED` 不是数值验证通过的证据。
- 本文是代码与文档检查后的操作流程，尚未在你的 H200 上运行。时间、峰值显存和干预效果必须实测。

建议给模型、环境、缓存和 8 个 lens 分片合计预留至少 200 GB 共享盘空间。模型权重索引记录约 55.6 GB；一个全深度 FP32 lens 分片约 6.6 GB。每个拟合进程先申请 96 GB CPU RAM。

### 当前 H200 依赖基线（Linux）

| 项目 | `pyproject.toml` 中的配置 | 当前 `uv.lock` 的安装结果 |
|---|---|---|
| Python | `>=3.11` | 新环境按 `.python-version` 使用 3.11；已有 3.12 环境也可继续使用 |
| PyTorch | `torch>=2.7`，另加 Linux 约束 `torch==2.11.0` | `2.11.0+cu128` |
| PyTorch 下载源 | Linux 的 torch 显式使用 `pytorch-cu128` index | `https://download.pytorch.org/whl/cu128` |
| CUDA 用户态库 | 由 torch 传递依赖引入，无须另加 `cuda` 依赖 | `cuda-toolkit==12.8.1`、`nvidia-cuda-runtime-cu12==12.8.90`；`torch.version.cuda` 应为 `12.8` |
| Triton | 由 torch / FLA 的依赖解析引入 | `3.6.0` |
| Transformers | `transformers>=5.8` | `5.17.0` |
| FLA | `flash-linear-attention[cuda]>=0.5`，仅 Linux 安装 | `flash-linear-attention==0.5.2`、`fla-core==0.5.2` |
| TileLang | `tilelang>=0.1.15,<0.2`，仅 Linux 安装 | `tilelang==0.1.15`；运行时还需 host `nvcc` |
| 可选卷积 kernel | `conv1d` extra：`causal-conv1d>=1.4` | 锁定 `1.7.0`，默认 `uv sync` 不安装 |

`torch>=2.7` 是声明的下限，H200 实际安装版本由 Linux 约束与下载源共同决定。锁文件还保留非 Linux 平台的 torch 2.14.0；在 Mac 上看到它不代表 Linux 会安装 CUDA 13。应在集群上使用 **`uv sync --locked`**，同时上传 `pyproject.toml` 和 `uv.lock`。不要在集群上执行 `uv lock --upgrade` 或用裸 `pip install torch` 替换这一组合。[uv 的 PyTorch 源配置说明](https://docs.astral.sh/uv/guides/integration/pytorch/)

PyTorch 官方提供 2.11.0 的 cu128 wheel，其基础 CUDA 运算使用 `.venv` 中的用户态库。本项目在 H200 上拟合时还需 TileLang 的 JIT 编译，因此应加载系统 CUDA 12.8 Toolkit，确保 `nvcc` 可用。仅安装 torch 的 CUDA runtime 不能满足这一要求。[PyTorch 官方版本安装表](https://pytorch.org/get-started/previous-versions/#v2110)、[TileLang 安装说明](https://github.com/tile-ai/tilelang/blob/v0.1.15/docs/get_started/Installation.md)

排查时区分这些版本：`nvidia-smi` 的 **Driver Version** 是节点驱动版本；它的 **CUDA Version** 表示驱动支持的 CUDA 版本；`torch.version.cuda` 才是 torch 的构建版本。`nvcc --version` 则是系统 Toolkit 的编译器版本。锁文件中的 `cuda-bindings==12.9.9` 是 Python 绑定包版本，也不表示 torch 使用 CUDA 12.9。

NVIDIA 列出的 CUDA 12.x minor compatibility 最低驱动系列是 525，CUDA 13.x 是 580，但 PTX/JIT 和新特性有额外限制。本项目依赖 Triton，不能仅凭驱动达到 525 就认定可运行；必须在 H200 上完成第 3 节的 BF16 检查和第 4 节的真实 kernel 验证。加载 CUDA module 不会升级驱动，也不会改变 torch wheel 的构建版本。[NVIDIA 驱动兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)

### 已安装 CUDA 13 环境的修复

用户日志显示 Python 3.12.15 + torch 2.14.0+cu130 报 `NVIDIA driver ... too old (found version 12080)`。12080 是 CUDA driver API 的版本编码，表示 12.8，不是 `nvidia-smi` 中的驱动发行版本号。FLA 回退 CPU 是 CUDA 初始化失败的后续症状。

把更新的 `pyproject.toml` 和 `uv.lock` 上传覆盖到 HPC，然后按第 2 节申请允许联网的 CPU 计算节点修复。先停止使用同一个 `.venv` 的作业。已有 3.12 环境无需为了这个问题再切换成 3.11；新建环境仍可按下文的 3.11 流程。

```bash
cd /scratch/$USER/jlens-cuda
if [ -f slurm/hpc.env ]; then source slurm/hpc.env; fi
export UV_PYTHON=3.12
export UV_LINK_MODE=hardlink
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE UV_OFFLINE UV_FROZEN UV_NO_SYNC
uv python pin 3.12
uv sync --locked --python "$UV_PYTHON"
uv pip check --python .venv/bin/python
.venv/bin/python -m pytest -q
```

若 `slurm/hpc.env` 中已设置 `UV_PYTHON=3.11`，将其改成 3.12，确保后续提交作业仍选择已安装的 Python。`.python-version`、`slurm/hpc.env` 和 `.venv` 的 Python 应保持一致。不要再次使用旧的 cu130 wheel。CUDA 12.8 包和相应依赖需要下载一次；执行第 2 节的“安装版本核对”确认替换成功。

接着重新申请 H200，在 GPU 节点执行 `.venv/bin/python scripts/check_env.py`，并实际运行 BF16 矩阵乘法。预期 torch 显示 `2.11.0+cu128`、CUDA 12.8、Triton 3.6.0，GPU 可用且 FLA 被选中。该依赖组合已通过锁文件解析；H200 kernel、完整模型与 Jacobian 验证仍需在集群实测。

### Hopper GDN 反向错误的修复

若 `jacobian_for_prompt` 在反向计算时报 `Triton >= 3.4.0 and < 3.7.1 on Hopper GPUs produces incorrect results`，说明已进入 GPU kernel 路径。FLA 为防止产生错误 Jacobian，主动中止了计算。[FLA 的保护检查](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/common/chunk_o.py)

torch 2.11.0 的 Linux wheel 要求 `triton==3.6.0`，这里保留这套 cu128 环境，使用 FLA 的 TileLang 替代实现。不要强装新 Triton、降级 FLA 或删除这个保护检查。FLA 的 TileLang 路径同时检查包是否安装、`nvcc` 是否可用、backend dispatch 是否启用。[TileLang backend 选择条件](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/common/backends/tilelang/__init__.py)

先上传更新后的 `pyproject.toml`、`uv.lock` 和 `scripts/check_env.py`，在允许联网的 CPU 节点同步环境（停止使用同一 `.venv` 的作业）：

```bash
cd /scratch/$USER/jlens-cuda
if [ -f slurm/hpc.env ]; then source slurm/hpc.env; fi
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE UV_OFFLINE UV_FROZEN UV_NO_SYNC
uv sync --locked --python "${UV_PYTHON:-3.11}"
uv pip check --python .venv/bin/python
.venv/bin/python -c 'from importlib.metadata import version; print(version("tilelang"))'
```

在分配到 H200 的 shell 中用 `module avail cuda` 查找 CUDA **12.8** Toolkit，加载集群实际提供的模块。例如模块确实名为 `cuda/12.8` 时，执行 `module load cuda/12.8`。将实际的加载命令写入 `slurm/common.sh` 的 cluster-specific 区域，确保批处理作业也能找到编译器；只在交互 shell 中加载不足以保证后续作业一致。

```bash
command -v nvcc
nvcc --version
export FLA_TILELANG=1
unset FLA_DISABLE_BACKEND_DISPATCH
.venv/bin/python scripts/check_env.py
```

`nvcc --version` 应显示 CUDA 12.8，检查脚本应显示 TileLang workaround 的前提可用。不要安装 `tilelang[nvcc]` 来补编译器：该 extra 引入 CUDA 13 工具链，不适合此处的驱动基线。若没有可用的 12.8 Toolkit，请管理员提供；`pip` 的 CUDA 12 `nvcc` 包并不包含 FLA 所检查的可执行 `nvcc`。

随后运行第 3 节的小规模 GDN 反向检查，再重跑第 4 节的 `verify_fit.py`。安装和前提检查通过不代表 kernel 或 Jacobian 已通过 H200 数值验证。

## 1. 登录、查询资源、上传项目

**Mac 终端：** 把 `YOUR_USER` 换成账户名，`HPC_LOGIN` 换成 HPC 登录地址。下文的 `/scratch` 也需按实际共享存储路径替换。

```bash
ssh YOUR_USER@HPC_LOGIN
```

**登录节点：**

```bash
mkdir -p /scratch/$USER/jlens-cuda
sinfo -o '%P %l %G'
sinfo -N -p gpu,gpu-short,gpu-interactive -o '%N %P %G %m %t'
scontrol show partition gpu
scontrol show partition gpu-short
```

确认能看到 `h200`。如作业被 QOS、账户或内存上限拒绝，按拒绝信息与账户权限调整，不要随意换成任意 GPU。

**Mac 的另一个终端：** 从当前实际工作目录上传，避免把 Mac 的虚拟环境带到 Linux。

```bash
rsync -av \
  --exclude '.venv/' --exclude '__pycache__/' --exclude '.pytest_cache/' \
  --exclude 'data/lens/' --exclude 'logs/' \
  /Users/bytedance/Dou/jlens-cuda/ \
  YOUR_USER@HPC_LOGIN:/scratch/YOUR_USER/jlens-cuda/
```

若已有需要上传的 lens，另行复制该文件及其 provenance，不要用上面的排除规则上传它。

## 2. 安装环境并准备联网数据

登录节点用于轻量编辑、传文件和提交作业。环境安装和 CPU 密集操作在计算节点进行，并遵守 HPC 的节点使用规则。

先申请有网络的 CPU 节点；若该节点没有出网权限，使用 HPC 允许的联网安装/数据传输节点完成本节，不要假设所有 GPU 节点都能下载。

```bash
srun -p short --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=16G --time=02:00:00 --pty /bin/bash
cd /scratch/$USER/jlens-cuda
```

若 `uv` 尚未安装，在允许联网的节点执行：

```bash
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/jlens-install-uv.sh
sh /tmp/jlens-install-uv.sh
export PATH="$HOME/.local/bin:$PATH"
```

新环境先建立后续每个 shell 都可加载的环境文件。**已有 `slurm/hpc.env` 时，跳过下方 `cat` 写文件步骤**，检查文件中的 Python 版本、模型路径与缓存设置，然后从 `source slurm/hpc.env` 开始执行。已有 3.12 环境设置 `UV_PYTHON=3.12`。

```bash
mkdir -p logs data
cat > slurm/hpc.env <<'EOF'
export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="/scratch/$USER/hf"
export UV_CACHE_DIR="/scratch/$USER/uv-cache"
export UV_PYTHON_INSTALL_DIR="/scratch/$USER/uv-python"
export UV_PYTHON=3.11
export UV_LINK_MODE=hardlink
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
if [ -f data/model_snapshot.txt ]; then
    export JLENS_MODEL="$(cat data/model_snapshot.txt)"
else
    export JLENS_MODEL="Qwen/Qwen3.8-27B"
fi
EOF
source slurm/hpc.env
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE UV_OFFLINE UV_FROZEN UV_NO_SYNC
uv python pin "$UV_PYTHON"
uv sync --locked --python "$UV_PYTHON"
uv pip check --python .venv/bin/python
.venv/bin/python -m pytest -q
```

通过标准：普通测试没有 failure；需要真实 GPU/模型的测试可以 skipped。不要照搬旧教程里固定的测试数量。

`tests/test_tiny_qwen35.py` 的四个数值测试始终在 CPU 上运行，fixture 会显式选择 PyTorch 的 GDN 与卷积参考实现。Linux 环境安装了 FLA，并不意味着这些 CPU 测试应调用 Triton。旧版测试若报 `0 active drivers ([]). There should only be one.`，先同步更新后的测试文件，再重跑上述命令；若仍失败，保留完整 traceback。CPU 测试通过后，仍需按第 3、4 节在分配到 H200 的作业中验证 CUDA 与 FLA 前向/反向 kernel。

上面的环境文件适合新建 Python 3.11 环境。已有 3.12 环境时，将其中 `UV_PYTHON=3.11` 改成 3.12，并保留已有的模型路径和缓存设置；无需为了修复 CUDA 重建成 3.11。每次更换 Python 都同步更新 `.python-version`、`slurm/hpc.env` 和 `.venv`，避免后续命令使用不同解释器。

`--locked` 会检查锁文件与项目声明是否一致，不一致就停止；`--frozen` 跳过该检查，且仍可能同步安装环境。安装完成后，下文使用 `--no-sync`；SLURM 的 `common.sh` 同样通过 `UV_NO_SYNC=1` 禁用自动同步。先在联网节点修复环境，再提交离线作业。

**安装版本核对（CPU 节点也可执行）：** 此处只检查包版本，CUDA 可用性留到 GPU 节点验证。

```bash
.venv/bin/python - <<'PY'
import sys
from importlib.metadata import version
import torch

print("Python:", sys.version.split()[0], "executable:", sys.executable)
expected = {
    "torch": "2.11.0+cu128",
    "triton": "3.6.0",
    "transformers": "5.17.0",
    "flash-linear-attention": "0.5.2",
    "fla-core": "0.5.2",
    "tilelang": "0.1.15",
    "cuda-toolkit": "12.8.1",
    "nvidia-cuda-runtime-cu12": "12.8.90",
}
for name, want in expected.items():
    got = version(name)
    print(f"{name}: {got}")
    assert got == want, f"{name}: expected {want}, got {got}; check uploaded files and .venv"
assert torch.version.cuda == "12.8", torch.version.cuda
print("Locked Linux dependency versions: OK")
PY
```

上述预期值对应本次锁文件。今后更新依赖时，应一起更新版本表与断言。CPU 节点没有 GPU，`torch.cuda.is_available()` 为 False 属于正常情况，不应在该节点运行 `scripts/check_env.py` 来判定 CUDA 安装失败。

Scratch 上出现 `Failed to reflink ... Operation not supported ... falling back` 是回退日志，不等于安装失败。这里显式设置 hardlink 来跳过 reflink 尝试；若缓存与环境不在同一文件系统或不支持硬链接，会回退复制，仍可能较慢。

暂不安装可选 `causal-conv1d`，也不启用 `--compile`。先减少首跑需要排查的因素。后续确需编译卷积扩展时，再准备与 torch CUDA 12.8 匹配的系统 Toolkit 和编译工具，并用 `uv sync --locked --extra conv1d --python "$UV_PYTHON"` 安装；该 extra 可能触发源码编译。

下载模型并保存这次实际取得的 snapshot 路径，后续统一使用该 checkpoint：

```bash
uv run --no-sync python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="Qwen/Qwen3.8-27B")
Path("data/model_snapshot.txt").write_text(p + "\n")
print("Model snapshot:", p)
PY
source slurm/hpc.env
uv run --no-sync python scripts/prepare_corpus.py \
  --n 1000 --min-chars 600 --source wikitext --tokenizer "$JLENS_MODEL"
```

通过标准：**1000 条、1000 条 unique**，检查下载日志是否确实来自 WikiText。loader 在 WikiText 失败时会补 c4 或内置文本；严格 WikiText 实验遇到这种情况要先处理下载问题、移走失败的缓存并重新准备，不能把混合语料称为纯 WikiText。

8 条试跑会复用 1000 条缓存的前 8 条，无须再联网。

保存复现信息：

```bash
git rev-parse HEAD > logs/code_commit.txt
sha256sum pyproject.toml uv.lock .python-version data/corpus/prompts_1000_600.jsonl > logs/input_hashes.txt
uv pip freeze > logs/packages.txt
```

准备完成后 `exit` 释放 CPU 节点，回到登录节点。以后每次登录执行：

```bash
cd /scratch/$USER/jlens-cuda
source slurm/hpc.env
mkdir -p logs
```

## 3. 首先申请一张 H200，检查 CUDA 与 BF16

**登录节点申请：**

```bash
srun -p gpu-interactive --nodes=1 --ntasks=1 --gres=gpu:h200:1 \
  --cpus-per-task=8 --mem=96G --time=02:00:00 --pty /bin/bash
```

**获得 GPU 节点后：**

```bash
cd /scratch/$USER/jlens-cuda
source slurm/hpc.env
source slurm/common.sh
command -v nvcc
nvcc --version
nvidia-smi
uv run --no-sync python scripts/check_env.py
uv run --no-sync python - <<'PY'
import torch
assert torch.__version__ == "2.11.0+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available(), "CUDA 初始化失败：检查节点驱动和 GPU 分配"
assert "H200" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
assert torch.cuda.is_bf16_supported(), "BF16 不可用"
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
y = x @ x
torch.cuda.synchronize()
assert torch.isfinite(y).all()
print(torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda, y.dtype)
PY
```

运行前按上面的 Hopper 修复节将 CUDA 12.8 的实际模块加载命令写入 `slurm/common.sh`。通过标准：torch 为 `2.11.0+cu128`、CUDA 为 `12.8`，设备是 H200，BF16 可用，矩阵乘成功，`torch_chunk_gated_delta_rule` 选中 FLA 实现，TileLang workaround 的前提可用，并打印 `[OK]`。卷积使用 torch fallback 可以接受；正式拟合的 GDN 使用慢速 fallback 应先处理。矩阵乘通过只验证 torch 的基础 CUDA 路径，还要验证 FLA/TileLang 的前向和反向 kernel。

**先做小规模 GDN 反向检查，不加载模型权重：**

```bash
.venv/bin/python - <<'PY'
import logging
import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

logging.basicConfig(level=logging.INFO)
assert torch.cuda.is_available()
torch.manual_seed(0)
q, k, v = [torch.randn(1, 128, 4, 128, device="cuda", dtype=torch.bfloat16,
                       requires_grad=True) for _ in range(3)]
g = (-torch.rand(1, 128, 4, device="cuda")).requires_grad_()
beta = torch.rand(1, 128, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
out, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta,
                               use_qk_l2norm_in_kernel=True)
assert torch.isfinite(out).all()
for i in range(2):
    grads = torch.autograd.grad(out.float().square().mean(), (q, k, v, g, beta),
                                retain_graph=(i == 0))
    assert all(torch.isfinite(t).all() for t in grads)
torch.cuda.synchronize()
print("GDN forward + two backward passes: OK")
PY
```

应看到 `[FLA Backend] common.chunk_bwd_dqkwg -> tilelang` 和最终 `OK`。这个检查验证 kernel 可执行、重复反向和结果有限，不能替代下一节独立 autograd 行对比的数值验证。

若依旧看到 CUDA 13，检查是否上传了更新后的 `pyproject.toml` 和 `uv.lock`，以及是否使用了正确的 `.venv/bin/python`。驱动仍不兼容时先停止拟合，保存 `nvidia-smi` 和环境日志。不要只用 `pip install` 改 `.venv`：后续 `uv sync` 会按锁文件恢复依赖。

| 症状 | 处理方式 |
|---|---|
| torch 仍为 `2.14.0+cu130` 或版本断言失败 | 回联网 CPU 节点，用同一份 `pyproject.toml` / `uv.lock` 重新 `uv sync --locked`；核对 `sys.executable` 指向本项目 `.venv` |
| `driver too old` / CUDA 初始化失败 | 保存当前节点 `nvidia-smi`；确认 cu128 环境后仍失败，联系集群管理员核对驱动；`module load` 无法修复驱动 |
| torch 矩阵乘成功，但 Triton 报 PTX / toolchain 不支持 | 保存完整 kernel 错误和驱动版本，请管理员核对 PTX/JIT 兼容性；不能凭矩阵乘结果继续正式拟合 |
| Hopper 上 gated `chunk_bwd_dqkwg` 拒绝 Triton 3.6.0 | 按 Hopper 修复节安装锁定 TileLang、加载 CUDA 12.8 `nvcc`、启用 backend dispatch；先通过小规模反向检查 |
| FLA 未安装、导入失败或 GDN 选中慢速 fallback | 检查第 2 节包版本和第 3 节日志；先修复 Linux 环境，再跑 Jacobian 验证 |
| 作业尝试联网或重新安装包 | 确认已加载 `slurm/hpc.env` / `slurm/common.sh`，运行使用 `--no-sync`，模型 snapshot 与缓存已准备好 |

## 4. 验证 Jacobian 数值并测量成本

仍在 H200 交互作业中，先从 `dim_batch=8` 开始：

```bash
set -o pipefail
uv run --no-sync python scripts/verify_fit.py \
  --dim-batch 8 --n-prompts 1000 --n-gpus 4 \
  2>&1 | tee logs/verify_fit_b8.log
```

这一步计算所有 63 个源层的完整矩阵，不是小矩阵测试；首次 Triton/TileLang 编译也可能较慢。它比较批量估计器与独立 batch=1 前向/反向得到的 Jacobian 行。

必须同时检查：

1. 没有 OOM、NaN、Inf 或反向 kernel 错误。
2. `correctness: ... -> OK`，所有检查行的 cosine 都大于 0.99。
3. relative error 处于可解释范围；脚本只用 cosine 决定 verdict，大的幅度误差仍需排查。
4. 记录 `s/prompt` 和 `peak GPU memory`。留出显存余量后，再尝试 `dim_batch=16` 并比较真实耗时。

不要仅凭作业 `COMPLETED` 判定通过，也不要把增大 `dim_batch` 当成必然加速。

正式估算：`GPU-hours ≈ 秒/条 × 1000 / 3600`；`理想计算墙钟 ≈ GPU-hours / 实际并发卡数`。还要加模型加载、checkpoint、语料差异和排队时间。比如实测 120 秒/条，总量约 33.3 GPU-hours，4 卡理想计算时间约 8.3 小时；这不是 H200 的性能承诺。

验证完 `exit`，释放交互作业。下面回到登录节点提交批处理。

## 5. 8 条 prompt：把完整管线先走一遍

```bash
cd /scratch/$USER/jlens-cuda
source slurm/hpc.env
N_PROMPTS=8 DIM_BATCH=8 TAG=qwen38_bf16_pilot \
  FIT_EXTRA_ARGS='--checkpoint-every 1' \
  sbatch -p gpu-short --gres=gpu:h200:1 --time=02:00:00 \
  --array=0-0 slurm/fit_array.sbatch
squeue -u "$USER"
```

将返回的 job ID 代入：

```bash
tail -f logs/fit_JOBID_0.out
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode,MaxRSS
```

如果实测 8 条不能在示例的 2 小时内完成，改用允许更长作业的 GPU partition，并按 HPC 时限调整 `--time`。首跑 checkpoint 每条一次，避免中断时所有进度丢失。

拟合完成后，用 CPU 节点合并；避免在登录节点做大矩阵处理：

```bash
srun -p short --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=32G --time=01:00:00 --pty /bin/bash
cd /scratch/$USER/jlens-cuda
source slurm/hpc.env
uv run --no-sync python scripts/merge_lens.py \
  data/lens/shards/qwen38_bf16_pilot_n8_shard00of01.pt \
  --out data/lens/qwen38_bf16_pilot_n8 --expect-n 8
exit
```

通过标准：产生 `.npz`、`.pt`、`.provenance.json`，prompt 数为 8。该 lens 用于试跑，不代表稳定的研究结果。

## 6. 正式 1000 条：单卡数组拟合

先根据实测选择分片数量。下面以 **8 个分片、最多同时 4 个任务、每作业 8 小时** 为示例，每个分片 125 条，每作业只申请 1 张 H200；并发数和时限须符合 HPC 的账户/QOS 限制：

```bash
source slurm/hpc.env
N_PROMPTS=1000 DIM_BATCH=8 TAG=qwen38_bf16_wt128 \
  sbatch -p gpu --gres=gpu:h200:1 --time=08:00:00 \
  --array=0-7%4 slurm/fit_array.sbatch
```

只有当 `125 × 秒/条 + 加载/保存余量` 能落在 8 小时内，才用此分片数。如果每条约 240 秒，单分片计算已约 8.3 小时，应增加到 `--array=0-15%4`，每片 62–63 条。个人并发权限小于 4 时降低 `%4`。

提交前，所有任务必须使用同一模型 snapshot、1000 条语料缓存、128 tokens、skip-first=16 和相同 source。改变这些实验设置或语料时使用新的 `TAG`，避免复用旧 checkpoint。

中断与恢复：

- 默认每 10 条 checkpoint；最多丢失上次保存之后的进度。若每条很慢，设置 `FIT_EXTRA_ARGS='--checkpoint-every 1'` 或 5，权衡大文件保存开销。
- **原封不动重提完整数组范围**，脚本会跳过已完成分片，并恢复未完成分片。`--requeue` 不保证超时任务会自动恢复。
- **不要只提交一个失败索引。** 模板把 `SLURM_ARRAY_TASK_COUNT` 用作分片总数；例如把 8 片数组改成 `--array=3` 会变成总数 1，从而改变分片和文件名。
- 不要在上一次相同 TAG 的数组仍运行时重提，避免两个任务写同一个 checkpoint。

## 7. 合并正式分片

全部分片完成后申请 CPU 节点。8 片先给 128 GB RAM：当前合并脚本会同时保留所有分片，每片约 6.6 GB；16 片建议先给 192 GB，并检查 partition 是否允许。

```bash
srun -p short --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=128G --time=02:00:00 --pty /bin/bash
cd /scratch/$USER/jlens-cuda
source slurm/hpc.env
uv run --no-sync python scripts/merge_lens.py \
  data/lens/shards/qwen38_bf16_wt128_n1000_shard*of08.pt \
  --out data/lens/qwen38_bf16_wt128_n1000 --expect-n 1000
exit
```

如果用了 16 分片，glob 改成 `*of16.pt`。不要用 `shards/*.pt`，它会混入试跑文件或 checkpoint。`--expect-n 1000` 检查总数，但仍需确认完整分片集合、相同实验参数和 provenance。

模型权重为 BF16；每条 Jacobian、累加和分片为 FP32；默认合并导出的 lens 为 FP16，加载后转成 FP32 运算。这是当前代码的精度策略，BF16 指模型权重，不代表整条管线都用 BF16。

## 8. 读出、缓存解码、因果干预的最终验证

申请一张 H200 的交互作业（同第 3 节），在获得的 GPU 节点中执行：

```bash
cd /scratch/$USER/jlens-cuda
source slurm/hpc.env
source slurm/common.sh
export JLENS_PATH=data/lens/qwen38_bf16_wt128_n1000.npz
set -o pipefail
uv run --no-sync python scripts/readout_smoke.py \
  --lens "$JLENS_PATH" 2>&1 | tee logs/readout_n1000.log
JLENS_SLOW_TESTS=1 uv run --no-sync pytest \
  tests/test_gpu_real_model.py -v -s 2>&1 | tee logs/gpu_validation_n1000.log
```

先试 pilot 时，把 `JLENS_PATH` 换成 `data/lens/qwen38_bf16_pilot_n8.npz`；pilot 的干预可能失败，正式 lens 必须再验证。

通过标准与解释：

- 模型加载日志应为 `dtype=torch.bfloat16`。
- readout 输出 J-lens / logit-lens / 模型下一个 token 的比较，观察答案概念何时浮现。top-1 完全相同的比例不是单独的质量阈值。
- GPU 测试共 3 项：缓存 stream 对完整 forward 的相对误差 `< 3e-2`；空干预不改变 greedy token；France→China 的 swap 从 Paris 基线转向 Beijing/China/Peking。
- 确认真实 GPU 测试是 **3 passed**，不能把 skipped 当通过。
- 当前干预测试默认层为 30,40,48。这是一个明确的因果检查案例，不能单凭它证明所有概念、prompt 或 global-workspace 理论。失败时保留基线、tokenization 和层号，分清模型基线变化、lens 质量和干预路径问题。

可选：测量功能分带。该脚本默认复用 fit 语料，因此是探索性统计；研究验证需另备独立 held-out 语料。

```bash
uv run --no-sync python scripts/measure_bands.py \
  --lens "$JLENS_PATH" --prompts 60
```

保存的边界是启发式结果。未运行时 UI 会用默认比例划分，不能把默认配色边界当成实测证据。

结束后 `exit` 释放交互作业。

## 9. 启动 Web App 并从 Mac 访问

**登录节点：**

```bash
cd /scratch/$USER/jlens-cuda
source slurm/hpc.env
JLENS_PATH=data/lens/qwen38_bf16_wt128_n1000.npz \
  sbatch -p gpu --gres=gpu:h200:1 --time=02:00:00 slurm/serve.sbatch
```

先用 2 小时验证 UI，只有需要更久且允许时才延长。查看 `logs/serve_JOBID.out`，取得实际计算节点名，等待服务启动完成。

**Mac 的另一个终端：** 将 `GPU_NODE` 换成日志里的节点。

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 8765:127.0.0.1:8765 \
  -J YOUR_USER@HPC_LOGIN YOUR_USER@GPU_NODE
```

在 Mac 打开 <http://localhost:8765/>。这个命令把端口转发到 **计算节点的 loopback**，与服务绑定的 `127.0.0.1` 相符。

若 HPC 不允许直接 SSH 到计算节点，使用 HPC 管理员支持的转发方式；需要时在仍绑定 loopback 的前提下建立计算节点与登录节点间的第二段隧道。不要把 login→GPU_NODE 的普通远端端口转发直接套在 loopback 服务上。

浏览器先用短 prompt、少量生成 token，验证读出网格和 Replace/Add/Remove/Erase。最后取消服务作业：

```bash
scancel SERVE_JOBID
```

## 10. 完成任务应保留的证据

| 证据 | 要回答的问题 |
|---|---|
| git commit、uv.lock 哈希、packages.txt | 用的哪版代码与依赖？ |
| model_snapshot.txt、模型 dtype 日志 | 用的哪个 Qwen3.8 checkpoint，是否为 BF16？ |
| corpus 哈希、来源和 token 长度统计 | 哪 1000 条文本，有无 fallback？ |
| verify_fit 日志、cosine 和 relative error | Jacobian 数值是否可信？ |
| 秒/条、GPU 显存、SLURM 状态和 RAM | 资源和耗时是多少？ |
| 最终 lens、各片 provenance、总 prompt 数 | lens 是否完整且可追溯？ |
| readout 和 GPU 测试日志 | 读出、缓存和因果干预是否有效？ |
| bands JSON 与 Web 示例截图 | 展示哪些现象，哪些仅为启发式？ |

任务完成后，把最终 lens、provenance、日志和结果备份到持久项目空间或其他允许的存储。Scratch 会清理，不能作为唯一副本。

## 查阅来源

- [当前项目依赖配置](../pyproject.toml) 与 [跨平台依赖锁文件](../uv.lock)
- [PyTorch 官方版本安装表：2.11.0 / CUDA 12.8](https://pytorch.org/get-started/previous-versions/#v2110)
- [uv 的 PyTorch 源与平台配置](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [Qwen3.8-27B 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)
- [模型权重索引](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/model.safetensors.index.json)
- [NVIDIA CUDA 驱动兼容表](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [本项目固定的 Anthropic Jacobian 估计器](https://github.com/anthropics/jacobian-lens/blob/581d398613e5602a5af361e1c34d3a92ea82ba8e/jlens/fitting.py)

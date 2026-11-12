#!/bin/bash
# ============================================================
# 安装 PyTorch（走阿里云镜像，60 秒内装完）
#
# 关键经验：
#   1. download.pytorch.org 在国内只有 6 KB/s，绝对不能直连
#   2. torch 2.10 需要一整套 nvidia-* CUDA 子包，缺一个就 ImportError
#   3. 用 --no-deps 先装主包，再手动补子包，避免 pip 解析冲突
# ============================================================
set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[X]${NC} $*"; }

ENV_PATH="${ENV_PATH:-/data/env/py312}"
source "$ENV_PATH/bin/activate"

TORCH_VER="${TORCH_VER:-2.10.0}"
TV_VER="${TV_VER:-0.25.0}"
TA_VER="${TA_VER:-2.10.0}"
CUDA_TAG="${CUDA_TAG:-cu128}"

PY_MINOR=$(python -c "import sys;print(sys.version_info[1])")
CP="cp3${PY_MINOR}"

echo "Python 3.${PY_MINOR} -> wheel 标签 $CP"
echo "目标: torch $TORCH_VER+$CUDA_TAG"

# ---------- 0) 检查是否已装好 ----------
if python -c "
import torch, sys
assert torch.cuda.is_available(), 'cuda off'
assert '+' in torch.__version__, 'cpu build'
print(torch.__version__)
" 2>/dev/null; then
    ok "torch 已可用: $(python -c 'import torch;print(torch.__version__)')"
    warn "跳过安装（如需重装：rm -rf $ENV_PATH 后重跑）"
    exit 0
fi

# ---------- 1) 测源 ----------
echo ""
echo "--- 测试镜像速度 ---"
ALIYUN="https://mirrors.aliyun.com/pytorch-wheels/$CUDA_TAG"
SPEED_ALIYUN=$(curl -L -o /dev/null -s -w "%{speed_download}" --max-time 12 \
    -r 0-10000000 "$ALIYUN/torch-$TORCH_VER%2B$CUDA_TAG-$CP-$CP-manylinux_2_28_x86_64.whl" 2>/dev/null || echo 0)
SPEED_ALIYUN=${SPEED_ALIYUN%.*}
echo "  阿里云 $(( SPEED_ALIYUN / 1024 )) KB/s"
if [ "${SPEED_ALIYUN:-0}" -lt 500000 ]; then
    warn "阿里云速度偏低，尝试官方源"
    USE_OFFICIAL=1
else
    USE_OFFICIAL=0
fi

# ---------- 2) 安装主包 ----------
echo ""
echo "--- 安装 torch 主包 ---"
mkdir -p /data/wheels && cd /data/wheels

install_pkg() {
    local name="$1" ver="$2"
    local fn="${name}-${ver}%2B${CUDA_TAG}-${CP}-${CP}-manylinux_2_28_x86_64.whl"
    local out="${name}-${ver}+${CUDA_TAG}-${CP}-${CP}-manylinux_2_28_x86_64.whl"

    if python -c "import ${name}; print(${name}.__version__)" 2>/dev/null | grep -q "${ver}"; then
        ok "$name $ver 已存在"
        return 0
    fi

    if [ "$USE_OFFICIAL" = "1" ]; then
        pip install "${name}==${ver}" --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" 2>&1 | tail -3
    else
        echo "  下载 $out ..."
        if curl -L --retry 3 --retry-delay 2 -C - -s -o "$out" "$ALIYUN/$fn"; then
            pip install --no-deps "$out" 2>&1 | tail -2
        else
            warn "阿里云失败，回退官方源"
            pip install "${name}==${ver}" --index-url "https://download.pytorch.org/whl/${CUDA_TAG}" 2>&1 | tail -3
        fi
    fi
}

install_pkg torch "$TORCH_VER"
install_pkg torchvision "$TV_VER"
install_pkg torchaudio "$TA_VER"

# ---------- 3) 补 nvidia 子包 ----------
echo ""
echo "--- 补齐 CUDA 运行时子包 ---"
pip install \
    nvidia-cublas-cu12 \
    nvidia-cuda-cupti-cu12 \
    nvidia-cuda-nvrtc-cu12 \
    nvidia-cuda-runtime-cu12 \
    nvidia-cudnn-cu12 \
    nvidia-cufft-cu12 \
    nvidia-cufile-cu12 \
    nvidia-curand-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-cusparselt-cu12 \
    nvidia-nccl-cu12 \
    nvidia-nvjitlink-cu12 \
    nvidia-nvshmem-cu12 \
    nvidia-nvtx-cu12 \
    cuda-bindings \
    triton \
    2>&1 | tail -6

# ---------- 4) 基础依赖 ----------
echo ""
echo "--- 安装基础依赖 ---"
pip install filelock fsspec jinja2 networkx sympy typing-extensions -q 2>&1 | tail -2

# ---------- 5) 验证 ----------
echo ""
echo "--- 验证 ---"
python - <<'PYEOF'
import torch
print("torch      ", torch.__version__)
print("cuda       ", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用")
print("gpu        ", torch.cuda.get_device_name(0))
print("vram GB    ", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
print("capability ", torch.cuda.get_device_capability(0))
x = torch.randn(512, 512, device="cuda", dtype=torch.float16)
print("matmul     ", bool((x @ x).sum().item() != 0))
PYEOF
ok "torch 安装完成"

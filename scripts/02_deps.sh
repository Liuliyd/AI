#!/bin/bash
# ============================================================
# 安装应用依赖
#
# 关键经验：
#   1. 一次性装全部会触发 ResolutionImpossible，必须分步
#   2. insightface / mediapipe 经常失败，失败要能继续
#   3. numpy 必须 1.26.x，2.x 会崩 opencv/insightface
#   4. 每步之后检查 torch 有没有被降级
# ============================================================
set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[X]${NC} $*"; }

ENV_PATH="${ENV_PATH:-/data/env/py312}"
source "$ENV_PATH/bin/activate"

IDX="--index-url https://pypi.tuna.tsinghua.edu.cn/simple"

TORCH_BEFORE=$(python -c "import torch;print(torch.__version__)" 2>/dev/null || echo none)
echo "安装前 torch: $TORCH_BEFORE"

check_torch() {
    local now
    now=$(python -c "import torch;print(torch.__version__)" 2>/dev/null || echo missing)
    if [ "$now" != "$TORCH_BEFORE" ]; then
        err "torch 被改成了 $now（原本 $TORCH_BEFORE）"
        err "修复: pip install --force-reinstall torch==$TORCH_BEFORE --index-url https://download.pytorch.org/whl/cu128"
        return 1
    fi
    return 0
}

step() { echo ""; echo "########## $* ##########"; }

step "1/8 numpy（锁 1.26）"
pip install $IDX "numpy==1.26.4" 2>&1 | tail -2
check_torch || warn "numpy 装完 torch 变了，继续但需警惕"

step "2/8 diffusers 生态"
pip install $IDX diffusers transformers accelerate safetensors 2>&1 | tail -4
ok "diffusers $(python -c 'import diffusers;print(diffusers.__version__)' 2>/dev/null)"

step "3/8 图像处理"
pip install $IDX \
    opencv-python-headless pillow scipy scikit-image \
    einops timm kornia 2>&1 | tail -4

step "4/8 onnx + 分割"
pip install $IDX protobuf onnx onnxruntime-gpu 2>&1 | tail -3
pip install $IDX rembg 2>&1 | tail -3 || warn "rembg 失败，可跳过（分割会退化为全图）"

step "5/8 insightface（易失败）"
if pip install $IDX insightface 2>&1 | tail -3; then
    ok "insightface 装好"
else
    warn "insightface 失败（人脸检测不可用，不影响换身主体）"
fi

step "6/8 mediapipe（易失败）"
if pip install $IDX mediapipe 2>&1 | tail -3; then
    ok "mediapipe 装好"
else
    warn "mediapipe 失败（会改用 controlnet-aux 的 OpenPose）"
fi

step "7/8 音频 + RVC 依赖"
pip install $IDX \
    soundfile librosa sounddevice av \
    faiss-cpu praat-parselmouth resampy \
    torchcrepe 2>&1 | tail -4 || warn "部分音频包失败"

step "8/8 服务框架"
pip install $IDX \
    fastapi "uvicorn[standard]" websockets aiohttp \
    python-multipart orjson 2>&1 | tail -4

step "可选项：controlnet-aux"
pip install $IDX controlnet-aux 2>&1 | tail -2 || warn "controlnet-aux 失败，位置提取改用 mediapipe"

# ---------- 最终检查 ----------
step "最终验证"
python - <<'PYEOF'
import importlib
mods = [
    ("torch", True), ("torchvision", True), ("numpy", True),
    ("diffusers", True), ("transformers", True), ("accelerate", True),
    ("cv2", True), ("PIL", True), ("scipy", True), ("skimage", True),
    ("einops", True), ("timm", True), ("onnxruntime", True),
    ("fastapi", True), ("uvicorn", True), ("websockets", True),
    ("librosa", True), ("soundfile", True),
    ("rembg", False), ("insightface", False), ("mediapipe", False),
]
required_missing = []
print("%-16s %-14s %s" % ("包", "版本", "状态"))
print("-" * 46)
for name, required in mods:
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", "ok")
        print("%-16s %-14s OK" % (name, str(v)[:14]))
    except Exception:
        print("%-16s %-14s %s" % (name, "-", "MISSING" if required else "可选缺失"))
        if required:
            required_missing.append(name)

import torch, numpy
print("-" * 46)
print("torch cuda :", torch.cuda.is_available())
print("numpy      :", numpy.__version__)

if required_missing:
    raise SystemExit("必需包缺失: " + ", ".join(required_missing))
if not numpy.__version__.startswith("1.26"):
    raise SystemExit("numpy 必须是 1.26.x，当前 " + numpy.__version__)
PYEOF

ok "依赖安装完成"

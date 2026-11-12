#!/bin/bash
# ============================================================
# 下载模型
#
# 关键经验：
#   1. hf-mirror 下大文件会卡死（unet 1.7G 卡了 10 分钟没动）
#   2. ModelScope 实测 14 MB/s，比 hf-mirror 快 5 倍
#   3. 用 curl 直下单个文件，比 snapshot_download 可靠得多
# ============================================================
set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[X]${NC} $*"; }

MODEL_DIR="${MODEL_DIR:-/data/models}"
MS="https://modelscope.cn/models/AI-ModelScope"
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

fetch() {
    local url="$1" out="$2" min_kb="${3:-1}"
    if [ -f "$out" ]; then
        local sz_kb=$(du -k "$out" | cut -f1)
        if [ "$sz_kb" -ge "$min_kb" ]; then
            ok "已存在 $(basename "$out")  ($(du -h "$out" | cut -f1))"
            return 0
        fi
        warn "文件不完整，重下 $(basename "$out")"
        rm -f "$out"
    fi
    mkdir -p "$(dirname "$out")"
    echo "  下载 $(basename "$out") ..."
    if curl -L --retry 3 --retry-delay 2 -C - -s -o "$out" "$url"; then
        ok "$(basename "$out")  $(du -h "$out" | cut -f1)"
        return 0
    fi
    err "失败: $url"
    return 1
}

# ---------------- SD-Turbo ----------------
echo "============================================"
echo "  1) SD-Turbo（实时出图核心）"
echo "============================================"
SD="$MODEL_DIR/sd-turbo"
mkdir -p "$SD"/{unet,text_encoder,vae,tokenizer,scheduler}

BASE="$MS/sd-turbo/resolve/master"

fetch "$BASE/unet/diffusion_pytorch_model.fp16.safetensors" \
      "$SD/unet/diffusion_pytorch_model.fp16.safetensors" 1000000

fetch "$BASE/text_encoder/model.fp16.safetensors" \
      "$SD/text_encoder/model.fp16.safetensors" 300000

fetch "$BASE/vae/diffusion_pytorch_model.fp16.safetensors" \
      "$SD/vae/diffusion_pytorch_model.fp16.safetensors" 100000

# 小配置文件
for f in model_index.json \
         unet/config.json text_encoder/config.json vae/config.json \
         tokenizer/vocab.json tokenizer/merges.txt \
         tokenizer/tokenizer_config.json tokenizer/special_tokens_map.json \
         scheduler/scheduler_config.json
do
    if [ ! -f "$SD/$f" ]; then
        curl -L -s -o "$SD/$f" "$BASE/$f" 2>/dev/null && echo "  got $f" || warn "fail $f"
    fi
done

# ---------- 完整性检查 ----------
echo ""
echo "--- SD-Turbo 文件检查 ---"
SD_OK=1
for f in unet/diffusion_pytorch_model.fp16.safetensors \
         text_encoder/model.fp16.safetensors \
         vae/diffusion_pytorch_model.fp16.safetensors \
         model_index.json tokenizer/vocab.json scheduler/scheduler_config.json
do
    if [ -f "$SD/$f" ]; then
        printf "  OK   %-52s %s\n" "$f" "$(du -h "$SD/$f"|cut -f1)"
    else
        printf "  MISS %s\n" "$f"
        SD_OK=0
    fi
done
[ "$SD_OK" -eq 1 ] || { err "SD-Turbo 不完整，无法继续"; exit 1; }
ok "SD-Turbo 完整  ($(du -sh "$SD" | cut -f1))"

# ---------------- ControlNet OpenPose ----------------
echo ""
echo "============================================"
echo "  2) ControlNet OpenPose（动作跟随）"
echo "============================================"
CN="$MODEL_DIR/controlnet_openpose"
mkdir -p "$CN"
fetch "$MS/controlnet-openpose-sdxl-1.0/resolve/master/diffusion_pytorch_model.safetensors" \
      "$CN/diffusion_pytorch_model.safetensors" 1000000 \
  || warn "ControlNet 下载失败，可稍后补（不影响基础换身）"

# ---------------- 可选：RMBG ----------------
echo ""
echo "============================================"
echo "  3) RMBG 人物分割（可选，rembg 会自己下）"
echo "============================================"
warn "rembg 首次运行会自动下载 u2netp（约 5MB），无需手动准备"

# ---------------- 汇总 ----------------
echo ""
echo "============================================"
echo "  模型汇总"
echo "============================================"
du -sh "$MODEL_DIR"/* 2>/dev/null | sort -h
echo ""
df -h "$MODEL_DIR" | tail -1
ok "模型准备完成"

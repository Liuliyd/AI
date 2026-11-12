#!/bin/bash
# 启动服务
set -e
ENV_PATH="${ENV_PATH:-/data/env/py312}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ENV_PATH/bin/activate"

export MODEL_DIR="${MODEL_DIR:-/data/models}"
export W="${W:-384}"
export H="${H:-384}"
export STEPS="${STEPS:-4}"
export STRENGTH="${STRENGTH:-0.55}"
export PORT="${PORT:-8000}"

echo "============================================================"
echo "  ZapSwap 服务"
echo "============================================================"
echo "  模型目录  $MODEL_DIR"
echo "  分辨率    ${W}x${H}"
echo "  步数      $STEPS"
echo "  强度      $STRENGTH"
echo "  端口      $PORT"
echo "============================================================"
echo ""

cd "$ROOT"
exec python server/app.py

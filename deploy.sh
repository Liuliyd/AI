#!/bin/bash
# ============================================================
# ZapSwap 一键部署
# 用法: bash deploy.sh [--skip-bench]
# ============================================================
set -e

cd "$(dirname "$0")"
ROOT="$(pwd)"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[X]${NC} $*"; }
step() { echo; echo "============================================================"; echo "  $*"; echo "============================================================"; }

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/data/hf}"
export ENV_PATH="${ENV_PATH:-/data/env/py312}"

SKIP_BENCH=0
for a in "$@"; do
    [ "$a" = "--skip-bench" ] && SKIP_BENCH=1
done

START_TS=$(date +%s)

step "步骤 0/6  环境检查"
bash "$ROOT/scripts/00_check.sh" || { err "环境检查未通过，见上方输出"; exit 1; }

step "步骤 1/6  创建 Python 环境"
if [ -f "$ENV_PATH/bin/python" ]; then
    ok "venv 已存在: $ENV_PATH"
else
    bash "$ROOT/scripts/00b_venv.sh" || exit 1
fi
source "$ENV_PATH/bin/activate"
ok "python: $(python -V 2>&1)"

step "步骤 2/6  安装 PyTorch"
bash "$ROOT/scripts/01_torch.sh" || exit 1

step "步骤 3/6  安装应用依赖"
bash "$ROOT/scripts/02_deps.sh" || exit 1

step "步骤 4/6  下载模型"
bash "$ROOT/scripts/03_models.sh" || exit 1

if [ "$SKIP_BENCH" -eq 0 ]; then
    step "步骤 5/6  性能基准"
    bash "$ROOT/scripts/04_bench.sh" || warn "基准测试失败，不影响启动"
else
    step "步骤 5/6  性能基准（已跳过）"
fi

step "步骤 6/6  部署完成"
END_TS=$(date +%s)
echo ""
ok "总耗时 $(( (END_TS - START_TS) / 60 )) 分 $(( (END_TS - START_TS) % 60 )) 秒"
echo ""
echo "下一步："
echo "  启动服务   bash scripts/99_start.sh"
echo "  开隧道     CF_TOKEN=<token> bash scripts/98_tunnel.sh"
echo "  本地客户端 python client/client.py --host https://你的域名 --cam 0"
echo ""
echo "环境变量已写入 $ROOT/.envrc，新终端执行："
echo "  source $ROOT/.envrc"
echo ""

cat > "$ROOT/.envrc" <<EOF
source $ENV_PATH/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/data/hf
export MODEL_DIR=/data/models
EOF
ok ".envrc 已生成"

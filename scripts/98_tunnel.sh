#!/bin/bash
# 启动 Cloudflare 隧道，把本地 8000 暴露到公网
#
# 用法:
#   CF_TOKEN="eyJh..." bash scripts/98_tunnel.sh
#   CF_URL=http://localhost:8000 CF_TOKEN="..." bash scripts/98_tunnel.sh
set -e
GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
ok()  { echo -e "${GREEN}[OK]${NC} $*"; }
err() { echo -e "${RED}[X]${NC} $*"; }

TARGET="${CF_URL:-http://localhost:8000}"
TOKEN="${CF_TOKEN:-}"

if [ -z "$TOKEN" ]; then
    err "需要 CF_TOKEN 环境变量"
    echo "获取方式:"
    echo "  1) cloudflared tunnel login"
    echo "  2) cloudflared tunnel create zapswap"
    echo "  3) cloudflared tunnel token <tunnel-id>"
    echo ""
    echo "或者用临时域名（免注册）:"
    echo "  cloudflared tunnel --url $TARGET"
    exit 1
fi

# 安装 cloudflared
if ! command -v cloudflared >/dev/null 2>&1; then
    echo "安装 cloudflared ..."
    ARCH=$(uname -m)
    case "$ARCH" in
        x86_64)  CF_URL_BIN="cloudflared-linux-amd64" ;;
        aarch64) CF_URL_BIN="cloudflared-linux-arm64" ;;
        *) err "不支持的架构 $ARCH"; exit 1 ;;
    esac
    curl -L -s -o /usr/local/bin/cloudflared \
        "https://github.com/cloudflare/cloudflared/releases/latest/download/$CF_URL_BIN"
    chmod +x /usr/local/bin/cloudflared
fi
ok "cloudflared $(cloudflared --version 2>&1 | head -1)"

echo ""
echo "隧道目标: $TARGET"
echo "启动中（保持此终端不要关）..."
echo ""
exec cloudflared tunnel --no-autoupdate run --token "$TOKEN" --url "$TARGET"

#!/bin/bash
# 创建 Python 虚拟环境
set -e
GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
ok()  { echo -e "${GREEN}[OK]${NC} $*"; }
err() { echo -e "${RED}[X]${NC} $*"; }

ENV_PATH="${ENV_PATH:-/data/env/py312}"
mkdir -p "$(dirname "$ENV_PATH")"

# 先确保 venv 模块可用
if ! python3 -c "import ensurepip" 2>/dev/null; then
    PV=$(python3 -c "import sys;print('%d.%d'%sys.version_info[:2])")
    echo "缺 ensurepip，安装 python${PV}-venv ..."
    if command -v apt-get >/dev/null 2>&1; then
        apt-get update -qq 2>/dev/null || true
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "python${PV}-venv" 2>&1 | tail -3
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-pip 2>&1 | tail -2
    fi
fi

if ! python3 -c "import ensurepip" 2>/dev/null; then
    err "ensurepip 仍不可用，请手动: apt install python3-venv"
    exit 1
fi

rm -rf "$ENV_PATH"
python3 -m venv "$ENV_PATH"
source "$ENV_PATH/bin/activate"

# 国内源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple >/dev/null 2>&1 || true
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn >/dev/null 2>&1 || true

pip install -U pip setuptools wheel -q
ok "venv: $ENV_PATH"
ok "pip: $(pip --version)"

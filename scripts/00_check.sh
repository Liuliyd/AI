#!/bin/bash
# 环境检查：GPU / CUDA / Python / 磁盘 / 网络
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[X]${NC} $*"; }

FAIL=0

echo "--- 系统 ---"
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "  $PRETTY_NAME"
fi
echo "  内核 $(uname -r)"
echo "  内存 $(free -g | awk '/^Mem:/{print $2}') GB"

echo ""
echo "--- GPU ---"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    err "nvidia-smi 不存在，没有 GPU 驱动"
    FAIL=1
else
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+' | head -1)

    echo "  型号   $GPU_NAME"
    echo "  显存   $((GPU_MEM / 1024)) GB"
    echo "  算力   ${GPU_CC:-未知}"
    echo "  CUDA   ${CUDA_VER:-未知}"

    if [ -n "$GPU_CC" ]; then
        CC_MAJOR=$(echo "$GPU_CC" | cut -d. -f1)
        if [ "$CC_MAJOR" -lt 8 ]; then
            warn "算力 $GPU_CC < 8.0（Turing 或更老）"
            warn "本方案在这类卡上只有 3~5 fps，不建议用于实时场景"
            warn "建议换 RTX 3090 / 4090 / A5000 及以上"
        elif [ "$CC_MAJOR" -eq 8 ]; then
            ok "Ampere 架构，性能可接受"
        else
            ok "Ada / Hopper 架构，性能优秀"
        fi
    fi

    if [ "$GPU_MEM" -lt 8000 ]; then
        err "显存 ${GPU_MEM}MB 太小，至少需要 12GB（推荐 24GB）"
        FAIL=1
    elif [ "$GPU_MEM" -lt 12000 ]; then
        warn "显存 ${GPU_MEM}MB 偏小，只能跑低分辨率"
    else
        ok "显存充足"
    fi
fi

echo ""
echo "--- Python ---"
PY3=$(command -v python3 || true)
if [ -z "$PY3" ]; then
    err "python3 不存在"
    FAIL=1
else
    PVER=$($PY3 -V 2>&1 | awk '{print $2}')
    echo "  $PY3  ($PVER)"
    PY_MINOR=$(echo "$PVER" | cut -d. -f2)
    if [ "$PY_MINOR" -lt 10 ]; then
        warn "Python $PVER 偏旧，建议 3.10~3.12"
    elif [ "$PY_MINOR" -eq 12 ]; then
        ok "Python 3.12，配 torch 2.10 无问题"
    else
        ok "Python 版本合适"
    fi
fi

echo ""
echo "--- venv 支持 ---"
if $PY3 -c "import venv" 2>/dev/null && $PY3 -c "import ensurepip" 2>/dev/null; then
    ok "venv + ensurepip 可用"
else
    warn "缺 venv/ensurepip，尝试自动安装 python3-venv"
    if command -v apt-get >/dev/null 2>&1; then
        PV=$(echo "$PVER" | cut -d. -f1,2)
        apt-get update -qq 2>/dev/null
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "python${PV}-venv" 2>&1 | tail -3
        if $PY3 -c "import ensurepip" 2>/dev/null; then
            ok "已安装 python${PV}-venv"
        else
            err "安装失败，请手动执行: apt install python${PV}-venv"
            FAIL=1
        fi
    else
        FAIL=1
    fi
fi

echo ""
echo "--- 磁盘 ---"
echo "  /      $(df -h / | awk 'NR==2{print $4}') 可用"
if [ -d /data ]; then
    DATA_AVAIL_KB=$(df -k /data | awk 'NR==2{print $4}')
    echo "  /data  $(df -h /data | awk 'NR==2{print $4}') 可用"
    if [ "$DATA_AVAIL_KB" -lt 20000000 ]; then
        err "/data 可用空间不足 20GB，模型放不下"
        FAIL=1
    else
        ok "/data 空间充足（环境与模型装这里）"
    fi
else
    ROOT_AVAIL_KB=$(df -k / | awk 'NR==2{print $4}')
    if [ "$ROOT_AVAIL_KB" -lt 25000000 ]; then
        err "根分区可用不足 25GB"
        FAIL=1
    fi
    warn "没有 /data，将装到根分区"
fi

echo ""
echo "--- 网络 ---"
test_url() {
    local n="$1" u="$2"
    printf "  %-14s " "$n"
    local code
    code=$(curl -o /dev/null -s -w "%{http_code}" --max-time 8 "$u" 2>/dev/null)
    if [ "$code" = "200" ] || [ "$code" = "302" ] || [ "$code" = "206" ]; then
        echo -e "${GREEN}$code${NC}"
    else
        echo -e "${RED}$code${NC}"
    fi
}
test_url "aliyun-pytorch" "https://mirrors.aliyun.com/pytorch-wheels/"
test_url "tsinghua-pypi" "https://pypi.tuna.tsinghua.edu.cn/simple"
test_url "modelscope" "https://modelscope.cn"
test_url "hf-mirror" "https://hf-mirror.com"

echo ""
echo "=== 检查结束 ==="
if [ $FAIL -ne 0 ]; then
    echo -e "${RED}存在阻断问题，请先解决${NC}"
    exit 1
fi
echo -e "${GREEN}环境检查通过${NC}"

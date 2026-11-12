# 踩坑记录

这里记录的是从零搭建过程中**实际遇到并解决**的问题。遇到报错先查这里。

---

## 一、PyTorch 相关

### 1.1 torch 下载只有 6 KB/s

**现象**

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
# 卡在 "Downloading torch-2.10.0+cu128...whl (916.9 MB)" 十几分钟没动
```

**原因**

`download.pytorch.org` 在国内实测 **6.4 KB/s**，916MB 要下 40 小时。

**解决**

用阿里云镜像直下 wheel：

```bash
curl -L -o torch.whl \
  "https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.10.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl"
pip install --no-deps torch.whl
```

实测速度 **17.9 MB/s**，1 分钟下完。

**测速对比**（同一台机器）

```
download.pytorch.org     6.4 KB/s
mirror.sjtu.edu.cn       0 B/s（不通）
mirrors.aliyun.com       17.9 MB/s
```

---

### 1.2 ImportError: libcusparseLt.so.0

**现象**

```python
import torch
# ImportError: libcusparseLt.so.0: cannot open shared object file
```

**原因**

torch 2.10 的 wheel 是 `--no-deps` 装的，缺 CUDA 运行时子包。

**解决**

```bash
pip install nvidia-cusparselt-cu12 nvidia-cublas-cu12 nvidia-cusparse-cu12 \
    nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cuda-runtime-cu12 \
    nvidia-nvjitlink-cu12 nvidia-cusolver-cu12 nvidia-cufile-cu12 \
    nvidia-nccl-cu12 nvidia-nvshmem-cu12 nvidia-nvtx-cu12 \
    nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cudnn-cu12 \
    cuda-bindings triton
```

完整清单见 `scripts/01_torch.sh`。

**验证**

```python
import torch
print(torch.__version__, torch.cuda.is_available())
x = torch.randn(512, 512, device="cuda", dtype=torch.float16)
print((x @ x).sum().item())
```

---

### 1.3 torch 被静默降级成 CPU 版

**现象**

本来 `torch 2.7.1+cu118`，装了个包之后变成 `torch 2.15.0+cpu`，`cuda.is_available()` 变 `False`。

**原因**

`pip install easyocr`（以及很多带 torch 依赖的包）会从默认源拉最新 torch，而默认源给的是 **CPU 版**。

**这是最危险的坑**：不会报错，只是 CUDA 悄悄失效。

**解决**

```bash
# 1) 卸载错误的
pip uninstall -y torch torchvision torchaudio

# 2) 重装 CUDA 版（注意 index-url）
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu118
```

**预防**

```bash
# 装任何可能依赖 torch 的包时，加 --no-deps
pip install easyocr --no-deps
# 或者装完立刻检查
pip list | grep -E "^torch |^numpy "
```

---

### 1.4 Python 3.12 装不了 torch 2.1.0

**现象**

```
ERROR: Could not find a version that satisfies the requirement torch==2.1.0
```

**原因**

```
torch 2.1.0  →  wheel 只有 cp38~cp311
Python 3.12  →  需要 torch >= 2.2.0（cp312 从 2.2 开始提供）
```

**解决**

要么换 Python 3.10/3.11（`uv venv --python 3.10`），要么把 torch 升到 2.2+。

**推荐**：Python 3.12 + torch 2.10（本仓库默认组合）。

---

## 二、依赖安装

### 2.1 ResolutionImpossible

**现象**

一次性装全部依赖时：

```
ERROR: ResolutionImpossible
```

**原因**

版本约束互相冲突，pip 无法同时满足。

**解决**

分步装，每步只装一组（见 `scripts/02_deps.sh`）：

```
① numpy==1.26.4
② diffusers transformers accelerate safetensors
③ opencv pillow scipy scikit-image einops timm kornia
④ protobuf onnx onnxruntime-gpu rembg
⑤ insightface（单独）
⑥ mediapipe（单独）
⑦ 音频相关
⑧ fastapi uvicorn websockets...
```

---

### 2.2 numpy 2.x 导致一堆库崩溃

**现象**

```
AttributeError: module 'numpy' has no attribute 'float'
```

**原因**

numpy 2.0 移除了大量别名，`opencv` / `insightface` / `mediapipe` 都还在用。

**解决**

```bash
pip install "numpy==1.26.4"
```

**注意**：torch 2.10 会建议 numpy 2.x，忽略这个建议，1.26 更稳。

---

### 2.3 insightface / mediapipe 装不上

**原因**

- 需要编译（缺 gcc / python-dev）
- 或版本和 Python 3.12 不匹配

**解决**

```bash
apt-get install -y build-essential python3.12-dev
pip install insightface
```

**如果还失败**：可以跳过。影响是：

```
insightface  →  人脸检测/对齐不可用（换身主体不受影响）
mediapipe    →  改用 controlnet-aux 的 OpenPose
```

---

## 三、模型下载

### 3.1 hf-mirror 下大文件卡死

**现象**

```python
snapshot_download(repo_id="stabilityai/sd-turbo", ...)
# 卡在 60%，unet 目录空的，10 分钟没动静
```

**原因**

hf-mirror 对大文件（>1GB）会走 xet-bridge，国内不稳定。

**解决**

**用 ModelScope 直下单个文件**：

```bash
curl -L -o unet/diffusion_pytorch_model.fp16.safetensors \
  "https://modelscope.cn/models/AI-ModelScope/sd-turbo/resolve/master/unet/diffusion_pytorch_model.fp16.safetensors"
```

**测速对比**

```
hf-mirror      2.9 MB/s（且会卡死）
modelscope    14.0 MB/s（稳定）
```

---

### 3.2 SD-Turbo 文件不完整

**必须有的文件**

```
sd-turbo/
├── model_index.json
├── unet/
│   ├── config.json
│   └── diffusion_pytorch_model.fp16.safetensors   1.7 GB
├── text_encoder/
│   ├── config.json
│   └── model.fp16.safetensors                     650 MB
├── vae/
│   ├── config.json
│   └── diffusion_pytorch_model.fp16.safetensors   160 MB
├── tokenizer/
│   ├── vocab.json
│   ├── merges.txt
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
└── scheduler/
    └── scheduler_config.json
```

缺任何一个都会在 `from_pretrained` 时失败。用 `scripts/03_models.sh` 自动补齐。

---

## 四、Python 环境

### 4.1 ensurepip is not available

**现象**

```
The virtual environment was not created successfully because ensurepip is not available
```

**解决**

```bash
apt-get install -y python3.12-venv
# 然后重建 venv
rm -rf /data/env/py312
python3 -m venv /data/env/py312
```

---

### 4.2 pip 报 externally-managed-environment

**现象**

```
error: externally-managed-environment
```

**原因**

PEP 668，Ubuntu 24.04 的系统 Python 禁止 pip 直接装包。

**解决**

**永远用 venv**，不要动系统 Python：

```bash
python3 -m venv /data/env/py312
source /data/env/py312/bin/activate
pip install ...
```

---

## 五、SSH / 远程操作

### 5.1 nohup ... & 在 ssh exec 里立刻退出

**现象**

```bash
ssh host 'nohup cmd > log 2>&1 &'
# 命令立刻返回，但服务没起来
```

**原因**

paramiko / ssh exec_command 会在通道关闭时杀掉子进程。

**解决**

```bash
# 用 setsid 脱离会话
ssh host 'setsid nohup bash script.sh > log 2>&1 < /dev/null &'
```

注意三个要素：`setsid` + `< /dev/null` + 重定向日志。

---

### 5.2 SSH 命令里的引号地狱

**现象**

```bash
ssh host 'python -c "import torch; print(torch.__version__)"'
# bash: -c: line 1: unexpected EOF
```

**解决**

**把脚本写成文件传上去执行**，不要在命令行里塞复杂引号：

```python
sftp.put("local.sh", "/root/x.sh")
ssh.exec_command("bash /root/x.sh")
```

---

## 六、性能

### 6.1 Turing 卡（sm_75）跑不动

**实测数据**（Quadro RTX 6000 24G）

```
512×512  4步   282 ms   3.5 fps
384×384  2步   148 ms   6.8 fps
256×256  2步    99 ms  10.1 fps
```

加上分割+ControlNet+合成后只剩 **2.5~5 fps**，无法实时。

**原因**

```
Turing sm_75   2018 年架构
Ada    sm_89   2022 年架构
同价格下算力差 4~6 倍
```

**解决**

换 Ampere（sm_86）及以上。租卡前必查：

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
# compute_cap 必须 >= 8.0
```

---

### 6.2 torch.compile 失败

**现象**

```
InductorError: CalledProcessError: Command '['/usr/bin/gcc', ...]'
cuda_utils.c:7:10: fatal error: Python.h: No such file or directory
```

**解决**

```bash
apt-get install -y python3.12-dev gcc
pip install xformers
```

装完 `torch.compile` 和 `xformers` 都能用，可能再快 15~25%。

---

### 6.3 显存不够 / OOM

**按显存选配置**

| 显存 | 分辨率 | 步数 | 备注 |
|---|---|---|---|
| 8 GB | 256×256 | 2 | 只能验证流程 |
| 12 GB | 384×384 | 4 | 勉强可用 |
| 16 GB | 512×512 | 4 | 可用 |
| 24 GB | 512×512 | 4 | 舒适，可加 IP-Adapter |
| 40 GB+ | 768×768 | 4~6 | 最佳 |

**省显存手段**

```python
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()   # 最省，但慢 30%
```

---

## 七、效果问题

### 7.1 画面闪烁（时序不一致）

**现象**

人物边缘每帧抖动，衣服纹理一直在变。

**原因**

扩散模型逐帧独立生成，没有时序约束。商用方案（Vidu / HeyGen）用专门训练的时序模型解决。

**缓解**

```bash
# 1) 提高 strength，让原始帧权重更高（换得少但更稳）
STRENGTH=0.35

# 2) 降低分辨率，噪点更少
W=320 H=320

# 3) 帧间平滑（需要改代码，加光流融合）
```

**根治**

换 AnimateDiff（5~8 fps，但稳定）。

---

### 7.2 换出来的形象不像参考图

**原因**

IP-Adapter 是"风格迁移"，不是像素级还原，相似度约 60~75%。

**解决**

```
① 参考图要清晰、正面、纯色背景
② 提高 IP-Adapter 权重
③ 顽固需求 → 训练 LoRA（20~30 分钟/个，相似度 85~90%）
```

---

### 7.3 背景被一起改了

**原因**

分割失败（`rembg` 没装或 mask 不准）。

**检查**

```bash
python -c "import rembg; print('ok')"
```

**解决**

```bash
pip install rembg onnxruntime-gpu
# 首次运行自动下载 u2netp 模型（5MB）
```

---

## 八、快速自检清单

部署完先跑这几条：

```bash
# 1) GPU 算力
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

# 2) torch
python -c "import torch;print(torch.__version__, torch.cuda.is_available())"

# 3) numpy
python -c "import numpy;print(numpy.__version__)"   # 必须 1.26.x

# 4) 模型文件
ls -lh /data/models/sd-turbo/unet/*.safetensors

# 5) 显存
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# 6) 服务
curl -s http://127.0.0.1:8000/status | python -m json.tool
```

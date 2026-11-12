# ZapSwap — 实时人物替换 + 变声（云 GPU 一键部署）

把真人视频流实时替换成目标形象（真人 / 动漫 / 换装），保留动作与背景，同时叠加 RVC 变声。

> 本仓库是从零搭建、踩完全部坑之后整理的可复现版本。**换到新卡只需 `bash deploy.sh`，10~15 分钟出结果。**

---

## 一、当前实测性能（重要）

不同显卡差异极大，**先看这张表再决定租什么卡**。

### SD-Turbo 512×512 / 4 步（纯生成，不含分割与合成）

| 显卡 | 架构 | 单帧 | 帧率 | 说明 |
|---|---|---|---|---|
| RTX 4090 24G | Ada sm_89 | ~35 ms | **28 fps** | ✅ 实时够用 |
| RTX 3090 24G | Ampere sm_86 | ~55 ms | **18 fps** | ✅ 可接受 |
| RTX A5000 24G | Ampere sm_86 | ~60 ms | **17 fps** | ✅ 可接受 |
| **Quadro RTX 6000 24G** | **Turing sm_75** | **233 ms** | **4.3 fps** | ❌ 不实时 |
| RTX 4070 Laptop 8G | Ada sm_89 | ~90 ms | 11 fps | ⚠️ 偏慢 |

### Quadro RTX 6000 上不同配置实测（本仓库实测数据）

```
512×512  4步   282 ms   3.5 fps
384×384  4步   201 ms   5.0 fps
384×384  2步   148 ms   6.8 fps
320×320  2步   122 ms   8.2 fps
256×256  2步    99 ms  10.1 fps
256×256  1步    71 ms  14.1 fps
```

### 加完整管线后的估算（分割 + ControlNet + IP-Adapter + 合成）

| 显卡 | 512×512 | 384×384 | 256×256 |
|---|---|---|---|
| 4090 | 18~22 fps | 25 fps | 30 fps |
| 3090 | 12~15 fps | 18 fps | 22 fps |
| **RTX 6000 (Turing)** | **2.5 fps** | **3.5 fps** | **5 fps** |

**结论**：Turing（sm_75）卡不适合本方案，至少要 Ampere（sm_86）以上。

---

## 二、租卡建议

```
优先     RTX 4090 24G      真·实时
次选     RTX 3090 24G      性价比最高
可用     RTX A5000 24G     稳定，兼容好
勉强     RTX 4070 / 4060Ti 16G   能跑但慢
避免     Quadro RTX 6000（Turing）
避免     任何 sm_75 及更老的卡
```

**必查**：租之前跑 `nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader`，`compute_cap` 低于 `8.0` 直接换。

---

## 三、镜像选择

```
系统      Ubuntu 22.04（24.04 也行）
Python    3.12
CUDA      12.8（对应 torch cu128 wheel）
PyTorch   2.10.0
```

**如果平台预装了 torch**，先验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

- `+cu128` 且 `True` → 跳过 torch 安装
- `+cpu` → 必须重装（见 `docs/TROUBLESHOOTING.md`）

---

## 四、一键部署

```bash
git clone <你的仓库地址> && cd zapswap
export HF_ENDPOINT=https://hf-mirror.com
bash deploy.sh
```

`deploy.sh` 会依次执行：

```
① 环境检查（GPU / CUDA / Python / 磁盘）
② 创建 venv（/data/env/py312）
③ 安装 PyTorch 2.10 + cu128（阿里云镜像，~60 秒）
④ 补齐全部 nvidia-* 依赖（cusparseLt / cudnn / nccl 等）
⑤ 安装应用依赖（分步安装，避免版本冲突）
⑥ 下载模型（ModelScope 14MB/s，~3 分钟）
⑦ 性能基准测试（给出你卡上的真实帧率）
```

**分步执行**（想看清每步）：

```bash
bash scripts/00_check.sh
bash scripts/01_torch.sh
bash scripts/02_deps.sh
bash scripts/03_models.sh
bash scripts/04_bench.sh
```

---

## 五、启动服务

```bash
bash scripts/99_start.sh
```

默认监听 `0.0.0.0:8000`，接口：

| 路径 | 说明 |
|---|---|
| `GET /status` | 服务状态、显存、平均帧率 |
| `POST /reference` | 上传参考图（base64），热切换目标形象 |
| `WS /ws` | 视频流：发 `{"frame":"<base64 jpeg>"}`，收 `{"frame":"..."}` |

### 可选：叠加 Cloudflare 隧道

```bash
export CF_TOKEN="<你的 tunnel token>"
bash scripts/98_tunnel.sh
```

---

## 六、本地客户端

```bash
pip install opencv-python websockets requests
python client/client.py --host https://ai.example.com --cam 0
```

客户端会：

```
摄像头采集 → WS 推帧 → 收处理结果 → 输出到本地窗口 / 虚拟摄像头
```

---

## 七、参考图怎么准备

一张图决定换出来的形象。

```
要求     正面 / 全身 / 纯色背景最佳
真人     正常照片即可
动漫     动漫风格立绘（需切基础模型，见 docs）
换装     让目标人物穿着那件衣服拍照
```

**换装说明**：不做虚拟试衣（VTON），而是把「穿那件衣服的目标形象」当参考图整体替换。这样真人 / 动漫 / 换装统一到一条链路，只换参考图。

---

## 八、延迟预算（4090 + 国内云）

```
采集              16 ms
上行（国内云）      30~50 ms
人物分割            8 ms
OpenPose 姿态      12 ms
SD 4步             35 ms
IP-Adapter         10 ms
背景合成            5 ms
RVC 变声           50~80 ms（并行，不叠加）
下行               30~50 ms
────────────────────────────
端到端            150~200 ms
```

**200 ms 以内**做直播对话观众基本无感。

---

## 九、已知限制

**① 时序闪烁**

StreamDiffusion / 逐帧扩散的通病：人物边缘和纹理每帧都在变。商用方案（Vidu / HeyGen）用专门训练的时序模型解决，开源方案要么闪（快），要么慢（稳）。

缓解手段：
- 加大 `STRENGTH` 让原始帧权重更高（更稳但换得少）
- 帧间光流平滑后处理（+20 ms）
- 换 AnimateDiff（5~8 fps，但稳定）

**② 相似度**

IP-Adapter 约 60~75% 相似。要精确还原需给每个目标形象训练 LoRA（20~30 分钟/个）。

**③ 单模型限制**

真人模型和动漫模型不能同时驻留显存，切换需 10~20 秒重建 pipeline。

---

## 十、目录结构

```
zapswap/
├── deploy.sh                    一键部署
├── scripts/
│   ├── 00_check.sh              环境检查
│   ├── 01_torch.sh              装 torch（阿里云镜像）
│   ├── 02_deps.sh               装依赖（分步防冲突）
│   ├── 03_models.sh             下模型（ModelScope）
│   ├── 04_bench.sh              性能基准
│   ├── 98_tunnel.sh             Cloudflare 隧道
│   └── 99_start.sh              启动服务
├── server/
│   ├── app.py                   FastAPI + WebSocket
│   └── pipeline.py              分割/姿态/生成/合成
├── client/
│   └── client.py                本地推拉流
└── docs/
    ├── TROUBLESHOOTING.md       踩坑记录（强烈建议看）
    └── PERFORMANCE.md           各卡实测数据
```

---

## 十一、和 ZapLive 的关系

ZapLive 走的是 Vidu 商用 API（云端人物替换），优点是质量稳定、无闪烁，缺点是画质被压到 952×544、按额度计费、不能自定义模型。

本仓库是**开源自建替代**：分辨率可控、模型可换、成本只有 GPU 租金，代价是要自己调参、有闪烁、画质取决于选的模型。

两者可以共存：**先本地/自建验证效果，决定长期用哪条路。**

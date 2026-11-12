#!/bin/bash
# 性能基准：测出你这张卡的真实帧率
set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }

ENV_PATH="${ENV_PATH:-/data/env/py312}"
MODEL_DIR="${MODEL_DIR:-/data/models}"
source "$ENV_PATH/bin/activate"

export MODEL_DIR

python - <<'PYEOF'
import os, time, torch
from diffusers import AutoPipelineForText2Image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

MODEL_DIR = os.environ.get("MODEL_DIR", "/data/models")
name = torch.cuda.get_device_name(0)
cc = torch.cuda.get_device_capability(0)
vram = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)

print("=" * 64)
print("GPU        %s" % name)
print("算力       sm_%d%d" % cc)
print("显存       %.1f GB" % vram)
print("=" * 64)

t = time.time()
pipe = AutoPipelineForText2Image.from_pretrained(
    os.path.join(MODEL_DIR, "sd-turbo"),
    torch_dtype=torch.float16, variant="fp16",
    safety_checker=None, local_files_only=True,
)
pipe.set_progress_bar_config(disable=True)
pipe.to("cuda")
pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
print("加载耗时   %.1f 秒" % (time.time() - t))
print("显存占用   %.2f GB" % (torch.cuda.memory_allocated() / 1024**3))
print()

def bench(w, h, steps, n=6):
    for _ in range(2):
        pipe(prompt="a person", num_inference_steps=steps,
             guidance_scale=0.0, width=w, height=h)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        pipe(prompt="a person", num_inference_steps=steps,
             guidance_scale=0.0, width=w, height=h)
    torch.cuda.synchronize()
    ms = (time.time() - t) / n * 1000
    return ms, 1000 / ms

print("--- 纯 SD 生成 ---")
print("%-16s %10s %10s" % ("配置", "单帧", "帧率"))
print("-" * 40)
results = {}
for (w, h, s) in [(512, 512, 4), (384, 384, 4), (384, 384, 2),
                  (320, 320, 2), (256, 256, 2)]:
    ms, fps = bench(w, h, s)
    key = "%dx%d %d步" % (w, h, s)
    results[key] = fps
    print("%-16s %8.1fms %8.1ffps" % (key, ms, fps))

print()
print("--- 完整管线估算（+分割 +ControlNet +合成 约 +120~150ms）---")
for key, fps in results.items():
    full_ms = 1000 / fps + 135
    full_fps = 1000 / full_ms
    flag = "✅ 实时" if full_fps >= 20 else ("⚠️ 可用" if full_fps >= 10 else "❌ 偏慢")
    print("  %-16s 完整约 %5.1f fps  %s" % (key, full_fps, flag))

print()
print("显存峰值   %.2f GB" % (torch.cuda.max_memory_allocated() / 1024**3))

# 结论
best = max(results.values())
print()
print("=" * 64)
if best >= 15:
    print("结论: 这张卡够用（纯生成 %.1f fps）" % best)
elif best >= 8:
    print("结论: 勉强可用，建议降分辨率或换卡")
else:
    print("结论: 这张卡不适合实时换身，建议换 Ampere(8.0+) 及以上")
print("=" * 64)
PYEOF

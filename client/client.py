#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZapSwap 本地客户端
==================
摄像头采集 -> WS 推帧 -> 收处理结果 -> 显示 / 推给虚拟摄像头

用法:
    python client.py --host https://ai.example.com --cam 0
    python client.py --host http://127.0.0.1:8000 --cam 0 --width 640
    python client.py --host ... --ref 参考图.jpg      # 上传参考图
    python client.py --host ... --no-show             # 不显示（只转发）

按键:
    q / ESC   退出
    r         重连
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import sys
import time

import cv2
import numpy as np

try:
    import websockets
except ImportError:
    print("缺 websockets: pip install websockets", file=sys.stderr)
    sys.exit(1)

try:
    import requests
except ImportError:
    requests = None


def encode(img, q=80):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def decode(b64):
    if not b64:
        return None
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def http_post(host, path, payload):
    if requests is None:
        print("缺 requests: pip install requests", file=sys.stderr)
        return None
    url = host.rstrip("/") + path
    try:
        r = requests.post(url, json=payload, timeout=15)
        return r.json()
    except Exception as e:
        print("POST %s 失败: %s" % (path, e))
        return None


def http_get(host, path):
    if requests is None:
        return None
    try:
        r = requests.get(host.rstrip("/") + path, timeout=10)
        return r.json()
    except Exception as e:
        print("GET %s 失败: %s" % (path, e))
        return None


async def run(args):
    host = args.host.rstrip("/")
    if host.startswith("https://"):
        ws_url = "wss://" + host[len("https://"):] + "/ws"
    elif host.startswith("http://"):
        ws_url = "ws://" + host[len("http://"):] + "/ws"
    else:
        ws_url = "wss://" + host + "/ws"

    print("=" * 58)
    print("  服务端  %s" % host)
    print("  WebSocket %s" % ws_url)
    print("=" * 58)

    st = http_get(host, "/status")
    if st:
        print("服务状态:", "就绪" if st.get("ok") else "未就绪")
        print("GPU:", st.get("gpu"), "| 显存 %.1fGB" % (st.get("vram_total_gb") or 0))
        s = st.get("stats") or {}
        if s.get("frames"):
            print("服务端已处理 %d 帧, 平均 %.0f ms (%.1f fps)"
                  % (s["frames"], s["avg_ms"], s["fps"]))
    else:
        print("警告: 无法读取 /status，继续尝试连接")

    # 上传参考图
    if args.ref:
        img = cv2.imread(args.ref)
        if img is None:
            print("参考图读取失败:", args.ref)
        else:
            print("上传参考图 %s (%dx%d) ..." % (args.ref, img.shape[1], img.shape[0]))
            r = http_post(host, "/reference", {"image": encode(img, 92)})
            print("  结果:", r)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("摄像头打开失败:", args.cam, file=sys.stderr)
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    print("摄像头已打开 %dx%d" % (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    frames_sent = 0
    t_start = time.time()
    fps_send = 0.0

    try:
        async with websockets.connect(
            ws_url, max_size=None, ping_interval=20, ping_timeout=30,
            open_timeout=20,
        ) as ws:
            print("已连接，按 q 退出\n")
            while True:
                ok, frame = cap.read()
                if not ok:
                    await asyncio.sleep(0.01)
                    continue

                import json
                await ws.send(json.dumps({"frame": encode(frame, 80)}))
                resp = await ws.recv()
                try:
                    out_b64 = json.loads(resp).get("frame", "")
                except Exception:
                    out_b64 = ""

                out = decode(out_b64)
                frames_sent += 1
                el = time.time() - t_start
                if el >= 1.0:
                    fps_send = frames_sent / el
                    frames_sent = 0
                    t_start = time.time()

                if out is not None and not args.no_show:
                    cv2.putText(out, "%.1f fps -> server" % fps_send,
                                (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)
                    cv2.imshow("ZapSwap", out)
                    k = cv2.waitKey(1) & 0xFF
                    if k in (ord("q"), 27):
                        break
                    if k == ord("r"):
                        print("重连...")
                        break
                elif args.no_show:
                    await asyncio.sleep(0)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print("连接异常:", type(e).__name__, str(e)[:200])
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("已退出")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True,
                    help="服务端地址，如 https://ai.example.com 或 http://127.0.0.1:8000")
    ap.add_argument("--cam", type=int, default=0, help="摄像头索引")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--ref", default="", help="参考图路径（上传后热切换形象）")
    ap.add_argument("--no-show", action="store_true", help="不显示窗口")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

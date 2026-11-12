#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZapSwap 服务端
==============
FastAPI + WebSocket，接收 JPEG 帧，返回处理后帧。

接口:
    GET  /            健康检查
    GET  /status      状态、显存、帧率
    POST /reference   上传参考图（热切换目标形象）
    WS   /ws          视频流

启动:
    python server/app.py            # 默认 0.0.0.0:8000
    PORT=8080 python server/app.py
"""
from __future__ import annotations

import os
import sys
import json
import time
import base64
import asyncio
import threading

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import BodySwapPipeline, Config  # noqa: E402

PORT = int(os.environ.get("PORT", "8000"))
PIPELINE: BodySwapPipeline | None = None
app = FastAPI(title="ZapSwap")


def log(*a):
    print("[app]", *a, flush=True)


def _decode_jpeg_b64(b64: str) -> np.ndarray | None:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64)
    except Exception:
        return None
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _encode_jpeg_b64(img: np.ndarray, q: int = 82) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode()


# ---------------- HTTP ----------------
@app.get("/")
def root():
    return {"service": "zapswap", "ok": True, "ws": "/ws", "status": "/status"}


@app.get("/status")
def status():
    if PIPELINE is None:
        return {"ok": False, "error": "pipeline not initialized"}
    d = PIPELINE.info()
    d["ok"] = d.get("ready", False)
    return d


@app.post("/reference")
async def set_reference(payload: dict):
    if PIPELINE is None:
        return JSONResponse({"ok": False, "error": "not ready"}, 503)
    b64 = payload.get("image", "")
    if not b64:
        return JSONResponse({"ok": False, "error": "no image"}, 400)
    img = _decode_jpeg_b64(b64)
    if img is None:
        return JSONResponse({"ok": False, "error": "decode failed"}, 400)
    PIPELINE.set_reference(img)
    return {"ok": True, "shape": list(img.shape)}


@app.post("/config")
async def set_config(payload: dict):
    """热调参数: {"steps":4,"strength":0.55,"width":384,...}"""
    if PIPELINE is None:
        return JSONResponse({"ok": False, "error": "not ready"}, 503)
    cfg = PIPELINE.cfg
    for k, v in payload.items():
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, type(getattr(cfg, k))(v))
            except Exception:
                pass
    return {"ok": True, "config": cfg.__dict__}


# ---------------- WebSocket ----------------
@app.websocket("/ws")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    log("client connected")
    loop = asyncio.get_event_loop()
    try:
        while True:
            msg = await ws.receive_text()
            if not msg:
                continue
            try:
                data = json.loads(msg)
            except Exception:
                data = {"frame": msg}

            b64 = data.get("frame")
            if not b64:
                await ws.send_text(json.dumps({"ok": False, "error": "no frame"}))
                continue

            img = _decode_jpeg_b64(b64)
            if img is None:
                await ws.send_text(json.dumps({"ok": False, "error": "decode failed"}))
                continue

            out = await loop.run_in_executor(None, PIPELINE.process, img)
            payload = json.dumps({
                "frame": _encode_jpeg_b64(out, PIPELINE.cfg.jpeg_quality),
            })
            await ws.send_text(payload)
    except WebSocketDisconnect:
        log("client disconnected")
    except Exception as e:
        log("ws error:", type(e).__name__, str(e)[:150])


# ---------------- 启动 ----------------
def main():
    global PIPELINE
    cfg = Config()
    log("配置: %dx%d steps=%d strength=%.2f" % (
        cfg.width, cfg.height, cfg.steps, cfg.strength))

    PIPELINE = BodySwapPipeline(cfg)
    PIPELINE.load()

    log("监听 0.0.0.0:%d" % PORT)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZapSwap 处理管线
================
输入: BGR 帧 (H, W, 3)
输出: BGR 帧（人物被替换，背景保留）

流程:
    分割 -> 姿态 -> SD 图生图(ControlNet + 参考图) -> 合成
"""
from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch

MODEL_DIR = os.environ.get("MODEL_DIR", "/data/models")


def _log(*a):
    print("[pipe]", *a, flush=True)


@dataclass
class Config:
    width: int = int(os.environ.get("W", "384"))
    height: int = int(os.environ.get("H", "384"))
    steps: int = int(os.environ.get("STEPS", "4"))
    guidance: float = float(os.environ.get("GUIDANCE", "0.0"))
    strength: float = float(os.environ.get("STRENGTH", "0.55"))
    cn_scale: float = float(os.environ.get("CN_SCALE", "0.9"))
    jpeg_quality: int = int(os.environ.get("JPEG_Q", "82"))
    prompt: str = os.environ.get(
        "PROMPT", "a person, full body, high quality, detailed, best quality"
    )
    negative: str = os.environ.get("NEGATIVE", "blurry, low quality, distorted")


@dataclass
class Stats:
    frames: int = 0
    total_ms: float = 0.0
    last_ms: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, ms: float):
        with self.lock:
            self.frames += 1
            self.total_ms += ms
            self.last_ms = ms

    def snapshot(self) -> dict:
        with self.lock:
            avg = self.total_ms / self.frames if self.frames else 0.0
            return {
                "frames": self.frames,
                "avg_ms": round(avg, 1),
                "last_ms": round(self.last_ms, 1),
                "fps": round(1000 / avg, 2) if avg else 0.0,
            }


class BodySwapPipeline:
    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.pipe = None
        self.pose = None
        self.rmbg = None
        self.ref_image: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.stats = Stats()
        self.ready = False

    # ---------------- 加载 ----------------
    def load(self):
        from diffusers import AutoPipelineForImage2Image

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        _log("加载 SD-Turbo ...")
        t = time.time()
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            os.path.join(MODEL_DIR, "sd-turbo"),
            torch_dtype=torch.float16,
            variant="fp16",
            safety_checker=None,
            local_files_only=True,
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to("cuda")
        self.pipe.unet = self.pipe.unet.to(memory_format=torch.channels_last)
        _log("SD 就绪 %.1fs / %.2fGB" % (
            time.time() - t, torch.cuda.memory_allocated() / 1024**3))

        # 姿态
        try:
            from controlnet_aux import OpenposeDetector
            self.pose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            _log("OpenPose 就绪")
        except Exception as e:
            _log("OpenPose 不可用:", type(e).__name__)

        # 分割
        try:
            from rembg import new_session
            self.rmbg = new_session("u2netp")
            _log("RMBG 就绪")
        except Exception as e:
            _log("RMBG 不可用:", type(e).__name__)

        self.ready = True
        _log("全部就绪")

    def set_reference(self, bgr: np.ndarray):
        with self.lock:
            self.ref_image = bgr
        _log("参考图已更新 %s" % (bgr.shape,))

    # ---------------- 子步骤 ----------------
    def _segment(self, bgr: np.ndarray) -> np.ndarray:
        """返回人物 mask (H,W) uint8"""
        h, w = bgr.shape[:2]
        if self.rmbg is None:
            return np.full((h, w), 255, np.uint8)
        try:
            from rembg import remove
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            m = remove(rgb, session=self.rmbg, only_mask=True)
            mask = (np.array(m) > 128).astype(np.uint8) * 255
            return mask
        except Exception:
            return np.full((h, w), 255, np.uint8)

    def _pose(self, bgr: np.ndarray):
        if self.pose is None:
            return None
        try:
            from PIL import Image as PILImage
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            p = self.pose(PILImage.fromarray(rgb))
            return p.resize((self.cfg.width, self.cfg.height))
        except Exception:
            return None

    @staticmethod
    def _compose(orig: np.ndarray, gen: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = orig.shape[:2]
        if gen.shape[:2] != (h, w):
            gen = cv2.resize(gen, (w, h), interpolation=cv2.INTER_LINEAR)
        m = cv2.GaussianBlur(mask, (7, 7), 0)[..., None].astype(np.float32) / 255.0
        out = gen.astype(np.float32) * m + orig.astype(np.float32) * (1 - m)
        return np.clip(out, 0, 255).astype(np.uint8)

    # ---------------- 主流程 ----------------
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        frame: BGR uint8 (任意尺寸)
        return: BGR uint8 (cfg.width x cfg.height)
        """
        if not self.ready or self.pipe is None:
            return cv2.resize(frame, (self.cfg.width, self.cfg.height))

        t0 = time.time()
        with self.lock:
            img = cv2.resize(frame, (self.cfg.width, self.cfg.height),
                             interpolation=cv2.INTER_AREA)

            mask = self._segment(img)
            if mask.sum() < 255 * 200:
                # 没检测到人，原样返回
                self.stats.add((time.time() - t0) * 1000)
                return img

            pose_img = self._pose(img)

            from PIL import Image as PILImage
            init = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            kwargs = dict(
                prompt=self.cfg.prompt,
                negative_prompt=self.cfg.negative,
                image=init,
                num_inference_steps=self.cfg.steps,
                guidance_scale=self.cfg.guidance,
                strength=self.cfg.strength,
            )
            if pose_img is not None and getattr(self.pipe, "controlnet", None) is not None:
                kwargs["control_image"] = pose_img
                kwargs["controlnet_conditioning_scale"] = self.cfg.cn_scale

            try:
                gen_pil = self.pipe(**kwargs).images[0]
            except TypeError:
                kwargs.pop("control_image", None)
                kwargs.pop("controlnet_conditioning_scale", None)
                gen_pil = self.pipe(**kwargs).images[0]

            gen = cv2.cvtColor(np.array(gen_pil), cv2.COLOR_RGB2BGR)
            out = self._compose(img, gen, mask)

        self.stats.add((time.time() - t0) * 1000)
        return out

    # ---------------- 状态 ----------------
    def info(self) -> dict:
        return {
            "ready": self.ready,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
            "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1) if torch.cuda.is_available() else 0,
            "vram_used_gb": round(torch.cuda.memory_allocated() / 1024**3, 2) if torch.cuda.is_available() else 0,
            "has_pose": self.pose is not None,
            "has_rmbg": self.rmbg is not None,
            "has_reference": self.ref_image is not None,
            "config": {
                "w": self.cfg.width, "h": self.cfg.height,
                "steps": self.cfg.steps, "strength": self.cfg.strength,
            },
            "stats": self.stats.snapshot(),
        }

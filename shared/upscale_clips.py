"""Upscale Wan clips to 1080p, streaming frame by frame (16 GB RAM can't hold a whole 4x clip).

Runs with ComfyUI's venv (it has torch + spandrel):
  <ComfyUI>\\venv\\Scripts\\python upscale_clips.py <in_dir> <out_dir> <ffmpeg>
Each clip keeps its own frame rate (48 fps from Wan x RIFE, 25 fps from InfiniteTalk); the assembler retimes them to 24 fps.
Model: Real-ESRGAN animevideov3 (x4, BSD-3), then downscaled to 1920x1080 on the GPU.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import spandrel
import torch
import torch.nn.functional as F

MODEL = Path(os.environ.get("COMFYUI_DIR", Path.home() / "comfyui")) / "models" / "upscale_models" / "realesr-animevideov3.pth"
OUT_W, OUT_H, BATCH = 1920, 1080, 8


def probe(ffmpeg, src):
    """(width, height, fps) of a clip, from ffmpeg's stream line."""
    err = subprocess.run([ffmpeg, "-i", str(src)], capture_output=True, text=True).stderr
    m = re.search(r"Video:.*?, (\d+)x(\d+).*?, ([\d.]+) fps", err)
    if not m:
        raise RuntimeError(f"can't read size of {src}")
    return int(m.group(1)), int(m.group(2)), m.group(3)


def upscale(model, ffmpeg, src, dst):
    w, h, fps = probe(ffmpeg, src)
    dec = subprocess.Popen([ffmpeg, "-loglevel", "error", "-i", str(src), "-f", "rawvideo",
                            "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OUT_W}x{OUT_H}",
                            "-r", fps, "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
                            "-pix_fmt", "yuv420p", str(dst)], stdin=subprocess.PIPE)
    frame_bytes, n = w * h * 3, 0
    while True:
        raw = dec.stdout.read(frame_bytes * BATCH)
        if not raw:
            break
        x = torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(-1, h, w, 3).cuda()
        x = x.permute(0, 3, 1, 2).half() / 255
        y = F.interpolate(model(x), size=(OUT_H, OUT_W), mode="bicubic", antialias=True)
        enc.stdin.write((y.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy().tobytes())
        n += x.shape[0]
    enc.stdin.close()
    if dec.wait() != 0 or enc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on {src}")
    return n


def main(in_dir, out_dir, ffmpeg):
    out_dir.mkdir(exist_ok=True)
    model = spandrel.ModelLoader().load_from_file(str(MODEL)).model.cuda().half().eval()
    stats = {}
    with torch.inference_mode():
        for src in sorted(in_dir.glob("*.mp4")):
            dst = out_dir / src.name
            if dst.exists():
                continue
            t = time.time()
            frames = upscale(model, ffmpeg, src, dst)
            stats[src.stem] = round(time.time() - t, 1)
            print(f"upscaled {src.stem}: {frames} frames in {stats[src.stem]}s", flush=True)
    print("UPSCALE", json.dumps(stats))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])

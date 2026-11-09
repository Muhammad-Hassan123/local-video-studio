"""Shot animation helpers. Every shot's first frame comes from the consistent character library and locations (the
renderer in make_cartoon.py); a video model animates it:
  - a character speaking on screen -> InfiniteTalk, lip-synced to the line's own audio (masks keep the listeners' mouths shut)
  - everything else               -> Wan 2.1 image-to-video, acted from a prompt built from the screenplay
Clips are 848x480, upscaled to 1080p by shared/upscale_clips.py, then the render stage adds camera moves, titles and fades.
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw

import sprites

GEN_W, GEN_H = 848, 480
I2V_MAX, I2V_FPS = 81, 16                     # Wan makes up to 81 frames at 16 fps (5 s); RIFE x3 -> 48 fps
STYLE = ("2D cartoon animation from an adult animated web series, flat colors, bold black outlines, the same character designs "
         "throughout, smooth natural motion, static camera.")
EXPR = {"neutral": "with a calm face", "smile": "smiling warmly", "happy": "laughing", "sad": "looking sad",
        "angry": "looking angry", "shocked": "looking shocked with wide eyes", "scared": "looking scared",
        "worried": "looking worried", "smug": "with a smug smirk", "confused": "looking confused",
        "serious": "looking serious", "crying": "crying"}
POSE = {"stand": "stands", "front": "faces the viewer", "point": "points forward", "cross": "stands with arms crossed",
        "hips": "stands with hands on hips", "think": "rubs their chin, thinking", "shrug": "shrugs", "wave": "waves hello",
        "phone": "looks at the smartphone in their hands", "hold": "holds up a small object", "hands": "raises both hands defensively",
        "fists": "clenches their fists", "slump": "slumps their shoulders", "walk": "walks"}


def prompt(cast, script, shot, talker, visible):
    """What the video model should act out, in plain words, from the shot's direction. Only `visible` characters are named."""
    if shot.get("cam", {}).get("type") == "insert":
        return f"{script['plates'][shot['insert']]['prompt']}. Subtle ambient movement, a slow gentle camera push in. {STYLE}"
    parts = []
    for a in shot["actors"]:
        if a["id"] not in visible:
            continue
        c = cast["characters"][a["id"]]
        pose = POSE.get(a["pose"]) or "stands " + sprites._words(c, script["poses"][a["pose"]]).lower()
        feel = EXPR[a["expr"]] + (f", then becomes {EXPR[a['expr2']].replace('looking ', '')}" if a.get("expr2") not in (None, a["expr"]) else "")
        if a["id"] == talker:
            doing = "talks expressively with natural hand gestures and head movement"
        else:
            doing = "listens and reacts with small natural movements, breathing and blinking"
        move = a.get("move")
        if move:
            doing = "has just walked in and stops" if move["kind"] == "enter" else f"turns and walks out of the frame to the {move['side']}"
        parts.append(f"{c['tag'][0].upper()}{c['tag'][1:]} {pose}, {feel}, and {doing}.")
    if shot.get("act"):
        parts.append(shot["act"])
    if parts:
        parts.append("Nobody else is in the shot.")
    return " ".join(parts) + " " + STYLE


def key(spec, keyframe):
    """Fingerprint of everything a clip depends on."""
    h = hashlib.sha1(json.dumps(spec, sort_keys=True).encode())
    h.update(keyframe.tobytes())
    return h.hexdigest()


def talk_audio(line_wav, lead, dur, out):
    """The shot's soundtrack for lip-sync: silence, the line at `lead` seconds, silence to the end of the shot."""
    x, sr = sf.read(line_wav, dtype="float32")
    track = np.zeros(int(dur * sr) + 1, np.float32)
    i = int(lead * sr)
    track[i:i + len(x)] = x[:len(track) - i]
    sf.write(out, track, sr)
    return out


def silence(out, sr=16000):
    sf.write(out, np.zeros(int(0.2 * sr), np.float32), sr)
    return out


def masks(boxes, talker, out_dir, sid):
    """Mask images for InfiniteTalk: the speaker's region, and everyone else's."""
    paths = []
    for name, ids in (("speaker", [talker]), ("others", [c for c in boxes if c != talker])):
        m = Image.new("L", (GEN_W, GEN_H), 0)
        d = ImageDraw.Draw(m)
        for c in ids:
            x0, y0, x1, y1 = (v * GEN_W / 1920 if i % 2 == 0 else v * GEN_H / 1080 for i, v in enumerate(boxes[c]))
            pad = 0.08 * (x1 - x0)
            d.rectangle((x0 - pad, y0 - pad, x1 + pad, y1 + pad), fill=255)
        path = out_dir / f"mask_{sid:03d}_{name}.png"
        Image.merge("RGB", (m, m, m)).save(path)
        paths.append(path)
    return paths


class ClipReader:
    """Frames of an upscaled clip by time, decoded once front to back. `stretch` > 1 slows a short clip to fill its shot."""

    def __init__(self, ffmpeg, path, stretch=1.0):
        err = subprocess.run([ffmpeg, "-i", str(path)], capture_output=True, text=True).stderr
        m = re.search(r"Video:.*?, (\d+)x(\d+).*?, ([\d.]+) fps", err)
        self.w, self.h, self.fps = int(m.group(1)), int(m.group(2)), float(m.group(3))
        self.stretch = stretch
        self.proc = subprocess.Popen([ffmpeg, "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                                     stdout=subprocess.PIPE)
        self.idx, self.last = -1, None

    def at(self, t):
        want = int(t / self.stretch * self.fps)
        while self.idx < want:
            raw = self.proc.stdout.read(self.w * self.h * 3)
            if len(raw) < self.w * self.h * 3:  # past the end: hold the last frame
                break
            self.last, self.idx = raw, self.idx + 1
        return Image.frombytes("RGB", (self.w, self.h), self.last)

    def close(self):
        self.proc.stdout.close()
        self.proc.wait()


def i2v_frames(dur):
    """Frames to generate for a shot: just enough to cover it (Wan wants 4n+1), at most one 5 s clip."""
    return min(I2V_MAX, max(17, 4 * -(-int(dur * I2V_FPS) // 4) + 1))

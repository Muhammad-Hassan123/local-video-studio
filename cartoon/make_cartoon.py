"""Cartoon pipeline: a screenplay in, a finished 2D animated episode out.

Usage:  venv\\Scripts\\python cartoon\\make_cartoon.py <episode> [stage ...|all]
  direct   screenplay.txt -> script.json (shots, framing, poses; director.py documents the format)
  tts      one wav per line in the speaker's own voice (Kokoro, voices blended per character)   -> audio/, timing.json
  sprites  every pose/expression/mouth/blink the episode needs, drawn once per character    -> characters/<id>/ (ComfyUI)
  plates   locations, painted once and reused by later episodes                               -> locations/ (ComfyUI)
  thumbs   thumbnail variants from the THUMB: lines                                           -> final/thumb_N.jpg
  animate  every shot animated by a video model from its drawn first frame, lip-synced dialogue -> clips/, clips_hd/ (ComfyUI)
           (animate:0-21 does only shots 0-21; render uses a shot's clip when there is one)
  render   every shot composed frame by frame (lip-sync, blinks, camera), shots in parallel   -> segments/, video.mp4
  music    instrumental beds per mood, composed once and shared by every episode             -> shared/music/ (ComfyUI)
  mix      voices + sound effects + ambience + music, music ducked under the voices           -> mix.wav
  qa       contact sheet of every shot and a length/hook check                                -> qa/
  final    loudness-normalised upload file, .srt subtitles, title/description/chapters        -> final/
Every stage caches: only changed lines are re-voiced, only missing sprites/locations are drawn, only changed shots re-render.

To replace a drawing you don't like:  make_cartoon.py redraw <character> <pose or pose_expression>   (e.g. redraw nia point_angry)
then run the episode again.
"""
import hashlib
import io
import json
import math
import multiprocessing as mp
import random
import subprocess
import sys
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

import animate
import director
import sprites

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
import comfy_client as cc  # noqa: E402
import music  # noqa: E402
import sfx  # noqa: E402

ROOT = Path(__file__).parent
MODELS = ROOT.parent / "models"
LOCS = ROOT / "locations"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
CAST = json.loads((ROOT / "cast.json").read_text(encoding="utf-8"))
STAGES = ["direct", "tts", "sprites", "plates", "thumbs", "animate", "render", "music", "mix", "qa", "final"]
OUT_W, OUT_H, FPS = 1920, 1080, 24
PLATE_W, PLATE_H = 1920, 1088      # generated size; plates are kept upscaled 2x for close-ups
LEAD, TAIL = 0.15, 0.3             # silence before / after each line inside its shot
FLOOR, BODY = 0.95, 0.66           # feet line and the height of a height-1.0 character, as fractions of the plate height
FONT = r"C:\Windows\Fonts\seguibl.ttf"
WORKERS = 8
RENDER_VERSION = 3                 # bump when the drawing code changes, to re-render every shot


def load(ep):
    return json.loads((ep / "script.json").read_text(encoding="utf-8"))


def stage_direct(ep, script):
    out = director.direct(ep / "screenplay.txt", ep / "script.json", CAST)
    print(f"DIRECT {len(out['shots'])} shots, {len(out['plates'])} locations, "
          f"{sum(len(v) for v in out['needs'].values())} character poses")


# ---------------------------------------------------------------- voices

def _speaker(script, who):
    return CAST["characters"][script["narrator"] if who == "narrator" else who]


def _audio_name(script, shot):
    c = _speaker(script, shot["who"])
    return hashlib.sha1(f"{c['voice']}|{c.get('speed', 1.0)}|{shot['text']}".encode()).hexdigest()[:12] + ".wav"


def stage_tts(ep, script):
    from kokoro_onnx import Kokoro
    adir = ep / "audio"
    adir.mkdir(exist_ok=True)
    k = Kokoro(str(MODELS / "kokoro-v1.0.onnx"), str(MODELS / "voices-v1.0.bin"))
    styles = {}
    timing, t, made = {}, 0.0, 0
    for shot in script["shots"]:
        entry = {}
        if shot["kind"] == "line":
            wav = adir / _audio_name(script, shot)
            if not wav.exists():
                c = _speaker(script, shot["who"])
                if c["voice"] not in styles:  # "af_heart*0.7+af_nova*0.3": a blend of Kokoro voices
                    parts = [p.partition("*") for p in c["voice"].split("+")]
                    styles[c["voice"]] = sum(k.get_voice_style(n.strip()) * float(w or 1) for n, _, w in parts)
                samples, sr = k.create(shot["text"], voice=styles[c["voice"]], speed=c.get("speed", 1.0), lang="en-us")
                sf.write(wav, samples, sr)
                made += 1
            dur = LEAD + sf.info(wav).duration + TAIL + shot.get("hold", 0.0)
            entry["audio"] = wav.name
        else:
            dur = shot["dur"]
        timing[shot["id"]] = {"start": round(t, 3), "dur": round(dur, 3), **entry}
        t += dur
    (ep / "timing.json").write_text(json.dumps(timing, indent=1))
    print(f"TTS: {len(timing)} shots, {t / 60:.1f} min, {made} lines voiced")


# ---------------------------------------------------------------- art

def stage_sprites(ep, script):
    needs = {c: {(p, e): t for p, e, t in v} for c, v in script["needs"].items()}
    sprites.ensure(CAST, needs, script["poses"], log=lambda m: print(m, flush=True))


def _plate_path(name, spec):
    return LOCS / f"{name}_c{spec.get('pick', 1)}_2x.png"


def _shot_plate(shot):
    """The picture a shot shows: its insert if it has one, else its scene's location."""
    return shot.get("insert") or shot.get("plate")


def stage_plates(ep, script):
    raw = LOCS / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, spec in script["plates"].items():
        prompt = spec["prompt"] + ("" if spec["insert"] else f". {director.STAGE}") + f". {CAST['plate_style']}"
        note = raw / f"{name}.txt"
        if note.exists() and note.read_text(encoding="utf-8") != prompt:  # the description changed: paint it again
            for f in list(raw.glob(f"{name}_c*.png")) + list(LOCS.glob(f"{name}_c*_2x.png")):
                f.unlink()
        note.write_text(prompt, encoding="utf-8")
        refs = [raw / f"{spec['ref']}_c{script['plates'][spec['ref']].get('pick', 1)}.png"] if "ref" in spec else []
        for n in (1, 2):
            out = raw / f"{name}_c{n}.png"
            if not out.exists():
                files, secs = cc.run(cc.klein_image(prompt, PLATE_W, PLATE_H, seed=700 + n, refs=[cc.to_input(r) for r in refs],
                                                    filename_prefix="cartoon/plate"), raw, out.stem)
                files[0].replace(out)
                print(f"{name}_c{n}: {secs:.0f}s", flush=True)
        final = _plate_path(name, spec)
        if not final.exists():
            files, _ = cc.run(cc.upscale(cc.to_input(raw / f"{name}_c{spec.get('pick', 1)}.png"), filename_prefix="cartoon/up"), LOCS, final.stem)
            files[0].replace(final)
    sheet = [Image.open(p).convert("RGB") for name in script["plates"] for p in sorted(raw.glob(f"{name}_c*.png"))]
    (ep / "qa").mkdir(exist_ok=True)
    _sheet(sheet, 4, 480, 272, ep / "qa" / "locations.jpg")
    print(f"PLATES {len(script['plates'])} locations (qa/locations.jpg shows both candidates; pick=2 on the @ line swaps)")


def _sheet(images, cols, tw, th, out):
    rows = math.ceil(len(images) / cols)
    s = Image.new("RGB", (cols * tw, rows * th), (24, 24, 28))
    for i, im in enumerate(images):
        s.paste(ImageOps.fit(im, (tw, th)), ((i % cols) * tw, (i // cols) * th))
    s.save(out, quality=85)


# ---------------------------------------------------------------- frame composition

def _smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def _font(size):
    return ImageFont.truetype(FONT, size)


def _text_image(text, size, max_w, fill=(255, 255, 255), last=(255, 214, 10)):
    """Big outlined text, wrapped to max_w, last line highlighted. Returns RGBA."""
    f = _font(size)
    words = text.split()
    n = math.ceil(f.getlength(text) / max_w)
    lines = [""]
    for w in words:  # balanced lines: break once a line reaches its share of the text
        if lines[-1] and len(lines) < n and f.getlength(lines[-1]) >= f.getlength(text) / n * 0.92:
            lines.append(w)
        else:
            lines[-1] = (lines[-1] + " " + w).strip()
    stroke = max(4, size // 9)
    lh = int(size * 1.08)
    w = int(max(f.getlength(ln) for ln in lines)) + 4 * stroke
    img = Image.new("RGBA", (w, lh * len(lines) + 3 * stroke), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        col = last if i == len(lines) - 1 and len(lines) > 1 else fill
        d.text(((w - f.getlength(ln)) / 2, i * lh + stroke), ln, font=f, fill=col, stroke_width=stroke, stroke_fill=(0, 0, 0))
    return img


def _envelope(path):
    """Mouth open/closed per frame from the line's loudness, with a little hysteresis and no long held-open mouths."""
    x, sr = sf.read(path, dtype="float32")
    hop = sr // FPS
    rms = np.sqrt(np.array([np.mean(x[i:i + hop] ** 2) for i in range(0, len(x), hop)]) + 1e-9)
    ref = np.percentile(rms, 90) or 1.0
    out, is_open, held = [], False, 0
    for v in rms / ref:
        is_open = v > 0.32 if not is_open else v > 0.2
        held = held + 1 if is_open else 0
        if held > 4:  # flap closed for a frame every few frames so long vowels still read as talking
            is_open, held = False, 0
        out.append(is_open)
    return out


class Ctx:
    def __init__(self, ep):
        self.ep = ep
        self.script = load(ep)
        self.shots = {s["id"]: s for s in self.script["shots"]}
        self.timing = {int(k): v for k, v in json.loads((ep / "timing.json").read_text()).items()}
        self.meta = {c: json.loads((sprites.CHARS / c / "sprites.json").read_text()) for c in self.script["needs"]}
        self._plates = {}
        self._tints = {}

    def tint(self, name):
        """How a location's light colours the characters in it: darker and cooler at night, a little warm by day."""
        if name not in self._tints:
            m = np.asarray(self.plate(name).resize((64, 36), Image.BOX), np.float32).reshape(-1, 3).mean(0) / 255
            bright = min(1.0, 0.45 + 0.9 * float(m.mean()))
            self._tints[name] = tuple(bright * (1 - 0.2 * (1 - v / m.max())) for v in m)
        return self._tints[name]

    def plate(self, name):
        if name not in self._plates:
            if len(self._plates) > 3:
                self._plates.pop(next(iter(self._plates)))
            self._plates[name] = Image.open(_plate_path(name, self.script["plates"][name])).convert("RGB")
        return self._plates[name]

    def char_h(self, cid):
        m = self.meta[cid]["stand_neutral"] if "stand_neutral" in self.meta[cid] else next(iter(self.meta[cid].values()))
        return m["feet"][1] - m["top"]


class ShotRenderer:
    """Everything one shot needs: the camera path, a pre-scaled canvas of the location and pre-scaled actor sprites."""

    def __init__(self, ctx, sid):
        self.ctx, self.shot = ctx, ctx.shots[sid]
        self.dur = ctx.timing[sid]["dur"]
        self.audio = ctx.timing[sid].get("audio")
        shot = self.shot
        self.plate = ctx.plate(_shot_plate(shot))
        pw, ph = self.plate.size
        self.pw, self.ph = pw, ph
        self.actors = [] if shot.get("cam", {}).get("type") == "insert" else shot.get("actors", [])
        # (x, feet y, height) in plate pixels; in groups of 3+, every other character stands a step further back,
        # so a scene is staged in depth instead of a line-up
        self.geo = {}
        for i, a in enumerate(self.actors):
            depth = 0.05 if len(self.actors) >= 3 and i % 2 == 1 else 0.0
            self.geo[a["id"]] = (a["x"] * pw, (FLOOR - depth) * ph, BODY * ph * CAST["characters"][a["id"]]["height"] * (1 - 1.2 * depth))
        self.mouth = _envelope(ctx.ep / "audio" / self.audio) if self.audio and any(a.get("talk") for a in self.actors) else []
        speech = sf.info(ctx.ep / "audio" / self.audio).duration if self.audio else self.dur
        self.switch_t = LEAD + 0.5 * speech
        self.base_rect = self._rect(shot.get("cam", {}).get("type", "wide"), shot.get("cam", {}).get("focus", []))
        # a crash zoom punches in on the focus character's face
        self.zoom_rect = self._rect("close", shot.get("cam", {}).get("focus", [])) if self.actors else (self.pw / 2, self.ph / 2, self.pw * 0.6)
        samples = [self.cam(self.dur * i / 8) for i in range(9)] + [self.cam(0.3)]
        x0 = max(0, min(c[0] - c[2] / 2 for c in samples))
        y0 = max(0, min(c[1] - c[2] * 9 / 32 for c in samples))
        x1 = min(pw, max(c[0] + c[2] / 2 for c in samples))
        y1 = min(ph, max(c[1] + c[2] * 9 / 32 for c in samples))
        self.k = min(OUT_W / min(c[2] for c in samples), 4200 / (x1 - x0))   # canvas pixels per plate pixel
        self.origin = (x0, y0)
        canvas = self.plate.resize((round((x1 - x0) * self.k), round((y1 - y0) * self.k)), Image.BICUBIC, box=(x0, y0, x1, y1))
        blur = {"close": 3.0, "medium": 1.4, "two": 0.8}.get(shot.get("cam", {}).get("type"), 0)
        self.canvas = canvas.filter(ImageFilter.GaussianBlur(blur)) if blur else canvas
        self.sprites = {}
        self.hidden = set()   # characters left out of the picture (the animate stage drops ones the frame edge cuts off)
        self.shadows = {}
        for a in self.actors:  # a soft contact shadow grounds each character on the floor
            h = self.geo[a["id"]][2] * self.k
            shadow = Image.new("L", (int(h * 0.42), int(h * 0.06)), 0)
            ImageDraw.Draw(shadow).ellipse((0, 0, shadow.width - 1, shadow.height - 1), fill=90)
            self.shadows[a["id"]] = shadow.filter(ImageFilter.GaussianBlur(h * 0.006))
        rnd = random.Random(sid)
        self.blinks = {a["id"]: self._blink_times(rnd) for a in self.actors}
        self.phase = {a["id"]: rnd.random() * 6.28 for a in self.actors}
        self.pop = _text_image(shot["pop"], 150 if shot["kind"] in ("title", "card") else 120, 1500) if shot.get("pop") else None

    def _blink_times(self, rnd):
        t, out = rnd.uniform(0.4, 2.0), []
        while t < self.dur:
            out.append(t)
            t += rnd.uniform(2.2, 4.8)
        return out

    def _rect(self, kind, focus):
        """Camera framing as (centre x, centre y, width) in plate pixels; the height is width * 9/16."""
        pw, ph = self.pw, self.ph
        if kind in ("wide", "insert") or not self.actors:
            return pw / 2, ph / 2, pw * 0.97
        focus = [f for f in focus if f in self.geo] or [self.actors[0]["id"]]
        xs = [self.geo[f][0] for f in focus]
        h = max(self.geo[f][2] for f in focus)
        top = min(self.geo[f][1] - self.geo[f][2] for f in focus)
        if kind == "two" or len(focus) > 1:
            w = max((FLOOR * ph - 0.3 * h - top + 0.08 * h) * 16 / 9, max(xs) - min(xs) + 0.75 * h)
            cx, cy = (min(xs) + max(xs)) / 2, top - 0.08 * h + w * 9 / 32
        else:
            span = 0.72 * h if kind == "medium" else 0.42 * h
            w = span * 16 / 9
            a = next(a for a in self.actors if a["id"] == focus[0])
            cx = xs[0] + (0.1 * w if not a["flip"] else -0.1 * w) * (a["pose"] != "front")
            cy = top - 0.06 * h + span / 2
        w = min(w, pw)
        return (min(max(cx, w / 2), pw - w / 2), min(max(cy, w * 9 / 32), ph - w * 9 / 32), w)

    def cam(self, t):
        cx, cy, w = self.base_rect
        u = t / max(self.dur, 0.01)
        if self.shot.get("zoom"):
            z = _smooth(t / 0.3)
            zx, zy, zw = self.zoom_rect
            cx, cy, w = cx + (zx - cx) * z, cy + (zy - cy) * z, w + (zw - w) * z
        elif self.shot.get("cam", {}).get("type") in ("wide", "insert") or not self.actors:
            w *= 1 - 0.04 * u
        else:
            w *= 1 - 0.07 * _smooth(u)
        w = min(w, self.pw)
        return (min(max(cx, w / 2), self.pw - w / 2), min(max(cy, w * 9 / 32), self.ph - w * 9 / 32), w)

    def _sprite(self, cid, name, flip):
        meta = self.ctx.meta[cid]
        if name not in meta:
            name = name.rsplit("_", 1)[0] if name.endswith(("_talk", "_blink")) else name
        key = (cid, name, flip)
        if key not in self.sprites:
            img = Image.open(sprites.CHARS / cid / f"{name}.png")
            s = self.geo[cid][2] * self.k / self.ctx.char_h(cid)
            img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
            r, g, b, a = img.split()  # light the character like the location, so they don't look pasted on
            img = Image.merge("RGBA", [ch.point(lambda v, k=k: v * k) for ch, k in zip((r, g, b), self.ctx.tint(_shot_plate(self.shot)))] + [a])
            fx, fy = meta[name]["feet"][0] * s, meta[name]["feet"][1] * s
            if flip:
                img, fx = ImageOps.mirror(img), img.width - fx
            self.sprites[key] = (img, fx, fy)
        return self.sprites[key]

    def _actor_x(self, a, t):
        x = self.geo[a["id"]][0]
        move = a.get("move")
        if not move:
            return x, a["flip"], False
        span = min(1.8, 0.6 * self.dur)
        off = -0.15 * self.pw if move["side"] == "left" else 1.15 * self.pw
        if move["kind"] == "enter":
            u = t / span
            pos = off + (x - off) * min(1.0, u)
        else:
            u = (t - (self.dur - span)) / span
            pos = x + (off - x) * max(0.0, min(1.0, u))
        moving = 0 < u < 1
        return pos, (off > x) == (move["kind"] == "enter") if moving else a["flip"], moving

    def frame(self, t):
        """The shot at time t. Also leaves self.boxes: each character's box in output pixels (for lip-sync masks)."""
        img = self.canvas.copy()
        x0, y0 = self.origin
        fi = int((t - LEAD) * FPS)
        drawn = {}
        for a in self.actors:
            cid = a["id"]
            if cid in self.hidden:
                continue
            late = t >= self.switch_t and ("pose2" in a or "expr2" in a)
            pose, expr = (a.get("pose2", a["pose"]), a.get("expr2", a["expr"])) if late else (a["pose"], a["expr"])
            x, flip, moving = self._actor_x(a, t)
            h = self.geo[cid][2] * self.k
            if moving:
                name = f"walk_{expr}" if int(t / 0.2) % 2 == 0 else f"{pose}_{expr}"
                bob = -abs(math.sin(t * math.pi / 0.2)) * 0.015 * h
            else:
                name = f"{pose}_{expr}"
                talking = a.get("talk") and 0 <= fi < len(self.mouth)
                if talking and self.mouth[fi]:
                    name += "_talk"
                elif any(0 <= t - b < 0.11 for b in self.blinks[cid]):
                    name += "_blink"
                bob = math.sin(t * 2.0 + self.phase[cid]) * 0.003 * h
                if talking:
                    bob -= abs(math.sin(t * 7.0)) * 0.004 * h
            spr, fx, fy = self._sprite(cid, name, flip)
            cx, cy = (x - x0) * self.k, (self.geo[cid][1] - y0) * self.k
            shadow = self.shadows[cid]
            img.paste((0, 0, 0), (int(cx - shadow.width / 2), int(cy - shadow.height / 2)), shadow)
            img.paste(spr, (int(cx - fx), int(cy - fy + bob)), spr)
            drawn[cid] = (cx - fx, cy - fy + bob, cx - fx + spr.width, cy - fy + bob + spr.height)
        cx, cy, w = self.cam(t)
        if self.shot.get("shake") and t < 0.5:
            amp = 0.012 * w * (1 - t / 0.5)
            cx, cy = cx + amp * math.sin(t * 90), cy + amp * math.cos(t * 70)
        bw, bh = w * self.k, w * 9 / 16 * self.k
        bx = min(max((cx - x0) * self.k - bw / 2, 0), img.width - bw)
        by = min(max((cy - y0) * self.k - bh / 2, 0), img.height - bh)
        sx, sy = OUT_W / bw, OUT_H / bh
        self.boxes = {c: ((b[0] - bx) * sx, (b[1] - by) * sy, (b[2] - bx) * sx, (b[3] - by) * sy) for c, b in drawn.items()
                      if b[2] > bx and b[0] < bx + bw}  # only characters at least partly in the frame
        return img.resize((OUT_W, OUT_H), Image.BILINEAR, box=(max(bx, 0), max(by, 0), min(bx + bw, img.width), min(by + bh, img.height)))


def _card_background(ctx, shot):
    plate = ctx.plate(shot["plate"]).resize((OUT_W, OUT_H), Image.BICUBIC).filter(ImageFilter.GaussianBlur(10))
    return ImageEnhance.Brightness(plate).enhance(0.45)


def _overlay_pop(frame, pop, t):
    s = 0.6 + 0.4 * _smooth(t / 0.18)
    img = pop.resize((max(1, int(pop.width * s)), max(1, int(pop.height * s))), Image.BICUBIC)
    frame.paste(img, ((OUT_W - img.width) // 2, (OUT_H - img.height) // 2), img)


def _caption_chunks(text, dur):
    words = text.split()
    chunks = [" ".join(words[i:i + 6]) for i in range(0, len(words), 6)]
    total = sum(len(c) for c in chunks)
    out, t = [], LEAD
    for c in chunks:
        d = dur * len(c) / total
        out.append((t, t + d, c))
        t += d
    return out


def _key_time(r):
    """When a shot's first frame for the video model is taken: after an entrance, so the walker is in the picture."""
    return next((min(1.8, 0.6 * r.dur) for a in r.actors if a.get("move", {}).get("kind") == "enter"), 0.0)


def _clip_info(ep, sid):
    path = ep / "clips" / f"shot_{sid:03d}.json"
    return json.loads(path.read_text()) if path.exists() and (ep / "clips_hd" / f"shot_{sid:03d}.mp4").exists() else None


def _clip_camera(frame, t, dur, shot, focus):
    """Camera moves on an animated clip: a gentle push in, or a crash zoom onto the focus character's face."""
    z, cx, cy = 1 + 0.04 * t / max(dur, 0.01), OUT_W / 2, OUT_H / 2
    if shot.get("zoom") and focus:
        s = _smooth(t / 0.3)
        z, cx, cy = 1 + 0.5 * s, cx + (focus[0] - cx) * s, cy + (focus[1] - cy) * s
    w, h = OUT_W / z, OUT_H / z
    if shot.get("shake") and t < 0.5:
        amp = 0.012 * OUT_W * (1 - t / 0.5)
        cx, cy = cx + amp * math.sin(t * 90), cy + amp * math.cos(t * 70)
    cx, cy = min(max(cx, w / 2), OUT_W - w / 2), min(max(cy, h / 2), OUT_H - h / 2)
    return frame.resize((OUT_W, OUT_H), Image.BILINEAR, box=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))


def _frames(ctx, sid):
    """Yield every frame of a shot as an RGB image: the animated clip if the animate stage made one, else the drawn shot."""
    shot = ctx.shots[sid]
    tm = ctx.timing[sid]
    n = round((tm["start"] + tm["dur"]) * FPS) - round(tm["start"] * FPS)
    if shot["kind"] in ("title", "card"):
        bg = _card_background(ctx, shot)
        pop = _text_image(shot["pop"], 150, 1500) if shot.get("pop") else None
        for f in range(n):
            frame = bg.copy()
            if pop:
                _overlay_pop(frame, pop, f / FPS)
            yield frame
        return
    r = ShotRenderer(ctx, sid)
    info = _clip_info(ctx.ep, sid)
    clip = focus = None
    if info:
        stretch = max(1.0, r.dur / info["seconds"]) if info["mode"] == "i2v" else 1.0
        clip = animate.ClipReader(FFMPEG, ctx.ep / "clips_hd" / f"shot_{sid:03d}.mp4", stretch)
        r.frame(_key_time(r))
        who = (shot.get("cam", {}).get("focus") or [None])[0]
        if who in r.boxes:
            b = r.boxes[who]
            focus = ((b[0] + b[2]) / 2, b[1] + 0.15 * (b[3] - b[1]))
    prev = ctx.shots.get(sid - 1)
    fade_in = prev is None or prev.get("plate") != shot["plate"] or prev["kind"] in ("title", "card")
    caps = []
    if ctx.script.get("captions") and r.audio:
        caps = [(a, b, _text_image(text, 64, 1600, last=(255, 255, 255)))
                for a, b, text in _caption_chunks(shot["text"], sf.info(ctx.ep / "audio" / r.audio).duration)]
    for f in range(n):
        t = f / FPS
        frame = _clip_camera(clip.at(t).resize((OUT_W, OUT_H)), t, r.dur, shot, focus) if clip else r.frame(t)
        if fade_in and t < 0.3:
            frame = Image.blend(Image.new("RGB", frame.size), frame, _smooth(t / 0.3))
        if r.pop:
            _overlay_pop(frame, r.pop, t)
        for a, b, cap in caps:
            if a <= t < b:
                frame.paste(cap, ((OUT_W - cap.width) // 2, int(OUT_H * 0.86) - cap.height // 2), cap)
        yield frame
    if clip:
        clip.close()


def stage_animate(ep, script, only=None):
    """Animate every shot from its drawn first frame (see animate.py), then upscale the clips to 1080p."""
    ctx = Ctx(ep)
    clips, hd = ep / "clips", ep / "clips_hd"
    clips.mkdir(exist_ok=True)
    hd.mkdir(exist_ok=True)
    todo = [s for s in script["shots"] if s["kind"] not in ("title", "card") and (only is None or s["id"] in only)]
    t_all, made = time.time(), 0
    for n, shot in enumerate(todo, 1):
        sid = shot["id"]
        r = ShotRenderer(ctx, sid)
        # a character the frame edge cuts off gets re-imagined (off-model) by the video model: leave them out of the shot
        r.frame(_key_time(r))
        r.hidden = {a["id"] for a in r.actors if a["id"] not in r.boxes or
                    min(r.boxes[a["id"]][2], OUT_W) - max(r.boxes[a["id"]][0], 0) < 0.7 * (r.boxes[a["id"]][2] - r.boxes[a["id"]][0])}
        first = r.frame(_key_time(r)).resize((animate.GEN_W, animate.GEN_H), Image.LANCZOS)
        visible = [a["id"] for a in r.actors if a["id"] not in r.hidden]
        talker = next((a["id"] for a in r.actors if a.get("talk") and a["id"] in visible), None) if r.audio else None
        text = animate.prompt(CAST, script, shot, talker, visible)
        others = sorted(c for c in r.boxes if c != talker) if talker else []
        spec = {"mode": "talk" if talker else "i2v", "prompt": text, "dur": r.dur, "audio": r.audio, "seed": sid, "others": others,
                "model": cc.WAN21_UNET}
        key = animate.key(spec, first)
        out, info = clips / f"shot_{sid:03d}.mp4", clips / f"shot_{sid:03d}.json"
        if out.exists() and info.exists() and json.loads(info.read_text())["key"] == key:
            continue
        (hd / out.name).unlink(missing_ok=True)
        first_png = clips / f"first_{sid:03d}.png"
        first.save(first_png)
        t0 = time.time()
        if talker:
            frames = math.ceil(r.dur * cc.TALK_FPS) + 1
            sizes = cc.talk_chunks(frames)
            audio_len = (sum(sizes) - cc.TALK_MOTION * (len(sizes) - 1)) / cc.TALK_FPS + 0.1
            wav = animate.talk_audio(ep / "audio" / r.audio, LEAD, max(r.dur, audio_len), clips / f"talk_{sid:03d}.wav")
            masks = [cc.to_input(p) for p in animate.masks(r.boxes, talker, clips, sid)] if others else None
            quiet = cc.to_input(animate.silence(clips / "silence.wav")) if others else None
            wf = cc.infinitetalk(cc.to_input(first_png), cc.to_input(wav), text, frames, animate.GEN_W, animate.GEN_H, seed=sid,
                                 masks=masks, silence=quiet)
            seconds = r.dur
        else:
            length = animate.i2v_frames(r.dur)
            wf = cc.wan21_i2v(cc.to_input(first_png), text, animate.GEN_W, animate.GEN_H, length=length, seed=sid)
            seconds = (length - 1) / animate.I2V_FPS
        files, _ = cc.run(wf, clips, f"shot_{sid:03d}_new")
        files[0].replace(out)
        info.write_text(json.dumps({"key": key, "mode": spec["mode"], "seconds": seconds, "prompt": text}, indent=1))
        made += 1
        print(f"[{n}/{len(todo)}] shot {sid} {spec['mode']} {r.dur:.1f}s: {time.time() - t0:.0f}s", flush=True)
    cc.free()  # the upscaler needs the GPU
    subprocess.run([str(cc.COMFY_DIR / "venv" / "Scripts" / "python.exe"), str(ROOT.parent / "shared" / "upscale_clips.py"),
                    str(clips), str(hd), FFMPEG], check=True)
    print(f"ANIMATE {made} new clips in {(time.time() - t_all) / 60:.0f} min")


_CTX = None


def _shot_key(ctx, sid):
    """Fingerprint of everything a shot's pixels depend on, so unchanged shots are never rendered twice."""
    shot = ctx.shots[sid]
    ids = [a["id"] for a in shot.get("actors", [])]
    name = _shot_plate(shot)
    plate = _plate_path(name, ctx.script["plates"][name]) if name else None
    data = [RENDER_VERSION, {k: v for k, v in shot.items() if k != "chapter"}, ctx.timing[sid], ctx.shots.get(sid - 1, {}).get("plate"), ctx.script.get("captions"),
            {c: ctx.meta[c] for c in ids}, str(plate), plate.stat().st_mtime if plate else 0, _clip_info(ctx.ep, sid)]
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _render_shot(args):
    global _CTX
    ep, sid = args
    if _CTX is None or _CTX.ep != ep:
        _CTX = Ctx(ep)
    ctx = _CTX
    out = ep / "segments" / f"shot_{sid:03d}.mp4"
    key, stamp = _shot_key(ctx, sid), out.with_suffix(".key")
    if out.exists() and stamp.exists() and stamp.read_text() == key:
        return sid, 0, 0.0
    t0 = time.time()
    enc = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OUT_W}x{OUT_H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
                            "-threads", "2", str(out)], stdin=subprocess.PIPE)
    n = 0
    for frame in _frames(ctx, sid):
        enc.stdin.write(frame.tobytes())
        n += 1
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on shot {sid}")
    stamp.write_text(key)
    return sid, n, time.time() - t0


def stage_render(ep, script):
    (ep / "segments").mkdir(exist_ok=True)
    t0, frames = time.time(), 0
    with mp.Pool(WORKERS) as pool:
        for sid, n, secs in pool.imap_unordered(_render_shot, [(ep, s["id"]) for s in script["shots"]]):
            if n:
                frames += n
                print(f"shot {sid}: {n} frames in {secs:.1f}s", flush=True)
    lst = ep / "segments" / "list.txt"
    lst.write_text("".join(f"file 'shot_{s['id']:03d}.mp4'\n" for s in script["shots"]))
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(ep / "video.mp4")], check=True)
    print(f"RENDER {frames} new frames in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- thumbnails

def stage_thumbs(ep, script):
    (ep / "final").mkdir(exist_ok=True)
    out = []
    for i, spec in enumerate(script["thumbs"], 1):
        W, H = 1280, 720
        plate = Image.open(_plate_path(spec["plate"], script["plates"][spec["plate"]])).convert("RGB")
        img = ImageOps.fit(plate, (W, H)).filter(ImageFilter.GaussianBlur(2))
        img = ImageEnhance.Color(ImageEnhance.Contrast(img).enhance(1.15)).enhance(1.3)
        for j, c in enumerate(spec["chars"][:2]):
            meta = json.loads((sprites.CHARS / c["id"] / "sprites.json").read_text())[f"{c['pose']}_{c['expr']}"]
            spr = Image.open(sprites.CHARS / c["id"] / f"{c['pose']}_{c['expr']}.png")
            h_char = meta["feet"][1] - meta["top"]
            target = H * (2.1 if j == 0 else 1.7)  # waist-up: the face is what sells the click
            s = target / h_char
            spr = spr.resize((int(spr.width * s), int(spr.height * s)), Image.LANCZOS)
            if j == 1:
                spr = ImageOps.mirror(spr)
            x = int(W * (0.74 if j == 0 else 0.2) - spr.width / 2)
            y = int(H * 0.06 - meta["top"] * s)
            glow = spr.split()[3].filter(ImageFilter.MaxFilter(15)).filter(ImageFilter.GaussianBlur(6))
            img.paste((255, 255, 255), (x, y), glow)
            img.paste(spr, (x, y), spr)
        text = _text_image(spec["text"].upper(), 130, int(W * 0.5))
        text = text.rotate(4, expand=True, resample=Image.BICUBIC)
        img.paste(text, (int(W * 0.03), int(H * 0.97 - text.height)), text)
        path = ep / "final" / f"thumb_{i}.jpg"
        img.save(path, quality=92)
        out.append(img)
    if out:
        (ep / "qa").mkdir(exist_ok=True)
        # top row as on a desktop, bottom row at the size a phone shows them
        sheet = Image.new("RGB", (640 * len(out), 360 + 200), (24, 24, 28))
        for i, o in enumerate(out):
            sheet.paste(o.resize((640, 360)), (640 * i, 0))
            sheet.paste(o.resize((240, 135)), (640 * i + 200, 392))
        sheet.save(ep / "qa" / "thumbs.jpg", quality=85)
    print(f"THUMBS {len(out)}")


# ---------------------------------------------------------------- sound

def _resample(x, sr_in, sr_out):
    n_out = int(round(len(x) * sr_out / sr_in))
    X = np.fft.rfft(x)
    Y = np.zeros(n_out // 2 + 1, dtype=complex)
    m = min(len(X), len(Y))
    Y[:m] = X[:m]
    return (np.fft.irfft(Y, n_out) * (sr_out / sr_in)).astype(np.float32)


def _place(track, x, at, gain_db=0.0):
    i = int(at * sfx.SR)
    if 0 <= i < len(track):
        x = x[:len(track) - i] * 10 ** (gain_db / 20)
        track[i:i + len(x)] += x


def _ambience_kind(prompt):
    p = prompt.lower()
    if "night" in p:
        return "night"
    if any(w in p for w in ("street", "city", "outside", "exterior", "sidewalk", "park")):
        return "city"
    return "room"


def stage_music(ep, script):
    music.ensure(log=lambda m: print(m, flush=True))


def stage_mix(ep, script):
    timing = {int(k): v for k, v in json.loads((ep / "timing.json").read_text()).items()}
    total = max(v["start"] + v["dur"] for v in timing.values())
    n = int(total * sfx.SR) + sfx.SR
    voice, fx, amb = (np.zeros(n, np.float32) for _ in range(3))
    beds = np.zeros((n, 2), np.float32)
    lib = {}

    def play(name, at, gain=0.0):
        if name not in lib:
            lib[name] = sfx.LIB[name]()
        _place(fx, lib[name], at, gain)

    prev_plate = None
    for shot in script["shots"]:
        sid = shot["id"]
        start, dur = timing[sid]["start"], timing[sid]["dur"]
        if shot["kind"] == "line":
            x, sr = sf.read(ep / "audio" / timing[sid]["audio"], dtype="float32")
            _place(voice, _resample(x, sr, sfx.SR), start + LEAD)
        for at, name in shot.get("sfx", []):
            play(name, start + at * dur)
        if shot.get("zoom"):
            play("whoosh", start - 0.15, -4)
        if shot.get("shake"):
            play("thump", start + 0.05, -2)
        if shot["kind"] == "title":
            play("whoosh", start, -3)
            play("boom", start + 0.15, -6)
        if shot.get("plate") and shot["plate"] != prev_plate and shot["kind"] not in ("title", "card"):
            play("whoosh_down", start, -10)
        for a in shot.get("actors", []):
            if "move" in a:
                span = min(1.8, 0.6 * dur)
                at = start if a["move"]["kind"] == "enter" else start + dur - span
                _place(fx, sfx.footsteps(span), at, -6)
            if a.get("expr2") in ("shocked", "scared") and a.get("expr") not in ("shocked", "scared"):
                play("pop", start + LEAD + 0.5 * (dur - LEAD - TAIL), -8)
        if shot.get("plate") and shot["kind"] not in ("title", "card"):
            spec = script["plates"][shot["plate"]]
            if not spec["insert"]:
                a = sfx.ambience(_ambience_kind(spec["prompt"]), dur + 0.1, seed=sid)
                ramp = np.minimum(1, np.minimum(np.arange(len(a)), np.arange(len(a))[::-1]) / (0.05 * sfx.SR)).astype(np.float32)
                _place(amb, a * ramp, start, -12)
            prev_plate = shot["plate"]
    runs = []
    for shot in script["shots"]:
        mood = shot.get("mood", "calm")
        if not runs or runs[-1][0] != mood:
            runs.append([mood, timing[shot["id"]]["start"]])
    for i, (mood, start) in enumerate(runs):
        end = runs[i + 1][1] if i + 1 < len(runs) else total
        length = end - start + 1.0
        bed = music.bed(mood, length, i)
        fade = np.ones(len(bed), np.float32)
        f = int(sfx.SR * 1.0)
        if i:
            fade[:f] = np.linspace(0, 1, f)
        fade[-f:] = np.linspace(1, 0, f)
        _place(beds, bed * fade[:, None], max(0.0, start - (0.5 if i else 0)), -6)
    env = np.convolve(np.abs(voice), np.ones(int(0.3 * sfx.SR), np.float32) / int(0.3 * sfx.SR), mode="same")
    duck = 1 - 0.6 * np.clip(env / 0.05, 0, 1)
    mix = np.tanh(((voice + fx * 0.9 + amb)[:, None] + beds * duck[:, None]) * 1.1) / 1.1
    sf.write(ep / "mix.wav", mix[:int(total * sfx.SR)], sfx.SR, subtype="FLOAT")
    print(f"MIX {total / 60:.1f} min")


# ---------------------------------------------------------------- review and delivery

def stage_qa(ep, script):
    timing = {int(k): v for k, v in json.loads((ep / "timing.json").read_text()).items()}
    qa = ep / "qa"
    qa.mkdir(exist_ok=True)
    tiles = []
    for shot in script["shots"]:
        seg = ep / "segments" / f"shot_{shot['id']:03d}.mp4"
        mid = timing[shot["id"]]["dur"] * 0.6
        png = subprocess.run([FFMPEG, "-loglevel", "error", "-ss", f"{mid:.2f}", "-i", str(seg), "-frames:v", "1", "-f", "image2pipe",
                              "-vcodec", "png", "-"], capture_output=True).stdout
        im = Image.open(io.BytesIO(png)).convert("RGB").resize((384, 216))
        ImageDraw.Draw(im).text((6, 4), f"{shot['id']} {shot.get('cam', {}).get('type', shot['kind'])}", fill=(255, 255, 0), font=_font(18))
        tiles.append(im)
    for i in range(0, len(tiles), 40):
        _sheet(tiles[i:i + 40], 5, 384, 216, qa / f"frames_{i // 40 + 1}.jpg")
    total = max(v["start"] + v["dur"] for v in timing.values())
    warn = []
    if total < 480:
        warn.append(f"runtime {total / 60:.2f} min is under 8:00 (no mid-roll ads)")
    if len(script["title"]) > 60:
        warn.append("title is over 60 characters")
    first = [s for s in script["shots"] if timing[s["id"]]["start"] < 10]
    if not any(s.get("zoom") or s.get("pop") or s.get("shake") for s in first):
        warn.append("nothing punchy (zoom/pop/shake) in the first 10 s")
    print(f"QA {len(tiles)} shots, {total / 60:.2f} min" + "".join(f"\n  WARNING: {w}" for w in warn))


def _srt_time(t):
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d},{int(t % 1 * 1000):03d}"


def stage_final(ep, script):
    (ep / "final").mkdir(exist_ok=True)
    target = "I=-14:TP=-1.5:LRA=11"
    err = subprocess.run([FFMPEG, "-i", str(ep / "mix.wav"), "-af", f"loudnorm={target}:print_format=json", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    m = json.loads(err[err.rindex("{"):err.rindex("}") + 1])
    af = (f"loudnorm={target}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}"
          f":measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    out = ep / "final" / f"{ep.name}.mp4"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(ep / "video.mp4"), "-i", str(ep / "mix.wav"), "-af", af, "-ar", "48000",
                    "-ac", "2", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-maxrate", "12M", "-bufsize", "24M",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)], check=True)
    timing = {int(k): v for k, v in json.loads((ep / "timing.json").read_text()).items()}
    srt, i = [], 1
    for shot in script["shots"]:
        if shot["kind"] != "line":
            continue
        speech = sf.info(ep / "audio" / timing[shot["id"]]["audio"]).duration
        for a, b, text in _caption_chunks(shot["text"], speech):
            t0 = timing[shot["id"]]["start"]
            srt.append(f"{i}\n{_srt_time(t0 + a)} --> {_srt_time(t0 + b)}\n{text}\n")
            i += 1
    (ep / "final" / f"{ep.name}.srt").write_text("\n".join(srt), encoding="utf-8")
    chapters = [(timing[s["id"]]["start"], s["chapter"]) for s in script["shots"] if s.get("chapter")]
    if chapters:
        chapters[0] = (0.0, chapters[0][1])
    cast = sorted({a["id"] for s in script["shots"] for a in s.get("actors", [])})
    lines = [script["title"], "", f"{script['title']}. An animated story from {CAST['series']}.", "",
             "Characters: " + ", ".join(CAST["characters"][c]["name"] for c in cast), ""]
    lines += [f"{int(t // 60)}:{int(t % 60):02d} {name}" for t, name in chapters]
    (ep / "final" / f"{ep.name}_upload.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("FINAL", out)


if __name__ == "__main__":
    if sys.argv[1] == "redraw":
        print("forgot:", ", ".join(sprites.redraw(sys.argv[2], sys.argv[3])) or "nothing (no such drawing yet)")
        sys.exit()
    ep = ROOT / "episodes" / sys.argv[1]
    bench = json.loads((ep / "bench.json").read_text()) if (ep / "bench.json").exists() else {}
    for stage in [s for a in sys.argv[2:] for s in (STAGES if a == "all" else [a])]:
        t0 = time.time()
        stage, _, shots = stage.partition(":")  # animate:0-21 animates only shots 0 to 21
        first, _, last = shots.partition("-")
        kwargs = {"only": set(range(int(first), int(last or first) + 1))} if shots else {}
        globals()[f"stage_{stage}"](ep, load(ep) if stage != "direct" else None, **kwargs)
        bench[stage] = round(time.time() - t0, 1)
        print(f"[{stage}] {bench[stage]}s", flush=True)
    (ep / "bench.json").write_text(json.dumps(bench, indent=1))

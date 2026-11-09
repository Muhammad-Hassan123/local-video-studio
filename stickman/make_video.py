"""Story-first pipeline: AI-painted plates (no characters) + code-drawn stick rigs + voices, captions, SFX and music.

Usage:  venv\\Scripts\\python make_video.py <episode> [direct|tts|plates|words|render|mix|final|all ...]
  direct   screenplay.txt -> script.json (automatic shot planning; director.py documents the screenplay format)
  tts      one wav per spoken shot, voice by speaker                -> audio/shot_XX.wav, timing.json
  plates   empty backgrounds in the cinematic look (2 candidates)   -> plates/<name>_cN.png   (ComfyUI must be running)
  words    word timings for captions (faster-whisper on each line)  -> words.json
  render   every shot drawn frame by frame, shots in parallel       -> segments/shot_XX.mp4, video.mp4
  mix      voices + SFX + ambience + music, ducked under the voice  -> mix.wav
  qa       contact sheets + hook / packaging warnings                 -> qa/
  thumbs   thumbnail variants from the THUMB: lines                  -> final/thumb_N.jpg, qa/thumbs.jpg
  final    loudness-normalised mux                                  -> final/<episode>.mp4
Shot direction (plate, camera, actors, props, pops, fx, sfx) lives in script.json; rig.py draws the characters, sfx.py the sounds.
"""
import hashlib
import json
import math
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import soundfile as sf
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
import comfy_client as cc  # noqa: E402
import overlays
import rig
import sfx

ROOT = Path(__file__).parent
MODELS = ROOT.parent / "models"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
# background looks (screenplay LOOK:); the character body is STYLE: (rig.BODIES)
PLATE_STYLES = {
    "cinematic": ("Cinematic painterly illustration, a detailed painted environment with realistic lighting, soft volumetric light, "
                  "atmospheric haze and depth, muted film color grade with deep shadows, serious mood, no people, no characters, "
                  "no animals, no text, no letters, no words"),
    "flat": ("Flat 2D vector cartoon illustration, clean thick black outlines, bright flat colors, no shading, no texture, simple "
             "shapes, explainer video style, no people, no characters, no animals, no text, no letters, no words"),
}
PLATE_W, PLATE_H = 1920, 1088
OUT_W, OUT_H, FPS = 1920, 1080, 24
LEAD, TAIL = 0.12, 0.28            # silence before / after every spoken line inside its shot
FONT_BOLD = r"C:\Windows\Fonts\segoeuib.ttf"
FONT_HUD = r"C:\Windows\Fonts\consolab.ttf"
WORKERS = 8
STAGES = ["direct", "tts", "plates", "words", "qa", "thumbs", "render", "mix", "final"]
CAST = json.loads((ROOT / "cast.json").read_text(encoding="utf-8"))
# music bed per scene mood: pad chords (MIDI notes) and seconds per chord; "dark" uses the drone instead
MOODS = {"calm": (((57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62)), 4.0),
         "happy": (((60, 64, 67), (67, 71, 74), (69, 72, 76), (65, 69, 72)), 3.0),
         "tense": (((50, 53, 57), (46, 50, 53), (43, 46, 50), (45, 49, 52)), 2.0)}


def _amb(script, plate):
    """Ambience bed for a plate: its amb= flag, else guessed from the plate prompt."""
    flag = script.get("plate_flags", {}).get(plate, {}).get("amb")
    if flag:
        return flag
    spec = script["plates"][plate]
    text = (spec["prompt"] if isinstance(spec, dict) else spec).lower()
    if "night" in text:
        return "night"
    if "bus" in text and "inside" in text:
        return "bus"
    if any(w in text for w in ("street", "city", "road", "park", "outside")):
        return "city"
    return "room"


def load(ep):
    return json.loads((ep / "script.json").read_text(encoding="utf-8"))


def _voice(script, who):
    return script.get("voices", {}).get(who) or (script["narrator_voice"] if who == "narrator" else CAST[who]["voice"])


def _audio_name(script, shot):
    """Lines are cached by voice + text, so editing the screenplay only regenerates the lines that changed."""
    return hashlib.sha1(f"{_voice(script, shot['who'])}|{shot['text']}".encode()).hexdigest()[:12] + ".wav"


def stage_tts(ep, script):
    from kokoro_onnx import Kokoro
    adir = ep / "audio"
    adir.mkdir(exist_ok=True)
    k = Kokoro(str(MODELS / "kokoro-v1.0.onnx"), str(MODELS / "voices-v1.0.bin"))
    timing, t, made = {}, 0.0, 0
    for shot in script["shots"]:
        entry = {}
        if "text" in shot:
            wav = adir / _audio_name(script, shot)
            if not wav.exists():
                samples, sr = k.create(shot["text"], voice=_voice(script, shot["who"]), speed=1.0, lang="en-us")
                sf.write(wav, samples, sr)
                made += 1
            dur = LEAD + sf.info(wav).duration + TAIL + shot.get("hold", 0.0)
            entry["audio"] = wav.name
        else:
            dur = shot["dur"]
        timing[shot["id"]] = {"start": round(t, 3), "dur": round(dur, 3), **entry}
        t += dur
    (ep / "timing.json").write_text(json.dumps(timing, indent=1))
    print(f"TTS: {len(timing)} shots, {t:.1f}s, {made} lines generated")


def stage_plates(ep, script):
    pdir = ep / "plates"
    pdir.mkdir(exist_ok=True)
    t0 = time.time()
    for name, spec in script["plates"].items():
        spec = spec if isinstance(spec, dict) else {"prompt": spec}
        refs = []
        if "ref" in spec:
            refs = [cc.to_input(pdir / f"{spec['ref']}_c{script.get('plate_picks', {}).get(spec['ref'], 1)}.png")]
        for n in (1, 2):
            if (pdir / f"{name}_c{n}.png").exists():
                continue
            style = PLATE_STYLES[script.get("look", "cinematic")]
            files, secs = cc.run(cc.klein_image(spec["prompt"] + ". " + style, PLATE_W, PLATE_H, seed=900 + n, refs=refs,
                                                filename_prefix=f"stickman/{ep.name}/plate"), pdir, f"{name}_c{n}")
            print(f"{name}_c{n}: {secs:.1f}s", flush=True)
    print(f"PLATES in {time.time() - t0:.0f}s")


def _resample(x, sr_in, sr_out):
    """Band-limited resample by zero-padding the spectrum (exact for an integer ratio like 24k -> 48k)."""
    n_out = int(round(len(x) * sr_out / sr_in))
    X = np.fft.rfft(x)
    Y = np.zeros(n_out // 2 + 1, dtype=complex)
    m = min(len(X), len(Y))
    Y[:m] = X[:m]
    return (np.fft.irfft(Y, n_out) * (sr_out / sr_in)).astype(np.float32)


def stage_words(ep, script):
    from faster_whisper import WhisperModel
    model = WhisperModel("small.en", device="cpu", compute_type="int8", download_root=str(MODELS / "whisper"))
    timing = json.loads((ep / "timing.json").read_text())
    old = json.loads((ep / "words.json").read_text()) if (ep / "words.json").exists() else {}
    words = {}
    for shot in script["shots"]:
        if "text" not in shot:
            continue
        name = timing[str(shot["id"])]["audio"]
        if name in old:
            words[name] = old[name]
            continue
        x, sr = sf.read(ep / "audio" / name, dtype="float32")
        segs, _ = model.transcribe(_resample(x, sr, 16000), language="en", word_timestamps=True, initial_prompt=shot["text"])
        heard = [(w.start, w.end) for s in segs for w in s.words]
        toks = shot["text"].split()
        if len(heard) != len(toks):  # whisper split or merged a word: spread the script's words by length instead
            total, acc = len(x) / sr, 0
            weights = [len(tk) + 2 for tk in toks]
            heard = []
            for wgt in weights:
                heard.append((acc / sum(weights) * total, (acc + wgt) / sum(weights) * total))
                acc += wgt
        words[name] = [[tk, round(a, 3), round(b, 3)] for tk, (a, b) in zip(toks, heard)]
    (ep / "words.json").write_text(json.dumps(words, indent=1))
    print("WORDS", sum(len(v) for v in words.values()))


# ---------------------------------------------------------------- frame composition

def _smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


class Ctx:
    """Everything a worker needs to draw any frame of the episode."""

    def __init__(self, ep):
        self.ep = ep
        self.script = load(ep)
        self.timing = {int(k): v for k, v in json.loads((ep / "timing.json").read_text()).items()}
        words = json.loads((ep / "words.json").read_text())
        self.words = {sid: words[t["audio"]] for sid, t in self.timing.items() if "audio" in t}
        self.shots = {s["id"]: s for s in self.script["shots"]}
        self.plates, self.tints, self.envs, self.fonts, self.shadows, self.fits = {}, {}, {}, {}, {}, {}
        yy, xx = np.mgrid[0:OUT_H, 0:OUT_W].astype(np.float32)
        r = np.sqrt(((xx - OUT_W / 2) / (OUT_W / 2)) ** 2 + ((yy - OUT_H / 2) / (OUT_H / 2)) ** 2)
        self.vignette = (1 - 0.32 * np.clip(r - 0.55, 0, 1) ** 1.5)[..., None]
        rng = np.random.default_rng(0)
        self.grain = [rng.normal(0, 3, (OUT_H, OUT_W, 1)).astype(np.float32) for _ in range(6)]

    def plate(self, name):
        if name not in self.plates:
            pick = self.script["plate_picks"][name]
            img = Image.open(self.ep / "plates" / f"{name}_c{pick}.png").convert("RGB")
            self.plates[name] = img
            # figures take the plate's light: its average colour (normalised) and brightness, multiplied onto the rig
            m = np.asarray(img.resize((64, 36))).reshape(-1, 3).mean(axis=0)
            lum = float(m @ [0.299, 0.587, 0.114])
            k = min(1.0, max(0.42, lum / 115))
            self.tints[name] = tuple(int(255 * k * (0.7 + 0.3 * c / m.max())) for c in m)
        return self.plates[name]

    def design(self, char):
        d = rig.extra(int(char.split(":")[1])) if char.startswith("extra:") else CAST[char]["design"]
        return dict(d, body=self.script.get("body", "slim"))

    def font(self, path, size):
        key = (path, size)
        if key not in self.fonts:
            self.fonts[key] = ImageFont.truetype(path, size)
        return self.fonts[key]

    def talk_env(self, sid):
        """Mouth openness per output frame of a spoken shot, from the line's loudness."""
        if sid not in self.envs:
            x, sr = sf.read(self.ep / "audio" / self.timing[sid]["audio"], dtype="float32")
            hop = sr // FPS
            frames = len(x) // hop
            rms = np.sqrt(np.mean(x[:frames * hop].reshape(frames, hop) ** 2, axis=1) + 1e-9)
            self.envs[sid] = np.clip((20 * np.log10(rms) + 38) / 18, 0, 1)
        return self.envs[sid]

    def word_time(self, sid, k):
        return LEAD + self.words[sid][k][1]


def _camera(shot, u):
    (cx0, cy0, z0), (cx1, cy1, z1) = shot["cam"]
    e = _smooth(u) * 0.6 + u * 0.4
    cx, cy, z = cx0 + (cx1 - cx0) * e, cy0 + (cy1 - cy0) * e, z0 + (z1 - z0) * e
    cw = PLATE_W / z
    ch = cw * OUT_H / OUT_W
    left = min(max(cx * PLATE_W - cw / 2, 0), PLATE_W - cw)
    top = min(max(cy * PLATE_H - ch / 2, 0), PLATE_H - ch)
    return left, top, cw, ch


def _draw_clock(frame, ctx, prop, t, dur, to_screen, s):
    v0, v1 = prop["spin"]
    ang = v0 * t + (v1 - v0) * t * t / (2 * dur)
    cx, cy = to_screen(prop["cx"], prop["cy"])
    r = prop["r"] * PLATE_H * s
    size = int(r * 2.4)
    k = 2
    layer = Image.new("RGBA", (size * k, size * k), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    c = size * k / 2
    for a, length, width, col in ((300 + ang / 12, 0.55, 0.075, (28, 22, 16, 255)), (60 + ang, 0.86, 0.05, (28, 22, 16, 255)),
                                  (ang * 6, 0.92, 0.016, (150, 32, 28, 255))):
        rad = math.radians(a)
        dx, dy = math.sin(rad), -math.cos(rad)
        px, py = -dy, dx
        L, W = length * r * k, width * r * k / 2
        tail = 0.15 * r * k
        pts = [(c - dx * tail + px * W, c - dy * tail + py * W), (c + dx * L, c + dy * L), (c - dx * tail - px * W, c - dy * tail - py * W)]
        d.polygon([(x + 4 * k, y + 5 * k) for x, y in pts], fill=(0, 0, 0, 90))
        d.polygon(pts, fill=col)
    d.ellipse((c - 0.07 * r * k, c - 0.07 * r * k, c + 0.07 * r * k, c + 0.07 * r * k), fill=(160, 128, 66, 255), outline=(40, 30, 20, 255), width=3)
    layer = layer.resize((size, size), Image.LANCZOS)
    frame.paste(layer, (int(cx - size / 2), int(cy - size / 2)), layer)


def _phone_pos(prop, u, hands, to_screen):
    keys = prop["keys"]

    def where(key):
        target = key[1]
        if isinstance(target, str):
            return hands[target.split(":")[1]]
        return to_screen(*target)

    if len(keys) == 1 or u <= keys[0][0]:
        return where(keys[0]), 0.0
    for a, b in zip(keys, keys[1:]):
        if a[0] <= u < b[0]:
            f = (u - a[0]) / (b[0] - a[0])
            ease = b[2] if len(b) > 2 else "lin"
            f = f * f if ease == "in" else 1 - (1 - f) ** 2 if ease == "out" else f
            p0, p1 = where(a), where(b)
            spin = 300 * f if ease == "in" else 300 * (1 - f) if ease == "out" else 0.0
            return (p0[0] + (p1[0] - p0[0]) * f, p0[1] + (p1[1] - p0[1]) * f), spin
    return where(keys[-1]), 0.0


def _draw_phone(frame, pos, angle, size):
    w, h = int(size * 0.5) * 2, int(size) * 2
    ph = Image.new("RGBA", (w + 8, h + 8), (0, 0, 0, 0))
    d = ImageDraw.Draw(ph)
    d.rounded_rectangle((4, 4, w + 4, h + 4), radius=w // 5, fill=(28, 28, 32, 255), outline=(10, 10, 10, 255), width=3)
    d.rounded_rectangle((4 + w * 0.12, 4 + h * 0.08, 4 + w * 0.88, 4 + h * 0.86), radius=w // 10, fill=(70, 110, 150, 255))
    ph = ph.resize((ph.width // 2, ph.height // 2), Image.LANCZOS).rotate(angle, expand=True, resample=Image.BICUBIC)
    frame.paste(ph, (int(pos[0] - ph.width / 2), int(pos[1] - ph.height / 2)), ph)


def _vhs(img, t, ctx):
    a = np.asarray(img).astype(np.float32)
    out = a.copy()
    out[:, 10:, 0] = a[:, :-10, 0]
    out[:, :-10, 2] = a[:, 10:, 2]
    gray = out.mean(axis=2, keepdims=True)
    out = out * 0.55 + gray * 0.45
    out[..., 2] += 22
    out[..., 0] -= 10
    out[::3] *= 0.8
    rng = np.random.default_rng(int(t * FPS) + 7)
    for _ in range(2):
        y0 = int(rng.integers(0, OUT_H - 80))
        out[y0:y0 + 70] = np.roll(out[y0:y0 + 70], int(rng.integers(15, 60)), axis=1) + 30
    out += rng.normal(0, 12, (OUT_H, OUT_W, 1)).astype(np.float32)
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    if int(t * 4) % 2 == 0:
        d.text((70, 60), "<< REWIND", font=ctx.font(FONT_HUD, 64), fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
    return img


def _fit(ctx, text, base):
    """Largest font size (from base down) at which the text wraps into at most three lines inside the frame."""
    key = (text, base)
    if key not in ctx.fits:
        for size in range(base, 40, -8):
            font = ctx.font(FONT_BOLD, size)
            lines, cur = [], ""
            for w in text.split():
                trial = f"{cur} {w}".strip()
                if cur and font.getlength(trial) > OUT_W * 0.84:
                    lines.append(cur)
                    cur = w
                else:
                    cur = trial
            lines.append(cur)
            if len(lines) <= 3 and max(font.getlength(x) for x in lines) <= OUT_W * 0.84:
                break
        ctx.fits[key] = (size, "\n".join(lines))
    return ctx.fits[key]


def _pop(frame, ctx, text, age, end_card=False):
    size, text = _fit(ctx, text, 110 if end_card else 150)
    size = int(size * (1 + 0.35 * max(0.0, 1 - age / 0.12)))
    font = ctx.font(FONT_BOLD, size)
    d = ImageDraw.Draw(frame)
    alpha = min(1.0, age / 0.6) if end_card else 1.0
    col = tuple(int(255 * alpha) for _ in range(3))
    jitter = 0 if end_card or age > 0.15 else int(6 * math.sin(age * 90))
    d.multiline_text((OUT_W / 2 + jitter, OUT_H / 2), text, font=font, anchor="mm", align="center", spacing=int(size * 0.15),
                     fill=col, stroke_width=10, stroke_fill=(0, 0, 0))


def _captions(frame, ctx, sid, t):
    words = ctx.words.get(sid)
    if not words:
        return
    lt = t - LEAD
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= 5 or w[0][-1] in ".?!,":
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
    active = None
    for i, ch in enumerate(chunks):
        end = chunks[i + 1][0][1] if i + 1 < len(chunks) else ch[-1][2] + 0.4
        if ch[0][1] - 0.05 <= lt < end:
            active = ch
    if not active:
        return
    font = ctx.font(FONT_BOLD, 60)
    d = ImageDraw.Draw(frame)
    space = font.getlength(" ")
    widths = [font.getlength(w[0]) for w in active]
    x = OUT_W / 2 - (sum(widths) + space * (len(active) - 1)) / 2
    y = OUT_H * 0.875
    for w, wd in zip(active, widths):
        on = w[1] <= lt + 0.02
        d.text((x, y), w[0], font=font, anchor="lm", fill=(255, 214, 64) if on and lt < w[2] + 0.1 else (255, 255, 255),
               stroke_width=6, stroke_fill=(0, 0, 0))
        x += wd + space


def compose(ctx, sid, t, finish=True):
    shot = ctx.shots[sid]
    dur = ctx.timing[sid]["dur"]
    t = max(0.0, t)   # a shot's first frame can sit a fraction of a frame before its start after rounding to frames
    u = t / dur
    if "rewind" in shot:  # replay the listed shots backwards, fast, through a VHS look
        srcs = shot["rewind"]
        total = sum(ctx.timing[s]["dur"] for s in srcs)
        back = total * (1 - u) - 1e-3
        for s in srcs:
            d = ctx.timing[s]["dur"]
            if back < d:
                return _vhs(compose(ctx, s, max(0.0, back), finish=False), t, ctx)
            back -= d
    if shot["plate"] is None:
        frame = Image.new("RGB", (OUT_W, OUT_H), (8, 8, 10))
        to_screen, s = None, 1.0
    else:
        left, top, cw, ch = _camera(shot, u)
        frame = ctx.plate(shot["plate"]).transform((OUT_W, OUT_H), Image.EXTENT, (left, top, left + cw, top + ch), Image.BILINEAR)
        s = OUT_W / cw

        def to_screen(px, py):
            return (px * PLATE_W - left) * s, (py * PLATE_H - top) * s
    if shot.get("dim"):
        frame = Image.eval(frame, lambda v: int(v * (1 - shot["dim"])))
    if shot.get("graphic"):
        frame = overlays.graphic_backdrop(frame)
    hands, held, badges = {}, [], []
    tint = ctx.tints.get(shot["plate"], (255, 255, 255))
    for k, actor in enumerate(sorted(shot.get("actors", []), key=lambda a: a["y"])):
        char = actor["char"]
        design = ctx.design(char)
        st = rig.actor_state(actor, t, dur, seed=k + sid)
        talk = st["yap"]
        if shot.get("who") == char:
            env = ctx.talk_env(sid)
            talk = float(env[min(len(env) - 1, max(0, int((t - LEAD) * FPS)))]) if t >= LEAD else 0.0
        h_px = st["h"] * PLATE_H * s
        img, (ax, ay), hp = rig.draw(design, st["pose"], st["view"], st["facing"], st["expr"], talk, st["blink"], st["look"],
                                     h_px, st["sweat"])
        img = ImageChops.multiply(img, Image.new("RGBA", img.size, tint + (255,)))
        if st["pose"].get("lie"):  # collapsed: the standing figure turned flat, head to the left, lying on the floor
            w0 = img.width
            img = img.rotate(90, expand=True, resample=Image.BICUBIC)
            hp = {k: (y, w0 - x) for k, (x, y) in hp.items()}
            ax, ay = img.width / 2, img.height - h_px * 0.02
        sx, sy = to_screen(st["x"], st["y"])
        ox, oy = int(sx - ax), int(sy - ay)
        if "fade" in actor:
            f = _smooth((u - actor["fade"][0]) / (actor["fade"][1] - actor["fade"][0]))
            img.putalpha(img.getchannel("A").point(lambda v: int(v * f)))
        if "clip_y" in actor:
            cut = int(to_screen(0, actor["clip_y"])[1] - oy)
            img = img.crop((0, 0, img.width, max(1, cut)))
        elif sy < OUT_H + 20:
            wsh = int(h_px * 0.32)
            if wsh not in ctx.shadows:
                ctx.shadows[wsh] = rig.contact_shadow(wsh)
            sh = ctx.shadows[wsh]
            frame.paste((0, 0, 0), (int(sx - sh.width / 2), int(sy - sh.height / 2)), sh)
        frame.paste(img, (ox, oy), img)
        hands[char] = (ox + hp["r"][0], oy + hp["r"][1])
        if actor.get("over"):
            badges.append((actor["over"], (ox + hp["head"][0], oy + hp["head"][1]), rig.HEAD_R * h_px))
        if st["pose"].get("_phone"):
            held.append((hands[char], h_px))
    if not any(p["prop"] == "phone" for p in shot.get("props", [])):
        for pos, h_px in held:
            _draw_phone(frame, pos, 0.0, 0.065 * h_px)
    for prop in shot.get("props", []):
        if prop["prop"] == "clock":
            _draw_clock(frame, ctx, prop, t, dur, to_screen, s)
        elif prop["prop"] == "phone":
            pos, ang = _phone_pos(prop, u, hands, to_screen)
            _draw_phone(frame, pos, ang, 0.04 * PLATE_H * s)
        elif prop["prop"] == "car":
            overlays.car(frame, prop, u, to_screen(0, 0.95)[1], 0.6 * PLATE_H * s)
    for fx in shot.get("fx", []):
        if fx["from"] <= u < fx["to"]:
            frame = _vhs(frame, t, ctx)
    if not finish:
        return frame
    a = np.asarray(frame).astype(np.float32) * ctx.vignette + ctx.grain[int(t * FPS) % len(ctx.grain)]
    frame = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    for spec, head, head_r in badges:
        overlays.badge(frame, spec, head, head_r, u, t, t if spec.get("new") else 9.0)
    if shot.get("graphic"):
        overlays.graphic(frame, shot["graphic"], u, t)
    if shot.get("hud"):
        overlays.hud(frame, shot["hud"], u, t)
    if shot.get("note"):
        overlays.note(frame, shot["note"], t)
    if shot.get("burst") and u >= shot["burst"][0]:
        overlays.burst(frame, shot["burst"][1], t - shot["burst"][0] * dur)
    pops = shot.get("pops")
    if pops:
        times = [ctx.word_time(sid, w) if isinstance(w, int) else w * dur for w, _ in pops]
        cur = [i for i, pt in enumerate(times) if pt <= t]
        if cur:
            _pop(frame, ctx, pops[cur[-1]][1], t - times[cur[-1]], end_card=shot["plate"] is None)
    elif "text" in shot:
        _captions(frame, ctx, sid, t)
    return frame


def _sheet(tiles, cols, tw, th, title, out):
    font = ImageFont.load_default(size=22)
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, 50 + rows * (th + 30)), "white")
    d = ImageDraw.Draw(sheet)
    d.text((10, 10), title, fill="black", font=ImageFont.load_default(size=30))
    for i, (label, img) in enumerate(tiles):
        x, y = (i % cols) * tw, 50 + (i // cols) * (th + 30)
        d.text((x + 6, y + 4), label, fill="black", font=font)
        img = img.copy()
        img.thumbnail((tw - 6, th - 6))
        sheet.paste(img, (x + 3, y + 28))
    sheet.save(out, quality=85)


def stage_qa(ep, script):
    """Contact sheets for a quick look before rendering: every plate candidate, and one frame from every shot."""
    qa = ep / "qa"
    qa.mkdir(exist_ok=True)
    plates = [(p.stem, Image.open(p)) for p in sorted((ep / "plates").glob("*.png"))]
    _sheet(plates, 4, 480, 272, "plates (pick= in the screenplay chooses the candidate)", qa / "plates.jpg")
    ctx = Ctx(ep)
    tiles = [(f"{s['id']} {s.get('who', '')}", compose(ctx, s["id"], 0.6 * ctx.timing[s["id"]]["dur"])) for s in script["shots"]]
    for k in range(0, len(tiles), 24):
        _sheet(tiles[k:k + 24], 4, 480, 272, f"frames {k + 1}-{k + len(tiles[k:k + 24])}", qa / f"frames_{k // 24 + 1}.jpg")
    print("QA sheets in", qa)
    # the opening has to show the premise, not just talk about it
    early = [sh for sh in script["shots"] if ctx.timing[sh["id"]]["start"] < 10]
    visual = any(sh.get(k) for sh in early for k in ("pops", "burst", "graphic", "props", "fx", "note")) or \
        any(a.get("over") or any(x.get("do") or x.get("expr") in ("shocked", "scared") for x in a["acts"])
            for sh in early for a in sh.get("actors", []))
    if not visual:
        print("WARNING hook: nothing visual happens in the first 10 s (add a badge, burst, graphic, move or shock)")
    if not script.get("thumbs"):
        print("WARNING packaging: no THUMB: lines in the screenplay")
    for th in script.get("thumbs", []):
        if len(th["text"].split()) > 4:
            print(f"WARNING packaging: thumbnail text {th['text']!r} is longer than 4 words")
    if len(script["title"]) > 60:
        print(f"WARNING packaging: title is {len(script['title'])} characters (keep it under 60)")


THUMB_W, THUMB_H = 1280, 720


def _thumb(ctx, spec):
    """One thumbnail: punchy plate, a huge character face with a white rim, the badge, and 1-3 giant words."""
    plate = ctx.plate(spec["plate"])
    cw = PLATE_W / 1.12
    ch = cw * THUMB_H / THUMB_W
    left, top = (PLATE_W - cw) / 2, (PLATE_H - ch) * 0.4
    frame = plate.transform((THUMB_W, THUMB_H), Image.EXTENT, (left, top, left + cw, top + ch), Image.BICUBIC)
    frame = ImageEnhance.Contrast(ImageEnhance.Color(frame).enhance(1.4)).enhance(1.15)
    shade = Image.linear_gradient("L").rotate(90).resize((THUMB_W, THUMB_H))       # darker on the text side
    frame = Image.composite(Image.eval(frame, lambda v: int(v * 0.45)), frame, shade.point(lambda v: int(v * 0.9)))
    h_px = THUMB_H * 2.5
    img, _, hp = rig.draw(ctx.design(spec["char"]), dict(rig.POSES[spec["pose"]]), "front", "right", spec["expr"], 0.0, False, -0.3, h_px)
    hx, hy = THUMB_W * 0.72, THUMB_H * 0.56
    head_r = rig.HEAD_R * h_px
    glow_col = overlays.battery_color(spec["over"]["to"]) if spec.get("over", {}).get("kind") == "battery" else overlays.YELLOW
    glow = Image.new("L", (THUMB_W, THUMB_H), 0)
    ImageDraw.Draw(glow).ellipse((hx - head_r * 2.6, hy - head_r * 2.4, hx + head_r * 2.6, hy + head_r * 2.4), fill=150)
    frame.paste(glow_col, (0, 0), glow.filter(ImageFilter.GaussianBlur(head_r)))
    ox, oy = int(hx - hp["head"][0]), int(hy - hp["head"][1])
    rim = img.getchannel("A").filter(ImageFilter.MaxFilter(19)).filter(ImageFilter.GaussianBlur(4))
    frame.paste((255, 255, 255), (ox, oy), rim)
    frame.paste(img, (ox, oy), img)
    if spec.get("over"):
        overlays.badge(frame, dict(spec["over"]), (hx, hy), head_r * 0.85, 1.0, 0.0, 9.0)
    words = spec["text"].split()
    if len(words) > 2:  # two lines, split where the longer line is as short as possible
        k = min(range(1, len(words)), key=lambda i: max(len(" ".join(words[:i])), len(" ".join(words[i:]))))
        lines = [" ".join(words[:k]), " ".join(words[k:])]
    else:
        lines = [spec["text"]]
    size = 190
    while size > 60 and max(overlays.font(size).getlength(l) for l in lines) > THUMB_W * 0.52:
        size -= 8
    layer = Image.new("RGBA", (THUMB_W, THUMB_H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    y = THUMB_H * 0.5 - (len(lines) - 1) * size * 0.55
    for i, l in enumerate(lines):
        d.text((THUMB_W * 0.29, y), l, font=overlays.font(size), anchor="mm", fill=overlays.YELLOW if i == len(lines) - 1 else (255, 255, 255),
               stroke_width=max(8, size // 14), stroke_fill=(10, 10, 12))
        y += size * 1.1
    layer = layer.rotate(4, resample=Image.BICUBIC, center=(THUMB_W * 0.29, THUMB_H * 0.5))
    frame.paste(layer, (0, 0), layer)
    return frame


def stage_thumbs(ep, script):
    """Every THUMB: line becomes a thumbnail; the sheet shows them at phone size, where they have to read in a second."""
    if not script.get("thumbs"):
        print("THUMBS: none defined (add THUMB: lines to the screenplay)")
        return
    ctx = Ctx(ep)
    (ep / "final").mkdir(exist_ok=True)
    tiles = []
    for i, spec in enumerate(script["thumbs"], 1):
        img = _thumb(ctx, spec)
        img.save(ep / "final" / f"thumb_{i}.jpg", quality=92)
        tiles += [(f"thumb_{i}", img), (f"thumb_{i} at phone size", img.resize((246, 138), Image.LANCZOS).resize((480, 270), Image.NEAREST))]
    _sheet(tiles, 2, 640, 370, "thumbnails (left: full size, right: as seen on a phone)", ep / "qa" / "thumbs.jpg")
    print(f"THUMBS {len(script['thumbs'])} -> final/thumb_N.jpg")


def _frame_range(ctx, sid):
    start = ctx.timing[sid]["start"]
    return round(start * FPS), round((start + ctx.timing[sid]["dur"]) * FPS)


_CTX = None


def _shot_key(ctx, sid):
    """Fingerprint of everything a shot's pixels depend on, so unchanged shots are never rendered twice."""
    shot = ctx.shots[sid]
    deps = [sid] + shot.get("rewind", [])
    data = [[{k: v for k, v in ctx.shots[d].items() if k != "chapter"}, ctx.timing[d], ctx.words.get(d)] for d in deps]
    data.append({p: ctx.script["plate_picks"].get(p) for p in {ctx.shots[d].get("plate") for d in deps} if p})
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _render_shot(args):
    global _CTX
    ep, sid = args
    if _CTX is None:
        _CTX = Ctx(ep)
    ctx = _CTX
    f0, f1 = _frame_range(ctx, sid)
    start = ctx.timing[sid]["start"]
    out = ep / "segments" / f"shot_{sid:02d}.mp4"
    key, stamp = _shot_key(ctx, sid), out.with_suffix(".key")
    if out.exists() and stamp.exists() and stamp.read_text() == key:
        return sid, 0, 0.0
    t0 = time.time()
    enc = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OUT_W}x{OUT_H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p",
                            "-threads", "2", str(out)], stdin=subprocess.PIPE)
    for f in range(f0, f1):
        enc.stdin.write(compose(ctx, sid, f / FPS - start).tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on shot {sid}")
    stamp.write_text(key)
    return sid, f1 - f0, time.time() - t0


def stage_render(ep, script, only=None):
    (ep / "segments").mkdir(exist_ok=True)
    ids = [s["id"] for s in script["shots"] if only is None or s["id"] in only]
    t0 = time.time()
    with mp.Pool(WORKERS) as pool:
        for sid, n, secs in pool.imap_unordered(_render_shot, [(ep, i) for i in ids]):
            if n:
                print(f"shot {sid}: {n} frames in {secs:.1f}s ({n / secs:.1f} fps)", flush=True)
    lst = ep / "segments" / "list.txt"
    lst.write_text("".join(f"file 'shot_{s['id']:02d}.mp4'\n" for s in script["shots"]))
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy",
                    str(ep / "video.mp4")], check=True)
    print(f"RENDER {len(ids)} shots in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- sound

def _place(track, x, at, gain_db=0.0):
    i = int(at * sfx.SR)
    if i >= len(track):
        return
    x = x[:len(track) - i] * 10 ** (gain_db / 20)
    track[i:i + len(x)] += x


def stage_mix(ep, script):
    timing = {int(k): v for k, v in json.loads((ep / "timing.json").read_text()).items()}
    by_audio = json.loads((ep / "words.json").read_text())
    words = {sid: by_audio[t["audio"]] for sid, t in timing.items() if "audio" in t}
    total = max(v["start"] + v["dur"] for v in timing.values())
    n = int(total * sfx.SR) + sfx.SR
    voice, fx, amb, music = (np.zeros(n, np.float32) for _ in range(4))
    rewinds = []
    lib = {}
    for shot in script["shots"]:
        sid = shot["id"]
        start, dur = timing[sid]["start"], timing[sid]["dur"]
        if "text" in shot:
            x, sr = sf.read(ep / "audio" / timing[sid]["audio"], dtype="float32")
            _place(voice, _resample(x, sr, sfx.SR), start + LEAD)
        for when, name, gain in shot.get("sfx", []):
            at = start + (LEAD + words[sid][int(when[1:])][1] if isinstance(when, str) else when * dur)
            if name not in lib:
                lib[name] = sfx.LIB[name]()
            _place(fx, lib[name], at, gain)
        if shot.get("plate"):
            a = sfx.ambience(_amb(script, shot["plate"]), dur + 0.1, seed=sid)
            ramp = np.minimum(1, np.minimum(np.arange(len(a)), np.arange(len(a))[::-1]) / (0.05 * sfx.SR)).astype(np.float32)
            _place(amb, a * ramp, start, -10)
        if "rewind" in shot:
            rewinds.append((start, start + dur))
        for f in shot.get("fx", []):
            rewinds.append((start + f["from"] * dur, start + f["to"] * dur))
    # one music bed per run of shots with the same mood, crossfaded over a second
    runs = []
    for shot in script["shots"]:
        mood = shot.get("mood", "calm")
        if not runs or runs[-1][0] != mood:
            runs.append([mood, timing[shot["id"]]["start"]])
    for i, (mood, start) in enumerate(runs):
        end = runs[i + 1][1] if i + 1 < len(runs) else total
        length = end - start + 1.0
        bed = sfx.drone(length) * 10 ** (8 / 20) if mood == "dark" else sfx.pad(length, *MOODS[mood])
        fade = np.ones(len(bed), np.float32)
        f = int(sfx.SR * 1.0)
        if i:
            fade[:f] = np.linspace(0, 1, f)
        fade[-f:] = np.linspace(1, 0, f)
        _place(music, bed * fade, max(0.0, start - (0.5 if i else 0)), -14)
    # duck the music under the voice (fast attack, slow release) and drop it out during rewinds
    env = np.abs(voice)
    k = int(0.3 * sfx.SR)
    env = np.convolve(env, np.ones(k, np.float32) / k, mode="same")
    duck = 1 - 0.55 * np.clip(env / 0.05, 0, 1)
    for a, b in rewinds:
        duck[int(a * sfx.SR):int(b * sfx.SR)] *= 0.15
    mix = voice + fx * 0.9 + amb + music * duck
    mix = np.tanh(mix * 1.1) / 1.1
    sf.write(ep / "mix.wav", mix[:int(total * sfx.SR)], sfx.SR, subtype="FLOAT")
    print(f"MIX {total:.1f}s")


def stage_final(ep, script):
    (ep / "final").mkdir(exist_ok=True)
    target = "I=-14:TP=-1.5:LRA=11"
    err = subprocess.run([FFMPEG, "-i", str(ep / "mix.wav"), "-af", f"loudnorm={target}:print_format=json", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    m = json.loads(err[err.rindex("{"):err.rindex("}") + 1])
    af = (f"loudnorm={target}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}"
          f":measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    out = ep / "final" / f"{ep.name}.mp4"
    # the segments are a fast high-quality intermediate; this pass makes an upload-sized file (about 10-16 Mbit/s)
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(ep / "video.mp4"), "-i", str(ep / "mix.wav"), "-af", af, "-ar", "48000",
                    "-ac", "2", "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-maxrate", "16M", "-bufsize", "32M",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)], check=True)
    # upload notes: title, a description stub and YouTube chapters (first one must be 0:00)
    timing = {int(k): v for k, v in json.loads((ep / "timing.json").read_text()).items()}
    chapters = [(timing[sh["id"]]["start"], sh["chapter"]) for sh in script["shots"] if sh.get("chapter")]
    if chapters:
        chapters[0] = (0.0, chapters[0][1])
    title = script["title"] + ("" if script["title"][-1:] in "?!." else "?")
    lines = [title, "", f"{title} An animated story with a twist you won't see coming. Which rule would you break? Tell us in the comments.", ""]
    lines += [f"{int(t // 60)}:{int(t % 60):02d} {name}" for t, name in chapters]
    (ep / "final" / f"{ep.name}_upload.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("FINAL", out)


def stage_direct(ep, script):
    import director
    out = director.direct(ep / "screenplay.txt", ep / "script.json", CAST)
    print(f"DIRECT {len(out['shots'])} shots, {len(out['plates'])} plates")


if __name__ == "__main__":
    ep = ROOT / "episodes" / sys.argv[1]
    bench = {}
    for stage in [s for a in sys.argv[2:] for s in (STAGES if a == "all" else [a])]:
        t0 = time.time()
        globals()[f"stage_{stage}"](ep, load(ep) if stage != "direct" else None)
        bench[stage] = round(time.time() - t0, 1)
        print(f"[{stage}] {bench[stage]}s", flush=True)
    (ep / "bench.json").write_text(json.dumps(bench, indent=1))

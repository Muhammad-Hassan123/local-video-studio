"""Build a full stickman episode from episodes/<name>/script.json.

Stages (each one caches its outputs, so a rerun only does missing work):
  tts        Kokoro narration per shot           -> audio/shot_XXX.wav
  keys       FLUX.2 klein keyframes per shot     -> keys/shot_XXX[a|b].png   (ComfyUI must be running)
  clips      Wan 2.2 I2V for shots with "anim"   -> clips/shot_XXX.mp4       (ComfyUI must be running)
  assemble   Ken Burns stills + clips + voice    -> final/<name>.mp4 + .srt

Usage:  venv\\Scripts\\python make_episode.py ep01_moon [stage ...]
Every stage appends its timings to episodes/<name>/benchmark.json.
"""
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import psutil
import soundfile as sf
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
import comfy_client as cc  # noqa: E402

ROOT = Path(__file__).parent
MODELS = ROOT.parent / "models"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
CHARS = json.loads((ROOT / "characters.json").read_text(encoding="utf-8"))

STYLE = ("flat 2D vector cartoon, clean thick black outlines, bright flat colors, no shading, no texture, "
         "simple shapes, explainer video style, no text, no letters, no words")
# only added when the shot has characters or people in it
FIGURE_RULES = ("Correct proportions: every object is drawn at its real-life size relative to the characters, handheld things "
                "like phones, cups, tablets and clipboards are small and fit in one hand. Stick figures always have exactly two "
                "arms and two separate legs drawn as single thick black lines, natural relaxed poses, elbows and knees bend "
                "naturally, two separate small black shoes. Faces stay simple: two dot eyes, a simple mouth, expressions shown "
                "with the mouth and small eyebrows. Any other people are simple stick figures in exactly the same style as the "
                "main characters: round white heads, single thick black line arms and legs, small round white hands, small black "
                "oval shoes, never tube-shaped limbs")
NIGHT_WORDS = ("night", "sky", "space", "dark", "stars")


LOCATION_PROPS = {}  # filled from script["location_props"]: objects that must appear in every shot at that location


def scene_text(shot):
    """Scene description plus the shot-level rules: Moon only where the story has it, the location's fixed objects,
    and always a real background."""
    scene, low, extra = shot["scene"], shot["scene"].lower(), []
    if shot.get("loc") in LOCATION_PROPS:
        extra.append(f"The scene includes {LOCATION_PROPS[shot['loc']]}, drawn exactly as in the location reference")
    if shot.get("moon"):
        extra.append("The Moon is light grey with a few darker grey round craters, exactly like the moon reference design")
    elif any(w in low for w in NIGHT_WORDS):
        extra.append("There is no moon anywhere in the sky")
    if "space" in low:
        extra.append("Only dark space with stars around, no ground, no buildings, no houses")
    elif "plain" not in low and "background" not in low:
        extra.append("Show a complete background that fits the scene, never a blank white background")
    return scene + ". " + ". ".join(extra) if extra else scene
# added everywhere unless the shot sets "faces_ok": the planet, moon, houses, rocks etc. are never characters
OBJECT_RULES = ("Inanimate things like planets, the Moon, the Sun, buildings, houses, rocks, plants, furniture, vehicles and "
                "tools have no faces, no eyes, no arms and no legs")


def style_for(shot=None, people=False):
    parts = [STYLE]
    if shot is not None and (shot["chars"] or "stick figure" in shot["scene"].lower()):
        parts.append(FIGURE_RULES)
    if not people and not (shot or {}).get("faces_ok"):
        parts.append(OBJECT_RULES)
    return ". ".join(parts)


KEY_W, KEY_H = 1920, 1088          # keyframe size (16:9, multiple of 16)
OUT_W, OUT_H, FPS = 1920, 1080, 24
WAN_W, WAN_H = 848, 480
WAN_MIN, WAN_MAX = 33, 81          # Wan clip length in frames at 16 fps (4n+1): 2 s .. 5 s
SAMPLE_RATE = 24000
PAUSE = 0.15                       # silence after every shot
SECTION_PAUSE = 0.5                # extra silence before a new section (shot ids in "sections")
REFRAME_EVERY = 4.0                # still shots change framing (wide / tight / wide) about this often, in seconds
MAX_SPEEDUP = 1.3                  # animated clips may play up to this much faster so the motion finishes in time
PRECUT = 2                         # the picture cuts this many frames before the voice line starts
COMFY_PY = cc.COMFY_DIR / "venv" / "Scripts" / "python.exe"


# ---------------------------------------------------------------- helpers

class Bench:
    def __init__(self, ep_dir):
        self.path = ep_dir / "benchmark.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def save(self, stage, info):
        # re-read first: another stage may be running in a second process and have written its own entry since
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.data[stage] = info
        self.path.write_text(json.dumps(self.data, indent=1))


class PeakSampler(threading.Thread):
    """Samples system RAM in use and GPU VRAM in use once a second."""
    def __init__(self):
        super().__init__(daemon=True)
        self.halt = threading.Event()
        self.ram = self.vram = 0.0

    def run(self):
        while not self.halt.is_set():
            vm = psutil.virtual_memory()
            self.ram = max(self.ram, (vm.total - vm.available) / 1e9)
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                self.vram = max(self.vram, float(out.strip()) / 1024)
            except Exception:
                pass
            time.sleep(1)


def sid(shot):
    return f"shot_{shot['id']:03d}"


MAX_REFS = 3


def plan_refs(shot, loc_ref, prop_refs):
    """Reference images for a shot (characters > location > props, capped at MAX_REFS) and the prompt
    prefix that tells the model what each reference is for."""
    refs, lines = [], []
    for c in shot["chars"]:
        refs.append(CHARS[c]["input"])
        lines.append(f"{CHARS[c]['name']} is the stick figure in reference image {len(refs)} ({CHARS[c]['traits']}), "
                     f"keep {CHARS[c]['name']} exactly like that reference")
    if loc_ref and len(refs) < MAX_REFS:
        refs.append(loc_ref)
        lines.append(f"reference image {len(refs)} shows the same location: keep the same place, background, "
                     f"scenery, colors and objects, but use it only for the setting")
    for p in shot.get("props", []):
        if len(refs) >= MAX_REFS:
            break
        refs.append(prop_refs[p])
        lines.append(f"the {p} must look exactly like the {p} in reference image {len(refs)}")
    prefix = ("; ".join(lines) + ". ") if lines else ""
    return refs, prefix


def timeline(script, durs):
    """(start, end) seconds of every shot. Audio starts at `start`; the picture runs until the next shot
    starts, so pauses (including section pauses) belong to the shot before them and nothing drifts."""
    sections = set(script.get("sections", []))
    starts, t = [], 0.0
    for shot in script["shots"]:
        if shot["id"] in sections and starts:
            t += SECTION_PAUSE
        starts.append(t)
        t += durs[sid(shot)] + PAUSE
    ends = starts[1:] + [t]
    return {sid(s): (a, b) for s, a, b in zip(script["shots"], starts, ends)}, t


def cut_frames(script, durs):
    """Frame count of every shot's picture. Cuts sit PRECUT frames before each voice line, computed from absolute
    times (not summed durations), so a 10 minute video can't drift."""
    tl, total = timeline(script, durs)
    cuts = [0] + [round(tl[sid(s)][0] * FPS) - PRECUT for s in script["shots"][1:]] + [round(total * FPS)]
    return {sid(s): cuts[i + 1] - cuts[i] for i, s in enumerate(script["shots"])}


# ---------------------------------------------------------------- stages

def stage_tts(ep, script, bench):
    from kokoro_onnx import Kokoro
    adir = ep / "audio"
    adir.mkdir(exist_ok=True)
    k = Kokoro(str(MODELS / "kokoro-v1.0.onnx"), str(MODELS / "voices-v1.0.bin"))
    t0, made = time.time(), 0
    durs = {}
    for shot in script["shots"]:
        wav = adir / f"{sid(shot)}.wav"
        if not wav.exists():
            samples, sr = k.create(shot["text"], voice=script["voice"], speed=script.get("speed", 1.0), lang="en-us")
            assert sr == SAMPLE_RATE, sr
            sf.write(wav, samples, sr)
            made += 1
        durs[sid(shot)] = sf.info(wav).duration
    (ep / "durations.json").write_text(json.dumps(durs, indent=1))

    # narration track = shots + pauses, laid out exactly like the video timeline
    tl, total = timeline(script, durs)
    track = np.zeros(int(round(total * SAMPLE_RATE)) + SAMPLE_RATE, dtype=np.float32)
    for shot in script["shots"]:
        s, _ = sf.read(adir / f"{sid(shot)}.wav", dtype="float32")
        i = int(round(tl[sid(shot)][0] * SAMPLE_RATE))
        track[i:i + len(s)] = s
    sf.write(ep / "narration.wav", track[:int(round(total * SAMPLE_RATE))], SAMPLE_RATE)
    words = sum(len(s["text"].split()) for s in script["shots"])
    info = {"seconds": round(time.time() - t0, 1), "shots_made": made, "audio_minutes": round(total / 60, 2),
            "words": words, "wpm": round(words / (sum(durs.values()) / 60))}
    bench.save("tts", info)
    print("TTS", info)


def _make(job, out_dir, stem, per):
    files, secs = cc.run(job, out_dir, stem)
    per.append(secs)
    print(f"{stem}: {secs:.1f}s", flush=True)
    return files[0]


def _stats(t0, per):
    return {"seconds": round(time.time() - t0, 1), "images_made": len(per),
            "avg_s": round(sum(per) / len(per), 1) if per else None,
            "min_s": round(min(per), 1) if per else None, "max_s": round(max(per), 1) if per else None}


def stage_keys(ep, script, bench):
    """Props + empty location plates (made once), then N candidate keyframes per shot (script "cands", default 2)."""
    kdir, pdir = ep / "keys" / "cands", ep / "props"
    kdir.mkdir(parents=True, exist_ok=True)
    pdir.mkdir(exist_ok=True)
    for c in CHARS.values():
        c["input"] = cc.to_input(ROOT / c["ref"])
    t0, per = time.time(), []

    prop_refs = {}  # master designs for recurring props (Earth, Moon, ...)
    for name, desc in script.get("props", {}).items():
        out = pdir / f"{name}.png"
        if not out.exists():
            _make(cc.klein_image(desc + ". " + style_for(), 1024, 1024, seed=77, filename_prefix=f"stickman/{ep.name}/prop"), pdir, name, per)
        prop_refs[name] = cc.to_input(out)

    plate_refs = {}  # empty establishing shot per location, so scenery and objects never drift
    for name, desc in script.get("locations", {}).items():
        out = pdir / f"plate_{name}.png"
        if not out.exists():
            _make(cc.klein_image(desc + ", nobody there, no people, no characters, no animals. " + style_for(), KEY_W, KEY_H, seed=55,
                                 filename_prefix=f"stickman/{ep.name}/plate"), pdir, out.stem, per)
        plate_refs[name] = cc.to_input(out)

    picks_file = ep / "qa" / "picks.json"
    picks = json.loads(picks_file.read_text()) if picks_file.exists() else {}
    for shot in script["shots"]:
        refs, prefix = plan_refs(shot, plate_refs.get(shot.get("loc")), prop_refs)
        if shot.get("ref_shot"):  # continuity with an earlier shot: its picked image is a reference too
            k = f"shot_{shot['ref_shot']:03d}"
            refs = refs + [cc.to_input(kdir / f"{k}_c{picks.get(k, 1)}.png")]
            prefix += (f"Reference image {len(refs)} is the previous shot: keep the same characters, animals, objects "
                       f"and place exactly as shown there. ")
        for n in range(1, shot.get("cands", 2) + 1):
            stem = f"{sid(shot)}_c{n}"
            if (kdir / f"{stem}.png").exists():
                continue
            _make(cc.klein_image(prefix + scene_text(shot) + ". " + style_for(shot), KEY_W, KEY_H, seed=shot["id"] * 1000 + n,
                                 refs=refs, filename_prefix=f"stickman/{ep.name}/key"), kdir, stem, per)
    bench.save("keys", _stats(t0, per))
    print("KEYS", bench.data["keys"])


def stage_finalize(ep, script, bench):
    """Copy the picked candidate (qa/picks.json, default 1) to keys/shot_XXXa.png. Long still shots get their
    second angle at assembly time as a tighter crop of this same image (AI close-ups drifted: lost glasses,
    stray houses), so nothing in the scene can change."""
    kdir = ep / "keys"
    picks_file = ep / "qa" / "picks.json"
    picks = json.loads(picks_file.read_text()) if picks_file.exists() else {}
    for b in kdir.glob("shot_*b.png"):
        b.unlink()
    for shot in script["shots"]:
        out_a = kdir / f"{sid(shot)}a.png"
        src = kdir / "cands" / f"{sid(shot)}_c{picks.get(sid(shot), 1)}.png"
        if not out_a.exists() or out_a.read_bytes() != src.read_bytes():
            out_a.write_bytes(src.read_bytes())
            for seg in (ep / "segments").glob(f"{sid(shot)}*.mp4"):
                seg.unlink()
    print("FINALIZE: picks copied")


def contact_sheets(ep, kind="keys", cols=4, thumb=(480, 272)):
    """Tile keyframes (or clip frames) into labelled grids for visual QA -> qa/<kind>_NN.png."""
    from PIL import ImageDraw
    qdir = ep / "qa"
    qdir.mkdir(exist_ok=True)
    items = []
    rows = 3
    if kind == "keys":  # one row per shot: all candidates side by side
        by_shot = {}
        for p in sorted((ep / "keys" / "cands").glob("*.png")):
            by_shot.setdefault(p.stem.split("_c")[0], []).append(p)
        cols = max(len(v) for v in by_shot.values())
        thumb = (1920 // cols, 1088 // cols)
        for shot, paths in by_shot.items():
            row = [(p.stem, Image.open(p)) for p in paths]
            items.extend(row + [("", Image.new("RGB", (8, 8), "white"))] * (cols - len(row)))
    elif kind == "bkeys":  # long still shots: picked image | its close-up
        cols, thumb = 2, (960, 544)
        for b in sorted((ep / "keys").glob("*b.png")):
            a = b.with_name(b.stem[:-1] + "a.png")
            items.extend([(a.stem, Image.open(a)), (b.stem, Image.open(b))])
    else:  # clips: first / middle / last frame of each clip side by side
        cols, thumb, rows = 3, (640, 362), 4
        for clip in sorted((ep / "clips").glob("*.mp4")):
            for t in ("0", "2.5", "4.9"):
                tmp = qdir / "_frame.png"
                subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-ss", t, "-i", str(clip), "-frames:v", "1", str(tmp)], check=True)
                items.append((f"{clip.stem}@{t}s", Image.open(tmp).copy()))
    for old in qdir.glob(f"{kind}_*.png"):
        old.unlink()
    sheets = []
    for n in range(0, len(items), cols * rows):
        page = Image.new("RGB", (cols * thumb[0], rows * (thumb[1] + 24)), "white")
        d = ImageDraw.Draw(page)
        for i, (label, im) in enumerate(items[n:n + cols * rows]):
            x, y = (i % cols) * thumb[0], (i // cols) * (thumb[1] + 24)
            page.paste(im.convert("RGB").resize(thumb), (x, y + 24))
            d.text((x + 6, y + 4), label, fill="black")
        out = qdir / f"{kind}_{n // (cols * rows) + 1:02d}.png"
        page.save(out)
        sheets.append(out)
    print("QA sheets:", len(sheets))
    return sheets


def wan_length(frames):
    """Wan frames (16 fps, 4n+1) covering a picture of `frames` at FPS, clamped to WAN_MIN..WAN_MAX."""
    need = math.ceil(frames / FPS * 16)
    return min(WAN_MAX, max(WAN_MIN, need + (-need) % 4 + 1))


def stage_clips(ep, script, bench):
    cdir = ep / "clips"
    cdir.mkdir(exist_ok=True)
    frames = cut_frames(script, json.loads((ep / "durations.json").read_text()))
    t0, per, wan_frames = time.time(), [], 0
    sampler = PeakSampler()
    sampler.start()
    for shot in script["shots"]:
        if not shot.get("anim"):
            continue
        out = cdir / f"{sid(shot)}.mp4"
        if out.exists():
            continue
        length = wan_length(frames[sid(shot)])
        start = cc.to_input(ep / "keys" / f"{sid(shot)}a.png")
        motion = shot["anim"] + ". Flat 2D cartoon animation, clean thick outlines, static camera."
        files, secs = cc.run(cc.wan_i2v(start, motion, WAN_W, WAN_H, length=length, seed=shot["id"] + 100000 * shot.get("clip_reroll", 0),
                                        filename_prefix=f"stickman/{ep.name}/clip"), cdir, sid(shot))
        per.append(secs)
        wan_frames += length
        print(f"clip {sid(shot)}: {length} frames in {secs:.1f}s (peak RAM {sampler.ram:.1f} GB, VRAM {sampler.vram:.1f} GB)", flush=True)
        bench.save("clips", {"seconds": round(time.time() - t0, 1), "clips_made": len(per), "wan_frames": wan_frames,
                             "avg_s": round(sum(per) / len(per), 1), "min_s": round(min(per), 1), "max_s": round(max(per), 1),
                             "s_per_wan_frame": round(sum(per) / wan_frames, 2),
                             "peak_ram_gb": round(sampler.ram, 1), "peak_vram_gb": round(sampler.vram, 1),
                             "settings": f"Wan2.2 I2V Q4_K_S + lightx2v 4-step + SageAttention, {WAN_W}x{WAN_H} @16fps, RIFE x3"})
    sampler.halt.set()
    print("CLIPS", bench.data.get("clips"))


def stage_upscale(ep, script, bench):
    """48 fps 480p clips -> 24 fps 1080p (Real-ESRGAN animevideov3), streamed in ComfyUI's venv."""
    cc.free()
    t0 = time.time()
    subprocess.run([str(COMFY_PY), str(ROOT.parent / "shared" / "upscale_clips.py"), str(ep / "clips"), str(ep / "clips_hd"), FFMPEG], check=True)
    bench.save("upscale", {"seconds": round(time.time() - t0, 1)})


def _ease(x):
    return x * x * (3 - 2 * x)


def _kenburns_box(cam, p, w, h):
    """Crop box (l, t, r, b) on a w x h source at progress p in [0, 1]."""
    p = _ease(p)
    if cam == "in":
        z = 1.0 + 0.20 * p
    elif cam == "out":
        z = 1.20 - 0.20 * p
    elif cam == "tight":  # second angle of a long still: a closer crop of the same image
        z = 1.32 + 0.10 * p
    else:
        z = 1.18
    cw, ch = w / z, h / z
    cx, cy = w / 2, h / 2
    slack_x, slack_y = (w - cw) / 2, (h - ch) / 2
    if cam == "left":
        cx = w / 2 + slack_x * (1 - 2 * p)
    elif cam == "right":
        cx = w / 2 - slack_x * (1 - 2 * p)
    elif cam == "up":
        cy = h / 2 + slack_y * (1 - 2 * p)
    elif cam == "down":
        cy = h / 2 - slack_y * (1 - 2 * p)
    return (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)


def _encode_args(out):
    return ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(FPS), str(out)]


def render_still(src, cam, frames, out, post=None):
    img = Image.open(src).convert("RGB")
    if post == "dark":  # a moonless night: same picture at a fraction of the light, slightly blue
        from PIL import ImageEnhance
        img = Image.blend(ImageEnhance.Brightness(img).enhance(0.22), Image.new("RGB", img.size, (8, 12, 40)), 0.25)
    w, h = img.size
    proc = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{OUT_W}x{OUT_H}", "-r", str(FPS), "-i", "-", *_encode_args(out)], stdin=subprocess.PIPE)
    for f in range(frames):
        box = _kenburns_box(cam, f / max(frames - 1, 1), w, h)
        proc.stdin.write(img.transform((OUT_W, OUT_H), Image.Transform.EXTENT, box, Image.Resampling.BICUBIC).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out}")


def render_clip(src, frames, out):
    """Play a 1080p 48 fps clip at FPS. If the clip is longer than the voice line it plays up to MAX_SPEEDUP faster so
    the motion still finishes (48 fps source keeps that smooth); if it is shorter, keep going with a slow push-in on
    its last frame instead of freezing or slowing the motion down."""
    have = _stream_end(src, "v")[0]
    step = 2 * min(max((have - 1) / 2 / max(frames - 1, 1), 1.0), MAX_SPEEDUP)  # source frames per output frame
    # one encode for clip + push-in, so there is no seam where the two meet
    dec = subprocess.Popen([FFMPEG, "-loglevel", "error", "-i", str(src), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{OUT_W}x{OUT_H}", "-r", str(FPS), "-i", "-", *_encode_args(out)], stdin=subprocess.PIPE)
    size, n, idx, last = OUT_W * OUT_H * 3, 0, 0, None
    while n < frames:
        raw = dec.stdout.read(size)
        if len(raw) < size:
            break
        if idx == round(n * step):
            enc.stdin.write(raw)
            n += 1
        last, idx = raw, idx + 1
    dec.kill()
    img, rest = Image.frombytes("RGB", (OUT_W, OUT_H), last), frames - n
    for f in range(rest):
        box = _kenburns_box("in", (f + 1) / rest, OUT_W, OUT_H)
        enc.stdin.write(img.transform((OUT_W, OUT_H), Image.Transform.EXTENT, box, Image.Resampling.BICUBIC).tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out}")


def mux_audio(ep, video, final):
    """Add the narration at YouTube's -14 LUFS. YouTube turns loud videos down but never quiet ones up, and one-pass
    loudnorm undershoots speech with pauses (measured -15.4), so measure first and apply linearly in a second pass."""
    target = "I=-14:TP=-1.5:LRA=11"
    err = subprocess.run([FFMPEG, "-i", str(ep / "narration.wav"), "-af", f"loudnorm={target}:print_format=json", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    m = json.loads(err[err.rindex("{"):err.rindex("}") + 1])
    af = (f"loudnorm={target}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}"
          f":measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(video), "-i", str(ep / "narration.wav"), "-af", af, "-ar", "48000",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(final)], check=True)


def stage_assemble(ep, script, bench):
    sdir = ep / "segments"
    sdir.mkdir(exist_ok=True)
    fdir = ep / "final"
    fdir.mkdir(exist_ok=True)
    durs = json.loads((ep / "durations.json").read_text())
    tl, total = timeline(script, durs)
    shot_frames = cut_frames(script, durs)
    t0 = time.time()
    parts = []
    for shot in script["shots"]:
        frames = shot_frames[sid(shot)]
        clip = ep / "clips_hd" / f"{sid(shot)}.mp4"
        if shot.get("anim") and clip.exists():
            out = sdir / f"{sid(shot)}.mp4"
            if not out.exists():
                render_clip(clip, frames, out)
            parts.append(out)
            continue
        key = ep / "keys" / f"shot_{shot.get('use_key', shot['id']):03d}a.png"
        n = max(1, round(frames / FPS / REFRAME_EVERY))
        cams = ([shot["cam"], "tight"] * n)[:n]  # wide / tight / wide ... cut on the same picture
        for i, cam in enumerate(cams):
            fr = frames // n + (1 if i < frames % n else 0)
            out = sdir / f"{sid(shot)}{'abcdef'[i]}.mp4"
            if not out.exists():
                render_still(key, cam, fr, out, shot.get("post"))
            parts.append(out)
        print(f"segment {sid(shot)} ({frames} frames)", flush=True)
    t_seg = time.time() - t0

    concat = sdir / "concat.txt"
    concat.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts))
    video = fdir / "video_only.mp4"
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(video)], check=True)
    mux_audio(ep, video, fdir / f"{ep.name}.mp4")

    def ts(x):
        ms = int(round(x * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    srt = []
    for i, shot in enumerate(script["shots"], 1):
        s, _ = tl[sid(shot)]
        srt.append(f"{i}\n{ts(s)} --> {ts(s + durs[sid(shot)])}\n{shot['text']}\n")
    (fdir / f"{ep.name}.srt").write_text("\n".join(srt), encoding="utf-8")

    info = {"seconds": round(time.time() - t0, 1), "segments_seconds": round(t_seg, 1),
            "video_minutes": round(total / 60, 2), "segments": len(parts), "output": str(final)}
    bench.save("assemble", info)
    print("ASSEMBLE", info)


def _stream_end(path, stream):
    """(frames, seconds) that ffmpeg decodes from one stream of a file."""
    err = subprocess.run([FFMPEG, "-i", str(path), "-map", f"0:{stream}", "-f", "null", "-"], capture_output=True, text=True).stderr
    last = err.rsplit("frame=" if stream == "v" else "size=", 1)[1]
    h, m, s = last.split("time=", 1)[1].split()[0].split(":")
    frames = int(last.split()[0]) if stream == "v" else None
    return frames, int(h) * 3600 + int(m) * 60 + float(s)


def stage_sync_check(ep, script, bench):
    """Listen to the finished video with faster-whisper and check every voice line starts where its cut is."""
    from faster_whisper import WhisperModel
    final = ep / "final" / f"{ep.name}.mp4"
    durs = json.loads((ep / "durations.json").read_text())
    tl, total = timeline(script, durs)
    shot_frames = cut_frames(script, durs)
    v_frames, v_secs = _stream_end(final, "v")
    _, a_secs = _stream_end(final, "a")

    t0 = time.time()
    model = WhisperModel("small.en", device="cpu", compute_type="int8", download_root=str(MODELS / "whisper"))
    # decode with our ffmpeg: faster-whisper's own loader breaks on the installed PyAV (unknown `metadata_errors` arg)
    pcm = subprocess.run([FFMPEG, "-loglevel", "error", "-i", str(final), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    segments, _ = model.transcribe(np.frombuffer(pcm, np.float32), word_timestamps=True, language="en")
    words = [("".join(c for c in w.word.lower() if c.isalnum()), w.start) for s in segments for w in s.words]

    # Whisper's word times are only good to ~0.5 s (first words after short pauses come out early), so it only checks
    # that each line is heard. Placement is measured exactly by finding each line's own waveform in the final audio.
    mix = np.frombuffer(subprocess.run([FFMPEG, "-loglevel", "error", "-i", str(final), "-ac", "1", "-ar", str(SAMPLE_RATE),
                                        "-f", "f32le", "-"], capture_output=True, check=True).stdout, np.float32)
    rows, cut = [], 0
    for shot in script["shots"]:
        k, expected = sid(shot), tl[sid(shot)][0]
        first = ["".join(c for c in w.lower() if c.isalnum()) for w in shot["text"].split()[:3]]
        heard = any([w for w, _ in words[j:j + len(first)]] == first and abs(words[j][1] - expected) < 5
                    for j in range(len(words) - len(first) + 1))
        ref = sf.read(ep / "audio" / f"{k}.wav", dtype="float32")[0][:int(0.8 * SAMPLE_RATE)]
        c = round(expected * SAMPLE_RATE)
        lo = max(0, c - SAMPLE_RATE)
        seg = mix[lo:c + SAMPLE_RATE + len(ref)]
        n = 1 << math.ceil(math.log2(len(seg) + len(ref)))
        corr = np.fft.irfft(np.fft.rfft(seg, n) * np.conj(np.fft.rfft(ref, n)), n)[:len(seg) - len(ref) + 1]
        offset_ms = (lo + int(np.argmax(corr)) - c) / SAMPLE_RATE * 1000
        rows.append({"shot": k, "voice_at": round(expected + offset_ms / 1000, 3), "cut_at": round(cut / FPS, 3),
                     "offset_ms": round(offset_ms, 1), "heard": heard})
        cut += shot_frames[k]
    off = [r["shot"] for r in rows if abs(r["offset_ms"]) > 1000 / FPS]
    info = {"video_frames": v_frames, "expected_frames": round(total * FPS), "video_s": round(v_secs, 2), "audio_s": round(a_secs, 2),
            "max_line_offset_ms": max(abs(r["offset_ms"]) for r in rows), "lines_off_by_over_1_frame": off,
            "lines_heard_by_whisper": f"{sum(r['heard'] for r in rows)}/{len(rows)}",
            "not_heard": [r["shot"] for r in rows if not r["heard"]], "seconds": round(time.time() - t0, 1)}
    (ep / "qa" / "sync_report.json").write_text(json.dumps({"summary": info, "lines": rows}, indent=1))
    bench.save("sync_check", info)
    print("SYNC", json.dumps(info))


STAGES = {"tts": stage_tts, "keys": stage_keys, "finalize": stage_finalize, "clips": stage_clips, "upscale": stage_upscale,
          "assemble": stage_assemble, "sync_check": stage_sync_check,
          "qa_keys": lambda ep, s, b: contact_sheets(ep, "keys"),
          "qa_bkeys": lambda ep, s, b: contact_sheets(ep, "bkeys"),
          "qa_clips": lambda ep, s, b: contact_sheets(ep, "clips")}

if __name__ == "__main__":
    name = sys.argv[1]
    ep_dir = ROOT / "episodes" / name
    script = json.loads((ep_dir / "script.json").read_text(encoding="utf-8"))
    LOCATION_PROPS.update(script.get("location_props", {}))
    bench = Bench(ep_dir)
    for st in (sys.argv[2:] or list(STAGES)):
        STAGES[st](ep_dir, script, bench)

"""Character library. Each character is drawn once by FLUX.2 klein (raw/master.png, from cast.json "look" + "seed"); every
pose is an edit of the master, and every expression, talking mouth and blink is an edit whose changed face is patched back
onto the pose drawing, so the body pixels never move. Sprites are keyed out of a flat key-colour background, upscaled 2x
and cached in characters/<id>/, shared by every episode.

Sprite names: <pose>_<expression>[_talk|_blink].png  (RGBA, cropped; anchors in characters/<id>/sprites.json)
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
import comfy_client as cc  # noqa: E402

ROOT = Path(__file__).parent
CHARS = ROOT / "characters"
GEN_W, GEN_H = 896, 1344
KEYS = {"green": "a solid flat chroma key green background (#00FF00)", "magenta": "a solid flat magenta background (#FF00FF)"}
PRONOUNS = {"she": ("she", "her", "her", "is"), "he": ("he", "him", "his", "is"), "they": ("they", "them", "their", "are")}
# pose -> edit instruction ({t} = them, {p} = their); "stand" is the master drawing itself
POSES = {
    "stand": None,
    "front": "Turn {t} to face the viewer directly, front view, standing straight with {p} arms relaxed at {p} sides",
    "point": "{P} right arm is extended forward, pointing with {p} index finger",
    "cross": "{P} arms are crossed over {p} chest",
    "hips": "Both of {p} hands are on {p} hips, elbows out, confident",
    "think": "One hand is raised to {p} chin, thinking",
    "shrug": "Shrugging, both forearms raised with palms up",
    "wave": "One hand is raised beside {p} head, waving hello",
    "phone": "Holding a smartphone in front of {p} chest with both hands, looking down at it",
    "hold": "Holding a small object up in front of {p} face with one hand, showing it",
    "hands": "Both hands are raised in front of {p} chest with palms out, defensive",
    "fists": "Both hands are clenched into fists at {p} sides, tense",
    "slump": "Shoulders slumped and head lowered, arms hanging, defeated",
    "walk": "Walking to the right in mid-stride, one leg forward and one back, arms swinging",
}
EXPRESSIONS = {
    "neutral": None,
    "smile": "a warm friendly closed-mouth smile with both eyes open and looking ahead",
    "happy": "laughing happily with {p} mouth open",
    "sad": "sad, eyebrows angled up, mouth turned down",
    "angry": "angry, furrowed brows and a hard frown",
    "shocked": "shocked, eyes wide open, mouth open in a round O",
    "scared": "scared, eyes wide, eyebrows raised, teeth clenched",
    "worried": "worried, inner eyebrows raised, a nervous frown",
    "smug": "smug, half-lidded eyes and a one-sided smirk",
    "confused": "confused, one eyebrow raised, mouth pulled to one side",
    "serious": "serious and focused, eyebrows low, mouth flat",
    "crying": "crying, tears running down {p} cheeks",
}
NO_BLINK = {"happy", "crying"}  # eyes already squeezed: a blink frame adds nothing


def _style(cast, key):
    return (f"{cast['style']}, full body from the top of the head to the shoes, standing, three-quarter view facing right, "
            f"centered with empty space above the head and below the feet, on {KEYS[key]}, no text, no shadow on the ground")


def _keep(key):
    return (" Keep everything else exactly the same: the same character, face, hair, outfit, colors and line style, the same size "
            f"and position in the frame, the full body visible, the same {KEYS[key].split(' (')[0].replace('a solid flat ', '')}.")


def _words(c, text):
    they, them, their, _ = PRONOUNS[c.get("pronoun", "they")]
    return text.format(t=them, p=their, P=their.capitalize(), y=they)


def _rgb(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32)


def _gen(prompt, out, seed, refs=()):
    files, secs = cc.run(cc.klein_image(prompt, GEN_W, GEN_H, seed=seed, refs=[cc.to_input(r) for r in refs], filename_prefix="cartoon/sprite"),
                         out.parent, out.stem)
    files[0].replace(out)
    return secs


def _key_strength(a, key):
    return a[..., 1] - np.maximum(a[..., 0], a[..., 2]) if key == "green" else np.minimum(a[..., 0], a[..., 2]) - a[..., 1]


def _background(a, k, k0, f=16):
    """The key colour as it actually is around every pixel (the model paints a soft vignette): block means of the
    background pixels, holes filled from their neighbours, smoothly upsampled."""
    bg = (k > 0.4 * k0)[..., None].astype(np.float32)
    h, w = a.shape[0] // f, a.shape[1] // f
    s = (a * bg)[:h * f, :w * f].reshape(h, f, w, f, 3).sum((1, 3))
    n = bg[:h * f, :w * f].reshape(h, f, w, f, 1).sum((1, 3))
    valid = n[..., 0] > f * f * 0.5
    col = np.where(valid[..., None], s / np.maximum(n, 1), 0)
    while not valid.all():
        acc, cnt = np.zeros_like(col), np.zeros(valid.shape)
        for shift in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            v = np.roll(valid, shift, (0, 1))
            acc += np.roll(col, shift, (0, 1)) * v[..., None]
            cnt += v
        grow = ~valid & (cnt > 0)
        col[grow] = acc[grow] / cnt[grow][:, None]
        valid |= grow
    return np.asarray(Image.fromarray(col.clip(0, 255).astype(np.uint8)).resize((a.shape[1], a.shape[0]), Image.BILINEAR)).astype(np.float32)


def _key_fields(a, key):
    """Per pixel: how much it leans to the key colour (k), the local background colour, and that colour's own lean (kb)."""
    k = _key_strength(a, key)
    border = np.concatenate([k[:6].ravel(), k[-6:].ravel(), k[:, :6].ravel(), k[:, -6:].ravel()])
    bgc = _background(a, k, float(np.median(border)))
    return k, bgc, np.maximum(_key_strength(bgc, key), 40)


def _alpha(a, key):
    """Soft matte from how strongly each pixel leans to the local key colour, then un-mix the key from the edge pixels."""
    k, bgc, kb = _key_fields(a, key)
    hard = k > 0.6 * kb
    # soft edges only next to real background: key-tinted eye whites or shading inside the figure stay opaque
    near = np.asarray(Image.fromarray(hard.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(9))) > 0
    alpha = np.where(near, np.clip((0.85 * kb - k) / (0.6 * kb), 0, 1), 1.0)
    rgb = np.clip((a - (1 - alpha[..., None]) * bgc) / np.maximum(alpha[..., None], 0.05), 0, 255)
    return rgb, alpha


def _ghost_area(path, key):
    """Pixels of solid half-key-coloured patches. A clean drawing has only 1-2 px blended edges; a stray limb the model
    blended into the key background leaves a patch that survives an opening."""
    k, _, kb = _key_fields(_rgb(path), key)
    soft = Image.fromarray((((k > 0.2 * kb) & (k < 0.75 * kb)) * 255).astype(np.uint8))
    return int((np.asarray(soft.filter(ImageFilter.MinFilter(5)).filter(ImageFilter.MaxFilter(5))) > 0).sum())


def _draw(prompt, out, seed, key, refs, log):
    """Draw a whole figure (a master or a pose); redraw with another seed if it has a ghost patch."""
    for attempt in range(3):
        secs = _gen(prompt, out, seed + 1000 * attempt, refs)
        if _ghost_area(out, key) <= 100:
            break
        log(f"  {out.stem}: ghost patch in the drawing, redrawing")
    return secs


def _head_box(a, key):
    """Rows and columns where the face can be: the top quarter of the figure."""
    _, alpha = _alpha(a, key)
    rows = np.nonzero(alpha.max(1) > 0.5)[0]
    top, bot = rows[0], rows[-1]
    h = bot - top
    y0, y1 = max(0, int(top - 0.02 * h)), int(top + 0.25 * h)
    cols = np.nonzero(alpha[y0:y1].max(0) > 0.5)[0]
    return y0, y1, max(0, int(cols[0] - 0.05 * h)), int(cols[-1] + 0.05 * h)


def _patch(base, var, box):
    """Paste only the densely changed part of `var` inside `box` onto `base` (an edit redraws every line a little;
    only the face change is wanted). Returns the patched image and the patched area fraction."""
    d = (np.abs(var - base).max(2) > 48).astype(np.uint8) * 255
    dense = np.asarray(Image.fromarray(d).filter(ImageFilter.BoxBlur(7))) > 100
    keep = np.zeros_like(dense)
    y0, y1, x0, x1 = box
    keep[y0:y1, x0:x1] = dense[y0:y1, x0:x1]
    mask = Image.fromarray(keep.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(17)).filter(ImageFilter.GaussianBlur(3))
    m = np.asarray(mask).astype(np.float32)[..., None] / 255
    ring = (m[..., 0] > 0.02) & (m[..., 0] < 0.6) & (np.abs(var - base).max(2) < 30)
    if ring.sum() > 50:  # the edit may shift colours slightly: match the variant to the base where they meet
        var = var + np.median((base - var)[ring], axis=0)
    return base * (1 - m) + var * m, float(keep.mean())


def _finish(raw, out, key, box=None):
    """Upscale 2x, key out the background, crop to `box` (or the figure) and save as RGBA. Returns the crop box."""
    files, _ = cc.run(cc.upscale(cc.to_input(raw), filename_prefix="cartoon/up"), out.parent / "raw", out.stem + "_up")
    rgb, alpha = _alpha(_rgb(files[0]), key)
    files[0].unlink()
    if box is None:
        ys, xs = np.nonzero(alpha > 0.5)
        pad = 24
        box = (max(0, xs.min() - pad), max(0, ys.min() - pad), min(alpha.shape[1], xs.max() + pad), min(alpha.shape[0], ys.max() + pad))
    x0, y0, x1, y1 = (int(v) for v in box)
    img = np.dstack([rgb, alpha * 255]).astype(np.uint8)[y0:y1, x0:x1]
    Image.fromarray(img, "RGBA").save(out)
    return box


def _anchors(path):
    """Feet anchor (centre of the lowest opaque rows) and head top, in sprite pixels."""
    a = np.asarray(Image.open(path))[..., 3] > 128
    rows = np.nonzero(a.any(1))[0]
    top, bot = int(rows[0]), int(rows[-1])
    xs = np.nonzero(a[bot - max(4, (bot - top) // 30):bot + 1].any(0))[0]
    return {"feet": [int((xs.min() + xs.max()) / 2), bot], "top": top, "size": list(Image.open(path).size)}


def _redraws(d):
    path = d / "redraws.json"
    return json.loads(path.read_text()) if path.exists() else {}


def redraw(cid, name):
    """Forget a pose (`point`) or one pose and expression (`point_angry`) of a character, so the next run draws it again
    with a new seed. The master (`stand`) is the character itself: change its seed in cast.json instead."""
    d = CHARS / cid
    if name.split("_")[0] == "stand" and name in ("stand", "stand_neutral"):
        raise SystemExit("'stand' is the master drawing. To redraw the character, change its seed in cast.json and delete "
                         f"cartoon/characters/{cid}.")
    if name.endswith("_neutral"):  # the neutral face is the pose drawing itself
        name = name[:-len("_neutral")]
    prefix = name + "_" if "_" not in name else name
    meta = json.loads((d / "sprites.json").read_text())
    gone = [n for n in meta if n.startswith(prefix)]
    for n in gone:
        del meta[n]
    for f in list(d.glob(f"{prefix}*.png")) + list((d / "raw").glob(f"{prefix}*.png")):
        f.unlink()
    (d / "sprites.json").write_text(json.dumps(meta, indent=1))
    bumps = _redraws(d)
    bumps[name] = bumps.get(name, 0) + 1
    (d / "redraws.json").write_text(json.dumps(bumps, indent=1))
    return gone


def _merge_meta(path, entries):
    """Add entries to sprites.json as it is on disk now, so two runs drawing the same character never drop each other's work."""
    meta = json.loads(path.read_text()) if path.exists() else {}
    meta.update(entries)
    path.write_text(json.dumps(meta, indent=1))
    return meta


def ensure(cast, needs, poses=None, log=print):
    """needs: {char_id: {(pose, expression): wants_talk}}; poses: extra pose instructions from the screenplay.
    Generates whatever is missing; returns the sprite metadata."""
    poses = dict(POSES, **(poses or {}))
    meta_all = {}
    for cid, combos in needs.items():
        c = cast["characters"][cid]
        key = c.get("key", "green")
        d = CHARS / cid
        (d / "raw").mkdir(parents=True, exist_ok=True)
        meta_path = d / "sprites.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        seed = c["seed"]
        bumps = _redraws(d)  # redraw() asks for a fresh seed per pose or pose_expression
        master = d / "raw" / "master.png"
        if not master.exists():
            secs = _draw(f"{_style(cast, key)}. The character: {c['look']}.", master, seed, key, [], log)
            Image.open(master).save(d / "master.png")
            log(f"{cid}: master drawn in {secs:.0f}s")
        for pose, expr in sorted(combos):
            name = f"{pose}_{expr}"
            todo = [name] + ([name + "_talk"] if combos[(pose, expr)] else []) + ([] if expr in NO_BLINK else [name + "_blink"])
            todo = [n for n in todo if n not in meta or not (d / f"{n}.png").exists()]
            if not todo:
                continue
            pose_raw = master if pose == "stand" else d / "raw" / f"{pose}_neutral.png"
            if not pose_raw.exists():  # a pose is an edit of the master, used whole
                secs = _draw(_words(c, f"Change {{p}} pose: {poses[pose]}. Keep {{p}} face and expression the same.") + _keep(key),
                             pose_raw, seed + 7919 * bumps.get(pose, 0), key, [master], log)
                log(f"{cid}: pose {pose} in {secs:.0f}s")
            raw = pose_raw if expr == "neutral" else d / "raw" / f"{name}.png"
            if not raw.exists():
                _variant(c, key, pose_raw, raw, f"Change only {{p}} facial expression: {EXPRESSIONS[expr]}.", seed + 7919 * bumps.get(name, 0), log)
            for vname in todo:
                vraw = raw if vname == name else d / "raw" / f"{vname}.png"
                if not vraw.exists():
                    what = ("Change only {p} mouth: open it as if {y} were in the middle of saying a word." if vname.endswith("_talk")
                            else "Change only {p} eyes: both eyes fully closed, mid-blink.")
                    _variant(c, key, raw, vraw, what, seed + 7919 * bumps.get(name, 0), log)
                # variants share the base sprite's crop, so swapping them never moves a pixel
                box = _finish(vraw, d / f"{vname}.png", key, meta[name]["box"] if vname != name else None)
                meta[vname] = dict(_anchors(d / f"{vname}.png"), box=[int(v) for v in box])
            meta = _merge_meta(meta_path, {n: meta[n] for n in todo})
            log(f"{cid}: {name} -> {', '.join(todo)}")
        meta_all[cid] = meta
    return meta_all


def _variant(c, key, base_raw, out, what, seed, log):
    """Edit `base_raw`, keep only the face change. Retries with another seed when the edit changed nothing."""
    base = _rgb(base_raw)
    box = _head_box(base, key)
    for attempt in range(3):
        _gen(_words(c, what) + f" The character: {c['look']}." + _keep(key), out, seed + 100 * attempt, [base_raw])
        img, area = _patch(base, _rgb(out), box)
        if area > 0.0004:
            break
        log(f"  {out.stem}: edit changed nothing, retrying")
    Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(out)

"""Automatic director: screenplay.txt -> script.json shots (the format make_video.py renders).

Screenplay format (one statement per line, '#' starts a comment):
  TITLE: What If You Could Rewind 10 Seconds
  NARRATOR: am_puck                                   narrator voice (Kokoro)
  STYLE: classic                                      character bodies: classic (Episode 1 stickman) or slim (grown-up)
  LOOK: flat                                          backgrounds: flat (bright vector) or cinematic (painted)
  @street: A wide city street in the morning, ...     location = one AI-painted plate, generated once
  @street_bus < street: The same street, now a bus    edit of another plate (keeps it consistent)
  @clock {free; clock=0.497,0.517,0.27}: ...          flags: free (no floor needed), clock hands, desk=<y>, amb=city|room|night|bus
  == street | max, zoe, extra:4 (tense)               scene: location | who is there, left to right | music mood
  NARRATOR: Text of the line. [max: run right -> shocked; zoe: smile] {sfx: door@0.4; fx: rewind}
  ZOE: Dialogue line. [smug]                          directions without a name apply to the speaker
  -- beat 1.5 {fx: replay 3}                          silent shot; replay = the last N shots backwards through the VHS look
  -- title 3 {pop: WHAT IF YOU COULD REWIND 10 SECONDS?}
  -- card 3 {pop: TO BE CONTINUED...}
  ## The rules                                        chapter marker: starts a YouTube chapter at the next shot
  THUMB: CAN'T DIE. | max smug | street | over=battery 1   thumbnail variant: text | character expression [pose] | plate | badge
Directions: steps split by '->' spread across the shot. A step mixes a pose (idle, shocked, fists, shrug, snap, point, hips,
phone, catch, slump, tell, laugh, watch, reach), an expression (smile, happy, smug, shocked, worried, sad, angry, squeeze,
scared, laugh, serious, neutral), a move (run/walk/enter/exit left|right), and flags (talk, sweat, shake, bounce, appear).
Tags: pop: A | B | C (big text, spread over the words), fx: rewind, sfx: name@frac, shot: wide|medium|close, focus: name, hold: sec,
  over: max=battery 100->87 | zoe=tag $40 | zoe=off      badge over a head, persists through the scene ('->' animates it)
  hud: YOUR BATTERY=battery 100->64 | DAY=text 3          corner widget, persists across scenes ('hud: off' clears all)
  note: 3 DAYS LATER                                     banner at the top for this shot
  graphic: number 4,106 | PEOPLE SAVED                   motion-graphic shot instead of the scene (number, chart up|down,
                                                         versus A | B | label, timeline A | B | C)
  burst: CRASH!@0.5                                      comic impact burst at that point of the shot
  car: right@0.2-0.7                                     a car drives across the frame
"""
import json
import re

import rig

POSE_ALIAS = {"tell": "tell_p", "laugh": "laugh_p", "watch": "watch_p", "reach": "reach_p", "sit": "sit_f"}
POSES = set(rig.POSES) | set(POSE_ALIAS)
EXPRS = set(rig.EXPR)
FLAGS = {"talk", "sweat", "shake", "bounce", "appear"}
SLOTS = {1: [0.5], 2: [0.36, 0.64], 3: [0.28, 0.5, 0.72], 4: [0.22, 0.41, 0.59, 0.78]}
FLOOR_Y, FLOOR_H = 0.95, 0.6                   # every plate is painted with an open floor across the bottom third
BACK_Y, BACK_H = 0.83, 0.42                    # extras stand further back
CLOSE_Y, CLOSE_H = 1.3, 1.15
COMPOSITION = ("eye-level camera, wide shot, an empty open floor area across the whole bottom third of the image, "
               "nothing blocking the foreground")
NARRATION_CYCLE = ["medium", "close", "medium", "wide", "close"]
TRANSIENT = {"tell_p", "laugh_p", "watch_p", "reach_p", "snap", "point", "shocked", "catch", "shrug"}


def _slots(n):
    return SLOTS.get(n) or [0.15 + 0.7 * i / (n - 1) for i in range(n)]


def _flags(text):
    out = {}
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition("=")
        out[k.strip()] = v.strip() if v else True
    return out


def _tags(text):
    tags = {}
    for block in re.findall(r"\{([^}]*)\}", text):
        for part in block.split(";"):
            if ":" in part:
                k, v = part.split(":", 1)
                tags.setdefault(k.strip(), []).append(v.strip())
    return tags


def _directions(text, speaker, protagonist):
    """'[max: run right -> shocked; zoe: smile]' -> {char: [step, ...]} where a step is a list of words."""
    out = {}
    for block in re.findall(r"\[([^\]]*)\]", text):
        for part in block.split(";"):
            part = part.strip()
            if not part:
                continue
            m = re.match(r"^([a-z0-9_:]+)\s*:\s*(.*)$", part)
            who, body = (m.group(1), m.group(2)) if m else (speaker or protagonist, part)
            out.setdefault(who, []).extend([s.split() for s in body.split("->")])
    return out


def _value(text):
    """'battery 100->87' -> battery badge animating 100 to 87; 'tag $8.7B' / 'text 3' -> text badge; 'off' -> None."""
    text = text.strip()
    if text == "off":
        return None
    kind, _, rest = text.partition(" ")
    if kind == "battery":
        a, _, b = rest.partition("->")
        return {"kind": "battery", "from": float(a), "to": float(b or a)}
    return {"kind": "tag", "text": rest.strip() or kind}


def _settle(spec):
    """After a shot, an animated badge keeps its end value and stops popping in."""
    spec.pop("new", None)
    if "to" in spec:
        spec["from"] = spec["to"]


def _change_sfx(spec):
    if spec["kind"] != "battery":
        return []
    d = spec["to"] - spec["from"]
    out = [[0.2, "drain", -6]] if d <= -15 else [[0.2, "charge", -6]] if d >= 15 else []
    return out + ([[0.12, "beep", -8]] if spec["to"] <= 5 else [])


def _split_sentences(text, limit=16):
    if len(text.split()) <= limit:
        return [text]
    parts = re.split(r"(?<=[.?!])\s+", text)
    return [p for p in parts if p]


class Scene:
    def __init__(self, plate, members, mood, flags):
        self.plate, self.mood, self.flags = plate, mood, flags
        mods = {m.split("@")[0]: set(m.split("@")[1:]) for m in members}   # name@desk (behind the desk), name@later (appears later)
        self.mains = [n for n in mods if not n.startswith("extra:")]
        extras = [n for n in mods if n.startswith("extra:")]
        self.pos = {}
        for name, x in zip(self.mains, _slots(len(self.mains)) if self.mains else []):
            self.pos[name] = (x, FLOOR_Y, FLOOR_H)
        taken = [self.pos[n][0] for n in self.mains]
        for name in extras:  # background people go in the gaps, so nobody (or their badge) hides behind someone else
            x = max((0.12 + 0.04 * i for i in range(20)), key=lambda c: min([abs(c - t) for t in taken] or [1.0]))
            taken.append(x)
            self.pos[name] = (x, BACK_Y, BACK_H)
        self.desk = {n for n, m in mods.items() if "desk" in m}
        self.state = {n: {"expr": "neutral", "pose": "idle"} for n in self.pos}
        self.present = {n for n, m in mods.items() if "later" not in m}
        self.over = {}
        self.count = 0


def parse(text, cast):
    script = {"title": "", "narrator_voice": "am_puck", "plates": {}, "plate_picks": {}, "plate_flags": {}, "shots": [], "thumbs": []}
    scene, sid, narr_i = None, 0, 0
    history, chapter, hud = [], None, {}
    for raw in text.splitlines():
        if raw.startswith("## "):
            chapter = raw[3:].strip()
            continue
        line = raw.split("#", 1)[0].rstrip() if not raw.lstrip().startswith("--") else raw.rstrip()
        line = line.strip()
        if not line:
            continue
        if line.startswith("TITLE:"):
            script["title"] = line[6:].strip()
        elif line.startswith("STYLE:"):
            script["body"] = line[6:].strip()
        elif line.startswith("LOOK:"):
            script["look"] = line[5:].strip()
        elif line.startswith("THUMB:"):
            parts = [x.strip() for x in line[6:].split("|")]
            who = parts[1].split()
            thumb = {"text": parts[0], "char": who[0], "expr": who[1] if len(who) > 1 else "shocked",
                     "pose": POSE_ALIAS.get(who[2], who[2]) if len(who) > 2 else "idle", "plate": parts[2]}
            for extra in parts[3:]:
                k, _, v = extra.partition("=")
                if k.strip() == "over":
                    thumb["over"] = _value(v)
            script["thumbs"].append(thumb)
        elif line.startswith("NARRATOR:") and scene is None:
            script["narrator_voice"] = line[9:].strip()
        elif line.startswith("@"):
            m = re.match(r"^@(\w+)\s*(?:<\s*(\w+))?\s*(?:\{([^}]*)\})?\s*:\s*(.+)$", line)
            name, ref, flags, desc = m.groups()
            flags = _flags(flags or "")
            prompt = desc if "free" in flags else f"{desc}, {COMPOSITION}"
            script["plates"][name] = {"prompt": prompt, "ref": ref} if ref else prompt
            script["plate_flags"][name] = flags
            script["plate_picks"][name] = int(flags.get("pick", 1))
        elif line.startswith("=="):
            m = re.match(r"^==\s*(\w+)\s*(?:\|\s*([^(]*))?(?:\((\w+)\))?\s*$", line)
            plate, members, mood = m.groups()
            members = [x.strip() for x in (members or "").split(",") if x.strip()]
            scene = Scene(plate, members, mood or "calm", script["plate_flags"].get(plate, {}))
        elif line.startswith("--"):
            m = re.match(r"^--\s*(\w+)\s+([\d.]+)\s*(.*)$", line)
            kind, dur, rest = m.group(1), float(m.group(2)), m.group(3)
            tags = _tags(rest)
            sid += 1
            shot = {"id": sid, "dur": dur, "mood": scene.mood if scene else "calm"}
            if kind == "card":
                shot.update(plate=None, pops=[[0.12, tags.get("pop", [""])[0]]], sfx=[[0.08, "boom", -3]])
            elif kind == "title":
                shot.update(_frame(scene, "plate", None, 0), dim=0.5, pops=[[0.1, tags.get("pop", [""])[0]]], sfx=[[0.08, "boom", -3]])
                shot["actors"] = []
            else:
                fx = tags.get("fx", [""])[0]
                if fx.startswith("replay"):
                    n = int(fx.split()[1]) if len(fx.split()) > 1 else 3
                    shot.update(rewind=history[-n:], sfx=[[0, "rewind", -2]])
                else:
                    shot.update(_frame(scene, "wide", None, 0))
                    shot["actors"], shot["sfx"] = _actors(scene, {}, "wide", None)
            if hud:
                shot["hud"] = [dict(w) for w in hud.values()]
            if chapter:
                shot["chapter"], chapter = chapter, None
            script["shots"].append(shot)
            if "rewind" not in shot:
                history.append(sid)
        else:
            m = re.match(r"^([A-Z][A-Z0-9_]*)\s*:\s*(.+)$", line)
            who, body = m.group(1).lower(), m.group(2)
            text_only = re.sub(r"\[[^\]]*\]|\{[^}]*\}", "", body).strip()
            dirs = _directions(body, who if who in scene.pos else None, scene.mains[0] if scene.mains else None)
            for name in dirs:
                if name not in scene.pos:
                    raise ValueError(f"{name!r} is directed but not in the scene: {raw.strip()}")
            tags = _tags(body)
            pieces = _split_sentences(text_only)
            for k, piece in enumerate(pieces):
                sid += 1
                first, last = k == 0, k == len(pieces) - 1
                shot = _direct_line(scene, sid, who, piece, dirs if first else {}, tags if last else {}, narr_i, hud)
                if who == "narrator":
                    narr_i += 1
                if first and history and scene.count == 1 and len(script["shots"]) > 0:
                    shot["sfx"].append([0, "whoosh", -12])
                if chapter:
                    shot["chapter"], chapter = chapter, None
                script["shots"].append(shot)
                history.append(sid)
    for name in script["plates"]:
        script["plate_picks"].setdefault(name, 1)
    return script


def _frame(scene, kind, focus, u_seed):
    """Camera move for a shot type: [[cx, cy, zoom] start, end], alternating push-ins and pull-outs."""
    xs = [scene.pos[n][0] for n in scene.present if not n.startswith("extra:")] or [0.5]
    cx = sum(xs) / len(xs)
    push = scene.count % 2 == 0
    drift = 0.02 if scene.count % 3 == 0 else -0.02
    if kind == "close" and focus:
        x = scene.pos[focus][0]
        a, b = [x, 0.5, 1.0], [x + drift * 0.5, 0.5, 1.09]
    elif kind == "medium":
        a, b = [cx - drift, 0.6, 1.38], [cx + drift, 0.6, 1.5]
    elif kind == "plate":
        a, b = [0.5, 0.52, 1.0], [0.5, 0.52, 1.18]
    else:
        a, b = [cx - drift, 0.56, 1.02], [cx + drift, 0.56, 1.12]
    if not push:
        a, b = b, a
    return {"plate": scene.plate, "cam": [a, b]}


def _step(words):
    """One direction step -> act fields."""
    act = {}
    i = 0
    while i < len(words):
        w = words[i]
        if w in ("run", "walk", "enter", "exit") and i + 1 < len(words) and words[i + 1] in ("left", "right"):
            act["move"] = (w, words[i + 1])
            i += 2
            continue
        if w == "look" and i + 1 < len(words):
            act["look"] = -0.8 if words[i + 1] == "left" else 0.8 if words[i + 1] == "right" else 0.0
            i += 2
            continue
        if w in POSES:
            act["pose"] = POSE_ALIAS.get(w, w)
        if w in EXPRS:
            act["expr"] = w
        if w in FLAGS:
            act[w] = True
        i += 1
    return act


def _actors(scene, dirs, kind, focus, speaker=None):
    """Actor specs for the shot; updates the scene's persistent expression / pose / presence state."""
    actors, sfx = [], []
    names = [focus] if kind == "close" and focus else [n for n in scene.pos if n in scene.present or n in dirs]
    for name in names:
        x, y, h = scene.pos[name]
        st = scene.state[name]
        steps = [_step(s) for s in dirs.get(name, [])] or [{}]
        acts = []
        x_now = x
        actor = {"char": name, "x": x, "y": y, "h": h}
        if kind == "close":
            actor.update(y=CLOSE_Y, h=CLOSE_H)
        if name in scene.desk:
            actor["clip_y"] = float(scene.flags.get("desk", 0.65))
            actor.update(y=0.98, h=0.62)
        others = [scene.pos[n][0] for n in scene.mains if n != name and n in scene.present]
        default_look = 0.0 if not others else (0.5 if others[0] > x else -0.5)
        if speaker and speaker != name and speaker in scene.pos:
            default_look = 0.6 if scene.pos[speaker][0] > x else -0.6
        for k, step in enumerate(steps):
            at = round(k / len(steps), 3)
            st["expr"] = step.get("expr", st["expr"])
            base = "idle" if k == 0 and st["pose"] in TRANSIENT else st["pose"]   # gestures don't carry into the next line
            pose = step.get("pose", base)
            st["pose"] = pose
            act = {"at": at, "pose": pose, "expr": st["expr"], "look": step.get("look", default_look)}
            if pose.endswith("_p"):
                act["face"] = "right" if default_look >= 0 else "left"
            for flag in ("sweat", "shake", "bounce"):
                if step.get(flag):
                    act[flag] = True
            if step.get("talk"):
                act["yap"] = True
            if step.get("appear"):
                actor["fade"] = [at, min(1.0, at + 0.18)]
                scene.present.add(name)
                sfx.append([at, "sting", -3])
            if st["expr"] in ("shocked", "scared") and step.get("expr") and k > 0:
                sfx.append([at, "pop", -8])
            if "move" in step:
                kind_m, d = step["move"]
                sgn = 1 if d == "right" else -1
                if kind_m == "enter":
                    start, end = x - sgn * 0.45, x
                    scene.present.add(name)
                elif kind_m == "exit":
                    start, end = x_now, x + sgn * 0.6
                    scene.present.discard(name)
                else:
                    start, end = x - sgn * 0.22, x + sgn * 0.22
                if k == 0:
                    actor["x"] = start
                act.update({"do": "walk" if kind_m == "walk" else "run" if kind_m == "run" else "walk", "to_x": end, "face": d})
                act.pop("pose", None)
                x_now = end
                sfx.append([at, "footsteps", -9])
            acts.append(act)
        actor["acts"] = acts
        if name in scene.present or any("move" in s for s in steps):
            actors.append(actor)
    return actors, sfx


def _direct_line(scene, sid, who, text, dirs, tags, narr_i, hud):
    speaker = who if who in scene.pos else None
    moving = any("move" in _step(s) for steps in dirs.values() for s in steps)
    if tags.get("graphic"):
        kind = "graphic"
    elif tags.get("shot"):
        kind = tags["shot"][0]
    elif not scene.pos:
        kind = "plate"
    elif scene.count == 0 or moving:
        kind = "wide"
    elif speaker:
        kind = "close"
    else:
        kind = NARRATION_CYCLE[narr_i % len(NARRATION_CYCLE)]
    focus = tags.get("focus", [None])[0] or speaker or next(iter(dirs), None) or (scene.mains[0] if scene.mains else None)
    if kind != "graphic" and (focus not in scene.pos or (kind == "close" and (focus in scene.desk or focus.startswith("extra:")))):
        kind = "medium" if scene.pos else "plate"
    shot = {"id": sid, "who": who, "text": text, "mood": scene.mood}
    shot.update(_frame(scene, "plate" if kind == "graphic" else kind, focus, sid))
    changed = []
    for item in tags.get("over", []):
        for part in item.split("|"):
            name, _, val = part.partition("=")
            name = name.strip()
            if name not in scene.pos:
                raise ValueError(f"badge for {name!r}, who is not in the scene: {text}")
            spec = _value(val)
            if spec is None:
                scene.over.pop(name, None)
            else:
                spec["new"] = name not in scene.over
                scene.over[name] = spec
                changed.append(spec)
    for item in tags.get("hud", []):
        if item.strip() == "off":
            hud.clear()
            continue
        for part in item.split("|"):
            label, _, val = part.partition("=")
            spec = _value(val)
            if spec is None:
                hud.pop(label.strip(), None)
            else:
                spec["label"] = label.strip()
                if spec["kind"] == "tag":
                    spec["kind"] = "text"
                hud[label.strip()] = spec
                changed.append(spec)
    if kind == "graphic":
        actors, sfx = [], []
        gtype, _, rest = tags["graphic"][0].partition(" ")
        parts = [x.strip() for x in rest.split("|")] if rest.strip() else []
        n_args = {"number": 1, "chart": 1, "versus": 2}.get(gtype, len(parts))
        shot["graphic"] = {"type": gtype, "args": parts[:n_args] or ["up"], "label": parts[n_args] if len(parts) > n_args else ""}
        sfx.append([0.02, "whoosh", -10])
        if gtype == "number":
            sfx.append([0.66, "ding", -8])
    else:
        actors, sfx = _actors(scene, dirs, kind, focus, speaker) if scene.pos else ([], [])
    for a in actors:
        if a["char"] in scene.over:
            a["over"] = dict(scene.over[a["char"]])
    if hud:
        shot["hud"] = [dict(w) for w in hud.values()]
    for spec in changed:
        sfx += _change_sfx(spec)
    if tags.get("note"):
        shot["note"] = tags["note"][0]
    if tags.get("burst"):
        word, _, at = tags["burst"][0].partition("@")
        shot["burst"] = [float(at or 0.4), word.strip()]
        sfx += [[float(at or 0.4), "boom", -4], [float(at or 0.4), "crash", -8]]
    for item in tags.get("car", []):
        d, _, span = item.partition("@")
        a, _, b = (span or "0.15-0.75").partition("-")
        shot.setdefault("props", []).append({"prop": "car", "dir": d.strip(), "from": float(a), "to": float(b)})
        sfx.append([float(a), "carpass", -4])
    if moving and kind == "wide":
        xs = [a["x"] for a in actors] + [a["acts"][-1].get("to_x", a["x"]) for a in actors]
        shot["cam"] = [[min(xs) + 0.1, 0.58, 1.2], [max(xs) - 0.1, 0.58, 1.2]]
    shot["actors"] = actors
    clock = scene.flags.get("clock")
    if clock:
        cx, cy, r = (float(v) for v in clock.split(","))
        shot.setdefault("props", []).append({"prop": "clock", "cx": cx, "cy": cy, "r": r, "spin": [-60, -600] if "fx" in tags else [-30, -60]})
    shot["sfx"] = sfx
    if tags.get("pop"):
        pops = [p.strip() for p in tags["pop"][0].split("|")]
        nw = len(text.split())
        shot["pops"] = [[min(nw - 1, round(i * nw / len(pops))), p] for i, p in enumerate(pops)]
        shot["sfx"] += [[f"w{w}", "boom" if i == len(pops) - 1 else "thump", -3 if i == len(pops) - 1 else -5]
                        for i, (w, _) in enumerate(shot["pops"])]
        shot["dim"] = 0.4
    for fx in tags.get("fx", []):
        if fx == "rewind":
            shot.setdefault("fx", []).append({"kind": "rewind", "from": 0.74, "to": 1.0})
            shot["sfx"].append([0.72, "rewind_short", -4])
    for s in tags.get("sfx", []):
        for item in s.split(","):
            name, _, at = item.strip().partition("@")
            shot["sfx"].append([float(at or 0), name, -4])
    if tags.get("hold"):
        shot["hold"] = float(tags["hold"][0])
    seen = set()
    shot["sfx"] = [x for x in shot["sfx"] if x[1] != "footsteps" or not (x[1] in seen or seen.add(x[1]))]
    for spec in list(scene.over.values()) + list(hud.values()):
        _settle(spec)
    scene.count += 1
    return shot


def direct(screenplay_path, out_path, cast):
    script = parse(screenplay_path.read_text(encoding="utf-8"), cast)
    out_path.write_text(json.dumps(script, indent=1, ensure_ascii=False), encoding="utf-8")
    return script

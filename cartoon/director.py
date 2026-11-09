"""Screenplay -> script.json for the cartoon pipeline: shots, framing, who stands where, poses, expressions, lip-sync.

Screenplay format (one statement per line, # starts a comment):
  TITLE: My Perfect Neighbor Had a Key to My Apartment
  NARRATOR: nia                          # a cast id: narration is in that character's voice
  THUMB: HE HAD A KEY | nia shocked | hallway | leo smile hold      # text | char expr [pose] | plate | more chars
  CAPTIONS: on                           # burn subtitles into the picture (an .srt is always written)
  POSE: mug = Holding a white coffee mug in front of {p} chest     # a story pose ({p} = his/her/their, {t} = him/her/them)
  @hallway: A narrow apartment hallway at night, ...                 # a location, painted once and reused
  @hallway_day < hallway: The same hallway in the morning, ...      # an edit of another location so they match
  @mailboxes {insert}: Close-up of a wall of brass mailboxes          # an insert: a full-frame picture, no characters
  ## Chapter name                         # YouTube chapter at the next shot
  == hallway | nia, leo@later (tense)     # scene: location | characters left to right (@later = enters later) (music mood)
  NARRATOR: Line. [nia: phone worried -> shocked] {shot: close; sfx: sting@0.4}
  LEO: Line. [hips smug; nia: angry]     # a direction without a name is the speaker's
  -- beat 1.2 [omar: shocked] {zoom}     # silent shot (a reaction)
  -- title 3 {pop: BIG TEXT}             # text card over the next location
  -- card 10 {pop: QUESTION FOR THE COMMENTS}

Directions [..]: `name: words; name: words`. Words are a pose, an expression, `->` (what follows starts halfway through
the line), `enter left|right`, `exit left|right`. Poses and expressions persist until changed (sprites.py lists them).
Tags {..}: shot: wide|two|medium|close, focus: a,b, insert: <location>, pop: TEXT, sfx: name@0.3, hold: 0.8,
zoom (crash zoom), shake, cam (the narrator says the line to camera), mood: calm|happy|tense|dark|funny.
"""
import json
import re

import sprites

EMOTIONAL = {"shocked", "angry", "scared", "crying", "sad"}
STAGE = ("a wide shot at eye level, the floor clearly visible across the lower part of the image, open empty floor space in the "
         "middle and foreground")
SLOTS = {1: [0.5], 2: [0.36, 0.64], 3: [0.28, 0.5, 0.72], 4: [0.2, 0.4, 0.6, 0.8], 5: [0.16, 0.33, 0.5, 0.67, 0.84]}
MOODS = {"calm", "happy", "tense", "dark", "funny"}
TRANSIENT = {"point", "shrug", "hands", "think", "wave", "fists"}  # gestures last one line; held poses (props, crossed arms) persist


def _trail(s, opener, closer):
    """Split a trailing {..} or [..] group off a line."""
    s = s.rstrip()
    if s.endswith(closer) and opener in s:
        i = s.rindex(opener)
        return s[:i].rstrip(), s[i + 1:-1].strip()
    return s, ""


def _tags(raw):
    tags = {}
    for part in filter(None, (p.strip() for p in raw.split(";"))):
        k, _, v = part.partition(":")
        tags[k.strip()] = v.strip() if v else True
    return tags


def _dirs(raw, default, cast, poses):
    out = {}
    for part in filter(None, (p.strip() for p in raw.split(";"))):
        who, sep, words = part.partition(":")
        if not sep or who.strip() not in cast:
            who, words = default, part
        who = who.strip()
        if who is None:
            raise ValueError(f"direction '{part}' needs a name on a narration line")
        d = out.setdefault(who, {})
        toks = words.split()
        for i, w in enumerate(toks):
            after = "->" in toks[:i]
            if w == "->":
                continue
            if w in ("enter", "exit"):
                d["move"] = {"kind": w, "side": toks[i + 1] if i + 1 < len(toks) else "left"}
            elif w in ("left", "right") and i and toks[i - 1] in ("enter", "exit"):
                continue
            elif w in poses:
                d["pose2" if after else "pose"] = w
            elif w in sprites.EXPRESSIONS:
                d["expr2" if after else "expr"] = w
            else:
                raise ValueError(f"unknown direction word '{w}' in [{raw}]")
    return out


class Scene:
    def __init__(self, plate, members, mood):
        self.plate, self.mood = plate, mood
        self.order = [m.split("@")[0] for m in members]
        self.present = [m.split("@")[0] for m in members if not m.endswith("@later")]
        xs = SLOTS[len(self.order)]
        self.x = dict(zip(self.order, xs))
        self.state = {m: {"pose": "stand", "expr": "neutral"} for m in self.order}
        self.n, self.last_speaker, self.last_cam = 0, None, None

    def facing_flip(self, who):
        x = self.x[who]
        others = [self.x[o] for o in self.present if o != who]
        if abs(x - 0.5) > 0.01:
            return x > 0.5
        return bool(others) and sum(others) / len(others) < 0.5


def direct(src, dst, cast):
    chars = cast["characters"]
    script = {"title": "", "narrator": None, "captions": False, "poses": {}, "thumbs": [], "plates": {}, "shots": []}
    poses = set(sprites.POSES)
    scene, chapter, needs = None, None, {}

    def need(who, pose, expr, talk=False):
        combos = needs.setdefault(who, {})
        combos[(pose, expr)] = combos.get((pose, expr), False) or talk

    for n, line in enumerate(src.read_text(encoding="utf-8").splitlines(), 1):
        comment = line.lstrip().startswith("#") and not line.lstrip().startswith("##")  # "## Chapter" is not a comment
        line = "" if comment else line.split(" #")[0].strip()
        if not line:
            continue
        try:
            head, _, rest = line.partition(":")
            if head in ("TITLE", "CAPTIONS") or (head == "NARRATOR" and scene is None and rest.strip() in chars):
                key = {"TITLE": "title", "NARRATOR": "narrator", "CAPTIONS": "captions"}[head]
                script[key] = rest.strip() if head != "CAPTIONS" else rest.strip().lower() == "on"
            elif head == "POSE":
                name, _, text = rest.partition("=")
                script["poses"][name.strip()] = text.strip()
                poses.add(name.strip())
            elif head == "THUMB":
                parts = [p.strip() for p in rest.split("|")]
                people = []
                for p in parts[1:2] + parts[3:]:
                    w = p.split()
                    people.append({"id": w[0], "expr": w[1] if len(w) > 1 else "neutral", "pose": w[2] if len(w) > 2 else "stand"})
                    need(w[0], people[-1]["pose"], people[-1]["expr"])
                script["thumbs"].append({"text": parts[0], "plate": parts[2], "chars": people})
            elif line.startswith("@"):
                name, _, prompt = line[1:].partition(":")
                name, flags = _trail(name, "{", "}")
                name, _, ref = name.partition("<")
                spec = {"prompt": prompt.strip(), "insert": "insert" in flags}
                if ref.strip():
                    spec["ref"] = ref.strip()
                m = re.search(r"pick=(\d)", flags)
                if m:
                    spec["pick"] = int(m.group(1))
                script["plates"][name.strip()] = spec
            elif line.startswith("##"):
                chapter = line[2:].strip()
            elif line.startswith("=="):
                body, mood = _trail(line[2:], "(", ")")
                plate, _, members = body.partition("|")
                scene = Scene(plate.strip(), [m.strip() for m in members.split(",") if m.strip()], mood or "calm")
                for m in scene.order:
                    if m not in chars:
                        raise ValueError(f"'{m}' is not in cast.json")
            elif line.startswith("--"):
                body, tags = _trail(line[2:], "{", "}")
                body, dirs = _trail(body, "[", "]")
                kind, *dur = body.split()
                shot = {"kind": kind, "dur": float(dur[0]) if dur else 1.0}
                _add(script, scene, shot, None, _dirs(dirs, None, chars, poses), _tags(tags), chapter, need)
                chapter = None
            elif head.lower() in chars or head == "NARRATOR":
                text, tags = _trail(rest, "{", "}")
                text, dirs = _trail(text, "[", "]")
                who = "narrator" if head == "NARRATOR" else head.lower()
                default = who if who != "narrator" else (script["narrator"] if scene and script["narrator"] in scene.present else None)
                _add(script, scene, {"kind": "line", "who": who, "text": text.strip()}, who, _dirs(dirs, default, chars, poses), _tags(tags), chapter, need)
                chapter = None
            else:
                raise ValueError("not a screenplay statement")
        except (ValueError, KeyError, IndexError) as e:
            raise SystemExit(f"screenplay line {n}: {e}\n  {line}")
    # cards show the next location (blurred); a card at the very end shows the last one
    plates = [s.get("plate") for s in script["shots"]]
    for i, s in enumerate(script["shots"]):
        if s["kind"] in ("title", "card"):
            s["plate"] = next((p for p in plates[i + 1:] if p), next((p for p in reversed(plates[:i]) if p), None))
    for combos in needs.values():  # the standing drawing sets every character's scale
        combos.setdefault(("stand", "neutral"), False)
    script["needs"] = {c: [[p, e, t] for (p, e), t in sorted(v.items())] for c, v in needs.items()}
    dst.write_text(json.dumps(script, indent=1), encoding="utf-8")
    return script


def _add(script, scene, shot, who, dirs, tags, chapter, need):
    shot["id"] = len(script["shots"])
    if chapter:
        shot["chapter"] = chapter
    for k in ("pop", "insert", "hold"):
        if k in tags:
            shot[k] = float(tags[k]) if k == "hold" else tags[k]
    if "sfx" in tags:
        shot["sfx"] = [[float(at or 0), name] for name, _, at in (s.partition("@") for s in tags["sfx"].split(","))]
    for k in ("zoom", "shake"):
        if k in tags:
            shot[k] = True
    if shot["kind"] in ("title", "card"):
        script["shots"].append(shot)
        return
    if scene is None:
        raise ValueError("a line before the first scene (==)")
    if "mood" in tags:
        if tags["mood"] not in MOODS:
            raise ValueError(f"unknown mood {tags['mood']}")
        scene.mood = tags["mood"]
    shot.update(plate=scene.plate, mood=scene.mood)
    narrator = script["narrator"]
    if "cam" in tags:  # the narrator talks to the audience, alone in the frame
        st = dict(scene.state.get(narrator, {"expr": "neutral"}), **dirs.get(narrator, {}))
        a = {"id": narrator, "x": 0.5, "flip": False, "pose": "front", "expr": st.get("expr", "neutral"), "talk": True}
        if "expr2" in st:
            a["expr2"] = st["expr2"]
        for e in {a["expr"], a.get("expr2", a["expr"])}:
            need(narrator, "front", e, True)
        shot.update(actors=[a], cam={"type": "medium", "focus": [narrator]})
        script["shots"].append(shot)
        return
    for m, d in dirs.items():
        if m not in scene.order:
            raise ValueError(f"'{m}' is not in this scene (add them to the == line)")
        if d.get("move", {}).get("kind") == "enter" and m not in scene.present:
            scene.present.append(m)
    actors = []
    for m in scene.order:
        if m not in scene.present:
            continue
        d, st = dirs.get(m, {}), scene.state[m]
        a = {"id": m, "x": scene.x[m], "flip": scene.facing_flip(m), "pose": d.get("pose", st["pose"]), "expr": d.get("expr", st["expr"])}
        if "pose2" in d or "expr2" in d:
            a["pose2"], a["expr2"] = d.get("pose2", a["pose"]), d.get("expr2", a["expr"])
        a["talk"] = shot["kind"] == "line" and who == m
        if "move" in d:
            a["move"] = d["move"]
            need(m, "walk", a["expr"])
        for p, e in {(a["pose"], a["expr"]), (a.get("pose2", a["pose"]), a.get("expr2", a["expr"]))}:
            need(m, p, e, a["talk"])
        st["pose"], st["expr"] = a.get("pose2", a["pose"]), a.get("expr2", a["expr"])
        if st["pose"] in TRANSIENT:
            st["pose"] = "stand"
        actors.append(a)
    shot["actors"] = actors
    shot["cam"] = _choose_cam(scene, shot, who, dirs, tags, narrator)
    for a in actors:
        if a.get("move", {}).get("kind") == "exit":
            scene.present.remove(a["id"])
    scene.n += 1
    script["shots"].append(shot)


def _choose_cam(scene, shot, who, dirs, tags, narrator):
    present = [a["id"] for a in shot["actors"]]
    if "insert" in tags:
        return {"type": "insert"}
    focus = [f.strip() for f in tags["focus"].split(",")] if "focus" in tags else [m for m in dirs if m in present]
    speaker = who if who in present else None
    if speaker and speaker not in focus:
        focus.insert(0, speaker)
    kind = tags.get("shot")
    if not kind:
        moving = any("move" in a for a in shot["actors"])
        if scene.n == 0 or moving or not present:
            kind = "wide"
        elif speaker:
            listener = scene.last_speaker if scene.last_speaker in present and scene.last_speaker != speaker else None
            emotional = any(a["id"] == speaker and (a["expr"] in EMOTIONAL or a.get("expr2") in EMOTIONAL) for a in shot["actors"])
            if listener and scene.n % 4 == 3:
                kind, focus = "two", [speaker, listener]
            else:
                kind = "close" if emotional else "medium"
        elif len(focus) >= 2:
            kind = "two"
        elif len(focus) == 1:
            kind = "close" if scene.last_cam == "medium" else "medium"
        else:
            kind = "wide" if scene.last_cam != "wide" else "two" if len(present) >= 2 else "medium"
            focus = present
        if kind == scene.last_cam == "wide" and present and not moving and scene.n:
            kind, focus = "medium", focus or [present[0]]
    if not focus:
        focus = present[:2]
    scene.last_cam = kind
    if speaker:
        scene.last_speaker = speaker
    return {"type": kind, "focus": focus}

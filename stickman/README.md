# Stickman video pipeline

You write a screenplay; the pipeline makes the finished video: painted backgrounds, code-drawn characters, voices, captions,
camera moves, on-screen graphics, sound effects, music and thumbnails. Everything runs locally (ComfyUI for the backgrounds,
the rest on the CPU).

Two looks, chosen per episode in the screenplay header:
- `STYLE: classic` + `LOOK: flat`: the Episode 1 look (big round head, line limbs, bright vector backgrounds).
- `STYLE: slim` + `LOOK: cinematic`: the Episode 2 look (grown-up proportions, painted backgrounds). This is the default.

`make_episode.py` is the original all-AI pipeline (AI keyframes animated by Wan 2.2) that made the first moon episode. It is
kept as it was, but it takes about 10x longer per minute of video.

Commands below run from the `stickman` folder; the Python environment is the repo's `..\venv`.

## Make a video

1. **Pick the topic.** Run `..\venv\Scripts\python topics.py`. It writes `research\topics_<date>.md`: what "What If" premises
   are getting views right now, and which are already taken. Choose a fresh premise with a **visible mechanic**
   (something you can draw over people: a battery %, a price tag, a countdown).
2. **Write the screenplay.** Copy `templates\whatif_v2.txt` (the structure, the craft rules and the packaging checklist)
   to `episodes\<name>\screenplay.txt`. `episodes\ep03_battery\screenplay.txt` is a full example.
3. **Start ComfyUI** (from its folder): `venv\Scripts\python main.py --use-sage-attention`
4. **Run** `..\venv\Scripts\python make_video.py <name> all`. The `qa` stage prints warnings if the first 10 s show
   nothing, or if the title or thumbnails break the packaging rules.
5. **Review.** Open `episodes\<name>\qa\`:
   - `plates.jpg` shows the backgrounds;
   - `frames_N.jpg` shows one frame per shot;
   - `thumbs.jpg` shows the thumbnails at full size and at phone size.
6. **Upload** `final\<name>.mp4` with the thumbnails `final\thumb_N.jpg` (use YouTube's Test & Compare with all of them)
   and the title, description and chapters from `final\<name>_upload.txt`.

Each stage caches its output, so after an edit only the changed work reruns. Stages, in order:
`direct tts plates words qa thumbs render mix final`.
- Voice lines are cached by voice and text.
- Rendered shots are cached by a fingerprint of everything they show.
- To change anything, edit the screenplay and run `all` again.
- To redo one background, delete `plates\<name>_c1.png`.
- If you changed the drawing code itself, delete `segments\` to force a full re-render.

## Benchmarks (RTX 4060 Ti 8 GB, 16 GB RAM, 8-minute episode)

| Stage | Time | Notes |
|---|---|---|
| direct | under 1 s | |
| tts | ~1.5 min | CPU |
| plates | ~25 s per new location (2 candidates) | GPU; reused locations are free |
| words | ~4 min | CPU, cached per line |
| qa + thumbs | ~1 min | |
| render | ~7 min | 8 CPU workers |
| mix | ~40 s | |
| final encode | ~4 min | |
| **Total** | **~20-25 min machine time** | Human time is mostly writing and approving the screenplay. |

## Characters: `cast.json`

One entry per character: a Kokoro voice and a design. The design fields are colours (`top`, `legs`, `shoes`,
`skirt`, `scarf`, `cap`, `hat`, `collar`, `hair_color`), `hair` (`short`, `bun`, `long`, `ponytail`), `glasses`, and `coat`.
The character looks the same in every shot, because it is drawn from this design, not by an AI model.
Background people are `extra:<number>`; each number is a different random look.

## Screenplay format (full reference at the top of `director.py`)

```
TITLE: What If You Could See Everyone's Battery?
NARRATOR: am_puck
THUMB: 1% LEFT? | walter scared | platform | over=battery 1      # thumbnail variant (add 2-3)
@platform: A subway platform in the morning, ...              # a location: one AI-painted background
@kitchen_night < kitchen: The same kitchen at night, ...      # an edit of another background, so they match
## One Percent                                                # YouTube chapter
== platform | max, walter, extra:3 (tense)                    # scene: location | who is there | music mood
NARRATOR: The line. [walter: lie] {over: walter=battery 1->0; burst: THUD!@0.45; hold: 0.8}
NARRATOR: The line. {hud: YOUR BATTERY=battery 100->80; note: RULE 1: EVERYTHING CHARGES YOU}
NARRATOR: The line. {graphic: number 214 | PEOPLE CHARGED}
ZOE: A dialogue line. [smug]
-- beat 1.6 {fx: replay 2}                                    # silent shot: the last 2 shots rewind on screen
-- title 3 {pop: BIG TITLE TEXT}
-- card 10 {pop: A QUESTION FOR THE COMMENTS}
```

**On-screen graphics (what makes the idea visible)**
- `over:` puts a badge over a head (`battery N`, `battery A->B`, `tag TEXT`). It stays for the whole scene.
- `hud:` adds corner widgets that stay across scenes.
- `note:` shows a banner for one shot.
- `graphic:` turns the shot into a motion graphic: `number`, `chart up|down`, `versus A | B | label`, or `timeline A | B | C`.
- `burst:` shows a comic impact word.
- `car:` drives a car across the frame.
- `pop:` shows big centre text.
- `fx: rewind` adds the VHS rewind look.

**Characters**
- Moods: `calm`, `happy`, `tense` and `dark`.
- Poses: `idle`, `shocked`, `fists`, `shrug`, `snap`, `point`, `hips`, `phone`, `catch`, `slump`, `tell`, `laugh`, `watch`, `reach`, `lie` (collapsed on the floor), and `desk` (with `name@desk`).
- Expressions: `smile`, `happy`, `smug`, `shocked`, `worried`, `sad`, `angry`, `squeeze`, `scared`, `laugh`, `serious` and `neutral`.
- Moves: `run`, `walk`, `enter` and `exit`, each followed by `left` or `right`.
- Flags: `talk`, `sweat`, `shake`, `bounce`, and `appear` (with `name@later`).

**Shots and sound**
- The director picks shot types itself: a wide shot opens each scene, dialogue is a close-up of the speaker, narration
  cycles medium, close and wide, and moves get a tracking shot. Override with `{shot: wide|medium|close}` or `{focus: name}`.
- Sound effects come automatically for scene changes, shocks, moves, big text, rewinds, bursts, cars and
  battery changes (beep at 5% or less, a charge-up or drain-down sound on big changes).
- All sounds are synthesized in `../shared/sfx.py`, so there are no licence questions.

## Files

- `make_video.py`: the stages, the frame compositor and the thumbnails.
- `director.py`: turns a screenplay into shots.
- `overlays.py`: badges, the HUD, banners, motion graphics, bursts and cars.
- `rig.py`: the characters, poses, expressions and walk/run cycles.
- `../shared/sfx.py`: the sounds and music beds (shared with the cartoon pipeline).
- `topics.py`: market research for picking topics.
- `templates\whatif_v2.txt`: the screenplay template and checklist.
- `../shared/comfy_client.py`: talks to ComfyUI (FLUX.2 klein).
- `make_episode.py`: the original all-AI pipeline (AI keyframes + Wan 2.2 animation).

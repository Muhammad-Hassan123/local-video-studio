# Cartoon pipeline

A screenplay goes in and a finished 2D animated episode comes out. Everything runs locally: FLUX.2 klein through ComfyUI
draws the characters and locations, Kokoro voices them, and the rest is CPU code.

## How the characters stay the same

Each character is drawn once from `cast.json` (`look` + `seed`) as `characters/<id>/master.png`. Everything else is an
edit of that drawing:
- every pose is an edit of the master;
- every expression is an edit of a pose drawing, with only the changed face pasted back onto it;
- every talking mouth and blink is an edit of an expression drawing, again pasting back only the changed face.

So the body pixels never change between a character's expressions, mouths and blinks. Sprites are cut out of a
green or magenta background, upscaled 2x with an anime upscaler, and cached in `characters/<id>/`. Every episode reuses
them, so a character looks the same everywhere and only new poses cost GPU time.

Locations work the same way: each is painted once (2 candidates), upscaled 2x, and cached in `locations/` for every later
episode. `@night < day` makes a location as an edit of another, so the two match.

## Make an episode

1. Add or adjust characters in `cast.json`:
   - `look`: what they look like, in one sentence.
   - `seed`: change it to get a different drawing.
   - `voice`: Kokoro voice blend, e.g. `af_heart*0.7+af_nova*0.3`.
   - `speed`, `height` and `bio` (their story).
   - `key`: `magenta` for characters who wear green.
2. Copy `templates\story_v1.txt` (the structure and the craft rules) to `episodes\<name>\screenplay.txt`.
   `episodes\ep01_the_key\screenplay.txt` is a full example. The format is documented at the top of `director.py`.
3. Start ComfyUI, then run from the repo root: `venv\Scripts\python cartoon\make_cartoon.py <name> all`.
4. Review in `episodes\<name>\qa\`:
   - `frames_N.jpg`: one frame per shot;
   - `locations.jpg`: both candidates per location (add `{pick=2}` to an `@` line to swap);
   - `thumbs.jpg`: the thumbnails at full size and phone size.
5. Upload `final\<name>.mp4` with:
   - the thumbnails `final\thumb_N.jpg`;
   - the subtitles `final\<name>.srt`;
   - the title, description and chapters from `final\<name>_upload.txt`.

Stages, in order: `direct tts sprites plates thumbs render mix qa final`. Each caches its work:
- Voice lines are cached by voice + text.
- Sprites and locations are cached forever, across episodes.
- Shots re-render only when something they show changed. Bump `RENDER_VERSION` after changing the drawing code.

## What the director does by itself

- Places the scene's characters left to right as listed and faces them toward each other.
- Picks the shots:
  - a wide shot to open each scene;
  - medium shots on speakers, and a close-up when they are shocked, angry, scared, sad or crying;
  - a two-shot every fourth exchange;
  - reaction shots for silent beats.
- Moves the camera: a slow push-in on every shot, and a crash zoom with `{zoom}`.
- Animates the characters:
  - lip-sync from the voice's loudness;
  - random blinks;
  - idle breathing and a talking bob;
  - a walk cycle for `enter` and `exit`.
- Adds sound: a whoosh on scene changes, footsteps on moves, a pop on shocks, a boom on the title, and ambience per location.
- Scores each scene with an instrumental bed for its mood. The beds are composed once by ACE-Step 1.5 (`music` stage,
  `shared/music.py`) and ducked under the voices.

## Benchmarks (RTX 4060 Ti 8 GB, 16 GB RAM)

Measured on Episode 1 (8:10, 130 shots, 5 characters, 14 locations, 71 pose/expression combinations):

| Stage | Episode 1 (new cast) | Later episodes, same cast | Notes |
|---|---|---|---|
| tts | 1.7 min | seconds to 1.5 min | CPU, cached per line |
| sprites | ~75 min (bf16 encoder); ~55 min with the fp8 encoder now in use | 0-10 min | ~13 s per edit with the fp8 encoder; only new poses and expressions cost anything |
| plates | 6.3 min | 0-2 min | ~27 s per new location |
| render | 1.9 min | under 2 min | 8 CPU workers, ~100 frames/s in total; only changed shots |
| music | 5.8 min, once ever | 0 | 10 beds shared by every episode |
| mix + qa + final | 3.2 min | 3.2 min | |
| **Total** | **~70 min** | **~10-20 min** | |

## Files

- `make_cartoon.py`: the stages, the compositor (camera, lip-sync, blinks, walks), the thumbnails, the mix.
- `director.py`: screenplay → shots (format reference at the top).
- `sprites.py`: the character library (drawing, editing, face patching, keying).
- `cast.json`: the series' characters, their voices and their stories.
- `characters/`, `locations/`: the cached art. Only `master.png` per character is in git; the rest regenerates.
- `templates/story_v1.txt`: the screenplay template and checklist.

# Video Studio

Two local pipelines that turn a screenplay into a finished, upload-ready YouTube episode of 8 minutes or more. Each episode
comes with thumbnails, chapters, subtitles and the upload text. No paid APIs: everything runs on one PC with an 8 GB
NVIDIA GPU.

| Pipeline | Look | Characters | Made for |
|---|---|---|---|
| [`stickman/`](stickman/README.md) | Stick figures. **classic** (the Episode 1 look) or **slim** (the Episode 2 look) | drawn by code, so they look identical in every shot | "What If You…" explainer stories |
| [`cartoon/`](cartoon/README.md) | 2D cartoon with flat colours and bold outlines | drawn once by AI, then edited into poses, expressions, mouths and blinks | animated storytime series with a recurring cast |

Both share `shared/`: the ComfyUI client and the synthesized sound effects and music. Voices are Kokoro TTS.

## Set up a PC

1. Install Python 3.12, git, and ComfyUI (portable or git + venv).
2. Run: `powershell -ExecutionPolicy Bypass -File setup.ps1 -ComfyUI <path to ComfyUI> -DownloadImageModels`.
   This creates `venv`, installs the requirements, downloads the voice model, and puts the image models
   (FLUX.2 klein 4B with its text encoder and VAE, an anime upscaler, and ACE-Step 1.5 for music, about 20 GB) into ComfyUI.
3. Start ComfyUI before the art stages by double-clicking `start_comfyui.bat`.

## Make an episode

Double-click `start_comfyui.bat`, then `make_cartoon.bat` or `make_stickman.bat`. Or from a terminal:

```
venv\Scripts\python cartoon\make_cartoon.py <episode> all
cd stickman  &&  ..\venv\Scripts\python make_video.py <episode> all
```

Every stage caches its work:
- A second render after a script edit only redoes the lines and shots that changed.
- Characters and locations drawn for one episode are reused by every later one.

## What's in the repo

The repo holds the code, the screenplays and each character's master drawing. Generated art, audio and video stay local
and are rebuilt from those files.

## Layout

```
setup.ps1, setup.sh, requirements.txt   new-PC setup (Windows / Linux cloud)
shared/      comfy_client.py (ComfyUI workflows), sfx.py (sounds and music)
stickman/    make_video.py, director.py, rig.py, overlays.py, topics.py, templates/, episodes/<ep>/screenplay.txt
cartoon/     make_cartoon.py, director.py, sprites.py, cast.json, templates/, characters/<id>/master.png, episodes/<ep>/screenplay.txt
models/      voice and caption models (downloaded by setup, not in git)
```

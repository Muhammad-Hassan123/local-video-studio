"""Music library: instrumental beds per mood, composed once by ACE-Step 1.5 (MIT licence, through ComfyUI) and reused by
every episode. Tracks live in shared/music/<mood>_<n>.flac; delete one to compose it again."""
from pathlib import Path

import numpy as np
import soundfile as sf

import comfy_client as cc
import sfx

LIB = Path(__file__).parent / "music"
TRACKS = 2           # variants per mood, so neighbouring scenes in the same mood don't reuse the same bed
SECONDS = 150.0
LEVEL = 0.08         # every bed is normalised to this RMS before mixing
# mood -> (style tags, bpm, key)
MOODS = {
    "calm": ("warm lo-fi background music for a story, gentle acoustic guitar, soft felt piano, light brushed drums, relaxed, "
             "instrumental, no vocals", 80, "C major"),
    "happy": ("cheerful light background music, ukulele, glockenspiel, pizzicato strings and claps, bright and warm, "
              "instrumental, no vocals", 112, "G major"),
    "funny": ("quirky comedic cartoon underscore, pizzicato strings, bassoon, woodblock and xylophone, bouncy and mischievous, "
              "instrumental, no vocals", 118, "F major"),
    "tense": ("suspenseful mystery underscore, low pulsing synth bass, ticking hi-hats, staccato strings, slowly building tension, "
              "instrumental, no vocals", 96, "D minor"),
    "dark": ("dark ambient underscore, low cello drones, sparse distant piano notes, eerie pads, ominous and slow, "
             "instrumental, no vocals", 60, "A minor"),
}


def ensure(log=print):
    LIB.mkdir(exist_ok=True)
    for i, (mood, (tags, bpm, key)) in enumerate(MOODS.items()):
        for n in range(1, TRACKS + 1):
            out = LIB / f"{mood}_{n}.flac"
            if not out.exists():
                files, secs = cc.run(cc.ace_music(tags, SECONDS, bpm, key, seed=100 * i + n, filename_prefix="music/bed"), LIB, out.stem)
                files[0].replace(out)
                log(f"music {out.name}: {secs:.0f}s")


def bed(mood, length, variant):
    """A stereo bed of `length` seconds at sfx.SR: the mood's track, looped with a 2 s crossfade if it is too short."""
    x, sr = sf.read(LIB / f"{mood}_{variant % TRACKS + 1}.flac", dtype="float32", always_2d=True)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    if sr != sfx.SR:
        t = np.arange(int(len(x) * sfx.SR / sr)) * sr / sfx.SR
        x = np.stack([np.interp(t, np.arange(len(x)), x[:, c]) for c in range(2)], axis=1).astype(np.float32)
    x *= LEVEL / (np.sqrt(np.mean(x ** 2)) + 1e-9)
    n, fade = int(length * sfx.SR), 2 * sfx.SR
    out = x
    while len(out) < n:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)[:, None]
        out = np.concatenate([out[:-fade], out[-fade:] * (1 - ramp) + x[:fade] * ramp, x[fade:]])
    return out[:n]

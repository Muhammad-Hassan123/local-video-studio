"""Sound effects, ambience and a music pad synthesized in numpy: no downloads, no license questions.
Every generator returns mono float32 at SR. Deterministic (seeded), so a re-render sounds the same."""
import numpy as np

SR = 48000


def _t(dur):
    return np.arange(int(dur * SR)) / SR


def _noise(dur, seed=0):
    return np.random.default_rng(seed).standard_normal(int(dur * SR)).astype(np.float32)


def _lp(x, cutoff):
    """Fast fixed low-pass via FFT brick-wall with a soft knee."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X *= 1 / (1 + (f / cutoff) ** 4)
    return np.fft.irfft(X, len(x)).astype(np.float32)


def _hp(x, cutoff):
    return (x - _lp(x, cutoff)).astype(np.float32)


def _bp_sweep(x, f0, f1, q=3.0):
    """Band-pass noise with a moving centre: state-variable filter, per-sample coefficients."""
    n = len(x)
    fc = np.geomspace(f0, f1, n)
    f = 2 * np.sin(np.pi * fc / SR)
    lo = bp = 0.0
    y = np.empty_like(x)
    damp = 1 / q
    for i in range(n):
        hi = x[i] - lo - damp * bp
        bp += f[i] * hi
        lo += f[i] * bp
        y[i] = bp
    return y


def _env(dur, attack, release, curve=2.0):
    t = _t(dur)
    e = np.minimum(1, t / max(attack, 1e-4)) * np.clip((dur - t) / max(release, 1e-4), 0, 1) ** curve
    return e.astype(np.float32)


def _norm(x, peak=0.9):
    m = np.abs(x).max()
    return (x / m * peak).astype(np.float32) if m > 0 else x


def whoosh(dur=0.45, up=True, seed=1):
    x = _bp_sweep(_noise(dur, seed), 300 if up else 3000, 3000 if up else 300, q=2.5)
    t = _t(dur)
    e = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 1.5
    return _norm(x * e, 0.6)


def tick(seed=0):
    x = _hp(_noise(0.03, seed), 2000) * np.exp(-_t(0.03) * 300)
    return _norm(x, 0.5)


def ticking(dur, every=0.5):
    out = np.zeros(int(dur * SR), np.float32)
    for k, start in enumerate(np.arange(0, dur - 0.05, every)):
        s = tick(k) * (1.0 if k % 2 else 0.7)
        i = int(start * SR)
        out[i:i + len(s)] += s[:len(out) - i]
    return out


def rewind(dur=1.6, seed=3):
    """Tape-rewind: a fast warbling squeal over hissing noise, pitch rising."""
    t = _t(dur)
    pitch = 900 * np.geomspace(1, 2.2, len(t)) * (1 + 0.25 * np.sin(2 * np.pi * 23 * t))
    phase = 2 * np.pi * np.cumsum(pitch) / SR
    squeal = np.sign(np.sin(phase)) * 0.25 + np.sin(2 * phase) * 0.3
    chatter = (np.sin(2 * np.pi * 61 * t) > 0.6).astype(np.float32) * 0.5 + 0.5
    hiss = _hp(_noise(dur, seed), 2500) * 0.35
    x = (_lp(squeal.astype(np.float32), 5000) * chatter + hiss) * _env(dur, 0.05, 0.12, 1.0)
    return _norm(x, 0.55)


def riser(dur=3.0, seed=4):
    t = _t(dur)
    x = _bp_sweep(_noise(dur, seed), 200, 5000, q=4) * (t / dur) ** 2
    tone = np.sin(2 * np.pi * np.cumsum(np.geomspace(110, 440, len(t))) / SR) * (t / dur) ** 3 * 0.4
    return _norm(x + tone.astype(np.float32), 0.6)


def boom(dur=1.6, seed=5):
    t = _t(dur)
    f = 60 * np.exp(-t * 2.2) + 32
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 2.6)
    click = _lp(_noise(dur, seed), 900) * np.exp(-t * 40) * 0.8
    return _norm((body + click).astype(np.float32), 0.95)


def thump(seed=6):
    t = _t(0.35)
    body = np.sin(2 * np.pi * np.cumsum(110 * np.exp(-t * 12) + 45) / SR) * np.exp(-t * 14)
    snap = _hp(_noise(0.35, seed), 1500) * np.exp(-t * 90) * 0.4
    return _norm((body + snap).astype(np.float32), 0.8)


def door(seed=7):
    """Pneumatic bus door: hiss then a rubbery thunk."""
    hiss = _hp(_noise(0.7, seed), 1800) * _env(0.7, 0.02, 0.5, 1.5)
    thunk = thump(seed + 1)
    out = np.zeros(int(0.9 * SR), np.float32)
    out[:len(hiss)] += hiss * 0.5
    i = int(0.55 * SR)
    out[i:i + len(thunk)] += thunk[:len(out) - i]
    return _norm(out, 0.7)


def engine(dur=3.0, away=True, seed=8):
    t = _t(dur)
    f = 48 * (np.linspace(1.0, 0.82, len(t)) if away else np.linspace(0.8, 1.0, len(t)))
    ph = 2 * np.pi * np.cumsum(f) / SR
    hum = (np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.25 * np.sign(np.sin(3 * ph))).astype(np.float32)
    rumble = _lp(_noise(dur, seed), 200)
    vol = np.linspace(1, 0.05, len(t)) if away else np.linspace(0.2, 1, len(t))
    return _norm(_lp(hum * 0.6 + rumble, 900) * vol.astype(np.float32), 0.6)


def brakes(seed=9):
    t = _t(1.0)
    squeal = np.sin(2 * np.pi * (2400 + 80 * np.sin(2 * np.pi * 7 * t)) * t) * _env(1.0, 0.1, 0.5) * 0.15
    hiss = _hp(_noise(1.0, seed), 2000) * _env(1.0, 0.4, 0.4) * 0.5
    return _norm((squeal + hiss).astype(np.float32), 0.5)


def clack(seed=10):
    t = _t(0.25)
    x = _hp(_noise(0.25, seed), 800) * np.exp(-t * 60) + np.sin(2 * np.pi * 1700 * t) * np.exp(-t * 50) * 0.3
    return _norm(x.astype(np.float32), 0.6)


def crickets(dur=1.5, seed=11):
    t = _t(dur)
    out = np.zeros_like(t, dtype=np.float32)
    for k, start in enumerate(np.arange(0.05, dur - 0.2, 0.42)):
        i = int(start * SR)
        n = int(0.16 * SR)
        tt = t[:n]
        chirp = np.sin(2 * np.pi * 4300 * tt) * (np.sin(2 * np.pi * 30 * tt) > 0) * np.sin(np.pi * tt / 0.16)
        out[i:i + n] += chirp[:len(out) - i] * (0.8 if k % 2 else 1.0)
    return _norm(out, 0.35)


def ding():
    t = _t(1.4)
    x = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t * d) for f, a, d in ((1568, 1, 3.5), (3136, 0.4, 5), (4704, 0.2, 7), (2093, 0.3, 4)))
    return _norm(x.astype(np.float32), 0.45)


def pop(seed=12):
    t = _t(0.12)
    x = np.sin(2 * np.pi * np.cumsum(np.geomspace(900, 300, len(t))) / SR) * np.exp(-t * 35)
    return _norm(x.astype(np.float32), 0.5)


def heartbeat(n=3, bpm=70):
    beat = 60 / bpm
    out = np.zeros(int((n * beat + 0.5) * SR), np.float32)
    for k in range(n):
        for off, g in ((0, 1.0), (0.28, 0.7)):
            s = thump(20 + k) * g
            s = _lp(s, 180)
            i = int((k * beat + off) * SR)
            out[i:i + len(s)] += s[:len(out) - i]
    return _norm(out, 0.9)


def sting(seed=13):
    """Horror hit: boom plus a dissonant high cluster with a short reverse swell."""
    t = _t(2.5)
    cluster = sum(np.sin(2 * np.pi * f * t) for f in (740, 784, 1109, 1175)) * np.exp(-t * 1.6) * 0.2
    b = boom(2.5, seed)
    return _norm((b + cluster).astype(np.float32), 0.95)


def footsteps(dur, every=0.31, seed=14):
    out = np.zeros(int(dur * SR), np.float32)
    for k, start in enumerate(np.arange(0.05, dur - 0.1, every)):
        t = _t(0.09)
        s = _lp(_noise(0.09, seed + k), 700) * np.exp(-t * 60)
        i = int(start * SR)
        out[i:i + len(s)] += s[:len(out) - i]
    return _norm(out, 0.4)


def ambience(kind, dur, seed=15):
    n = _noise(dur, seed)
    if kind == "city":
        x = _lp(n, 500) * 0.6 + _bp_sweep(_noise(dur, seed + 1) * 0.2, 900, 1100, q=8) * 0.3
    elif kind == "night":
        x = _lp(n, 250) * 0.5
    elif kind == "bus":
        x = _lp(n, 160) * 0.8 + engine(dur, away=False, seed=seed)[:len(n)] * 0.15
    else:  # room
        x = _lp(n, 300) * 0.3
    return _norm(x.astype(np.float32), 0.25)


def drone(dur, seed=16):
    t = _t(dur)
    x = sum(np.sin(2 * np.pi * f * t + np.sin(2 * np.pi * 0.1 * t) * 2) for f in (55, 82.4, 110.5)) / 3
    x = x * (0.7 + 0.3 * np.sin(2 * np.pi * 0.2 * t)) + _lp(_noise(dur, seed), 120) * 0.3
    return _norm(x.astype(np.float32) * _env(dur, 0.8, 0.8, 1.0), 0.4)


def pad(dur, chords=((57, 60, 64), (53, 57, 60), (48, 52, 55), (55, 59, 62)), bar=4.0, seed=17):
    """Soft synth pad: detuned saws, low-passed, one chord per bar (A minor - F - C - G by default)."""
    t = _t(dur)
    out = np.zeros_like(t, dtype=np.float32)
    xf = int(0.6 * SR)
    for b in range(int(np.ceil(dur / bar))):
        notes = chords[b % len(chords)]
        i0, i1 = int(b * bar * SR), min(len(t), int((b + 1) * bar * SR) + xf)
        tt = t[: i1 - i0]
        seg = np.zeros(len(tt), np.float32)
        for m in notes:
            f = 440 * 2 ** ((m - 69) / 12)
            for det in (-0.12, 0.0, 0.11):
                ph = (f * (1 + det / 100) * tt) % 1.0
                seg += (2 * ph - 1).astype(np.float32) * 0.2
        e = np.minimum(1, tt / 0.6) * np.minimum(1, (len(tt) / SR - tt) / 0.6)
        out[i0:i1] += seg * e.astype(np.float32)
    return _norm(_lp(out, 900), 0.5)


LIB = {"whoosh": whoosh, "whoosh_down": lambda: whoosh(0.5, up=False, seed=2), "rewind": rewind, "riser": riser, "boom": boom,
       "thump": thump, "door": door, "engine_away": lambda: engine(3.0, True), "brakes": brakes, "clack": clack,
       "crickets": crickets, "ding": ding, "pop": pop, "heartbeat": heartbeat, "sting": sting,
       "ticking": lambda: ticking(2.0), "rewind_short": lambda: rewind(0.6, 31), "footsteps": lambda: footsteps(2.5)}


def crash(seed=18):
    """Ceramic pot shattering: a noisy hit plus a scatter of short high pings."""
    t = _t(1.2)
    hit = _hp(_noise(1.2, seed), 600) * np.exp(-t * 9)
    rng = np.random.default_rng(seed)
    pings = np.zeros_like(t, dtype=np.float32)
    for _ in range(14):
        i = int(rng.uniform(0.0, 0.5) * SR)
        f = rng.uniform(2500, 6500)
        n = int(0.08 * SR)
        pings[i:i + n] += (np.sin(2 * np.pi * f * t[:n]) * np.exp(-t[:n] * 60) * rng.uniform(0.2, 0.6))[:len(pings) - i]
    return _norm((hit + pings + _lp(_noise(1.2, seed + 1), 300) * np.exp(-t * 6) * 0.8).astype(np.float32), 0.8)


def horn(dur=0.8):
    t = _t(dur)
    x = sum(np.sign(np.sin(2 * np.pi * f * t)) for f in (415, 523)) * 0.5
    return _norm(_lp(x.astype(np.float32), 2500) * _env(dur, 0.02, 0.1, 1.0), 0.55)


def screech(dur=1.1, seed=19):
    t = _t(dur)
    x = _bp_sweep(_noise(dur, seed), 3200, 2600, q=20) * (1 + 0.4 * np.sin(2 * np.pi * 13 * t))
    return _norm((x * _env(dur, 0.05, 0.4, 1.0)).astype(np.float32), 0.5)


def beep(n=3):
    """Low-battery beeps."""
    out = np.zeros(int((n * 0.32 + 0.2) * SR), np.float32)
    for k in range(n):
        t = _t(0.14)
        x = np.sin(2 * np.pi * 1320 * t) * _env(0.14, 0.005, 0.03, 1.0)
        i = int(k * 0.32 * SR)
        out[i:i + len(x)] += x.astype(np.float32)
    return _norm(out, 0.4)


def charge():
    """Rising three-note chime, like a phone starting to charge."""
    out = np.zeros(int(0.9 * SR), np.float32)
    for k, f in enumerate((660, 880, 1320)):
        t = _t(0.45)
        x = (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t)) * np.exp(-t * 7)
        i = int(k * 0.12 * SR)
        out[i:i + len(x)] += x.astype(np.float32)[:len(out) - i]
    return _norm(out, 0.45)


def drain(dur=0.9):
    """Falling power-down tone."""
    t = _t(dur)
    f = np.geomspace(700, 90, len(t))
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(dur, 0.02, 0.3, 1.0)
    return _norm(_lp(x.astype(np.float32), 2500), 0.45)


def carpass(dur=2.0, seed=21):
    """Car driving past: engine hum rising then falling in pitch and level."""
    t = _t(dur)
    f = 70 * (1.15 - 0.3 * (t / dur))
    ph = 2 * np.pi * np.cumsum(f) / SR
    hum = np.sin(ph) + 0.5 * np.sin(2 * ph)
    vol = np.exp(-((t / dur - 0.5) ** 2) / 0.04)
    road = _lp(_noise(dur, seed), 900) * 0.6
    return _norm(((hum.astype(np.float32) * 0.6 + road) * vol).astype(np.float32), 0.6)


LIB.update({"crash": crash, "horn": horn, "screech": screech, "beep": beep, "charge": charge, "drain": drain, "carpass": carpass})

"""Code-drawn stick characters. Each character is one fixed design posed by joint angles, so every shot shows exactly the
same figure: no AI redraw drift, hands can't break, and anything can move every frame (breathing, blinks, lip flap).

Angles are in degrees, 0 = pointing straight down, positive = rotated toward +x (screen right, or "forward" in profile).
Profile figures are drawn facing right and mirrored for facing left. Lengths are in units of the figure height.
"""
import math
import random

from PIL import Image, ImageDraw, ImageFilter

SS = 3                                   # supersampling factor for clean edges
# Body styles (a channel picks one). slim: grown-up figures in clothes. classic: the Episode 1 stickman, a big round head,
# a small white torso, thin single-line arms and legs, mitten hands and oval shoes.
BODIES = {
    "slim": {"head_r": 0.083, "neck": 0.022, "torso": 0.29, "uarm": 0.15, "farm": 0.135, "hand_r": 0.021, "thigh": 0.225,
             "shin": 0.215, "arm_w": 0.036, "leg_w": 0.046, "shoulder_w": 0.07, "hip_w": 0.04, "shoe": (0.036, 0.028, 0.014)},
    "classic": {"head_r": 0.165, "neck": -0.012, "torso": 0.27, "uarm": 0.15, "farm": 0.14, "hand_r": 0.032, "thigh": 0.15,
                "shin": 0.145, "arm_w": 0.017, "leg_w": 0.019, "shoulder_w": 0.1, "hip_w": 0.05, "shoe": (0.05, 0.042, 0.021)},
}
HEAD_R = BODIES["slim"]["head_r"]         # slim default, used by callers that size things off a figure
OUTLINE = (22, 22, 24, 255)
SKIN = (243, 240, 233, 255)
MOUTH_IN = (70, 22, 24, 255)

# named poses: front view uses l = screen-left limb; profile uses l = far limb, r = near limb
POSES = {
    "idle": {"la": -9, "le": -5, "ra": 9, "re": 5, "ll": -3, "rl": 3},
    "idle_p": {"la": -4, "le": 10, "ra": 4, "re": 12, "ll": -2, "rl": 3},
    "shocked": {"la": -150, "le": -25, "ra": 150, "re": 25, "ll": -6, "rl": 6},
    "fists": {"la": -12, "le": 125, "ra": 12, "re": -125, "ll": -4, "rl": 4, "crouch": 0.01},
    "shrug": {"la": -35, "le": -85, "ra": 35, "re": 85, "ll": -3, "rl": 3},
    "snap": {"la": -9, "le": -5, "ra": 140, "re": 30, "ll": -3, "rl": 3},
    "point": {"la": -6, "le": -4, "ra": 95, "re": 0, "ll": -3, "rl": 3},
    "hips": {"la": -40, "le": 95, "ra": 40, "re": -95, "ll": -5, "rl": 5},
    "reach_p": {"la": 10, "le": 15, "ra": 80, "re": 10, "ll": -2, "rl": 3, "lean": 8},
    "catch_p": {"la": 10, "le": 15, "ra": 70, "re": 40, "ll": -2, "rl": 3, "lean": 4},
    "tell_p": {"la": -4, "le": 10, "ra": 60, "re": 70, "ll": -2, "rl": 3},
    "laugh_p": {"la": 20, "le": 90, "ra": 25, "re": 95, "ll": -2, "rl": 3, "lean": -6, "head": -12},
    "sit_p": {"la": 20, "le": 60, "ra": 25, "re": 65, "ll": 88, "lk": 88, "rl": 92, "rk": 92},
    "sit_f": {"la": -14, "le": 35, "ra": 14, "re": -35, "ll": -38, "lk": -40, "rl": 38, "rk": 40, "sit": True},
    "desk": {"la": -25, "le": 70, "ra": 25, "re": -70, "ll": -3, "rl": 3},
    "slump": {"la": -3, "le": -2, "ra": 3, "re": 2, "ll": -3, "rl": 3, "head": 18, "lean": 6},
    "watch_p": {"la": 40, "le": 100, "ra": 4, "re": 12, "ll": -2, "rl": 3, "head": 15},
    "phone": {"la": -9, "le": -5, "ra": 22, "re": -115, "ll": -3, "rl": 3, "head": 8},
    "catch": {"la": -9, "le": -5, "ra": 35, "re": -35, "ll": -6, "rl": 6, "crouch": 0.015, "head": 10},
    "lie": {"la": -12, "le": 8, "ra": 12, "re": -8, "ll": -3, "rl": 3, "lie": True},   # drawn standing, rotated flat by the compositor
}
POSE_KEYS = ("la", "le", "ra", "re", "ll", "lk", "rl", "rk", "lean", "head", "crouch")

# expression -> (eyebrow inner raise, mouth shape)
EXPR = {"neutral": (0, "line"), "smile": (0.2, "smile"), "happy": (0.3, "grin"), "smug": (0, "smirk"), "shocked": (1, "O"),
        "worried": (0.8, "frown"), "sad": (0.7, "frown"), "angry": (-0.8, "line"), "squeeze": (-0.9, "tight"),
        "scared": (1, "o"), "laugh": (0.4, "laugh"), "serious": (-0.3, "line")}


def _v(a):
    r = math.radians(a)
    return math.sin(r), math.cos(r)


def _add(p, a, length):
    dx, dy = _v(a)
    return p[0] + dx * length, p[1] + dy * length


def cycle(kind, t, speed=1.0):
    """Run / walk cycle pose (profile, facing right) at time t."""
    if kind == "run":
        p = 2 * math.pi * t / (0.62 / speed)
        s = math.sin(p)
        return {"ll": 38 * s, "lk": 12 + 50 * (1 - s) / 2, "rl": -38 * s, "rk": 12 + 50 * (1 + s) / 2,
                "la": -42 * s, "le": 85, "ra": 42 * s, "re": 85, "lean": 13, "crouch": -0.018 * abs(math.cos(p))}
    p = 2 * math.pi * t / (1.05 / speed)
    s = math.sin(p)
    return {"ll": 22 * s, "lk": 4 + 26 * (1 - s) / 2, "rl": -22 * s, "rk": 4 + 26 * (1 + s) / 2,
            "la": -18 * s, "le": 12, "ra": 18 * s, "re": 12, "lean": 3, "crouch": -0.008 * abs(math.cos(p))}


def extra(seed):
    """A background person: same rig, muted random clothes and hair."""
    rnd = random.Random(seed)
    tops = [(96, 104, 112), (120, 98, 80), (70, 92, 84), (110, 84, 92), (140, 132, 118), (64, 72, 96), (150, 110, 70)]
    hair = rnd.choice([None, "short", "short", "bun", "long"])
    top = rnd.choice(tops)
    return {"top": top, "torso": top, "legs": rnd.choice([(48, 48, 52), (70, 66, 60), (40, 46, 60)]), "shoes": (24, 22, 20),
            "hair": hair, "hair_color": rnd.choice([(40, 30, 24), (88, 60, 38), (30, 30, 30), (120, 100, 70)])}


# ---------------------------------------------------------------- drawing

class _Pen:
    def __init__(self, img, scale, ox, oy):
        self.d = ImageDraw.Draw(img)
        self.s, self.ox, self.oy = scale, ox, oy

    def p(self, pt):
        return (pt[0] * self.s + self.ox, pt[1] * self.s + self.oy)

    def limb(self, pts, width, color, outline=OUTLINE):
        pts = [self.p(q) for q in pts]
        w = width * self.s
        o = max(1.5, 0.006 * self.s)
        for col, ww in ((outline, w + 2 * o), (color, w)):
            self.d.line(pts, fill=col, width=int(ww), joint="curve")
            for q in (pts[0], pts[-1]):
                self.d.ellipse((q[0] - ww / 2, q[1] - ww / 2, q[0] + ww / 2, q[1] + ww / 2), fill=col)

    def circle(self, c, r, fill, outline=OUTLINE, ow=0.006):
        c = self.p(c)
        r *= self.s
        self.d.ellipse((c[0] - r, c[1] - r, c[0] + r, c[1] + r), fill=fill, outline=outline, width=max(1, int(ow * self.s)))

    def ellipse(self, c, rx, ry, fill, outline=OUTLINE, ow=0.005):
        c = self.p(c)
        self.d.ellipse((c[0] - rx * self.s, c[1] - ry * self.s, c[0] + rx * self.s, c[1] + ry * self.s), fill=fill,
                       outline=outline, width=max(1, int(ow * self.s)) if outline else 0)

    def poly(self, pts, fill, outline=OUTLINE, ow=0.006):
        self.d.polygon([self.p(q) for q in pts], fill=fill, outline=outline, width=max(1, int(ow * self.s)))

    def line(self, pts, fill=OUTLINE, w=0.006):
        self.d.line([self.p(q) for q in pts], fill=fill, width=max(1, int(w * self.s)), joint="curve")

    def arc(self, c, rx, ry, a0, a1, fill=OUTLINE, w=0.006):
        c = self.p(c)
        self.d.arc((c[0] - rx * self.s, c[1] - ry * self.s, c[0] + rx * self.s, c[1] + ry * self.s), a0, a1, fill=fill,
                   width=max(1, int(w * self.s)))


def _rgba(c):
    return tuple(c) + (255,) if len(c) == 3 else c


def _shade(c, k):
    return tuple(int(v * k) for v in c[:3]) + (255,)


def _skeleton(pose, view, b):
    """Joint positions with the hip at the origin (y down), for body proportions b."""
    g = lambda k: pose.get(k, 0.0)
    hip = (0.0, 0.0)
    lean = g("lean")
    neck = _add(hip, 180 - lean, b["torso"])
    head = _add(neck, 180 - lean - g("head"), b["neck"] + b["head_r"])
    if view == "front":
        perp = _v(90 - lean)
        sw = b["shoulder_w"]
        sh_l = (neck[0] - perp[0] * sw, neck[1] - perp[1] * sw + 0.02)
        sh_r = (neck[0] + perp[0] * sw, neck[1] + perp[1] * sw + 0.02)
        hp_l, hp_r = (-b["hip_w"], 0.0), (b["hip_w"], 0.0)
    else:
        sh_l = sh_r = _add(neck, -lean, 0.025)
        hp_l = hp_r = hip
    j = {"hip": hip, "neck": neck, "head": head, "sh_l": sh_l, "sh_r": sh_r, "hp_l": hp_l, "hp_r": hp_r}
    for s in "lr":
        a, e = g(s + "a"), g(s + "e")
        j["el_" + s] = _add(j["sh_" + s], a + lean * 0.5, b["uarm"])
        j["ha_" + s] = _add(j["el_" + s], a + e + lean * 0.5, b["farm"])
        th = g(s + "l")
        thigh = b["thigh"] * (0.38 if pose.get("sit") else 1.0)
        j["kn_" + s] = _add(j["hp_" + s], th, thigh)
        j["an_" + s] = _add(j["kn_" + s], th - g(s + "k"), b["shin"])
    return j


def draw(design, pose, view="front", facing="right", expr="neutral", talk=0.0, blink=False, look=0.0, height=500, sweat=False):
    """Render one figure. Returns (RGBA image, (ax, ay), hands): (ax, ay) is the ground point under the feet and hands
    maps "l"/"r" to hand centres, all in image pixels."""
    body = design.get("body", "slim")
    b = BODIES[body]
    classic = body == "classic"
    HEAD_R = b["head_r"]
    fs = HEAD_R / BODIES["slim"]["head_r"]          # face features scale with the head
    j = _skeleton(pose, view, b)
    feet_y = max(j["an_l"][1], j["an_r"][1]) + 0.012
    crouch = pose.get("crouch", 0.0)
    xs = [p[0] for p in j.values()]
    ys = [p[1] for p in j.values()]
    m = 0.16 + (HEAD_R - BODIES["slim"]["head_r"])
    x0, x1, y0, y1 = min(xs) - m, max(xs) + m, min(ys) - m - 0.06, feet_y + 0.03
    scale = height * SS
    w, h = int((x1 - x0) * scale), int((y1 - y0) * scale)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    pen = _Pen(img, scale, -x0 * scale, -y0 * scale)
    top, legs, shoes = _rgba(design["top"]), _rgba(design["legs"]), _rgba(design["shoes"])
    profile = view != "front"
    near, far = ("r", "l")

    sp, sf, sh = b["shoe"]

    def leg(s, k=1.0):
        col = OUTLINE if classic else _shade(legs, k)
        pen.limb([j["hp_" + s], j["kn_" + s], j["an_" + s]], b["leg_w"], col)
        a = j["an_" + s]
        if profile:
            pen.ellipse((a[0] + sp * 0.55, a[1] + 0.004), sp, sh, _shade(shoes, k))
        else:
            pen.ellipse((a[0] + (sf * 0.42 if s == "r" else -sf * 0.42), a[1] + 0.004), sf, sh, _shade(shoes, k))

    def arm(s, k=1.0):
        pen.limb([j["sh_" + s], j["el_" + s], j["ha_" + s]], b["arm_w"], OUTLINE if classic else _shade(top, k))
        pen.circle(j["ha_" + s], b["hand_r"], SKIN, ow=0.006 * (1.3 if classic else 1))

    def torso():
        n, hp, lean = j["neck"], j["hip"], pose.get("lean", 0)
        if classic:  # a soft white (or shirt-coloured) block from the shoulders to the hips
            fill = _rgba(design.get("torso", SKIN))
            hw = (0.095 if not profile else 0.065)
            top_y = n[1] - b["neck"] - 0.004
            bottom = 0.03 + (b["thigh"] * 0.6 if design.get("coat") else 0.0)
            pen.d.rounded_rectangle((*pen.p((min(n[0], hp[0]) - hw, top_y)), *pen.p((max(n[0], hp[0]) + hw * 1.05, hp[1] + bottom))),
                                    radius=0.05 * pen.s, fill=fill, outline=OUTLINE, width=max(1, int(0.008 * pen.s)))
            if design.get("straps"):
                for sx in (-1, 1):
                    pen.line([(n[0] + sx * hw * 0.75, top_y + 0.01), (n[0] + sx * hw * 0.95, hp[1] - 0.06)],
                             fill=_rgba(design["straps"]), w=0.016)
            if design.get("collar"):
                pen.poly([(n[0] - 0.03, top_y + 0.004), (n[0] + 0.03, top_y + 0.004), (n[0], top_y + 0.07)], _rgba(design["collar"]))
            return
        sw, hw = (b["shoulder_w"] + 0.012, b["hip_w"] + 0.022) if not profile else (0.04, 0.045)
        bottom = 0.0
        if design.get("coat"):
            hw, bottom = hw + 0.03, b["thigh"] * 0.75
        px, py = _v(90 - lean)
        pts = [(n[0] - px * sw, n[1] + 0.01), (n[0] + px * sw, n[1] + 0.01),
               (hp[0] + hw, hp[1] + 0.015 + bottom), (hp[0] - hw, hp[1] + 0.015 + bottom)]
        if profile:
            pts = [(n[0] - 0.035, n[1] + 0.01), (n[0] + 0.03, n[1] + 0.01), (hp[0] + 0.045, hp[1] + 0.015 + bottom),
                   (hp[0] - 0.045, hp[1] + 0.015 + bottom)]
        pen.poly(pts, top)
        if design.get("collar"):
            pen.poly([(n[0] - 0.02, n[1] + 0.008), (n[0] + 0.02, n[1] + 0.008), (n[0], n[1] + 0.05)], _rgba(design["collar"]))

    def scarf():
        n = j["neck"]
        c = _rgba(design["scarf"])
        if classic:  # Episode 1 scarf: a band at the neck and one pointed tie hanging down the chest
            y = n[1] - b["neck"] + 0.012
            pen.ellipse((n[0], y), 0.075 if not profile else 0.055, 0.03, c)
            sway = 0.006 * math.sin(pose.get("_t", 0) * 2.3)
            pen.poly([(n[0] - 0.022, y + 0.02), (n[0] + 0.022, y + 0.02), (n[0] + 0.02 + sway, y + 0.1), (n[0] + sway, y + 0.125),
                      (n[0] - 0.02 + sway, y + 0.1)], c)
            return
        pen.ellipse((n[0], n[1] + 0.012), 0.05 if not profile else 0.04, 0.022, c)
        sway = 0.012 * math.sin(pose.get("_t", 0) * 2.3)
        tx = n[0] + (0.018 if not profile else -0.02)
        for dx in (0.0, 0.02):
            pen.poly([(tx + dx - 0.012, n[1] + 0.02), (tx + dx + 0.012, n[1] + 0.02),
                      (tx + dx + 0.013 + sway, n[1] + 0.13), (tx + dx - 0.011 + sway, n[1] + 0.13)], c)

    def head():
        hc = j["head"]
        lk = look if not profile else 0.55
        hair = design.get("hair")
        hcol = _rgba(design.get("hair_color", (30, 30, 30)))
        if hair == "ponytail":  # a rounded teardrop from the top of the head, swinging a little
            side = -1 if profile else 1
            sway = 0.05 * HEAD_R * math.sin(pose.get("_t", 0) * 2.1)
            pts = []
            for i in range(25):
                a = math.pi * i / 24
                pts.append((hc[0] + side * HEAD_R * (0.95 + 0.38 * math.sin(a)) + side * sway * math.sin(a),
                            hc[1] - HEAD_R * 0.7 + HEAD_R * 1.25 * (1 - math.cos(a)) / 2 + HEAD_R * 0.35 * math.sin(a) ** 4))
            pts += [(hc[0] + side * HEAD_R * 0.8, hc[1] + HEAD_R * 0.3), (hc[0] + side * HEAD_R * 0.7, hc[1] - HEAD_R * 0.6)]
            pen.poly(pts, hcol)
        pen.circle(hc, HEAD_R, SKIN)
        if hair == "ponytail":
            pen.d.chord(_box(pen, hc, HEAD_R * 1.04), 190, 350, fill=hcol, outline=OUTLINE, width=max(1, int(0.006 * pen.s)))
            pen.poly([(hc[0] - HEAD_R * 0.75, hc[1] - HEAD_R * 0.55), (hc[0] + HEAD_R * 0.75, hc[1] - HEAD_R * 0.55),
                      (hc[0] + HEAD_R * 0.6, hc[1] - HEAD_R * 0.32), (hc[0] - HEAD_R * 0.6, hc[1] - HEAD_R * 0.32)], hcol, outline=None)
        if hair in ("short", "bun", "long"):
            pen.d.chord(_box(pen, hc, HEAD_R * 1.04), 185, 355, fill=hcol, outline=OUTLINE, width=max(1, int(0.006 * pen.s)))
            if hair == "bun":
                pen.circle((hc[0] - (0.03 * fs if profile else 0), hc[1] - HEAD_R * 1.1), 0.032 * fs, hcol)
            if hair == "long":
                for sx in ((-1, 1) if not profile else (-1,)):
                    pen.poly([(hc[0] + sx * HEAD_R * 0.95, hc[1] - 0.01), (hc[0] + sx * HEAD_R * 1.05, hc[1] - 0.01),
                              (hc[0] + sx * HEAD_R * 1.0, hc[1] + 0.1), (hc[0] + sx * HEAD_R * 0.8, hc[1] + 0.1)], hcol)
        bx, by = hc[0] + lk * 0.022 * fs, hc[1]
        raise_, mouth = EXPR[expr]
        big = expr in ("shocked", "scared")
        lw = 0.0055 * (fs ** 0.6)            # line weights grow slower than the head, like a hand-drawn face
        for i, sx in enumerate((-1, 1)):
            ex = bx + sx * 0.031 * fs * (0.75 if profile else 1)
            if profile and sx == -1:
                ex += 0.012 * fs
            ey = by - 0.006 * fs
            if expr == "squeeze":
                q = 0.012 * fs
                pen.line([(ex - q, ey - q / 2), (ex + q / 3 * sx, ey), (ex - q, ey + q / 2)] if sx < 0 else
                         [(ex + q, ey - q / 2), (ex - q / 3, ey), (ex + q, ey + q / 2)], w=lw * 1.1)
            elif blink:
                pen.line([(ex - 0.009 * fs, ey), (ex + 0.009 * fs, ey)], w=lw)
            else:
                pen.circle((ex, ey), (0.0125 if big else 0.0098) * fs, OUTLINE, ow=0.001)
                pen.circle((ex + 0.0035 * fs, ey - 0.0035 * fs), 0.003 * fs, SKIN, outline=None, ow=0.001)
            # eyebrow: inner end raised by `raise_`
            inner, outer = (ex + 0.011 * fs * -sx, ex + 0.011 * fs * sx)
            byy = ey - (0.024 + (0.007 if big else 0)) * fs
            if expr == "smug" and sx == 1:
                byy -= 0.008 * fs
            pen.line([(outer, byy), (inner, byy - 0.008 * fs * raise_)], w=lw)
            if design.get("glasses"):
                pen.circle((ex, ey), 0.019 * fs, None, ow=0.0045 * fs ** 0.6)
        if design.get("glasses"):
            pen.line([(bx - 0.012 * fs, by - 0.006 * fs), (bx + 0.012 * fs, by - 0.006 * fs)], w=0.004 * fs ** 0.6)
        mx, my = bx, by + 0.033 * fs
        open_ = max(talk, {"O": 0.75, "o": 0.4, "laugh": 0.6, "grin": 0.0}.get(mouth, 0.0))
        if open_ > 0.12:
            wdt = (0.02 if mouth in ("O", "o") else 0.026) * fs
            pen.ellipse((mx, my + 0.002 * fs), wdt, (0.004 + 0.019 * open_) * fs, MOUTH_IN, ow=0.005 * fs ** 0.6)
        elif mouth == "smile":
            pen.arc((mx, my - 0.012 * fs), 0.022 * fs, 0.016 * fs, 25, 155, w=lw)
        elif mouth == "grin":
            pen.d.chord(_box(pen, (mx, my - 0.008 * fs), 0.024 * fs), 0, 180, fill=MOUTH_IN, outline=OUTLINE,
                        width=max(1, int(0.005 * fs ** 0.6 * pen.s)))
        elif mouth == "smirk":
            pen.line([(mx - 0.018 * fs, my + 0.002 * fs), (mx + 0.008 * fs, my + 0.002 * fs), (mx + 0.02 * fs, my - 0.008 * fs)], w=lw)
        elif mouth == "frown":
            pen.arc((mx, my + 0.014 * fs), 0.018 * fs, 0.012 * fs, 205, 335, w=lw)
        elif mouth == "tight":
            pen.line([(mx - 0.014 * fs, my), (mx - 0.005 * fs, my - 0.003 * fs), (mx + 0.005 * fs, my + 0.003 * fs), (mx + 0.014 * fs, my)], w=lw)
        else:
            pen.line([(mx - 0.013 * fs, my), (mx + 0.013 * fs, my)], w=lw)
        if design.get("cap"):
            cap = _rgba(design["cap"])
            a0, a1 = (204, 336) if classic else (197, 343)        # the big classic head keeps its eyebrows visible
            pen.d.chord(_box(pen, (hc[0], hc[1] - 0.05 * HEAD_R), HEAD_R * 1.07), a0, a1, fill=cap, outline=OUTLINE,
                        width=max(1, int(0.006 * pen.s)))
            by_ = hc[1] - (0.56 if classic else 0.5) * HEAD_R
            reach = 1.35 if classic else 1.7
            if profile:
                pen.poly([(hc[0] + 0.02 * fs, by_ - 0.006 * fs), (hc[0] + HEAD_R * reach, by_ + 0.004 * fs),
                          (hc[0] + HEAD_R * (reach - 0.05), by_ + 0.013 * fs), (hc[0] + 0.02 * fs, by_ + 0.008 * fs)], _shade(cap, 0.85))
            else:
                pen.ellipse((hc[0] + lk * 0.02 * fs, by_ + 0.002 * fs), HEAD_R * 0.98, 0.011 * fs, _shade(cap, 0.85))
        if design.get("hat"):
            hat = _rgba(design["hat"])
            pen.poly([(hc[0] - HEAD_R * 0.8, hc[1] - HEAD_R * 0.45), (hc[0] + HEAD_R * 0.8, hc[1] - HEAD_R * 0.45),
                      (hc[0] + HEAD_R * 0.62, hc[1] - HEAD_R * 1.35), (hc[0] - HEAD_R * 0.62, hc[1] - HEAD_R * 1.35)], hat)
            pen.ellipse((hc[0], hc[1] - HEAD_R * 0.45), HEAD_R * 1.55, 0.018 * fs, _shade(hat, 0.8))
            shade = Image.new("RGBA", img.size, (0, 0, 0, 0))
            sd = ImageDraw.Draw(shade)
            c = pen.p(hc)
            r = HEAD_R * pen.s
            sd.chord((c[0] - r, c[1] - r, c[0] + r, c[1] + r), 180, 360, fill=(0, 0, 0, 150))
            img.alpha_composite(Image.composite(shade, Image.new("RGBA", img.size, (0, 0, 0, 0)), _head_mask(img.size, c, r)))
        if sweat:
            sx, sy = hc[0] + HEAD_R * 0.9, hc[1] - HEAD_R * 0.5
            pen.poly([(sx, sy - 0.02 * fs), (sx + 0.011 * fs, sy + 0.004 * fs), (sx - 0.011 * fs, sy + 0.004 * fs)], (150, 200, 235, 255), ow=0.003)
            pen.circle((sx, sy + 0.006 * fs), 0.011 * fs, (150, 200, 235, 255), ow=0.003)

    def skirt():
        hp = j["hip"]
        if classic:  # Episode 1 skirt: a short band at the bottom of the torso
            hw = 0.1 if not profile else 0.07
            pen.d.rounded_rectangle((*pen.p((hp[0] - hw, hp[1] - 0.035)), *pen.p((hp[0] + hw * 1.05, hp[1] + 0.05))),
                                    radius=0.015 * pen.s, fill=_rgba(design["skirt"]), outline=OUTLINE, width=max(1, int(0.008 * pen.s)))
            return
        tw, bw = (0.05, 0.088) if not profile else (0.045, 0.07)
        pen.poly([(hp[0] - tw, hp[1] - 0.02), (hp[0] + tw, hp[1] - 0.02), (hp[0] + bw, hp[1] + 0.12), (hp[0] - bw, hp[1] + 0.12)],
                 _rgba(design["skirt"]))

    if profile:
        arm(far, 0.72)
        leg(far, 0.72)
        torso()
        leg(near)
        if design.get("skirt"):
            skirt()
        if design.get("scarf"):
            scarf()
        head()
        arm(near)
    else:
        leg("l")
        leg("r")
        torso()
        if design.get("skirt"):
            skirt()
        if design.get("scarf"):
            scarf()
        arm("l")
        arm("r")
        head()

    img = img.resize((max(1, w // SS), max(1, h // SS)), Image.LANCZOS)
    ax, ay = -x0 * height, (feet_y - y0) * height + crouch * height
    hands = {s: ((j["ha_" + s][0] - x0) * height, (j["ha_" + s][1] - y0) * height) for s in "lr"}
    hands["head"] = ((j["head"][0] - x0) * height, (j["head"][1] - y0) * height)
    if facing == "left" and profile:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        ax = img.width - ax
        hands = {s: (img.width - x, y) for s, (x, y) in hands.items()}
    return img, (ax, ay), hands


def _box(pen, c, r):
    c = pen.p(c)
    r *= pen.s
    return (c[0] - r, c[1] - r, c[0] + r, c[1] + r)


def _head_mask(size, c, r):
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).ellipse((c[0] - r, c[1] - r, c[0] + r, c[1] + r), fill=255)
    return m


def contact_shadow(width):
    """Soft ellipse to put under a figure's feet."""
    w, h = max(8, int(width)), max(4, int(width * 0.18))
    s = Image.new("L", (w * 2, h * 4), 0)
    ImageDraw.Draw(s).ellipse((w * 0.5, h * 1.5, w * 1.5, h * 2.5), fill=120)
    return s.filter(ImageFilter.GaussianBlur(h * 0.6))


# ---------------------------------------------------------------- acting

def _smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def _pose_of(act, local_t):
    """(pose dict, view) of one act at local time."""
    do = act.get("do", "stand")
    if do in ("run", "walk"):
        return cycle(do, local_t, act.get("speed", 1.0)), "profile"
    name = act.get("pose", "idle_p" if act.get("view") == "profile" else "idle")
    view = act.get("view", "profile" if name.endswith("_p") else "front")
    return dict(POSES[name]), view


def actor_state(actor, t, dur, seed=0):
    """Evaluate an actor spec at shot time t: position, pose, view, facing, expression, blink."""
    acts = actor.get("acts", [{"at": 0}])
    starts = [a["at"] * dur for a in acts]
    i = max(k for k, s in enumerate(starts) if s <= t + 1e-6) if t >= starts[0] else 0
    act, t_act = acts[i], starts[i]
    # position: walk/run acts move x from where the previous act left it to act["to_x"]
    x = actor["x"]
    for k in range(i + 1):
        a = acts[k]
        end = starts[k + 1] if k + 1 < len(acts) else dur
        if "to_x" in a:
            f = 1.0 if k < i else (t - starts[k]) / max(1e-3, end - starts[k])
            f = f if a.get("do") in ("run", "walk") else _smooth(f)
            x = x + (a["to_x"] - x) * min(1.0, max(0.0, f))
    pose, view = _pose_of(act, t - t_act)
    if i > 0 and t - t_act < 0.22:
        prev, pview = _pose_of(acts[i - 1], t_act - starts[i - 1])
        if pview == view:
            f = _smooth((t - t_act) / 0.22)
            sit = pose.get("sit", False)
            pose = {k: prev.get(k, 0) + (pose.get(k, 0) - prev.get(k, 0)) * f for k in POSE_KEYS}
            pose["sit"] = sit
    if act.get("do") not in ("run", "walk"):
        breath = math.sin(2 * math.pi * (t + seed * 0.37) / 3.2)
        pose["crouch"] = pose.get("crouch", 0) + 0.003 * breath
        pose["head"] = pose.get("head", 0) + 1.5 * breath
        if act.get("bounce"):
            pose["crouch"] = pose.get("crouch", 0) + 0.012 * abs(math.sin(t * 13))
            pose["head"] = pose.get("head", 0) - 6 * abs(math.sin(t * 13))
        if act.get("shake"):
            pose["la"] = pose.get("la", 0) + 3 * math.sin(t * 40)
            pose["ra"] = pose.get("ra", 0) + 3 * math.sin(t * 43)
    pose["_t"] = t
    pose["_phone"] = act.get("pose") in ("phone", "catch")
    pose["lie"] = act.get("pose") == "lie"
    facing = act.get("face", actor.get("face", "right"))
    blink = ((t + seed * 1.13) % 3.6) < 0.11 and act.get("expr") != "squeeze"
    return {"x": x, "y": actor["y"], "h": actor["h"], "pose": pose, "view": view, "facing": facing,
            "expr": act.get("expr", actor.get("expr", "neutral")), "look": act.get("look", 0.0), "blink": blink,
            "sweat": act.get("sweat", False),
            "yap": 0.45 + 0.45 * math.sin(t * 17) * math.sin(t * 5.3) if act.get("yap") else 0.0}

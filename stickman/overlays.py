"""On-screen graphics that make a video's idea visible: badges over heads (battery %, price tags), corner HUD widgets,
note banners, motion-graphic scenes (big number, chart, versus, timeline), comic impact bursts and a passing car.
All drawn with Pillow straight onto the frame, so they are crisp, consistent and cost almost nothing to render."""
import math

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_BOLD = r"C:\Windows\Fonts\segoeuib.ttf"
INK = (14, 14, 18)
WHITE = (250, 250, 250)
YELLOW = (255, 214, 64)
_fonts = {}


def font(size):
    size = max(8, int(size))
    if size not in _fonts:
        _fonts[size] = ImageFont.truetype(FONT_BOLD, size)
    return _fonts[size]


def _ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def value_at(spec, u):
    """Animated value of a badge/widget: 'from' -> 'to' across the middle of the shot."""
    a, b = spec.get("from"), spec.get("to", spec.get("from"))
    if a is None:
        return None
    return a + (b - a) * _ease((u - 0.15) / 0.6)


def battery_color(v):
    return (80, 200, 255) if v > 100 else (70, 210, 90) if v > 50 else (250, 196, 40) if v > 20 else (235, 60, 50)


def _battery(d, x, y, w, v, t):
    """Battery icon with fill and percentage, top-left at (x, y), width w."""
    h = w * 0.46
    blink = v <= 5 and int(t * 3) % 2 == 0
    col = battery_color(v)
    lw = max(2, int(w * 0.05))
    d.rounded_rectangle((x, y, x + w, y + h), radius=h * 0.22, fill=(20, 20, 24), outline=WHITE, width=lw)
    d.rounded_rectangle((x + w, y + h * 0.3, x + w * 1.07, y + h * 0.7), radius=lw, fill=WHITE)
    if not blink:
        pad = lw * 1.8
        fw = max(0.0, (w - 2 * pad) * min(v, 100) / 100)
        if fw > 1:
            d.rounded_rectangle((x + pad, y + pad, x + pad + fw, y + h - pad), radius=h * 0.12, fill=col)
    d.text((x + w / 2, y + h / 2), f"{round(v)}%", font=font(h * 0.62), anchor="mm", fill=WHITE, stroke_width=max(2, int(h * 0.08)),
           stroke_fill=INK)
    return h


def badge(frame, spec, head_xy, head_r, u, t, age):
    """A badge floating above a character's head. spec: {"kind": "battery"|"tag", "from", "to", "text"}."""
    d = ImageDraw.Draw(frame)
    k = 1 + 0.4 * max(0.0, 1 - age / 0.15)          # pop-in when it first appears
    bob = math.sin(t * 2.4) * head_r * 0.06
    cx, top = head_xy[0], head_xy[1] - head_r * (1.55 + 0.1) + bob
    if spec["kind"] == "battery":
        w = head_r * 1.9 * k
        h = w * 0.46
        _battery(d, cx - w / 2, top - h, w, value_at(spec, u), t)
    else:
        f = font(head_r * 0.62 * k)
        text = spec["text"]
        tw = f.getlength(text)
        h = head_r * 0.85 * k
        d.rounded_rectangle((cx - tw / 2 - h * 0.35, top - h, cx + tw / 2 + h * 0.35, top), radius=h * 0.3,
                            fill=spec.get("color", (20, 20, 24)), outline=WHITE, width=max(2, int(h * 0.07)))
        d.text((cx, top - h / 2), text, font=f, anchor="mm", fill=spec.get("ink", YELLOW))


def hud(frame, widgets, u, t):
    """Corner widgets, stacked top-left: [{"label", "kind": "battery"|"text", "from", "to", "text"}]."""
    d = ImageDraw.Draw(frame)
    y = 46
    for w in widgets:
        lf = font(30)
        d.text((60, y), w["label"], font=lf, fill=WHITE, stroke_width=4, stroke_fill=INK)
        y += 40
        if w["kind"] == "battery":
            y += _battery(d, 60, y, 150, value_at(w, u), t) + 24
        else:
            d.text((60, y), w["text"], font=font(54), fill=YELLOW, stroke_width=5, stroke_fill=INK)
            y += 74


def note(frame, text, age):
    """Banner chip in the lower-left, above the captions (e.g. 'DAY 21', 'RULE 1: ...'), sliding in as it appears.
    Heads and their badges sit high in the frame, so down here it never covers them. 'LABEL: text' gets two lines."""
    d = ImageDraw.Draw(frame)
    head, sep, body = text.partition(": ")
    lines = [(head, font(30), YELLOW), (body, font(40), WHITE)] if sep and len(text) > 18 else [(text, font(42), WHITE)]
    tw = max(f.getlength(t) for t, f, _ in lines)
    h = 30 + sum(f.size * 1.2 for _, f, _ in lines)
    slide = 80 * (1 - _ease(age / 0.25))
    x0, y1 = 50 - slide, frame.height * 0.8
    d.rounded_rectangle((x0, y1 - h, x0 + tw + 56, y1), radius=16, fill=(16, 16, 20), outline=YELLOW, width=4)
    y = y1 - h + 15
    for t, f, col in lines:
        d.text((x0 + 28, y), t, font=f, fill=col)
        y += f.size * 1.2


def burst(frame, text, age):
    """Comic impact burst: radial lines, a jagged star and a word ('CRASH!', 'BEEP!')."""
    if age > 0.9:
        return
    W, H = frame.size
    k = 1 + 0.5 * max(0.0, 1 - age / 0.12)
    alpha = 1.0 if age < 0.6 else 1 - (age - 0.6) / 0.3
    layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, cy = W / 2, H / 2
    for i in range(28):
        a = i / 28 * 2 * math.pi
        r0, r1 = H * 0.32 * k, H * (0.55 + 0.08 * (i % 3)) * k
        d.line((cx + math.cos(a) * r0, cy + math.sin(a) * r0, cx + math.cos(a) * r1, cy + math.sin(a) * r1),
               fill=(255, 255, 255, int(220 * alpha)), width=6)
    pts = []
    for i in range(24):
        a = i / 24 * 2 * math.pi
        r = H * (0.27 if i % 2 == 0 else 0.17) * k
        pts.append((cx + math.cos(a) * r * 1.35, cy + math.sin(a) * r))
    d.polygon(pts, fill=(235, 70, 45, int(255 * alpha)), outline=(255, 255, 255, int(255 * alpha)), width=6)
    d.text((cx, cy), text, font=font(H * 0.11 * k), anchor="mm", fill=(255, 255, 255, int(255 * alpha)), stroke_width=8,
           stroke_fill=(14, 14, 18, int(255 * alpha)))
    frame.paste(layer, (0, 0), layer)


def car(frame, spec, u, floor_y, scale):
    """A simple dark car silhouette with headlights crossing the frame between spec['from'] and spec['to'] (fractions)."""
    a, b = spec.get("from", 0.2), spec.get("to", 0.6)
    if not a <= u <= b:
        return
    f = (u - a) / (b - a)
    W = frame.width
    cw, ch = 0.5 * scale, 0.17 * scale
    x = (-cw + (W + 2 * cw) * f) if spec.get("dir", "right") == "right" else (W + cw - (W + 2 * cw) * f)
    y = floor_y
    layer = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.ellipse((x - cw * 0.55, y - ch * 0.1, x + cw * 0.55, y + ch * 0.12), fill=(0, 0, 0, 90))
    d.rounded_rectangle((x - cw / 2, y - ch * 0.75, x + cw / 2, y - ch * 0.2), radius=ch * 0.18, fill=(26, 28, 34, 255))
    d.rounded_rectangle((x - cw * 0.28, y - ch * 1.15, x + cw * 0.22, y - ch * 0.7), radius=ch * 0.2, fill=(26, 28, 34, 255))
    d.rectangle((x - cw * 0.22, y - ch * 1.05, x + cw * 0.16, y - ch * 0.76), fill=(90, 110, 130, 255))
    for wx in (-0.3, 0.3):
        d.ellipse((x + cw * wx - ch * 0.24, y - ch * 0.42, x + cw * wx + ch * 0.24, y + ch * 0.06), fill=(10, 10, 12, 255))
    front = 1 if spec.get("dir", "right") == "right" else -1
    hx = x + front * cw * 0.47
    d.ellipse((hx - ch * 0.1, y - ch * 0.62, hx + ch * 0.1, y - ch * 0.44), fill=(255, 246, 200, 255))
    glow = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    ImageDraw.Draw(glow).polygon([(hx, y - ch * 0.53), (hx + front * cw * 1.6, y - ch * 1.1), (hx + front * cw * 1.6, y + ch * 0.1)],
                                 fill=(255, 240, 190, 70))
    layer = Image.alpha_composite(glow.filter(ImageFilter.GaussianBlur(ch * 0.3)), layer)
    frame.paste(layer, (0, 0), layer)


def graphic(frame, spec, u, t):
    """Motion-graphic scene drawn over a darkened, blurred plate.
    number: big count-up ("4,106", label) / chart: line going up or down / versus: two big values / timeline: dots with labels."""
    W, H = frame.size
    d = ImageDraw.Draw(frame)
    p = _ease(u / 0.7)
    kind, args = spec["type"], spec["args"]
    label = spec.get("label", "")
    if kind == "number":
        raw = args[0]
        shown = _count(raw, p)
        d.text((W / 2, H * 0.46), shown, font=font(H * 0.2), anchor="mm", fill=YELLOW, stroke_width=10, stroke_fill=INK)
        if label:
            d.text((W / 2, H * 0.64), label, font=font(H * 0.06), anchor="mm", fill=WHITE, stroke_width=6, stroke_fill=INK)
    elif kind == "chart":
        up = args[0] != "down"
        x0, x1, y0, y1 = W * 0.18, W * 0.82, H * 0.72, H * 0.28
        d.line((x0, y0, x1, y0), fill=WHITE, width=5)
        d.line((x0, y0, x0, y1), fill=WHITE, width=5)
        pts = []
        for i in range(41):
            f = i / 40
            if f > p:
                break
            v = f ** 1.6 if up else 1 - f ** 0.7
            v += 0.03 * math.sin(f * 23)
            pts.append((x0 + (x1 - x0) * f, y0 - (y0 - y1) * max(0.0, min(1.0, v))))
        if len(pts) > 1:
            d.line(pts, fill=(70, 210, 90) if up else (235, 60, 50), width=12, joint="curve")
            ex, ey = pts[-1]
            d.ellipse((ex - 14, ey - 14, ex + 14, ey + 14), fill=WHITE)
        if label:
            d.text((W / 2, H * 0.18), label, font=font(H * 0.06), anchor="mm", fill=WHITE, stroke_width=6, stroke_fill=INK)
    elif kind == "versus":
        a, b = (args + ["", ""])[:2]
        d.line((W / 2, H * 0.22, W / 2, H * 0.78), fill=WHITE, width=6)
        for i, (txt, cx) in enumerate(((a, W * 0.27), (b, W * 0.73))):
            if p > i * 0.35:
                d.text((cx, H * 0.47), txt, font=font(H * 0.14), anchor="mm", fill=YELLOW if i else WHITE, stroke_width=9, stroke_fill=INK)
        d.text((W / 2, H * 0.47), "VS", font=font(H * 0.06), anchor="mm", fill=INK, stroke_width=5, stroke_fill=WHITE)
        if label:
            d.text((W / 2, H * 0.72), label, font=font(H * 0.055), anchor="mm", fill=WHITE, stroke_width=6, stroke_fill=INK)
    elif kind == "timeline":
        n = len(args)
        y = H * 0.52
        x0, x1 = W * 0.14, W * 0.86
        d.line((x0, y, x0 + (x1 - x0) * p, y), fill=WHITE, width=6)
        for i, txt in enumerate(args):
            x = x0 + (x1 - x0) * (i / max(1, n - 1))
            if p >= i / max(1, n - 1) - 1e-3:
                d.ellipse((x - 18, y - 18, x + 18, y + 18), fill=YELLOW, outline=INK, width=4)
                d.text((x, y + 60), txt, font=font(H * 0.045), anchor="mm", fill=WHITE, stroke_width=5, stroke_fill=INK)
        if label:
            d.text((W / 2, H * 0.3), label, font=font(H * 0.06), anchor="mm", fill=WHITE, stroke_width=6, stroke_fill=INK)


def _count(raw, p):
    """'$8,742,619,300' at progress p -> the same text counting up from zero. Decimals ('$2.2 MILLION') stay as written."""
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits or "." in raw:
        return raw
    n = round(int(digits) * p)
    i = raw.index(digits[0])
    j = len(raw) - raw[::-1].index(digits[-1])
    return raw[:i] + (f"{n:,}" if "," in raw else str(n)) + raw[j:]


def graphic_backdrop(plate_frame):
    """Darken and blur the scene so the graphic reads on top of it."""
    small = plate_frame.resize((plate_frame.width // 8, plate_frame.height // 8)).filter(ImageFilter.GaussianBlur(3))
    return Image.eval(small.resize(plate_frame.size, Image.BILINEAR), lambda v: int(v * 0.32))

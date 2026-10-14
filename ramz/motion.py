"""Bespoke motion graphics for SIR RAMZ, rendered frame-by-frame with Pillow.

Every graphic is drawn at 2x and downsampled (clean anti-aliasing), uses
eased entrances/exits, and is written as an RGBA PNG sequence that the
renderer overlays on the final timeline.

Graphic types (all text comes from the plan — nothing is invented here):
  title      — brand line + two-line mask-reveal headline + light sweep (hook)
  strike     — "lead text" + big word that gets struck through
  diagram    — schematic fake-out: level, price path breaking out and failing.
               Always tagged ILUSTRASI so it is never mistaken for market data.
  compare    — FROM chip (struck) -> arrow -> TO chip, optional footnote
  statement  — two-line statement, second line large + light sweep
  keyword    — compact single-phrase panel
  endcard    — brand line + question + call to action
  chips      — heading + row of chips that pop in as each is named, optional note
  checklist  — heading + 2x2 grid of items; each ticks when it is spoken
"""
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SS = 2  # supersampling factor
FONT_DIR = "/usr/share/fonts/opentype/inter"
FONTS = {
    "display": "InterDisplay-ExtraBold.otf",
    "display_bold": "InterDisplay-Bold.otf",
    "semibold": "Inter-SemiBold.otf",
    "medium": "Inter-Medium.otf",
    "bold": "Inter-Bold.otf",
}
FALLBACK = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

_font_cache = {}


def font(kind, size):
    key = (kind, size)
    if key not in _font_cache:
        path = os.path.join(FONT_DIR, FONTS.get(kind, kind))
        _font_cache[key] = ImageFont.truetype(path if os.path.exists(path) else FALLBACK, int(size * SS))
    return _font_cache[key]


def rgb(hexstr, a=255):
    h = hexstr.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), a)


# ----------------------------------------------------------------- easing
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def prog(t, start, dur):
    return clamp((t - start) / max(dur, 1e-6))


def ease_out(p):
    return 1 - (1 - p) ** 3


def ease_in_out(p):
    return 4 * p ** 3 if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2


def back_out(p, s=1.6):
    p -= 1
    return p * p * ((s + 1) * p + s) + 1


def envelope(t, dur, enter=0.35, exit_=0.3):
    """Global in/out: (opacity, y-offset px) for the whole graphic."""
    a = 1.0 if enter <= 0 else ease_out(prog(t, 0, enter))
    b = 1 - ease_in_out(prog(t, dur - exit_, exit_))
    k = min(a, b)
    off = (1 - a) * 18 - (1 - b) * 10
    return k, off


# ------------------------------------------------------------- primitives
def text_img(text, fnt, fill, tracking=0):
    """Tight RGBA image of a single text line (optional letter tracking)."""
    if tracking:
        widths = [fnt.getlength(c) for c in text]
        w = int(sum(widths) + tracking * SS * (len(text) - 1)) + 4
    else:
        w = int(fnt.getlength(text)) + 4
    asc, desc = fnt.getmetrics()
    img = Image.new("RGBA", (max(w, 1), asc + desc + 4), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if tracking:
        x = 0
        for c, cw in zip(text, widths):
            d.text((x, 0), c, font=fnt, fill=fill)
            x += cw + tracking * SS
    else:
        d.text((0, 0), text, font=fnt, fill=fill)
    return img


def fade(img, k):
    if k >= 0.999:
        return img
    out = img.copy()
    a = np.asarray(out.getchannel("A"), dtype=np.float32) * k
    out.putalpha(Image.fromarray(a.astype(np.uint8)))
    return out


def reveal(img, p):
    """Mask reveal: line slides up from below its own baseline."""
    if p <= 0:
        return None
    e = ease_out(p)
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.paste(img, (0, int((1 - e) * img.size[1] * 0.9)))
    return fade(out, clamp(p * 1.6))


def sweep(img, p, width=0.18, strength=0.85):
    """Diagonal light sweep across the opaque pixels of img."""
    if p <= 0 or p >= 1:
        return img
    w, h = img.size
    arr = np.asarray(img, dtype=np.float32).copy()
    xs = np.arange(w)[None, :]
    ys = np.arange(h)[:, None]
    band = w * width
    cx = -band + (w + 2 * band + h * 0.35) * p
    g = np.exp(-(((xs + ys * 0.35) - cx) / band) ** 2) * strength
    a = arr[..., 3:4] / 255.0
    arr[..., :3] = arr[..., :3] + (255 - arr[..., :3]) * (g[..., None] * a)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def panel(size, radius=30, fill=(9, 11, 15, 222), border=(217, 181, 106, 80)):
    """Dark glass panel with soft drop shadow (returned with padding)."""
    w, h = size
    pad = 40 * SS
    img = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(sh).rounded_rectangle((pad, pad + 10 * SS, pad + w, pad + h + 10 * SS), radius * SS,
                                         fill=(0, 0, 0, 90))
    img = Image.alpha_composite(img, sh.filter(ImageFilter.GaussianBlur(18 * SS)))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((pad, pad, pad + w, pad + h), radius * SS, fill=fill, outline=border, width=2 * SS)
    return img, pad


class Frame:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.img = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))

    def paste(self, layer, x, y, k=1.0):
        if layer is None or k <= 0:
            return
        layer = fade(layer, k)
        self.img.alpha_composite(layer, (int(x), int(y)))

    def draw(self):
        return ImageDraw.Draw(self.img)

    def final(self, opacity=1.0, yoff=0.0):
        out = self.img.resize((self.w, self.h), Image.LANCZOS)
        if yoff:
            shifted = Image.new("RGBA", out.size, (0, 0, 0, 0))
            shifted.alpha_composite(out, (0, int(round(yoff))))
            out = shifted
        return fade(out, opacity)


# ------------------------------------------------------------- graphics
class Graphic:
    """Base: region is (x, y, w, h) on the 1080x1920 canvas."""
    HEIGHT = 400

    def __init__(self, params, dur, brand):
        self.p = params
        self.dur = dur
        self.gold = brand.get("gold", "#D9B56A")
        self.white = brand.get("white", "#F5F4EF")
        self.alert = brand.get("alert", "#E25D4F")
        self.region = tuple(params.get("region", (60, 130, 960, self.HEIGHT)))
        self._static = {}

    def brand_line(self, f, cx, y, t, label="SIR RAMZ", at=0.0):
        p = prog(t, at, 0.5)
        if p <= 0:
            return y
        lbl = text_img(label, font("bold", 24), rgb(self.gold), tracking=7)
        e = ease_out(p)
        rule = int(70 * SS * e)
        d = f.draw()
        ly = y + lbl.size[1] // 2
        gap = 22 * SS
        d.line((cx - lbl.size[0] // 2 - gap - rule, ly, cx - lbl.size[0] // 2 - gap, ly), fill=rgb(self.gold, 200), width=2 * SS)
        d.line((cx + lbl.size[0] // 2 + gap, ly, cx + lbl.size[0] // 2 + gap + rule, ly), fill=rgb(self.gold, 200), width=2 * SS)
        f.paste(lbl, cx - lbl.size[0] // 2, y, k=e)
        return y + lbl.size[1] + 18 * SS

    def panel_layer(self, w, h):
        key = ("panel", w, h)
        if key not in self._static:
            self._static[key] = panel((w, h))
        return self._static[key]

    def frame(self, t):
        raise NotImplementedError


class Title(Graphic):
    HEIGHT = 372

    def frame(self, t):
        x, y, w, h = self.region
        f = Frame(w, h)
        # opening frame doubles as the thumbnail: panel is there from frame 0
        k, off = envelope(t, self.dur, enter=0)
        pw, ph = w * SS - 40 * SS, h * SS - 40 * SS
        pnl, pad = self.panel_layer(pw, ph)
        f.paste(pnl, 20 * SS - pad, 20 * SS - pad)
        cx = w * SS // 2
        yy = self.brand_line(f, cx, 58 * SS, t, at=0.0)
        l1 = text_img(self.p["line1"], font("semibold", self.p.get("size1", 56)), rgb(self.white))
        l2 = text_img(self.p["line2"], font("display", self.p.get("size2", 118)), rgb(self.gold))
        l2 = sweep(l2, prog(t, 0.45, 0.75))  # headline is complete on frame 0 (thumbnail); the sweep is the motion
        f.paste(reveal(l1, prog(t, -0.45, 0.45)), cx - l1.size[0] // 2, yy)
        f.paste(reveal(l2, prog(t, -0.38, 0.45)), cx - l2.size[0] // 2, yy + l1.size[1] + 2 * SS)
        return f.final(k, off)


class Strike(Graphic):
    HEIGHT = 300

    def frame(self, t):
        x, y, w, h = self.region
        f = Frame(w, h)
        k, off = envelope(t, self.dur)
        pw, ph = w * SS - 40 * SS, h * SS - 40 * SS
        pnl, pad = self.panel_layer(pw, ph)
        f.paste(pnl, 20 * SS - pad, 20 * SS - pad)
        cx = w * SS // 2
        lead = text_img(self.p["lead"], font("semibold", 54), rgb(self.white))
        word = text_img(self.p["word"], font("display", 128), rgb(self.gold))
        top = 70 * SS
        f.paste(reveal(lead, prog(t, 0.05, 0.4)), cx - lead.size[0] // 2, top)
        wy = top + lead.size[1]
        wx = cx - word.size[0] // 2
        f.paste(reveal(word, prog(t, 0.2, 0.45)), wx, wy)
        ps = ease_in_out(prog(t, self.p.get("strike_at", 0.8), 0.35))
        if ps > 0:
            d = f.draw()
            ly = wy + int(word.size[1] * 0.52)
            d.line((wx - 10 * SS, ly, wx - 10 * SS + (word.size[0] + 20 * SS) * ps, ly),
                   fill=rgb(self.alert), width=11 * SS)
        return f.final(k, off)


class Diagram(Graphic):
    HEIGHT = 440

    PATH = [(0.00, 0.78), (0.10, 0.62), (0.19, 0.70), (0.31, 0.52), (0.41, 0.60), (0.53, 0.44),
            (0.62, 0.30), (0.67, 0.17), (0.73, 0.50), (0.84, 0.66), (1.00, 0.80)]
    LEVEL = 0.36
    SPIKE = 7

    def frame(self, t):
        x, y, w, h = self.region
        f = Frame(w, h)
        k, off = envelope(t, self.dur)
        pw, ph = w * SS - 40 * SS, h * SS - 40 * SS
        pnl, pad = self.panel_layer(pw, ph)
        f.paste(pnl, 20 * SS - pad, 20 * SS - pad)
        d = f.draw()
        L, T = 60 * SS, 48 * SS
        title = text_img(self.p.get("title", "FAKE OUT"), font("bold", 30), rgb(self.gold), tracking=3)
        tag = text_img("ILUSTRASI", font("medium", 20), (200, 200, 200, 150), tracking=4)
        f.paste(title, L, T, k=ease_out(prog(t, 0.0, 0.3)))
        f.paste(tag, w * SS - 60 * SS - tag.size[0], T + 6 * SS, k=ease_out(prog(t, 0.0, 0.3)))
        # plot area
        ax0, ax1 = L, w * SS - 60 * SS
        ay0, ay1 = T + 70 * SS, h * SS - 60 * SS
        X = lambda u: ax0 + (ax1 - ax0) * u
        Y = lambda v: ay0 + (ay1 - ay0) * v
        # level line (dashed), drawn left -> right
        pl = ease_in_out(prog(t, 0.1, 0.45))
        if pl > 0:
            xe = X(pl)
            xx = ax0
            while xx < xe:
                d.line((xx, Y(self.LEVEL), min(xx + 18 * SS, xe), Y(self.LEVEL)), fill=rgb(self.gold, 200), width=3 * SS)
                xx += 30 * SS
            lab = text_img(self.p.get("level_label", "Level"), font("medium", 22), rgb(self.gold, 220))
            f.paste(lab, ax1 - lab.size[0], Y(self.LEVEL) + 10 * SS, k=pl)
        # price path drawn progressively by length
        pts = [(X(u), Y(v)) for u, v in self.PATH]
        seg = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
        total = sum(seg)
        pp = ease_in_out(prog(t, 0.35, self.p.get("draw", 1.35))) * total
        acc, drawn = 0.0, [pts[0]]
        for (a, b), sl in zip(zip(pts, pts[1:]), seg):
            if acc + sl <= pp:
                drawn.append(b)
                acc += sl
            else:
                r = (pp - acc) / sl if sl else 0
                drawn.append((a[0] + (b[0] - a[0]) * r, a[1] + (b[1] - a[1]) * r))
                break
        if len(drawn) > 1 and pp > 0:
            d.line(drawn, fill=rgb(self.white), width=6 * SS, joint="curve")
            hx, hy = drawn[-1]
            d.ellipse((hx - 7 * SS, hy - 7 * SS, hx + 7 * SS, hy + 7 * SS), fill=rgb(self.white))
        # spike marker + label once the path passes the spike
        spike_len = sum(seg[:self.SPIKE])
        if pp >= spike_len:
            ps = prog(t, 0.35 + self.p.get("draw", 1.35) * (spike_len / total), 0.5)
            sx, sy = pts[self.SPIKE]
            r = (10 + 22 * ease_out(ps)) * SS
            d.ellipse((sx - r, sy - r, sx + r, sy + r), outline=rgb(self.gold, int(230 * (1 - ps * 0.6))), width=3 * SS)
            lab = text_img(self.p.get("spike_label", "Breakout palsu"), font("semibold", 26), rgb(self.white))
            f.paste(lab, sx - lab.size[0] - 44 * SS, sy - lab.size[1] - 6 * SS, k=ease_out(ps))
        # drop arrow after reversal
        drop_len = sum(seg[:self.SPIKE + 1])
        if pp >= drop_len:
            pa = ease_out(prog(t, 0.35 + self.p.get("draw", 1.35) * (drop_len / total), 0.3))
            ax, ay = pts[self.SPIKE + 1][0] + 40 * SS, pts[self.SPIKE + 1][1] - 30 * SS
            d.line((ax, ay - 50 * SS * pa, ax, ay + 10 * SS), fill=rgb(self.alert), width=5 * SS)
            d.polygon([(ax - 13 * SS, ay + 4 * SS), (ax + 13 * SS, ay + 4 * SS), (ax, ay + 24 * SS)], fill=rgb(self.alert, int(255 * pa)))
        return f.final(k, off)


class Compare(Graphic):
    """FROM chip (struck) -> one or more TO chips that pop in as each is named."""
    HEIGHT = 505

    def chip(self, text, filled, size=48):
        fnt = font("display_bold", size)
        tw = int(fnt.getlength(text))
        ch = int(size * 1.85) * SS
        cw = tw + int(size * 1.25) * SS
        img = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        if filled:
            d.rounded_rectangle((0, 0, cw - 1, ch - 1), ch // 2, fill=rgb(self.gold))
            col = (12, 14, 18, 255)
        else:
            d.rounded_rectangle((0, 0, cw - 1, ch - 1), ch // 2, outline=(255, 255, 255, 120), width=3 * SS,
                                fill=(255, 255, 255, 18))
            col = rgb(self.white)
        asc, desc = fnt.getmetrics()
        d.text(((cw - tw) // 2, (ch - asc - desc) // 2 + 2 * SS), text, font=fnt, fill=col)
        return img

    def frame(self, t):
        x, y, w, h = self.region
        f = Frame(w, h)
        k, off = envelope(t, self.dur)
        pw, ph = w * SS - 40 * SS, h * SS - 40 * SS
        pnl, pad = self.panel_layer(pw, ph)
        f.paste(pnl, 20 * SS - pad, 20 * SS - pad)
        cx = w * SS // 2
        head = text_img(self.p.get("title", "TUKAR MARKET"), font("bold", 28), rgb(self.gold), tracking=5)
        f.paste(head, cx - head.size[0] // 2, 58 * SS, k=ease_out(prog(t, 0, 0.3)))
        d = f.draw()
        # row 1: the market being left
        a = self.chip(self.p["from"], False, 46)
        ay = 112 * SS
        ax = cx - a.size[0] // 2
        f.paste(a, ax, ay, k=ease_out(prog(t, 0.1, 0.35)))
        st = ease_in_out(prog(t, 0.45, 0.3))
        if st > 0:
            my = ay + a.size[1] // 2
            d.line((ax + 18 * SS, my, ax + 18 * SS + (a.size[0] - 36 * SS) * st, my), fill=rgb(self.alert), width=7 * SS)
        # connector arrow down
        pa = ease_in_out(prog(t, 0.6, 0.35))
        top = ay + a.size[1] + 14 * SS
        if pa > 0:
            y1 = top + 44 * SS * pa
            d.line((cx, top, cx, y1), fill=rgb(self.gold), width=4 * SS)
            if pa > 0.7:
                d.polygon([(cx - 12 * SS, y1 - 4 * SS), (cx + 12 * SS, y1 - 4 * SS), (cx, y1 + 14 * SS)], fill=rgb(self.gold))
        # row 2: destinations, each pops in when named
        tos = self.p["to"] if isinstance(self.p["to"], list) else [self.p["to"]]
        ats = self.p.get("to_at", [1.1] * len(tos))
        ats = ats if isinstance(ats, list) else [ats]
        chips = [self.chip(tx, True, 46) for tx in tos]
        gap = 26 * SS
        total = sum(c.size[0] for c in chips) + gap * (len(chips) - 1)
        bx = cx - total // 2
        by = top + 74 * SS
        for c, at in zip(chips, ats):
            pb = prog(t, at, 0.45)
            if pb > 0:
                s = max(0.01, back_out(pb))
                cc = c.resize((max(1, int(c.size[0] * s)), max(1, int(c.size[1] * s))), Image.LANCZOS)
                f.paste(cc, bx + (c.size[0] - cc.size[0]) // 2, by + (c.size[1] - cc.size[1]) // 2, k=clamp(pb * 2))
            bx += c.size[0] + gap
        if self.p.get("note"):
            note = text_img(self.p["note"], font("semibold", 40), rgb(self.white))
            f.paste(reveal(note, prog(t, self.p.get("note_at", 2.0), 0.45)), cx - note.size[0] // 2,
                    by + chips[0].size[1] + 26 * SS)
        return f.final(k, off)


class Statement(Graphic):
    HEIGHT = 330

    def frame(self, t):
        x, y, w, h = self.region
        f = Frame(w, h)
        k, off = envelope(t, self.dur, enter=0.25)
        pw, ph = w * SS - 40 * SS, h * SS - 40 * SS
        pnl, pad = self.panel_layer(pw, ph)
        f.paste(pnl, 20 * SS - pad, 20 * SS - pad, k=ease_out(prog(t, 0, 0.25)))
        cx = w * SS // 2
        l1 = text_img(self.p["line1"], font("semibold", 54), rgb(self.white))
        l2 = text_img(self.p["line2"], font("display", self.p.get("size2", 134)), rgb(self.gold))
        l2 = sweep(l2, prog(t, 0.55, 0.65))
        top = 66 * SS
        f.paste(reveal(l1, prog(t, 0.0, 0.35)), cx - l1.size[0] // 2, top)
        f.paste(reveal(l2, prog(t, 0.15, 0.4)), cx - l2.size[0] // 2, top + l1.size[1])
        return f.final(k, off)


class Keyword(Graphic):
    HEIGHT = 250

    def frame(self, t):
        x, y, w, h = self.region
        f = Frame(w, h)
        k, off = envelope(t, self.dur, enter=0.25, exit_=0.25)
        txt = text_img(self.p["text"], font("display", self.p.get("size", 86)),
                       rgb(self.gold if self.p.get("gold") else self.white))
        pw, ph = txt.size[0] + 110 * SS, txt.size[1] + 70 * SS
        pnl, pad = self.panel_layer(pw, ph)
        cx = w * SS // 2
        px = cx - pw // 2
        f.paste(pnl, px - pad, 30 * SS - pad, k=ease_out(prog(t, 0, 0.22)))
        f.paste(reveal(txt, prog(t, 0.05, 0.35)), cx - txt.size[0] // 2, 30 * SS + 35 * SS)
        return f.final(k, off)


class EndCard(Graphic):
    HEIGHT = 380

    def frame(self, t):
        x, y, w, h = self.region
        f = Frame(w, h)
        k, off = envelope(t, self.dur, enter=0.3, exit_=0.35)
        pw, ph = w * SS - 40 * SS, h * SS - 40 * SS
        pnl, pad = self.panel_layer(pw, ph)
        f.paste(pnl, 20 * SS - pad, 20 * SS - pad, k=ease_out(prog(t, 0, 0.3)))
        cx = w * SS // 2
        q = text_img(self.p["question"], font("semibold", self.p.get("size1", 54)), rgb(self.white))
        cta = text_img(self.p["cta"], font("display", self.p.get("size2", 108)), rgb(self.gold))
        cta = sweep(cta, prog(t, 1.0, 0.7))
        top = 70 * SS
        f.paste(reveal(q, prog(t, 0.05, 0.4)), cx - q.size[0] // 2, top)
        f.paste(reveal(cta, prog(t, 0.3, 0.45)), cx - cta.size[0] // 2, top + q.size[1] + 4 * SS)
        self.brand_line(f, cx, top + q.size[1] + cta.size[1] + 34 * SS, t, at=0.7)
        return f.final(k, off)


class Chips(Compare):
    """Heading + chips that pop in one by one as each is named (no 'from' row)."""
    HEIGHT = 290

    def frame(self, t):
        x, y, w, h = self.region
        f = Frame(w, h)
        k, off = envelope(t, self.dur)
        pw, ph = w * SS - 40 * SS, h * SS - 40 * SS
        pnl, pad = self.panel_layer(pw, ph)
        f.paste(pnl, 20 * SS - pad, 20 * SS - pad, k=ease_out(prog(t, 0, 0.25)))
        cx = w * SS // 2
        head = text_img(self.p["title"], font("bold", 28), rgb(self.gold), tracking=5)
        f.paste(head, cx - head.size[0] // 2, 50 * SS, k=ease_out(prog(t, 0.05, 0.3)))
        items = self.p["items"]
        ats = self.p.get("at_each", [0.4 + 0.5 * i for i in range(len(items))])
        chips = [self.chip(tx, True, 44) for tx in items]
        gap = 26 * SS
        total = sum(c.size[0] for c in chips) + gap * (len(chips) - 1)
        bx, by = cx - total // 2, 100 * SS
        for c, at in zip(chips, ats):
            pb = prog(t, at, 0.45)
            if pb > 0:
                s = max(0.01, back_out(pb))
                cc = c.resize((max(1, int(c.size[0] * s)), max(1, int(c.size[1] * s))), Image.LANCZOS)
                f.paste(cc, bx + (c.size[0] - cc.size[0]) // 2, by + (c.size[1] - cc.size[1]) // 2, k=clamp(pb * 2))
            bx += c.size[0] + gap
        if self.p.get("note"):
            note = text_img(self.p["note"], font("semibold", 38), rgb(self.white))
            f.paste(reveal(note, prog(t, self.p.get("note_at", 2.0), 0.45)), cx - note.size[0] // 2,
                    by + chips[0].size[1] + 20 * SS)
        return f.final(k, off)


class Checklist(Graphic):
    """Heading + 2x2 grid; each item's ring draws and ticks at its own time."""
    HEIGHT = 280

    def frame(self, t):
        x, y, w, h = self.region
        f = Frame(w, h)
        k, off = envelope(t, self.dur)
        pw, ph = w * SS - 40 * SS, h * SS - 40 * SS
        pnl, pad = self.panel_layer(pw, ph)
        f.paste(pnl, 20 * SS - pad, 20 * SS - pad, k=ease_out(prog(t, 0, 0.25)))
        cx = w * SS // 2
        head = text_img(self.p["title"], font("bold", 28), rgb(self.gold), tracking=5)
        f.paste(head, cx - head.size[0] // 2, 48 * SS, k=ease_out(prog(t, 0.05, 0.3)))
        items, ats = self.p["items"], self.p["tick_at"]
        d = f.draw()
        r = 19 * SS
        col_w = (pw - 60 * SS) // 2
        for i, (txt, at) in enumerate(zip(items, ats)):
            ox = 20 * SS + 40 * SS + (i % 2) * col_w
            oy = 102 * SS + (i // 2) * 78 * SS
            ccx, ccy = ox + r, oy + r + 4 * SS
            pt = prog(t, at, 0.45)
            done = pt > 0
            label = text_img(txt, font("semibold", 38), rgb(self.white, 255 if done else 150))
            f.paste(label, ox + 2 * r + 18 * SS, oy, k=ease_out(prog(t, 0.15 + 0.08 * i, 0.3)))
            d.ellipse((ccx - r, ccy - r, ccx + r, ccy + r), outline=(255, 255, 255, 90), width=2 * SS)
            if done:
                e = ease_out(pt)
                d.arc((ccx - r, ccy - r, ccx + r, ccy + r), -90, -90 + 360 * e, fill=rgb(self.gold), width=4 * SS)
                pc = ease_in_out(prog(t, at + 0.2, 0.25))
                if pc > 0:
                    a, b, c = (ccx - 8 * SS, ccy), (ccx - 2 * SS, ccy + 7 * SS), (ccx + 9 * SS, ccy - 7 * SS)
                    if pc < 0.4:
                        q = pc / 0.4
                        d.line((a, (a[0] + (b[0] - a[0]) * q, a[1] + (b[1] - a[1]) * q)), fill=rgb(self.gold), width=4 * SS)
                    else:
                        q = (pc - 0.4) / 0.6
                        d.line((a, b, (b[0] + (c[0] - b[0]) * q, b[1] + (c[1] - b[1]) * q)), fill=rgb(self.gold),
                               width=4 * SS, joint="curve")
        return f.final(k, off)


TYPES = {"chips": Chips, "checklist": Checklist, "title": Title, "strike": Strike, "diagram": Diagram, "compare": Compare,
         "statement": Statement, "keyword": Keyword, "endcard": EndCard}


def render(kind, params, dur, fps, out_dir, brand):
    """Write the PNG sequence; return (x, y) placement on the canvas."""
    g = TYPES[kind](params, dur, brand)
    os.makedirs(out_dir, exist_ok=True)
    n = max(1, int(round(dur * fps)))
    for i in range(n):
        g.frame(i / fps).save(os.path.join(out_dir, f"f{i:04d}.png"), compress_level=1)
    return g.region[0], g.region[1]

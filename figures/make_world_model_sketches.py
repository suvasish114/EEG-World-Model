"""Pen-stroke sketches of world model v1 (world_model_alice.py) and v3 (world_model_v3.py).

Outputs (white background, 300 dpi): figures/world_model_v1_sketch.png, figures/world_model_v3_sketch.png
Same hand-drawn toolkit as make_sense_alice_sketch.py (seeded jitter, 680 px wide canvas).
Uses the Caveat handwriting font if it can be downloaded into figures/fonts/, else DejaVu Sans.
"""
import math
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects, rcParams
from matplotlib.patches import Circle, PathPatch, Rectangle
from matplotlib.path import Path as MPath
from matplotlib.transforms import Affine2D

HERE = Path(__file__).resolve().parent
FONT = HERE / "fonts" / "Caveat.ttf"
URLS = ["https://cdn.jsdelivr.net/gh/google/fonts@main/ofl/caveat/Caveat%5Bwght%5D.ttf",
        "https://github.com/google/fonts/raw/main/ofl/caveat/Caveat%5Bwght%5D.ttf"]
INK, PX = "#1f1e1d", 0.72
CORAL, PURPLE, TEAL, AMBER = "#D85A30", "#7F77DD", "#1D9E75", "#BA7517"
rcParams["lines.scale_dashes"] = False
M, C3, L = MPath.MOVETO, MPath.CURVE3, MPath.LINETO


def font_family():
    if not FONT.exists():
        FONT.parent.mkdir(exist_ok=True)
        for u in URLS:
            try:
                urllib.request.urlretrieve(u, FONT)
                break
            except Exception as e:
                print(f"Caveat download failed ({u}): {e}")
                FONT.unlink(missing_ok=True)
    if not FONT.exists():
        return ["DejaVu Sans"], 0.8  # wider font: shrink text so it fits the boxes
    font_manager.fontManager.addfont(str(FONT))
    return [font_manager.FontProperties(fname=str(FONT)).get_name(), "DejaVu Sans"], 1.0


FAM, FSCALE = font_family()
ax, seed = None, 1


def new(H, s):
    global ax, seed
    seed = s
    fig = plt.figure(figsize=(6.8, H / 100), dpi=300, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 680)
    ax.set_ylim(H, 0)
    ax.axis("off")
    return fig


def r():
    global seed
    seed = seed * 16807 % 2147483647
    return seed / 2147483647


def j(a):
    return (r() - .5) * 2 * a


def patch(verts, codes, c=None, w=1.7, dash=None, op=1.0):
    p = PathPatch(MPath(verts, codes), fill=False, edgecolor=c or INK, lw=w * PX, alpha=op,
                  capstyle="round", joinstyle="round")
    if dash:
        p.set_linestyle((0, [d * PX for d in dash]))
    ax.add_patch(p)


def quad(x1, y1, cx, cy, x2, y2, **o):
    patch([(x1, y1), (cx, cy), (x2, y2)], [M, C3, C3], **o)


def poly(pts, **o):
    patch(pts, [M] + [L] * (len(pts) - 1), **o)


def ln(x1, y1, x2, y2, c=None, w=1.7, dash=None):
    for p in range(2):
        mx, my = (x1 + x2) / 2 + j(2.5), (y1 + y2) / 2 + j(2.5)
        a, b = x1 + j(1.5), y1 + j(1.5)
        e, f = x2 + j(1.5), y2 + j(1.5)
        quad(a, b, mx, my, e, f, c=c, w=1 if p else w, dash=dash, op=.45 if p else 1)


def head(fx, fy, x, y, c=None):
    a = math.atan2(y - fy, x - fx)
    for s in (-.45, .45):
        poly([(x, y), (x - 10 * math.cos(a + s), y - 10 * math.sin(a + s))], c=c, w=1.7)


def arr(x1, y1, x2, y2, c=None, dash=None):
    ln(x1, y1, x2, y2, c=c, dash=dash)
    head(x1, y1, x2, y2, c)


def curve(x1, y1, cx, cy, x2, y2, c=None, dash=None):
    quad(x1, y1, cx, cy, x2, y2, c=c, dash=dash)
    quad(x1 + 1.5, y1 + 1, cx + j(4), cy + j(4), x2 + 1, y2 - 1, c=c, dash=dash, w=1, op=.4)
    head(cx, cy, x2, y2, c)


def box(x, y, w, h, fill=None, c=None, dash=None, e=3, alpha=.22):
    if fill:
        ang = j(1.2)
        ax.add_patch(Rectangle((x + 3, y + 4), w - 4, h - 5, facecolor=fill, edgecolor="none", alpha=alpha,
                               transform=Affine2D().rotate_deg_around(x + w / 2, y + h / 2, ang) + ax.transData))
    ln(x - e, y, x + w + e, y, c=c, dash=dash)
    ln(x + w, y - e, x + w, y + h + e, c=c, dash=dash)
    ln(x + w + e, y + h, x - e, y + h, c=c, dash=dash)
    ln(x, y + h + e, x, y - e, c=c, dash=dash)


def tx(x, y, s, z=20, a="middle", w=400):
    pe = [patheffects.withStroke(linewidth=0.6, foreground=INK)] if w == 600 else None
    ax.text(x, y, s, fontsize=z * PX * FSCALE, ha={"middle": "center", "start": "left"}[a], va="center",
            family=FAM, color=INK, path_effects=pe)


def wave(x, y, w, c, amp=4):
    pts, i = [(x, y)], 1
    while i <= w / 3:
        v = math.sin(i * .9 + r() * 2) * amp
        v += j(3)
        pts.append((x + i * 3, y + v))
        i += 1
    poly(pts, c=c, w=1.4)


def tokens(x, y, n, fill, null_from=None):
    """Row of small hand-drawn token squares; squares from null_from on are dashed null tokens."""
    for i in range(n):
        xi = x + i * 20
        if null_from is not None and i >= null_from:
            ax.add_patch(Rectangle((xi, y), 14, 14, fill=False, edgecolor=INK, lw=1.2 * PX, ls=(0, (2, 2))))
            tx(xi + 7, y + 7.5, "∅", 14)
        else:
            ax.add_patch(Circle((xi + 7, y + 7), 7.5, facecolor=fill, alpha=.35, edgecolor="none"))
            poly([(xi + 7 + (7 + j(.8)) * math.cos(t / 10 * 6.5), y + 7 + (7 + j(.8)) * math.sin(t / 10 * 6.5))
                  for t in range(11)], w=1.4)


def spec(x, y, cols, rows, cw, ch):
    for c in range(cols):
        for rw in range(rows):
            op = round(.12 + .75 * abs(math.sin(c * .8 + rw * 1.3 + r())), 2)
            ax.add_patch(Rectangle((x + c * (cw + 1), y + rw * (ch + 1)), cw, ch, facecolor=TEAL,
                                   edgecolor="none", alpha=op))
    box(x - 3, y - 3, cols * (cw + 1) + 5, rows * (ch + 1) + 5)


# ===================================================================== v1: EEG = state, audio = action
def v1():
    fig = new(380, 11)
    for k in range(3):
        wave(30, 80 + k * 18, 90, CORAL)
    tx(75, 140, "EEG", 22, w=600)
    arr(124, 98, 155, 98)
    box(160, 75, 55, 46, fill=PURPLE)
    tx(187, 98, "E", 24, w=600)
    arr(220, 98, 245, 98)
    tokens(250, 90, 6, PURPLE)
    tx(310, 72, "z", 22, w=600)
    arr(375, 98, 425, 98)
    box(430, 72, 80, 52, fill=TEAL)
    tx(470, 90, "g", 24, w=600)
    tx(470, 110, "bi-dir", 16)
    arr(515, 88, 553, 72)
    spec(561, 49, 8, 5, 10, 7)
    tx(604, 32, "mel", 20, w=600)
    arr(515, 112, 553, 126)
    tokens(565, 120, 4, TEAL)
    tx(604, 154, "≈ u", 20)

    wave(30, 330, 90, TEAL, amp=7)
    tx(75, 362, "audio", 22, w=600)
    arr(124, 330, 155, 330)
    box(160, 307, 55, 46, fill=TEAL)
    tx(187, 330, "A", 24, w=600)
    arr(220, 330, 245, 330)
    tokens(250, 322, 6, TEAL)
    tx(310, 356, "u", 22, w=600)

    box(430, 190, 80, 62, fill=AMBER)
    tx(470, 210, "f", 24, w=600)
    tx(470, 232, "causal", 16)
    arr(330, 112, 426, 205)
    arr(330, 318, 426, 240)
    arr(515, 221, 553, 221)
    tokens(565, 213, 4, PURPLE)
    tx(604, 248, "≈ EMA(z) next", 18)
    tx(470, 274, "vs shuffled audio", 16)
    fig.savefig(HERE / "world_model_v1_sketch.png", dpi=300, facecolor="white")
    plt.close(fig)


# ===================================================================== v3: audio = state, EEG = action
def v3():
    fig = new(380, 23)
    spec(31, 63, 5, 5, 15, 7)
    tx(70, 124, "audio", 22, w=600)
    arr(120, 84, 155, 84)
    box(160, 61, 55, 46, fill=TEAL)
    tx(187, 84, "A", 24, w=600)
    arr(220, 84, 245, 84)
    tokens(250, 76, 6, TEAL, null_from=4)
    tx(310, 58, "s", 22, w=600)

    for k in range(3):
        wave(30, 290 + k * 18, 90, CORAL)
    tx(75, 352, "EEG", 22, w=600)
    arr(124, 308, 155, 308)
    box(160, 285, 55, 46, fill=PURPLE)
    tx(187, 308, "E", 24, w=600)
    arr(220, 308, 245, 308)
    tokens(250, 300, 6, PURPLE)
    tx(310, 336, "e", 22, w=600)

    box(430, 165, 80, 62, fill=AMBER)
    tx(470, 185, "f", 24, w=600)
    tx(470, 207, "causal", 16)
    arr(330, 96, 426, 180)
    arr(330, 296, 426, 215)
    tx(402, 300, "+1 ahead", 16)
    arr(515, 196, 553, 196)
    spec(563, 177, 6, 5, 11, 7)
    tx(600, 238, "next mel", 20, w=600)
    curve(598, 170, 560, 20, 374, 68, c=AMBER, dash=(5, 4))
    tx(515, 112, "rollout", 18)
    tx(470, 247, "vs shuffled EEG", 16)
    ax.add_patch(Rectangle((430, 323), 14, 14, fill=False, edgecolor=INK, lw=1.2 * PX, ls=(0, (2, 2))))
    tx(437, 330.5, "∅", 14)
    tx(456, 331, "null: dropped / EEG-only", 17, "start")
    fig.savefig(HERE / "world_model_v3_sketch.png", dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    v1()
    v3()
    print("wrote", HERE / "world_model_v1_sketch.png", "and", HERE / "world_model_v3_sketch.png")

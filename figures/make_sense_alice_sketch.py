"""Pen-stroke sketch of the SENSE-Alice architecture -> figures/sense_alice_sketch.png (white background).

Exact port of the inline SVG sketch: same seeded jitter (seed 11, Park-Miller RNG), same coordinates
(680 x 430 px canvas), same Caveat handwriting font (downloaded once into figures/fonts/).
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
OUT = HERE / "sense_alice_sketch.png"
FONT = HERE / "fonts" / "Caveat.ttf"
URL = "https://github.com/google/fonts/raw/main/ofl/caveat/Caveat%5Bwght%5D.ttf"
INK = "#1f1e1d"
PX = 0.72  # one canvas px in points (6.8 in figure for 680 px)
rcParams["lines.scale_dashes"] = False


def font_family():
    if not FONT.exists():
        try:
            FONT.parent.mkdir(exist_ok=True)
            urllib.request.urlretrieve(URL, FONT)
        except Exception as e:
            print("Caveat download failed, using fallback font:", e)
            return ["DejaVu Sans"]
    font_manager.fontManager.addfont(str(FONT))
    return [font_manager.FontProperties(fname=str(FONT)).get_name(), "DejaVu Sans"]


seed = 11


def r():
    global seed
    seed = seed * 16807 % 2147483647
    return seed / 2147483647


def j(a):
    return (r() - .5) * 2 * a


fig = plt.figure(figsize=(6.8, 4.3), dpi=300, facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 680)
ax.set_ylim(430, 0)
ax.axis("off")
FAM = font_family()
M, C3, L = MPath.MOVETO, MPath.CURVE3, MPath.LINETO


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
    ccx, ccy = cx + j(4), cy + j(4)
    quad(x1 + 1.5, y1 + 1, ccx, ccy, x2 + 1, y2 - 1, c=c, dash=dash, w=1, op=.4)
    head(cx, cy, x2, y2, c)


def box(x, y, w, h, fill=None, c=None, dash=None):
    if fill:
        ang = j(1.2)
        ax.add_patch(Rectangle((x + 3, y + 4), w - 4, h - 5, facecolor=fill, edgecolor="none", alpha=.22,
                               transform=Affine2D().rotate_deg_around(x + w / 2, y + h / 2, ang) + ax.transData))
    e = 3
    ln(x - e, y, x + w + e, y, c=c, dash=dash)
    ln(x + w, y - e, x + w, y + h + e, c=c, dash=dash)
    ln(x + w + e, y + h, x - e, y + h, c=c, dash=dash)
    ln(x, y + h + e, x, y - e, c=c, dash=dash)


def tx(x, y, s, z=20, a="middle", w=400):
    pe = [patheffects.withStroke(linewidth=0.6, foreground=INK)] if w == 600 else None
    ax.text(x, y, s, fontsize=z * PX, ha={"middle": "center", "start": "left"}[a], va="center",
            family=FAM, color=INK, path_effects=pe)


def blob(cx, cy, R):
    for p in range(2):
        pts = []
        for i in range(25):
            t = i / 24 * 6.6 + p * .3
            rr = R + j(1.6)
            pts.append((cx + rr * math.cos(t), cy + rr * math.sin(t)))
        poly(pts, w=1 if p else 1.7, op=.45 if p else 1)


def wave(x, y, w, c):
    pts, i = [(x, y)], 1
    while i <= w / 3:
        v = math.sin(i * .9 + r() * 2) * 4
        v += j(3)
        pts.append((x + i * 3, y + v))
        i += 1
    poly(pts, c=c, w=1.4)


CORAL, PURPLE, TEAL, AMBER = "#D85A30", "#7F77DD", "#1D9E75", "#BA7517"

# ---- encoder row
for k in range(4):
    wave(28, 78 + k * 20, 90, CORAL)
tx(73, 178, "EEG", 22, w=600)
arr(124, 108, 152, 108)
blob(176, 108, 20)
tx(176, 108, "σ", 22)
tx(176, 148, "gate", 18)
arr(198, 98, 238, 80)
arr(198, 118, 238, 150)
box(242, 55, 96, 50, fill=PURPLE)
g = [(262, 70), (290, 64), (318, 74), (275, 92), (306, 92)]
for a_, b_ in [(0, 1), (1, 2), (0, 3), (1, 3), (1, 4), (2, 4), (3, 4)]:
    ln(*g[a_], *g[b_], w=1.1)
for x, y in g:
    ax.add_patch(Circle((x, y), 3.5, facecolor=INK, edgecolor="none"))
box(242, 125, 96, 50, fill=PURPLE)
tx(290, 150, "cnn", 22)
arr(340, 80, 366, 106)
arr(340, 150, 366, 124)
blob(378, 115, 13)
tx(378, 114, "+", 22)
arr(392, 115, 404, 115)
for k in range(3, -1, -1):
    box(408 + k * 7, 72 + k * 7, 62, 64, fill=None if k else TEAL)
tx(439, 106, "S4 ×4", 22, w=600)
arr(492, 115, 512, 115)
box(516, 62, 30, 108, fill=TEAL)
tx(531, 48, "M", 24, w=600)

# ---- decoder column
arr(531, 178, 531, 236)
box(476, 240, 112, 50, fill=TEAL)
tx(532, 265, "decoder", 22)
arr(532, 296, 532, 322)
for c in range(10):
    for rw in range(6):
        op = round(.12 + .75 * abs(math.sin(c * .8 + rw * 1.3 + r())), 2)
        ax.add_patch(Rectangle((478 + c * 11, 328 + rw * 9), 10, 8, facecolor=TEAL, edgecolor="none", alpha=op))
box(474, 324, 114, 58)
tx(531, 402, "mel", 22, w=600)

# ---- training-only heads
D = (6, 5)
curve(516, 160, 300, 180, 100, 246, dash=D)
curve(516, 166, 330, 200, 225, 246, dash=D)
curve(518, 170, 420, 205, 350, 246, dash=D)
box(50, 250, 100, 44, dash=D)
tx(100, 272, "phonemes", 20)
box(175, 250, 100, 44, dash=D)
wave(188, 272, 74, CORAL)
box(300, 250, 100, 44, dash=D)
tx(350, 272, "CLIP", 20)
tx(225, 312, "recon", 17)
tx(100, 312, "ctc", 17)
tx(350, 312, "esc", 17)
arr(402, 272, 472, 266, c=AMBER, dash=(3, 4))
ln(430, 260, 436, 280, c=AMBER, w=1.5)
ln(437, 259, 443, 279, c=AMBER, w=1.5)
tx(438, 244, "g", 22, w=600)
ln(50, 380, 90, 380, dash=D)
tx(98, 380, "training only", 18, "start")
ln(50, 406, 90, 406, c=AMBER, dash=(3, 4))
tx(98, 406, "stop-grad FiLM", 18, "start")

fig.savefig(OUT, dpi=300, facecolor="white")
print("wrote", OUT)

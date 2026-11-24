"""Pen-stroke sketch of the Alice preprocessing pipeline (prepare_alice_dataset.py, --align marker).

Output: figures/preprocessing_sketch.png (white background, 300 dpi). Reuses the hand-drawn toolkit of
make_world_model_sketches.py.
"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_world_model_sketches as sk  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

CORAL, PURPLE, TEAL, AMBER = sk.CORAL, sk.PURPLE, sk.TEAL, sk.AMBER


def head(cx, cy, R):
    sk.poly([(cx + (R + sk.j(1.2)) * math.cos(t / 24 * 6.5), cy + (R + sk.j(1.2)) * math.sin(t / 24 * 6.5))
             for t in range(25)], w=1.7)
    sk.poly([(cx - 5, cy - R + 1), (cx, cy - R - 7), (cx + 5, cy - R + 1)], w=1.5)  # nose
    for dx, dy in [(-12, -8), (0, -12), (12, -8), (-14, 6), (14, 6), (-5, 14), (6, 14), (0, 1)]:
        if (dx, dy) == (12, -8):  # rejected channel -> interpolated
            sk.ln(cx + dx - 4, cy + dy - 4, cx + dx + 4, cy + dy + 4, c=CORAL, w=1.8)
            sk.ln(cx + dx - 4, cy + dy + 4, cx + dx + 4, cy + dy - 4, c=CORAL, w=1.8)
        else:
            sk.ax.add_patch(Circle((cx + dx, cy + dy), 2.6, facecolor=sk.INK, edgecolor="none"))


def main():
    fig = sk.new(400, 31)
    # ---------------- EEG row
    for k in range(3):
        sk.wave(30, 62 + k * 18, 80, CORAL)
    sk.tx(70, 122, "EEG", 22, w=600)
    sk.arr(114, 80, 140, 80)
    head(172, 82, 24)
    sk.arr(204, 82, 220, 82)
    sk.box(224, 58, 62, 48)
    sk.poly([(230, 98), (240, 70), (270, 70), (280, 98)], w=1.6, c=PURPLE)
    sk.tx(255, 124, "0.1-40 Hz", 16)
    sk.arr(292, 82, 306, 82)
    for i in range(6):
        x, h = 312 + i * 7, 14 * math.sin(i * 1.1 + .4)
        sk.ln(x, 82, x, 82 - h, w=1.2)
        sk.ax.add_patch(Circle((x, 82 - h), 2.2, facecolor=sk.INK, edgecolor="none"))
    sk.tx(330, 124, "120 Hz", 16)
    sk.arr(358, 82, 372, 82)
    pts = [(376 + i * 2, 82 - max(-9, min(9, 16 * math.sin(i * .35)))) for i in range(33)]
    sk.poly(pts, c=CORAL, w=1.5)
    sk.ln(374, 73, 444, 73, c=sk.INK, w=1, dash=(3, 3))
    sk.ln(374, 91, 444, 91, c=sk.INK, w=1, dash=(3, 3))
    sk.tx(410, 124, "clip ±20", 16)
    sk.curve(448, 82, 470, 110, 476, 170, c=None)

    # ---------------- audio row
    sk.wave(30, 330, 80, TEAL, amp=7)
    sk.tx(70, 362, "audio", 22, w=600)
    sk.arr(114, 324, 144, 306)
    sk.spec(152, 288, 6, 4, 10, 7)
    sk.tx(185, 276, "mel", 18, w=600)
    sk.arr(114, 338, 144, 356)
    sk.box(150, 342, 66, 30, fill=TEAL)
    sk.tx(183, 357, "w2v", 18)
    sk.arr(224, 306, 256, 324)
    sk.arr(224, 357, 256, 338)
    sk.box(262, 316, 30, 30)
    sk.tx(277, 331, "z", 18)
    sk.curve(298, 331, 420, 320, 470, 256)

    # ---------------- alignment + windows
    sk.wave(476, 185, 174, CORAL, amp=2)
    sk.wave(476, 245, 174, TEAL, amp=2)
    sk.ln(488, 158, 488, 188, w=1.6)
    sk.ax.add_patch(Polygon([(488, 158), (504, 164), (488, 170)], closed=True, facecolor=AMBER, edgecolor=sk.INK,
                            lw=1.2 * sk.PX))
    sk.ln(488, 192, 488, 240, c=AMBER, w=1.3, dash=(4, 3))
    sk.tx(510, 164, "marker", 16, "start")
    sk.box(538, 172, 86, 26, fill=PURPLE)
    sk.box(520, 232, 86, 26, fill=TEAL)
    sk.arr(563, 215, 581, 215, c=AMBER)
    sk.tx(600, 215, "+150 ms", 15, "start")
    sk.tx(581, 156, "3 s", 18, w=600)

    # ---------------- sentence-block splits
    sk.arr(563, 264, 563, 312)
    cols = [PURPLE, PURPLE, AMBER, PURPLE, PURPLE, CORAL, PURPLE, AMBER, PURPLE, PURPLE]
    for i, c in enumerate(cols):
        sk.box(470 + i * 18, 320, 14, 18, fill=c, e=1, alpha=.7)
    sk.tx(560, 362, "train · val · test", 16)
    fig.savefig(HERE / "preprocessing_sketch.png", dpi=300, facecolor="white")
    print("wrote", HERE / "preprocessing_sketch.png")


if __name__ == "__main__":
    main()

"""Training curves of world model v1 (world_model_alice.py) and v3 (world_model_v3.py).

Reads only saved outputs (no GPU, no model loading):
  train : the per-100-step loss lines of the SLURM log (full run = last section of the log)
  val   : runs/<run>/val_history.json (every 500 steps, seen subjects, val sentences)
  test  : runs/<run>/metrics.json (best.pt only -> one point per group at the best step)
  mTRF  : the baseline numbers stored in metrics.json (subject model = seen, pooled = unseen)

Usage:  python plot_world_models.py [--v1-log logs/out_553059.log] [--v3-log logs/out_553161.log]
Output: runs/plots/world_model_v1_v3.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
BLUE, ORANGE = "#2a78d6", "#eb6834"          # v1, v3 (fixed categorical slots 1, 2)
LOSS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
GREY, INK, MUTED = "#8a8985", "#0b0b0b", "#52514e"


def train_curves(log: Path) -> dict[str, list]:
    """step -> loss parts from 'step N | k v k v ... | lr ...' lines of the LAST run in the log."""
    lines = log.read_text().splitlines()
    start = max(i for i, l in enumerate(lines) if l.startswith("seen ") and "subjects, held out" in l)
    out: dict[str, list] = {"step": []}
    for l in lines[start:]:
        m = re.match(r"step (\d+) \| (.*?) \| lr", l)
        if not m:
            continue
        out["step"].append(int(m.group(1)))
        toks = m.group(2).split()
        for k, v in zip(toks[::2], toks[1::2]):
            out.setdefault(k, []).append(float(v))
    return out


def val_curves(run: Path) -> dict[str, list]:
    hist = json.loads((run / "val_history.json").read_text())
    return {k: [h[k] for h in hist] for k in hist[0]}


def style(ax, title, ylabel=None):
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    ax.set_xlabel("step", fontsize=8, color=MUTED)
    ax.tick_params(labelsize=7, colors=MUTED)
    ax.grid(True, color="#e6e5e1", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")


def best_line(ax, step, color):
    ax.axvline(step, color=color, lw=0.8, ls=":", alpha=0.8)


def test_points(ax, step, seen, unseen, color, label=None):
    ax.plot([step], [seen], "o", color=color, ms=8, mec="white", mew=1.5, zorder=5,
            label=label and f"{label} test seen")
    ax.plot([step], [unseen], "^", color=color, ms=8, mec="white", mew=1.5, zorder=5,
            label=label and f"{label} test unseen")


def mtrf_lines(ax, base_seen, base_unseen):
    ax.axhline(base_seen, color=GREY, lw=1.2, ls="--", label="mTRF subject (seen)")
    ax.axhline(base_unseen, color=GREY, lw=1.2, ls="-.", label="mTRF pooled (unseen)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--v1-run", default=str(ROOT / "runs" / "world_model"))
    p.add_argument("--v3-run", default=str(ROOT / "runs" / "world_model_v3"))
    p.add_argument("--v1-log", default=str(ROOT / "logs" / "out_553059.log"))
    p.add_argument("--v3-log", default=str(ROOT / "logs" / "out_553161.log"))
    p.add_argument("--out", default=str(ROOT / "runs" / "plots" / "world_model_v1_v3.png"))
    a = p.parse_args()

    t1, t3 = train_curves(Path(a.v1_log)), train_curves(Path(a.v3_log))
    v1, v3 = val_curves(Path(a.v1_run)), val_curves(Path(a.v3_run))
    m1 = json.loads((Path(a.v1_run) / "metrics.json").read_text())
    m3 = json.loads((Path(a.v3_run) / "metrics.json").read_text())
    b1, b3 = m1["checkpoint_step"], m3["checkpoint_step"]
    bs, bu = m1["seen"]["baseline"], m1["unseen"]["baseline"]

    fig, ax = plt.subplots(3, 3, figsize=(17, 13))
    fig.patch.set_facecolor("white")

    # ---- row 1: training losses + train/val contrastive accuracy
    for axx, t, name, b in ((ax[0, 0], t1, "v1 (EEG = state, audio = action)", b1),
                            (ax[0, 1], t3, "v3 (audio = state, EEG = action)", b3)):
        keys = [k for k in t if k not in ("step", "anc_acc")]
        for k, c in zip(keys, LOSS_COLORS):
            axx.plot(t["step"], t[k], color=c, lw=2, label=k)
        axx.set_yscale("log")
        best_line(axx, b, INK)
        style(axx, f"Train losses, {name}", "loss (log scale, 100-step mean)")
        axx.legend(fontsize=7, ncol=2, frameon=False)

    axx = ax[0, 2]
    axx.plot(t1["step"], t1["anc_acc"], color=BLUE, lw=2, label="v1 train anc_acc (true vs 3 shuffled)")
    axx.plot(t3["step"], t3["anc_acc"], color=ORANGE, lw=2, label="v3 train anc_acc (true vs 3 shuffled)")
    axx.plot(v1["step"], v1["wm_mm_acc"], color=BLUE, lw=2, ls="--", marker="o", ms=4,
             label="v1 val wm_mm_acc (true vs 1 shuffled)")
    axx.plot(v3["step"], v3["wm_mm_acc"], color=ORANGE, lw=2, ls="--", marker="o", ms=4,
             label="v3 val wm_mm_acc (true vs 1 shuffled)")
    test_points(axx, b1, m1["seen"]["wm_mm_acc"], m1["unseen"]["wm_mm_acc"], BLUE, "v1")
    test_points(axx, b3, m3["seen"]["wm_mm_acc"], m3["unseen"]["wm_mm_acc"], ORANGE, "v3")
    axx.axhline(0.25, color=GREY, lw=0.8, ls=":")
    axx.axhline(0.5, color=GREY, lw=0.8, ls=":")
    axx.text(axx.get_xlim()[1], 0.25, " chance train", fontsize=7, color=MUTED, va="bottom", ha="right")
    axx.text(axx.get_xlim()[1], 0.5, " chance val/test", fontsize=7, color=MUTED, va="bottom", ha="right")
    style(axx, "World-model contrastive accuracy: train vs val vs test", "accuracy")
    axx.legend(fontsize=6.5, frameon=False, loc="upper left")

    # ---- row 2: decoding metrics (val curves, test points at best step, mTRF reference)
    for axx, k1, title, ylabel, base in (
            (ax[1, 0], "mm_acc", "Match-mismatch accuracy (decoded envelope)", "accuracy (chance 0.5)", "mm_acc"),
            (ax[1, 1], "r_env", "Envelope reconstruction r", "Pearson r", "r_env"),
            (ax[1, 2], "r_above_null", "Envelope r above null (r_env - r_env_null)", "Δ Pearson r", "r_above_null")):
        axx.plot(v1["step"], v1[k1], color=BLUE, lw=2, marker="o", ms=4, label="v1 val (IDM decoder)")
        axx.plot(v3["step"], v3[f"eeg_only_{k1}"], color=ORANGE, lw=2, marker="o", ms=4,
                 label="v3 val eeg_only (fair)")
        axx.plot(v3["step"], v3[f"rollout_{k1}"], color=ORANGE, lw=2, ls="--", marker="s", ms=4,
                 label="v3 val rollout (sees 1 s true audio)")
        test_points(axx, b1, m1["seen"][k1], m1["unseen"][k1], BLUE, "v1")
        test_points(axx, b3, m3["seen"][f"eeg_only_{k1}"], m3["unseen"][f"eeg_only_{k1}"], ORANGE, "v3 eeg_only")
        mtrf_lines(axx, bs[base], bu[base])
        if k1 == "mm_acc":
            axx.axhline(0.5, color=GREY, lw=0.8, ls=":")
        best_line(axx, b1, BLUE)
        best_line(axx, b3, ORANGE)
        style(axx, title, ylabel)
        axx.legend(fontsize=6.5, frameon=False, ncol=2, loc="lower right" if k1 != "mm_acc" else "lower right")

    # ---- row 3: does the model use its conditioning input?
    axx = ax[2, 0]
    axx.plot(v1["step"], v1["dnll_null"], color=BLUE, lw=2, marker="o", ms=4, label="v1 val dNLL_null (latent MSE)")
    axx.plot(v1["step"], v1["dnll_shuf"], color=BLUE, lw=2, ls="--", marker="s", ms=4,
             label="v1 val dNLL_shuf (latent MSE)")
    axx.axhline(0, color=INK, lw=0.8)
    best_line(axx, b1, BLUE)
    style(axx, "v1: dynamics error with null / shuffled audio minus true audio", "Δ error (> 0 = audio used)")
    axx.legend(fontsize=7, frameon=False)

    axx = ax[2, 1]
    axx.plot(v3["step"], v3["dnll_null"], color=ORANGE, lw=2, marker="o", ms=4, label="v3 val dNLL_null (Mel L1)")
    axx.plot(v3["step"], v3["dnll_shuf"], color=ORANGE, lw=2, ls="--", marker="s", ms=4,
             label="v3 val dNLL_shuf (Mel L1)")
    axx.axhline(0, color=INK, lw=0.8)
    best_line(axx, b3, ORANGE)
    style(axx, "v3: next-Mel error with null / shuffled EEG minus true EEG", "Δ error (> 0 = EEG used)")
    axx.legend(fontsize=7, frameon=False)

    axx = ax[2, 2]
    for mode, ls, mk in (("eeg_only", "-", "o"), ("rollout", "--", "s"), ("tf", ":", "^")):
        axx.plot(v3["step"], v3[f"{mode}_eeg_gain"], color=ORANGE, lw=2, ls=ls, marker=mk, ms=4,
                 label=f"v3 val {mode}")
        axx.plot([b3], [m3["seen"][f"{mode}_eeg_gain"]], marker=mk, color=ORANGE, ms=9, mec=INK, mew=1, ls="none")
        axx.plot([b3], [m3["unseen"][f"{mode}_eeg_gain"]], marker=mk, color="white", ms=9, mec=ORANGE, mew=2,
                 ls="none")
    axx.plot([], [], "o", color=ORANGE, mec=INK, label="test seen (filled)")
    axx.plot([], [], "o", color="white", mec=ORANGE, mew=2, label="test unseen (hollow)")
    axx.axhline(0, color=INK, lw=0.8)
    best_line(axx, b3, ORANGE)
    style(axx, "v3 EEG gain: r_env(true EEG) - r_env(shuffled EEG)", "Δ Pearson r")
    axx.legend(fontsize=7, frameon=False)

    fig.suptitle("World model v1 (job 553059) vs v3 (job 553161) on Alice D_T_processed_v2   "
                 "·   dotted vertical = best.pt step   ·   ● test seen  ▲ test unseen (at best step)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"saved {out}")

    # short text summary next to the figure
    print(f"v1: {len(t1['step'])} train points to step {t1['step'][-1]}, {len(v1['step'])} val evals, best {b1}")
    print(f"v3: {len(t3['step'])} train points to step {t3['step'][-1]}, {len(v3['step'])} val evals, best {b3}")


if __name__ == "__main__":
    main()

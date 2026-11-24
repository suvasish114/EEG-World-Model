"""Train / validation loss and accuracy of the mTRF baseline as a function of ridge lambda.

The mTRF (mtrf_baseline.py) is a closed-form ridge regression, so it has no epochs and no
training history. The only "training trajectory" is the regularisation path: for every
lambda of the baseline grid (10^-3 .. 10^3, relative to the mean eigenvalue of X^T X) this
script refits each subject-specific model exactly as mtrf_baseline.py does (same lags,
targets, non-overlapping train windows, sentence-block splits) and records

  loss      : MSE between prediction and z-scored target (envelope + 8 coarse Mel bands)
  accuracy  : r_env  (mean per-window Pearson r of the envelope)
              mm_acc (3 s match-mismatch accuracy, chance 0.5)

on the train, val and test windows (non-overlapping). The baseline picks, per subject, the
lambda with the best mean val r over all 9 targets; that choice is marked in the plots.

Outputs (default figures/mtrf_lambda/):
  lambda_curves.png      4 panels: train/val loss, train/val r_env, train/val mm_acc, chosen lambda
  per_subject_lambda.csv one row per (subject, lambda, split)

Usage: python mtrf_lambda_curves.py --data D_T_processed_v2 --out figures/mtrf_lambda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mtrf_baseline import Ridge, Sums, chunks, corr_rows, evaluate, non_overlapping, predict_windows, valid

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=ROOT / "D_T_processed_v2")
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "mtrf_lambda")
    ap.add_argument("--subjects", nargs="*", default=None, help="default: all subjects in meta.json")
    ap.add_argument("--lag-min", type=float, default=-0.10)
    ap.add_argument("--lag-max", type=float, default=0.25)
    ap.add_argument("--bands", type=int, default=8)
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw lambda_curves.png from the saved per_subject_lambda.csv (no refitting)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    grid = 10.0 ** np.arange(-3, 4)
    if a.plot_only:
        df = pd.read_csv(a.out / "per_subject_lambda.csv")
        val = df[df.split == "val"]
        chosen = val.loc[val.groupby("subject").r_all.idxmax(), ["subject", "lam_rel"]]
        plot(df, chosen, grid, a.out)
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.loads((a.data / "meta.json").read_text())
    fs, T = meta["fs"], meta["window_samples"]
    lags = np.arange(int(round(a.lag_min * fs)), int(round(a.lag_max * fs)) + 1)
    subjects = a.subjects or meta["subjects"]
    print(f"device={dev} subjects={len(subjects)} lags {lags[0]}..{lags[-1]} lambda grid {grid.tolist()}",
          flush=True)

    # targets exactly as mtrf_baseline.py: envelope + coarse bands, z-scored on train frames
    mel = np.load(a.data / "speech_mel.npy").astype(np.float32)
    Y = np.concatenate([mel.mean(1, keepdims=True), mel.reshape(len(mel), a.bands, -1).mean(2)], 1)
    windows = pd.read_csv(a.data / "windows.csv")
    tr = windows[windows.split == "train"].drop_duplicates("audio_start")
    frames = np.unique(np.concatenate([np.arange(s, s + T) for s in tr.audio_start]))
    Y = (Y - Y[frames].mean(0)) / (Y[frames].std(0) + 1e-6)

    rows = []
    for i, s in enumerate(subjects):
        eeg = np.load(a.data / "eeg" / f"{s}.npy").astype(np.float32)

        def sw(split):
            w = windows[(windows.subject == s) & (windows.split == split)].sort_values("audio_start")
            return non_overlapping(valid(w, eeg.shape[1], T, lags), T)

        wins = {k: sw(k) for k in ("train", "val", "test")}
        st = Sums(eeg.shape[0] * len(lags), Y.shape[1], dev)
        for X, y in chunks(eeg, wins["train"], Y, T, lags):
            st.add(X, y, dev)
        model = Ridge(st)
        for lam in grid:
            W = model.weights(lam)
            for split, w in wins.items():
                if not len(w):
                    continue
                P, Yt = predict_windows(model, W, eeg, w, Y, T, lags)
                ev = evaluate(P, Yt, w.audio_start.values, T)
                rows.append(dict(subject=s, lam_rel=lam, split=split, n_windows=len(w),
                                 mse=float(((P - Yt) ** 2).mean()), mse_env=float(((P - Yt)[..., 0] ** 2).mean()),
                                 r_all=float(corr_rows(P, Yt).mean()), **ev))
        df_s = pd.DataFrame([r for r in rows if r["subject"] == s])
        v = df_s[df_s.split == "val"].set_index("lam_rel").r_all
        print(f"[{i + 1}/{len(subjects)}] {s}: chosen lambda {v.idxmax():g} | "
              + " ".join(f"{l:g}:{x:.3f}" for l, x in v.items()), flush=True)
        del st, model
        if dev == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(a.out / "per_subject_lambda.csv", index=False)
    val = df[df.split == "val"]
    chosen = val.loc[val.groupby("subject").r_all.idxmax(), ["subject", "lam_rel"]]
    chosen.to_csv(a.out / "chosen_lambda.csv", index=False)
    plot(df, chosen, grid, a.out)


def plot(df: pd.DataFrame, chosen: pd.DataFrame, grid: np.ndarray, out: Path) -> None:
    """4 panels; the loss panel uses a log y-axis and the median (a few subjects blow up at tiny lambda)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = df.groupby(["split", "lam_rel"])
    mean, sem = g.mean(numeric_only=True), g.sem(numeric_only=True)
    q25, med, q75 = (g.quantile(q, numeric_only=True) for q in (0.25, 0.5, 0.75))
    n_subj = df.subject.nunique()
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    style = {"train": dict(color="C0", marker="o"), "val": dict(color="C1", marker="s"),
             "test": dict(color="grey", marker="^", ls=":", alpha=0.7)}

    # loss: median and interquartile range on a log axis
    a_ = ax[0, 0]
    for split in ("train", "val", "test"):
        m = med.loc[split]["mse"]
        lo, hi = q25.loc[split]["mse"], q75.loc[split]["mse"]
        a_.plot(m.index, m.values, label=f"{split} (median)", **style[split])
        a_.fill_between(m.index, lo.values, hi.values, color=style[split]["color"], alpha=0.15)
    a_.axhline(1.0, c="k", lw=0.7, ls=":")
    a_.text(grid[-1], 1.0, "predict 0 ", va="bottom", ha="right", fontsize=7)
    lo_all = min(q25.loc[s]["mse"].min() for s in ("train", "val", "test"))
    hi_all = max(q75.loc[s]["mse"].max() for s in ("train", "val", "test"))
    pad = 0.1 * (hi_all - lo_all)
    a_.set(xscale="log", yscale="log", ylim=(lo_all - pad, hi_all + pad),
           xlabel="ridge lambda (relative to mean eigenvalue)", ylabel="MSE (z-scored targets, log scale)",
           title="Loss (envelope + 8 Mel bands): median, IQR band")
    big = mean.loc[(slice(None), grid[0]), "mse"]
    a_.text(0.02, 0.97, f"means at lambda={grid[0]:g} (a few subjects blow up): "
            + ", ".join(f"{s} {v:.2f}" for (s, _), v in big.items()),
            transform=a_.transAxes, fontsize=7, va="top")
    a_.legend(fontsize=8, loc="lower right")

    panels = [(ax[0, 1], "r_env", "envelope Pearson r", "Accuracy: envelope reconstruction r"),
              (ax[1, 0], "mm_acc", "match-mismatch accuracy", "Accuracy: 3 s match-mismatch (chance 0.5)")]
    for a_, key, ylab, title in panels:
        for split in ("train", "val", "test"):
            m, e = mean.loc[split][key], sem.loc[split][key]
            a_.errorbar(m.index, m.values, yerr=e.values, capsize=3, label=split, **style[split])
        a_.set(xscale="log", xlabel="ridge lambda (relative to mean eigenvalue)", ylabel=ylab, title=title)
        if key == "mm_acc":
            a_.axhline(0.5, c="k", lw=0.7)
        a_.legend(fontsize=8)
    cnt = chosen.lam_rel.value_counts().reindex(grid, fill_value=0)
    ax[1, 1].bar(range(len(grid)), cnt.values, color="C2")
    ax[1, 1].set_xticks(range(len(grid)))
    ax[1, 1].set_xticklabels([f"{l:g}" for l in grid])
    ax[1, 1].set(xlabel="ridge lambda", ylabel="number of subjects",
                 title="Lambda chosen on validation (best mean val r)")
    fig.suptitle(f"mTRF subject-specific models: train / val / test vs regularisation "
                 f"({n_subj} subjects; accuracy = mean ± SEM)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "lambda_curves.png", dpi=130)
    plt.close(fig)

    print("\n== mean over subjects ==\n" + mean[["mse", "r_env", "mm_acc"]].round(4).to_string())
    print("\n== median MSE ==\n" + med[["mse"]].round(4).to_string())
    print("\n== chosen lambda counts ==\n" + cnt.to_string())
    print(f"saved {out / 'lambda_curves.png'}")


if __name__ == "__main__":
    main()

"""Mel spectrogram plots of mTRF (linear backward model) predictions vs the true audio.

Reuses the ridge code of mtrf_baseline.py (same lags, splits, non-overlapping training
windows, lambda picked on the subject's val windows). mtrf_baseline.py itself predicts the
envelope + 8 coarse Mel bands; to get a real spectrogram this script fits two decoders
per subject:
  baseline : envelope + 8 coarse bands      (exactly the mtrf_baseline.py targets)
  full     : envelope + all 120 Mel bands   (same model, finer targets)

Figures (default figures/mtrf_mel/):
  <subject>_w<k>.png        one 3 s test window: true vs predicted 120-band Mel,
                            true vs predicted 8 bands, envelope overlay with r
  band_correlation.png      test r per Mel band (120-band model), one line per subject
  predictions_<subject>.npz true / predicted arrays of the plotted windows

Usage: python mtrf_mel_plots.py --data D_T_processed_v2 --out figures/mtrf_mel
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mtrf_baseline import Ridge, Sums, chunks, corr_rows, non_overlapping, pick_lambda, predict_windows, valid

ROOT = Path(__file__).resolve().parent


def make_targets(mel: np.ndarray, bands: int, frames: np.ndarray) -> np.ndarray:
    """[frames, 1+bands]: envelope + coarse bands, z-scored on training frames (as mtrf_baseline.py)."""
    b = mel.reshape(len(mel), bands, -1).mean(2)
    Y = np.concatenate([mel.mean(1, keepdims=True), b], 1)
    return (Y - Y[frames].mean(0)) / (Y[frames].std(0) + 1e-6)


def pick_subjects(results: Path, n: int) -> list[str]:
    """Best, median and weakest subject of the subject-specific baseline (by r_env)."""
    df = pd.read_csv(results)
    df = df[df.model == "subject"].sort_values("r_env", ascending=False).reset_index(drop=True)
    idx = sorted({0, len(df) // 2, len(df) - 1})[:n]
    return df.subject.iloc[idx].tolist()


def fit(eeg, wtr, wva, Y, T, lags, dev, grid):
    st = Sums(eeg.shape[0] * len(lags), Y.shape[1], dev)
    for X, y in chunks(eeg, wtr, Y, T, lags):
        st.add(X, y, dev)
    model = Ridge(st)
    lam = pick_lambda(model, grid, [(eeg, wva)], Y, T, lags)
    return model, model.weights(lam), lam


def plot_window(path, title, true120, pred120, true8, pred8, env_t, env_p, fs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = true120.shape[0]
    ext = [0, T / fs, 0, None]
    fig, ax = plt.subplots(5, 1, figsize=(11, 13), gridspec_kw=dict(height_ratios=[3, 3, 1.4, 1.4, 1.6]))

    def show(a, data, label, nb):
        lim = np.percentile(np.abs(data), 98) + 1e-6
        im = a.imshow(data.T, origin="lower", aspect="auto", cmap="magma", vmin=-lim, vmax=lim,
                      extent=[ext[0], ext[1], 0, nb])
        a.set_ylabel(label)
        fig.colorbar(im, ax=a, pad=0.01, label="z")

    r120 = corr_rows(pred120[None], true120[None])[0]
    r8 = corr_rows(pred8[None], true8[None])[0]
    r_env = corr_rows(env_p[None, :, None], env_t[None, :, None])[0, 0]
    show(ax[0], true120, "Mel band\n(true)", 120)
    ax[0].set_title("True log-Mel spectrogram (120 bands, standardized)")
    show(ax[1], pred120, "Mel band\n(predicted)", 120)
    ax[1].set_title(f"mTRF predicted log-Mel, 120-band model (own colour scale; mean band r = {r120.mean():.3f})")
    show(ax[2], true8, "band\n(true)", 8)
    ax[2].set_title("True 8 coarse Mel bands")
    show(ax[3], pred8, "band\n(pred)", 8)
    ax[3].set_title(f"mTRF predicted 8 coarse bands, baseline setting (mean band r = {r8.mean():.3f})")
    z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    t = np.arange(T) / fs
    ax[4].plot(t, z(env_t), label="true envelope", lw=1.2)
    ax[4].plot(t, z(env_p), label=f"mTRF envelope (r = {r_env:.3f})", lw=1.2)
    ax[4].set(xlabel="time in window (s)", ylabel="z", xlim=(0, T / fs))
    ax[4].legend(fontsize=8, loc="upper right")
    for a in ax[:4]:
        a.set_xticklabels([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return float(r_env), float(r8.mean()), float(r120.mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=ROOT / "D_T_processed_v2")
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "mtrf_mel")
    ap.add_argument("--results", type=Path, default=ROOT / "runs" / "mtrf_baseline_v2" / "results.csv",
                    help="baseline results used to pick best / median / weakest subject")
    ap.add_argument("--subjects", nargs="*", default=None, help="default: best, median, weakest subject")
    ap.add_argument("--n-windows", type=int, default=3, help="test windows plotted per subject")
    ap.add_argument("--lag-min", type=float, default=-0.10)
    ap.add_argument("--lag-max", type=float, default=0.25)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.loads((a.data / "meta.json").read_text())
    fs, T = meta["fs"], meta["window_samples"]
    lags = np.arange(int(round(a.lag_min * fs)), int(round(a.lag_max * fs)) + 1)
    grid = 10.0 ** np.arange(-3, 4)
    subjects = a.subjects or (pick_subjects(a.results, 3) if a.results.exists() else meta["subjects"][:3])
    print(f"device={dev} subjects={subjects} lags {lags[0]}..{lags[-1]} ({len(lags)})", flush=True)

    windows = pd.read_csv(a.data / "windows.csv")
    mel = np.load(a.data / "speech_mel.npy").astype(np.float32)
    tr = windows[windows.split == "train"].drop_duplicates("audio_start")
    frames = np.unique(np.concatenate([np.arange(s, s + T) for s in tr.audio_start]))
    Y8, Y120 = make_targets(mel, 8, frames), make_targets(mel, 120, frames)

    band_r, summary = {}, []
    for s in subjects:
        eeg = np.load(a.data / "eeg" / f"{s}.npy").astype(np.float32)

        def sw(split, nonover):
            w = windows[(windows.subject == s) & (windows.split == split)].sort_values("audio_start")
            w = valid(w, eeg.shape[1], T, lags)
            return non_overlapping(w, T) if nonover else w

        wtr, wva, wte = sw("train", True), sw("val", True), sw("test", True)
        m8, W8, lam8 = fit(eeg, wtr, wva, Y8, T, lags, dev, grid)
        m120, W120, lam120 = fit(eeg, wtr, wva, Y120, T, lags, dev, grid)

        # band-wise test correlation of the 120-band model over all non-overlapping test windows
        P, Yt = predict_windows(m120, W120, eeg, wte, Y120, T, lags)
        band_r[s] = corr_rows(P, Yt).mean(0)  # [121]: envelope + 120 bands

        pick = wte.iloc[np.linspace(0, len(wte) - 1, min(a.n_windows, len(wte))).astype(int)]
        P8, T8 = predict_windows(m8, W8, eeg, pick, Y8, T, lags)
        P120, T120 = predict_windows(m120, W120, eeg, pick, Y120, T, lags)
        np.savez_compressed(a.out / f"predictions_{s}.npz", true_mel120=T120[..., 1:], pred_mel120=P120[..., 1:],
                            true_bands8=T8[..., 1:], pred_bands8=P8[..., 1:], true_env=T8[..., 0],
                            pred_env=P8[..., 0], audio_start=pick.audio_start.to_numpy())
        for k, r in enumerate(pick.itertuples()):
            t0 = (r.audio_start - meta["clip_offsets"][r.audio_clip - 1]) / fs
            title = (f"{s} | test window {k + 1}: clip {r.audio_clip}, {t0:.1f}-{t0 + T / fs:.1f} s | "
                     f"lambda 8-band {lam8:g}, 120-band {lam120:g}\n\"{r.text}\"")
            re, r8, r120 = plot_window(a.out / f"{s}_w{k + 1}.png", title, T120[k, :, 1:], P120[k, :, 1:],
                                       T8[k, :, 1:], P8[k, :, 1:], T8[k, :, 0], P8[k, :, 0], fs)
            summary.append(dict(subject=s, window=k + 1, audio_start=int(r.audio_start), clip=int(r.audio_clip),
                                t0_s=round(t0, 2), r_env=re, mean_r_8bands=r8, mean_r_120bands=r120,
                                text=r.text))
        print(f"{s}: {len(wtr)} train / {len(wte)} test windows, lambda {lam8:g} / {lam120:g}, "
              f"test mean band r (120) {band_r[s][1:].mean():.3f}, env r {band_r[s][0]:.3f}", flush=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for s, r in band_r.items():
        ax.plot(np.arange(120), r[1:], label=f"{s} (env r {r[0]:.3f})")
    ax.axhline(0, c="k", lw=0.7)
    ax.set(xlabel="Mel band (low -> high frequency)", ylabel="test Pearson r",
           title="mTRF reconstruction accuracy per Mel band (120-band model, non-overlapping 3 s test windows)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out / "band_correlation.png", dpi=130)
    plt.close(fig)

    pd.DataFrame(summary).to_csv(a.out / "windows_plotted.csv", index=False)
    print(pd.DataFrame(summary).drop(columns="text").to_string(index=False))
    print(f"saved figures to {a.out}")


if __name__ == "__main__":
    main()

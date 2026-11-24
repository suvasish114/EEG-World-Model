"""Linear backward-model (mTRF-style) baseline for EEG -> speech on the Alice data.

Reconstructs the speech envelope and a few coarse mel bands from time-lagged EEG
with ridge regression. This is the floor every deep model (SENSE-style or other)
has to beat, and it shows which subjects actually track the audio.

Input: the output of prepare_alice_dataset.py (D_T_processed/)
  eeg/Sxx.npy      [61, time] at 120 Hz, robust-scaled
  speech_mel.npy   [frames, 120] at 120 Hz, standardized per band
  windows.csv      3 s windows with subject, split, eeg_start, audio_start
                   (eeg_start already includes the 150 ms brain delay)

Targets (per frame): envelope = mean of the log-mel bands, plus --bands coarse
bands (averages of adjacent mel bands). All targets are z-scored on train frames.

Models
  subject : trained on that subject's train windows
  pooled  : trained on every OTHER subject's train windows (unseen subject);
            ridge lambda picked on the val windows of other subjects only
Lambda for the subject model is picked on that subject's val windows.
Splits are the sentence-block splits from prepare_alice_dataset.py, so test
sentences were never heard in training by any subject.

Metrics on test windows (all 3 s test windows of the subject)
  r_env, r_bands   : mean Pearson r per window between prediction and truth
  r_env_null       : same, but prediction paired with a different test window
                     whose audio does not overlap (chance level, should be ~0)
  mm_acc           : match-mismatch accuracy on the envelope: fraction of
                     (true, wrong) pairs where the prediction correlates more
                     with the true window (chance = 0.5)

Output (runs/mtrf_baseline/): results.csv, summary.json, mtrf_baseline.png

Usage:  python mtrf_baseline.py [--data D_T_processed] [--lag-min -0.1 --lag-max 0.25]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent


# ----------------------------------------------------------------- data utils
def non_overlapping(df: pd.DataFrame, length: int) -> pd.DataFrame:
    """Greedy subset of windows whose audio does not overlap (one subject, one split)."""
    keep, last = [], -10 ** 9
    for idx, a in zip(df.index, df.audio_start):
        if a >= last + length:
            keep.append(idx)
            last = a
    return df.loc[keep]


def lagged(eeg: np.ndarray, start: int, T: int, lags: np.ndarray) -> np.ndarray:
    """[T, C*L] design matrix: X[t, c, l] = eeg[c, start + t + lags[l]]."""
    seg = eeg[:, start + lags[0]: start + T + lags[-1]]
    X = np.lib.stride_tricks.sliding_window_view(seg, len(lags), axis=1)  # [C, T, L]
    return X.transpose(1, 0, 2).reshape(T, -1)


def valid(win: pd.DataFrame, n_eeg: int, T: int, lags: np.ndarray) -> pd.DataFrame:
    s = win.eeg_start
    return win[(s + lags[0] >= 0) & (s + T + lags[-1] <= n_eeg)]


def design(eeg, win, T, lags):
    return np.concatenate([lagged(eeg, s, T, lags) for s in win.eeg_start]).astype(np.float32)


def targets(Y, win, T):
    return np.concatenate([Y[a:a + T] for a in win.audio_start]).astype(np.float32)


def chunks(eeg, win, Y, T, lags, size: int = 24):
    """Yield (X [b*T, F], Y [b*T, K]) over groups of windows, to bound memory."""
    for i in range(0, len(win), size):
        w = win.iloc[i:i + size]
        yield design(eeg, w, T, lags), targets(Y, w, T)


# ---------------------------------------------------------------------- ridge
class Sums:
    """Sufficient statistics for centered ridge regression."""

    def __init__(self, F: int, K: int, dev):
        self.xx = torch.zeros(F, F, dtype=torch.float64, device=dev)
        self.xy = torch.zeros(F, K, dtype=torch.float64, device=dev)
        self.x = torch.zeros(F, dtype=torch.float64, device=dev)
        self.y = torch.zeros(K, dtype=torch.float64, device=dev)
        self.n = 0

    def add(self, X: np.ndarray, Y: np.ndarray, dev, chunk: int = 20000):
        for i in range(0, len(X), chunk):
            x = torch.from_numpy(X[i:i + chunk]).to(dev)
            y = torch.from_numpy(Y[i:i + chunk]).to(dev)
            self.xx += (x.T @ x).double()
            self.xy += (x.T @ y).double()
            self.x += x.double().sum(0)
            self.y += y.double().sum(0)
            self.n += len(x)

    def __iadd__(self, o: "Sums"):
        self.xx += o.xx; self.xy += o.xy; self.x += o.x; self.y += o.y; self.n += o.n
        return self

    def __isub__(self, o: "Sums"):
        self.xx -= o.xx; self.xy -= o.xy; self.x -= o.x; self.y -= o.y; self.n -= o.n
        return self

    def to(self, dev, dtype=torch.float64) -> "Sums":
        s = Sums.__new__(Sums)
        s.xx, s.xy, s.x, s.y = (t.to(device=dev, dtype=dtype, copy=True)
                                for t in (self.xx, self.xy, self.x, self.y))
        s.n = self.n
        return s


class Ridge:
    """All lambdas from one eigendecomposition of the centered covariance."""

    def __init__(self, s: Sums):
        mx, my = s.x / s.n, s.y / s.n
        cxx = s.xx - s.n * torch.outer(mx, mx)
        cxy = s.xy - s.n * torch.outer(mx, my)
        self.w, self.V = torch.linalg.eigh(cxx)
        self.Vtc = self.V.T @ cxy
        self.mx, self.my = mx, my
        self.scale = float(self.w.clamp(min=0).mean())

    def weights(self, lam_rel: float) -> torch.Tensor:
        lam = lam_rel * self.scale
        return self.V @ (self.Vtc / (self.w.clamp(min=0) + lam)[:, None])

    def predict(self, X: np.ndarray, W: torch.Tensor) -> np.ndarray:
        x = torch.from_numpy(X).to(W.device, torch.float64)
        return ((x - self.mx) @ W + self.my).float().cpu().numpy()


# -------------------------------------------------------------------- metrics
def corr_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r along axis -2 (time) for [..., T, K] arrays -> [..., K]."""
    a = a - a.mean(-2, keepdims=True)
    b = b - b.mean(-2, keepdims=True)
    return (a * b).sum(-2) / (np.sqrt((a ** 2).sum(-2) * (b ** 2).sum(-2)) + 1e-12)


def evaluate(pred: np.ndarray, true: np.ndarray, audio_start: np.ndarray, T: int) -> dict:
    """pred/true: [n_win, T, K]; column 0 is the envelope."""
    n = len(pred)
    r = corr_rows(pred, true)  # [n, K]
    env_p = pred[:, :, 0]
    env_t = true[:, :, 0]
    zp = (env_p - env_p.mean(1, keepdims=True)) / (env_p.std(1, keepdims=True) + 1e-12)
    zt = (env_t - env_t.mean(1, keepdims=True)) / (env_t.std(1, keepdims=True) + 1e-12)
    R = zp @ zt.T / T  # R[i, j] = corr(pred_i, true_j)
    ok = np.abs(audio_start[:, None] - audio_start[None, :]) >= T  # non-overlapping audio
    wins = (np.diag(R)[:, None] > R) & ok
    mm = wins.sum() / max(ok.sum(), 1)
    null = R[ok].mean() if ok.any() else np.nan
    return dict(n_test_windows=n, r_env=float(r[:, 0].mean()),
                r_bands=float(r[:, 1:].mean()) if r.shape[1] > 1 else np.nan,
                r_env_null=float(null), mm_acc=float(mm))


def predict_windows(model: Ridge, W, eeg, win, Y, T, lags):
    """-> pred, true: [n_win, T, K]."""
    P, Yt = [], []
    for X, y in chunks(eeg, win, Y, T, lags):
        P.append(model.predict(X, W))
        Yt.append(y)
    K = Y.shape[1]
    return (np.concatenate(P).reshape(len(win), T, K),
            np.concatenate(Yt).reshape(len(win), T, K))


def pick_lambda(model: Ridge, grid, val_sets, Y, T, lags) -> float:
    """val_sets: list of (eeg, windows). Mean per-window r over all targets."""
    if not any(len(w) for _, w in val_sets):
        return 1.0
    scores = []
    for lam in grid:
        W = model.weights(lam)
        rs = [corr_rows(*predict_windows(model, W, e, w, Y, T, lags)).mean()
              for e, w in val_sets if len(w)]
        scores.append(np.mean(rs))
    return float(grid[int(np.argmax(scores))])


# ----------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=ROOT / "D_T_processed")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "mtrf_baseline")
    ap.add_argument("--lag-min", type=float, default=-0.10,
                    help="s, relative to eeg_start (which is already audio + 150 ms)")
    ap.add_argument("--lag-max", type=float, default=0.25)
    ap.add_argument("--bands", type=int, default=8, help="coarse mel bands (must divide 120)")
    ap.add_argument("--subjects", nargs="*", default=None, help="default: all in meta.json")
    ap.add_argument("--no-pooled", action="store_true", help="skip the unseen-subject model")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.loads((args.data / "meta.json").read_text())
    fs, T = meta["fs"], meta["window_samples"]
    lags = np.arange(int(round(args.lag_min * fs)), int(round(args.lag_max * fs)) + 1)
    subjects = args.subjects or meta["subjects"]
    print(f"device={dev} fs={fs} window={T} samples lags={lags[0]}..{lags[-1]} "
          f"({len(lags)}) subjects={len(subjects)}", flush=True)

    # targets: envelope + coarse bands, z-scored on train frames
    mel = np.load(args.data / "speech_mel.npy").astype(np.float32)
    bands = mel.reshape(len(mel), args.bands, -1).mean(2)
    Y = np.concatenate([mel.mean(1, keepdims=True), bands], 1)
    windows = pd.read_csv(args.data / "windows.csv")
    tr = windows[windows.split == "train"].drop_duplicates("audio_start")
    frames = np.unique(np.concatenate([np.arange(a, a + T) for a in tr.audio_start]))
    Y = (Y - Y[frames].mean(0)) / (Y[frames].std(0) + 1e-6)
    K = Y.shape[1]

    grid = 10.0 ** np.arange(-3, 4)  # relative to mean eigenvalue
    eegs = {s: np.load(args.data / "eeg" / f"{s}.npy", mmap_mode="r") for s in subjects}
    C = eegs[subjects[0]].shape[0]
    F = C * len(lags)

    def subject_windows(s, split, nonover):
        w = windows[(windows.subject == s) & (windows.split == split)].sort_values("audio_start")
        w = valid(w, eegs[s].shape[1], T, lags)
        return non_overlapping(w, T) if nonover else w

    # pass 1: per-subject train sums (kept on CPU in float32 for the pooled model)
    sums, rows, preds_example = {}, [], None
    for i, s in enumerate(subjects):
        eeg = np.asarray(eegs[s], dtype=np.float32)
        wtr = subject_windows(s, "train", True)
        st = Sums(F, K, dev)
        for X, y in chunks(eeg, wtr, Y, T, lags):
            st.add(X, y, dev)

        # subject-specific model
        model = Ridge(st)
        lam = pick_lambda(model, grid, [(eeg, subject_windows(s, "val", True))], Y, T, lags)
        wte = subject_windows(s, "test", False)
        P, Yte = predict_windows(model, model.weights(lam), eeg, wte, Y, T, lags)
        res = evaluate(P, Yte, wte.audio_start.values, T)
        rows.append(dict(subject=s, model="subject", lam_rel=lam, n_train_windows=len(wtr), **res))
        print(f"[{i + 1}/{len(subjects)}] {s} subject: r_env={res['r_env']:.3f} "
              f"null={res['r_env_null']:.3f} r_bands={res['r_bands']:.3f} "
              f"mm_acc={res['mm_acc']:.3f} (lam={lam:g}, {len(wtr)} train / {len(wte)} test win)",
              flush=True)
        if preds_example is None:
            preds_example = (s, P[:4, :, 0], Yte[:4, :, 0])
        if not args.no_pooled:
            sums[s] = st.to("cpu", torch.float32)
        del st, model
        if dev == "cuda":
            torch.cuda.empty_cache()

    # pass 2: pooled model, leave one subject out
    if not args.no_pooled and len(subjects) > 1:
        total = Sums(F, K, "cpu")
        for st in sums.values():
            total += st.to("cpu")
        for i, s in enumerate(subjects):
            st = total.to("cpu")
            st -= sums[s].to("cpu")
            model = Ridge(st.to(dev))
            # lambda from the val windows of 3 other subjects (never the held-out one)
            others = [subjects[(i + k) % len(subjects)] for k in (1, 2, 3) if len(subjects) > k]
            val_sets = [(np.asarray(eegs[o], dtype=np.float32), subject_windows(o, "val", True))
                        for o in others]
            lam = pick_lambda(model, grid, val_sets, Y, T, lags)
            eeg = np.asarray(eegs[s], dtype=np.float32)
            wte = subject_windows(s, "test", False)
            P, Yte = predict_windows(model, model.weights(lam), eeg, wte, Y, T, lags)
            res = evaluate(P, Yte, wte.audio_start.values, T)
            rows.append(dict(subject=s, model="pooled", lam_rel=lam, n_train_windows=None, **res))
            print(f"[{i + 1}/{len(subjects)}] {s} pooled:  r_env={res['r_env']:.3f} "
                  f"null={res['r_env_null']:.3f} r_bands={res['r_bands']:.3f} "
                  f"mm_acc={res['mm_acc']:.3f} (lam={lam:g})", flush=True)
            del model, st
            if dev == "cuda":
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "results.csv", index=False)
    summary = {}
    for m, g in df.groupby("model"):
        summary[m] = {k: dict(mean=float(g[k].mean()), sem=float(g[k].std() / np.sqrt(len(g))))
                      for k in ("r_env", "r_env_null", "r_bands", "mm_acc")}
        summary[m]["n_subjects"] = int(len(g))
        summary[m]["subjects_above_null"] = int((g.r_env > g.r_env_null + 0.02).sum())
    summary["settings"] = dict(lags_samples=[int(lags[0]), int(lags[-1])], fs=fs,
                               window_samples=T, bands=args.bands, lambda_grid=grid.tolist())
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
        for m, g in df.groupby("model"):
            ax[0].plot(g.subject, g.r_env, "o-", ms=3, label=f"{m} r_env")
            ax[0].plot(g.subject, g.r_env_null, "x:", ms=3, label=f"{m} null")
            ax[1].plot(g.subject, g.mm_acc, "o-", ms=3, label=m)
        ax[0].set(title="Envelope reconstruction (test)", ylabel="Pearson r")
        ax[1].axhline(0.5, c="k", lw=0.8)
        ax[1].set(title="Match-mismatch accuracy (3 s)", ylabel="accuracy")
        for a in ax[:2]:
            a.tick_params(axis="x", rotation=90, labelsize=6)
            a.legend(fontsize=7)
        s, p, t = preds_example
        tt = np.arange(p.shape[1]) / fs
        ax[2].plot(tt, (t[0] - t[0].mean()) / t[0].std(), label="true envelope")
        ax[2].plot(tt, (p[0] - p[0].mean()) / p[0].std(), label="EEG reconstruction")
        ax[2].set(title=f"{s}, first test window", xlabel="s")
        ax[2].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.out / "mtrf_baseline.png", dpi=120)
        print(f"saved {args.out / 'mtrf_baseline.png'}")
    except Exception as e:  # plotting is optional
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()

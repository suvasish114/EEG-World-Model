"""Per-clip EEG/audio alignment check for every subject.

Why: in mtrf_baseline.py most failing subjects form one ID block (S22-S36), and
some of them (S22, S11, S27, S30) decode better with a decoder from OTHER
subjects than with their own, i.e. their test EEG tracks the audio but their
own training part may be misaligned. This script finds, for every
(subject, audio clip), the time offset at which the EEG best matches the audio.

Method
  1. Clip start per subject/clip is recovered from windows.csv exactly as
     prepare_alice_dataset.py placed it (eeg_start - audio position - 150 ms).
  2. A linear envelope decoder (same lags as mtrf_baseline.py) is trained on
     the best subjects of runs/mtrf_baseline/results.csv, split in two halves
     A/B. A subject in A is decoded with B's decoder and vice versa; all other
     subjects use B. So no subject is ever decoded by a model trained on itself.
  3. The decoder reconstructs a continuous envelope from the subject's whole
     recording. For each clip, the correlation with the true clip envelope is
     scanned over offsets of +-6 s around the assumed start.
     offset 0 = alignment used in D_T_processed; a sharp peak elsewhere means
     that clip is shifted by that amount.
  4. The assumed start is also compared with the hardware marker "Stimulus,k"
     in the BrainVision .vmrk file.

Output (runs/clip_alignment/)
  clips.csv     one row per (subject, clip): offset_ms of the best match,
                peak_r, r_at_0, peak_z (peak height in SDs of the scan),
                marker_diff_ms, fraction of the clip's windows in each split,
                verdict: ok / shifted / no_signal
  subjects.csv  per subject: clips ok / shifted / no_signal, median offset
  clip_alignment.png  heatmaps subject x clip of offset and peak_r

Usage:  python check_clip_alignment.py [--data D_T_processed] [--n-ref 16]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, sosfiltfilt

from mtrf_baseline import Ridge, Sums, lagged


def bandpass(x: np.ndarray, sos) -> np.ndarray:
    """Zero-phase band-pass that leaves NaN edges (from reconstruct) as NaN."""
    out = np.full_like(x, np.nan, dtype=np.float32)
    ok = ~np.isnan(x)
    if ok.sum() > 100:
        i0, i1 = np.flatnonzero(ok)[[0, -1]]
        out[i0:i1 + 1] = sosfiltfilt(sos, x[i0:i1 + 1]).astype(np.float32)
    return out

ROOT = Path(__file__).resolve().parent
SHIFT_S = 0.15  # brain delay already built into eeg_start by prepare_alice_dataset.py


# --------------------------------------------------------------------- inputs
def clip_starts(windows: pd.DataFrame, offs: list[int], shift: int) -> dict:
    """{subject: {clip: EEG sample (120 Hz) where audio frame 0 of the clip sits}}."""
    out = {}
    for s, g in windows.groupby("subject"):
        rel = g.eeg_start - (g.audio_start - g.audio_clip.map(lambda c: offs[c - 1])) - shift
        out[s] = {int(c): int(round(np.median(r))) for c, r in rel.groupby(g.audio_clip)}
    return out


def vmrk_markers(path: Path) -> dict[int, float]:
    """{clip number: marker time in s} from 'Stimulus,<k>' entries (500 Hz, 1-based)."""
    if not path.exists():
        return {}
    text = path.read_text(errors="ignore")
    sf = 500.0
    vhdr = path.with_suffix(".vhdr")
    if vhdr.exists():
        m = re.search(r"SamplingInterval=(\d+)", vhdr.read_text(errors="ignore"))
        if m:
            sf = 1e6 / float(m.group(1))
    out = {}
    for typ, desc, pos in re.findall(r"Mk\d+=([^,]*),([^,]*),(\d+),", text):
        d = re.sub(r"\D", "", desc)
        if typ.strip() == "Stimulus" and d and 1 <= int(d) <= 12 and int(d) not in out:
            out[int(d)] = (int(pos) - 1) / sf
    return out


# -------------------------------------------------------------------- decoder
def frame_design(eeg, start, n, lags, shift):
    """Rows for audio frames start..start+n-1 (EEG positions), lags after the brain delay."""
    return lagged(eeg, start + shift, n, lags).astype(np.float32)


def fit_decoder(subjects, eegs, starts, lens, env, offs, lags, shift, dev, lam_rel):
    st = Sums(eegs[subjects[0]].shape[0] * len(lags), 1, dev)
    for s in subjects:
        eeg = np.asarray(eegs[s], dtype=np.float32)
        for c, s0 in starts[s].items():
            n = lens[c - 1]
            if s0 + shift + lags[0] < 0 or s0 + shift + n + lags[-1] > eeg.shape[1]:
                continue
            for i in range(0, n, 3000):
                m = min(3000, n - i)
                X = frame_design(eeg, s0 + i, m, lags, shift)
                y = env[offs[c - 1] + i: offs[c - 1] + i + m, None]
                st.add(X, y, dev)
    model = Ridge(st)
    return model, model.weights(lam_rel)


def reconstruct(model, W, eeg, lags, shift):
    """Envelope estimate rec[m] for every EEG sample m where the lags fit (NaN elsewhere)."""
    N = eeg.shape[1]
    rec = np.full(N, np.nan, dtype=np.float32)
    first = max(0, -(shift + lags[0]))
    last = N - (shift + lags[-1])  # exclusive
    for i in range(first, last, 4000):
        m = min(4000, last - i)
        rec[i:i + m] = model.predict(frame_design(eeg, i, m, lags, shift), W)[:, 0]
    return rec


def offset_scan(rec, s0, e, max_off):
    """corr(rec[s0+d : s0+d+n], e) for d in -max_off..max_off (NaN if out of range)."""
    n = len(e)
    ez = (e - e.mean()) / (e.std() + 1e-12)
    d = np.arange(-max_off, max_off + 1)
    r = np.full(len(d), np.nan)
    lo, hi = s0 - max_off, s0 + max_off + n
    a, b = max(lo, 0), min(hi, len(rec))
    seg = rec[a:b]
    if len(seg) < n:
        return d, r
    win = np.lib.stride_tricks.sliding_window_view(seg, n)  # offsets a-s0 .. b-n-s0
    valid = ~np.isnan(win).any(1)
    wz = (win - np.nanmean(win, 1, keepdims=True)) / (np.nanstd(win, 1, keepdims=True) + 1e-12)
    rr = np.where(valid, (wz * ez).mean(1), np.nan)
    k0 = a - lo
    r[k0:k0 + len(rr)] = rr
    return d, r


# ----------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=ROOT / "D_T_processed")
    ap.add_argument("--raw", type=Path, default=ROOT / "Dataset" / "eeg")
    ap.add_argument("--baseline", type=Path, default=ROOT / "runs" / "mtrf_baseline" / "results.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "clip_alignment")
    ap.add_argument("--n-ref", type=int, default=16, help="best subjects used to train the decoders")
    ap.add_argument("--lag-min", type=float, default=-0.10)
    ap.add_argument("--lag-max", type=float, default=0.25)
    ap.add_argument("--lam", type=float, default=0.1, help="relative ridge lambda (baseline's usual pick)")
    ap.add_argument("--max-offset", type=float, default=6.0, help="s, offset scan range")
    ap.add_argument("--shift-ms", type=float, default=100.0, help="|offset| above this = shifted")
    ap.add_argument("--min-z", type=float, default=4.0, help="peak_z below this = no_signal")
    ap.add_argument("--band-lo", type=float, default=1.0, help="Hz, scan band-pass")
    ap.add_argument("--band-hi", type=float, default=8.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.loads((args.data / "meta.json").read_text())
    fs, offs, lens = meta["fs"], meta["clip_offsets"], meta["clip_lengths"]
    shift = int(round(SHIFT_S * fs))
    lags = np.arange(int(round(args.lag_min * fs)), int(round(args.lag_max * fs)) + 1)
    max_off = int(round(args.max_offset * fs))

    mel = np.load(args.data / "speech_mel.npy").astype(np.float32)
    env = mel.mean(1)
    env = (env - env.mean()) / env.std()
    windows = pd.read_csv(args.data / "windows.csv")
    subjects = meta["subjects"]
    starts = clip_starts(windows, offs, shift)
    eegs = {s: np.load(args.data / "eeg" / f"{s}.npy", mmap_mode="r") for s in subjects}

    base = pd.read_csv(args.baseline)
    ranked = (base[base.model == "subject"].sort_values("r_env", ascending=False).subject.tolist())
    ref = [s for s in ranked if s in subjects][:args.n_ref]
    ref_a, ref_b = ref[0::2], ref[1::2]
    print(f"device={dev} reference A={ref_a}\n reference B={ref_b}", flush=True)

    dec = {}
    for name, group in (("A", ref_a), ("B", ref_b)):
        dec[name] = fit_decoder(group, eegs, starts, lens, env, offs, lags, shift, dev, args.lam)
        print(f"decoder {name} trained on {len(group)} subjects", flush=True)

    split_frac = (windows.groupby(["subject", "audio_clip"]).split
                  .value_counts(normalize=True).unstack(fill_value=0.0))

    # Whole-clip correlations are dominated by slow drifts (pauses, EEG drift), which
    # flatten the offset scan. Band-pass both signals to the 1-8 Hz tracking band.
    sos = butter(4, [args.band_lo, args.band_hi], btype="band", fs=fs, output="sos")
    env_bp = np.concatenate([bandpass(env[o:o + n], sos) for o, n in zip(offs, lens)])

    rows, scans = [], {}
    for s in subjects:
        which = "B" if s in ref_a else "A" if s in ref_b else "B"
        model, W = dec[which]
        eeg = np.asarray(eegs[s], dtype=np.float32)
        rec = bandpass(reconstruct(model, W, eeg, lags, shift), sos)
        marks = vmrk_markers(args.raw / f"{s}.vmrk")
        for c in range(1, len(lens) + 1):
            row = dict(subject=s, clip=c, decoder=which, ref=s in ref)
            if c not in starts[s]:
                rows.append(dict(row, verdict="missing"))
                continue
            s0 = starts[s][c]
            e = env_bp[offs[c - 1]: offs[c - 1] + lens[c - 1]]
            d, r = offset_scan(rec, s0, e, max_off)
            scans[(s, c)] = r
            if np.all(np.isnan(r)):
                rows.append(dict(row, verdict="missing"))
                continue
            k = int(np.nanargmax(r))
            z = (r[k] - np.nanmean(r)) / (np.nanstd(r) + 1e-12)
            off_ms = 1e3 * d[k] / fs
            verdict = ("no_signal" if z < args.min_z else
                       "shifted" if abs(off_ms) > args.shift_ms else "ok")
            fr = split_frac.loc[(s, c)] if (s, c) in split_frac.index else {}
            rows.append(dict(
                row, start_s=s0 / fs, offset_ms=off_ms, peak_r=float(r[k]),
                r_at_0=float(r[max_off]), peak_z=float(z),
                marker_diff_ms=(1e3 * (marks[c] - s0 / fs) if c in marks else np.nan),
                frac_train=float(fr.get("train", 0)), frac_val=float(fr.get("val", 0)),
                frac_test=float(fr.get("test", 0)), verdict=verdict))
        g = pd.DataFrame([x for x in rows if x["subject"] == s])
        print(f"{s} [{which}] r~{g.peak_r.median():.3f} z~{g.peak_z.median():.1f}: " + " ".join(
            f"{int(x.clip)}:{x.verdict[0]}{'' if x.verdict != 'shifted' else f'({x.offset_ms:+.0f})'}"
            for x in g.itertuples()), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "clips.csv", index=False)

    subj = []
    for s, g in df.groupby("subject", sort=False):
        ok = g[g.verdict == "ok"]
        subj.append(dict(
            subject=s, ref=bool(g.ref.iloc[0]),
            n_ok=int((g.verdict == "ok").sum()), n_shifted=int((g.verdict == "shifted").sum()),
            n_no_signal=int((g.verdict == "no_signal").sum()), n_missing=int((g.verdict == "missing").sum()),
            median_peak_r=float(g.peak_r.median()) if "peak_r" in g else np.nan,
            median_offset_ms_ok=float(ok.offset_ms.median()) if len(ok) else np.nan,
            shifted_clips=" ".join(f"{int(c)}({o:+.0f}ms)" for c, o in
                                   zip(g.loc[g.verdict == "shifted", "clip"],
                                       g.loc[g.verdict == "shifted", "offset_ms"])),
            max_abs_marker_diff_ms=float(g.marker_diff_ms.abs().max()) if "marker_diff_ms" in g else np.nan))
    sdf = pd.DataFrame(subj)
    sdf.to_csv(args.out / "subjects.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print("\n=== per subject ===")
        print(sdf.to_string(index=False))
    print("\nverdict counts:", df.verdict.value_counts().to_dict())
    ok = df[df.verdict == "ok"]
    if len(ok):
        print(f"offset of 'ok' clips: median {ok.offset_ms.median():.0f} ms, "
              f"IQR {ok.offset_ms.quantile(.25):.0f}..{ok.offset_ms.quantile(.75):.0f} ms")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        off = df.pivot(index="subject", columns="clip", values="offset_ms").reindex(subjects)
        pk = df.pivot(index="subject", columns="clip", values="peak_z").reindex(subjects)
        fig, ax = plt.subplots(1, 3, figsize=(16, 9), gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
        im = ax[0].imshow(np.clip(off.values.astype(float), -1000, 1000), cmap="coolwarm",
                          vmin=-1000, vmax=1000, aspect="auto")
        ax[0].set(title="best offset (ms, clipped +-1 s)", xlabel="clip")
        fig.colorbar(im, ax=ax[0], shrink=0.6)
        im = ax[1].imshow(pk.values.astype(float), cmap="viridis", vmin=0, vmax=12, aspect="auto")
        ax[1].set(title="peak height (z)", xlabel="clip")
        fig.colorbar(im, ax=ax[1], shrink=0.6)
        for a in ax[:2]:
            a.set_yticks(range(len(subjects)), subjects, fontsize=6)
            a.set_xticks(range(len(lens)), range(1, len(lens) + 1), fontsize=7)
        t = np.arange(-max_off, max_off + 1) / fs
        for s in [x for x in ("S10", "S22", "S27", "S32") if x in subjects]:
            r = np.nanmean([scans[(s, c)] for c in range(1, len(lens) + 1) if (s, c) in scans], 0)
            ax[2].plot(t, r, lw=0.8, label=s)
        ax[2].axvline(0, c="k", lw=0.8)
        ax[2].set(title="offset scan, mean over clips", xlabel="offset (s)", ylabel="r")
        ax[2].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(args.out / "clip_alignment.png", dpi=110)
        print(f"saved {args.out / 'clip_alignment.png'}")
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()

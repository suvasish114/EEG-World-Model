"""Test: do the failing subjects decode once clips are aligned from the .vmrk markers?

Background (runs/clip_alignment/clips.csv): for most subjects the BrainVision
marker "Stimulus,k" sits ~50 ms before the clip start that prepare_alice_dataset.py
takes from the authors' trial table (proc.trl). Two groups disagree:
  * S24 S26 S27 S30 S32 S34 S35 S36: trl clip k sits where clip k+1 starts
    (no marker 1, no trl clip 12) -> suspected off-by-one clip labels
  * S02: markers ~2.55 s before the trl start on every clip
All of them decoded at chance in mtrf_baseline.py.

This script rebuilds the 3 s windows for every subject under two alignments
that differ ONLY in where each clip starts in the EEG:
  trl     clip start from windows.csv (what D_T_processed uses)
  marker  clip start = vmrk marker time + typical marker->trl delay
          (median over all subject/clips that agree within 0.5 s)
Same words, same sentence-block splits, same EEG (D_T_processed/eeg), no trial
rejection in either case, so differences come from alignment alone.

Two decoders are evaluated on each subject's test windows (held-out sentences):
  subject  ridge trained on the subject's own train windows (lambda on its val)
  cross    ridge trained on the train windows of the best baseline subjects
           (trl alignment; two halves A/B so a subject is never in its own
           decoder; fixed relative lambda). Independent of the subject's own
           training data, so it tests the alignment of the test part directly.

Output (runs/realign_markers/)
  results.csv        per subject: r_env / mm_acc for {subject, cross} x {trl, marker}
  clip_starts.json   per subject and clip: trl and marker start (s), for fixing
                     prepare_alice_dataset.py afterwards
Usage:  python realign_from_markers.py [--data D_T_processed] [--n-ref 16]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from check_clip_alignment import clip_starts, vmrk_markers
from mtrf_baseline import (Ridge, Sums, chunks, evaluate, non_overlapping, pick_lambda,
                           predict_windows, valid)
from prepare_alice_dataset import assign_splits, candidate_windows

ROOT = Path(__file__).resolve().parent
SHIFT_S = 0.15


def build_windows(base, start, fs, n_eeg, T, lags):
    """base windows placed with clip starts `start` ({clip: EEG sample at 120 Hz})."""
    w = base[base.audio_clip.isin(list(start))].copy()
    w["eeg_start"] = [start[c] + int(round((t0 + SHIFT_S) * fs))
                      for c, t0 in zip(w.audio_clip, w.audio_t0)]
    return valid(w, n_eeg, T, lags).sort_values("audio_start")


def split_of(w, name):
    return w[w.split == name]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=ROOT / "D_T_processed")
    ap.add_argument("--dataset", type=Path, default=ROOT / "Dataset")
    ap.add_argument("--baseline", type=Path, default=ROOT / "runs" / "mtrf_baseline" / "results.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "realign_markers")
    ap.add_argument("--n-ref", type=int, default=16)
    ap.add_argument("--lag-min", type=float, default=-0.10)
    ap.add_argument("--lag-max", type=float, default=0.25)
    ap.add_argument("--bands", type=int, default=8)
    ap.add_argument("--cross-lam", type=float, default=0.1)
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.loads((args.data / "meta.json").read_text())
    fs, T, offs = meta["fs"], meta["window_samples"], meta["clip_offsets"]
    shift = int(round(SHIFT_S * fs))
    lags = np.arange(int(round(args.lag_min * fs)), int(round(args.lag_max * fs)) + 1)
    subjects = args.subjects or meta["subjects"]
    grid = 10.0 ** np.arange(-3, 4)

    # ---- all candidate windows (same rule and splits as prepare_alice_dataset.py)
    words = pd.read_csv(args.dataset / "AliceChapterOne-EEG.csv")
    words["split"] = words.Sentence.map(assign_splits(words))
    n_clips = int(words.Segment.max())
    durations = [sf.info(args.dataset / "audio" / f"DownTheRabbitHoleFinal_SoundFile{c}.wav").duration
                 for c in range(1, n_clips + 1)]
    base = candidate_windows(words, durations, offs)
    print(f"{len(base)} candidate windows {base.split.value_counts().to_dict()}", flush=True)

    # ---- targets, identical to mtrf_baseline.py
    mel = np.load(args.data / "speech_mel.npy").astype(np.float32)
    Y = np.concatenate([mel.mean(1, keepdims=True), mel.reshape(len(mel), args.bands, -1).mean(2)], 1)
    frames = np.unique(np.concatenate([np.arange(a, a + T) for a in split_of(base, "train").audio_start]))
    Y = (Y - Y[frames].mean(0)) / (Y[frames].std(0) + 1e-6)
    K = Y.shape[1]

    # ---- clip starts: trial table vs markers
    trl = clip_starts(pd.read_csv(args.data / "windows.csv"), offs, shift)
    marks = {s: vmrk_markers(args.dataset / "eeg" / f"{s}.vmrk") for s in meta["subjects"]}
    diffs = [trl[s][c] / fs - m for s in meta["subjects"] for c, m in marks[s].items()
             if c in trl.get(s, {}) and abs(trl[s][c] / fs - m) < 0.5]
    delay = float(np.median(diffs))
    print(f"marker -> trial-table delay: median {1e3 * delay:.1f} ms over {len(diffs)} agreeing clips",
          flush=True)
    marker = {s: {c: int(round((m + delay) * fs)) for c, m in marks[s].items()} for s in meta["subjects"]}
    (args.out / "clip_starts.json").write_text(json.dumps(
        {s: {str(c): dict(trl_s=trl[s].get(c, None) and trl[s][c] / fs,
                          marker_s=marker[s].get(c, None) and marker[s][c] / fs)
             for c in range(1, n_clips + 1)} for s in meta["subjects"]}, indent=1))

    eegs = {s: np.load(args.data / "eeg" / f"{s}.npy", mmap_mode="r") for s in meta["subjects"]}
    C = eegs[subjects[0]].shape[0]
    F = C * len(lags)

    # ---- cross decoders from the best baseline subjects (trl alignment, train split only)
    res = pd.read_csv(args.baseline)
    ranked = res[res.model == "subject"].sort_values("r_env", ascending=False).subject.tolist()
    ref = [s for s in ranked if s in meta["subjects"]][:args.n_ref]
    ref_a, ref_b = ref[0::2], ref[1::2]
    dec = {}
    for name, group in (("A", ref_a), ("B", ref_b)):
        st = Sums(F, K, dev)
        for s in group:
            eeg = np.asarray(eegs[s], dtype=np.float32)
            w = non_overlapping(split_of(build_windows(base, trl[s], fs, eeg.shape[1], T, lags), "train"), T)
            for X, y in chunks(eeg, w, Y, T, lags):
                st.add(X, y, dev)
        m = Ridge(st)
        dec[name] = (m, m.weights(args.cross_lam))
        del st
        print(f"cross decoder {name}: {group}", flush=True)

    # ---- evaluate every subject under both alignments
    rows = []
    for i, s in enumerate(subjects):
        eeg = np.asarray(eegs[s], dtype=np.float32)
        cross_model, cross_W = dec["B" if s in ref_a else "A" if s in ref_b else "B"]
        diff_ms = [1e3 * (trl[s][c] - marker[s][c]) / fs for c in marker[s] if c in trl[s]]
        row = dict(subject=s, ref=s in ref, n_clips_trl=len(trl[s]), n_clips_marker=len(marker[s]),
                   trl_minus_marker_ms=float(np.median(diff_ms)) if diff_ms else np.nan)
        for align, start in (("trl", trl[s]), ("marker", marker[s])):
            w = build_windows(base, start, fs, eeg.shape[1], T, lags)
            te = split_of(w, "test")
            if not len(te):
                continue
            # subject-specific decoder
            tr = non_overlapping(split_of(w, "train"), T)
            st = Sums(F, K, dev)
            for X, y in chunks(eeg, tr, Y, T, lags):
                st.add(X, y, dev)
            m = Ridge(st)
            lam = pick_lambda(m, grid, [(eeg, non_overlapping(split_of(w, "val"), T))], Y, T, lags)
            r = evaluate(*predict_windows(m, m.weights(lam), eeg, te, Y, T, lags), te.audio_start.values, T)
            row[f"subj_r_{align}"], row[f"subj_mm_{align}"] = r["r_env"], r["mm_acc"]
            # cross decoder (never trained on this subject)
            r = evaluate(*predict_windows(cross_model, cross_W, eeg, te, Y, T, lags), te.audio_start.values, T)
            row[f"cross_r_{align}"], row[f"cross_mm_{align}"] = r["r_env"], r["mm_acc"]
            row[f"null_{align}"] = r["r_env_null"]
            row[f"n_test_{align}"] = len(te)
            del st, m
        if dev == "cuda":
            torch.cuda.empty_cache()
        rows.append(row)
        g = lambda k: row.get(k, np.nan)
        print(f"[{i + 1}/{len(subjects)}] {s} trl-marker={row['trl_minus_marker_ms']:+.0f} ms | "
              f"subject r trl={g('subj_r_trl'):.3f} marker={g('subj_r_marker'):.3f} | "
              f"cross r trl={g('cross_r_trl'):.3f} marker={g('cross_r_marker'):.3f} | "
              f"cross mm trl={g('cross_mm_trl'):.3f} marker={g('cross_mm_marker'):.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "results.csv", index=False)
    df["disagree"] = df.trl_minus_marker_ms.abs() > 200  # marker starts already include `delay`
    print("\n=== mean r_env by group (disagree = trl and markers differ by > 200 ms) ===")
    cols = ["subj_r_trl", "subj_r_marker", "cross_r_trl", "cross_r_marker",
            "cross_mm_trl", "cross_mm_marker"]
    print(df.groupby("disagree")[cols].mean().round(3).to_string())
    print("\n=== subjects where trl and markers disagree ===")
    print(df[df.disagree][["subject", "trl_minus_marker_ms"] + cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

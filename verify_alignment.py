"""Check that the speech features and the EEG are aligned in time.

Four independent checks, using the same alignment logic as prepare_alice_dataset.py:

1. Trial table: the EEG onset of every word (proc.trl) minus its CSV onset
   should be the same constant within each audio clip. The spread around the
   per-clip median is the alignment jitter (should be ~1 EEG sample = 2 ms).
2. Hardware markers: the estimated clip start should coincide with a marker
   in the BrainVision .vmrk file.
3. Word-onset ERP: averaging the processed EEG at word onsets should show an
   auditory N1/P2 response ~100-200 ms after onset (not before 0).
4. Envelope tracking: the EEG should correlate best with the speech envelope
   when the EEG lags the audio by roughly 100-200 ms. The code assumes 150 ms.

Read-only: writes alignment_check.png and prints a summary.
Usage:  python verify_alignment.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

from prepare_alice_dataset import FS, load_proc

ROOT = Path(__file__).resolve().parent
DATA, PROC = ROOT / "Dataset", ROOT / "D_T_processed"


def sfreq_of(vhdr: Path) -> float:
    m = re.search(r"SamplingInterval=(\d+)", vhdr.read_text(errors="ignore"))
    return 1e6 / float(m.group(1))


def markers(vmrk: Path) -> pd.DataFrame:
    rows = []
    for line in vmrk.read_text(errors="ignore").splitlines():
        m = re.match(r"Mk\d+=([^,]*),([^,]*),(\d+),", line)
        if m:
            rows.append(dict(type=m.group(1), desc=m.group(2).strip(), pos=int(m.group(3)) - 1))
    return pd.DataFrame(rows)


def main() -> None:
    meta = json.loads((PROC / "meta.json").read_text())
    words = pd.read_csv(DATA / "AliceChapterOne-EEG.csv")
    csv_on = words.set_index("Order").onset
    mel = np.load(PROC / "speech_mel.npy").astype(np.float32)
    offs, lens = meta["clip_offsets"], meta["clip_lengths"]

    sos = butter(4, [1, 8], btype="band", fs=FS, output="sos")
    lags = np.arange(int(-0.3 * FS), int(0.6 * FS) + 1)
    erp_t = np.arange(int(-0.2 * FS), int(0.6 * FS))
    xcorr_all, erp_all, jitter, marker_diff = [], [], [], []

    first = True
    for subj in meta["subjects"]:
        sf = sfreq_of(DATA / "eeg" / f"{subj}.vhdr")
        proc = load_proc(DATA / "proc" / f"{subj}.mat")
        if first:  # show the raw trial-table layout once
            trl = np.asarray(loadmat(DATA / "proc" / f"{subj}.mat", squeeze_me=True,
                                     struct_as_record=False)["proc"].trl, dtype=float)
            print(f"[{subj}] trl offset column values: {np.unique(trl[:, 2])[:5]}  "
                  f"epoch length (samples): {np.unique(trl[:, 1] - trl[:, 0])[:5]}  sfreq={sf}")
            first = False

        # 1) jitter of (EEG onset - CSV onset) within each clip
        rel = proc["onset_sample"] / sf - csv_on.loc[proc["order"]].values
        clip_start = {}
        for c in np.unique(proc["clip"]):
            r = rel[proc["clip"] == c]
            clip_start[c] = float(np.median(r))
            jitter.append(dict(subject=subj, clip=c, n=len(r),
                               max_abs_ms=1e3 * np.abs(r - clip_start[c]).max(),
                               std_ms=1e3 * r.std()))

        # 2) nearest hardware marker to each estimated clip start
        mk = markers(DATA / "eeg" / f"{subj}.vmrk")
        mk = mk[mk.type != "New Segment"]
        for c, t in clip_start.items():
            if len(mk):
                d = (mk.pos.values / sf - t)
                i = np.abs(d).argmin()
                marker_diff.append(dict(subject=subj, clip=c, marker=mk.desc.iloc[i], diff_ms=1e3 * d[i]))

        eeg = np.load(PROC / "eeg" / f"{subj}.npy").astype(np.float32)

        # 3) ERP at word onsets (kept trials only), baseline -200..0 ms
        kept = [i for i in range(len(proc["order"])) if i in proc["kept_trials"]]
        epochs = []
        for i in kept:
            c, o = proc["clip"][i], csv_on.loc[proc["order"][i]]
            s = int(round((clip_start[c] + o) * FS))
            if s + erp_t[0] >= 0 and s + erp_t[-1] < eeg.shape[1]:
                epochs.append(eeg[:, s + erp_t])
        ep = np.stack(epochs)
        ep -= ep[:, :, erp_t < 0].mean(-1, keepdims=True)
        erp_all.append(ep.mean(0))

        # 4) envelope-EEG cross-correlation (1-8 Hz), RMS over channels
        filt = sosfiltfilt(sos, eeg, axis=1)
        filt = (filt - filt.mean(1, keepdims=True)) / (filt.std(1, keepdims=True) + 1e-8)
        acc = np.zeros(len(lags))
        nclip = 0
        for c, t in clip_start.items():
            env = sosfiltfilt(sos, mel[offs[c - 1]:offs[c - 1] + lens[c - 1]].mean(1))
            env = (env - env.mean()) / env.std()
            e0, n = int(round(t * FS)), len(env)
            if e0 + lags[0] < 0 or e0 + lags[-1] + n > filt.shape[1]:
                continue
            r = np.stack([filt[:, e0 + L:e0 + L + n] @ env / n for L in lags])  # [lags, ch]
            acc += np.sqrt((r ** 2).mean(1))
            nclip += 1
        xcorr_all.append(acc / max(nclip, 1))
        print(f"{subj}: {len(ep)} ERP epochs, {nclip} clips, "
              f"envelope peak lag {1e3 * lags[np.argmax(acc)] / FS:.0f} ms", flush=True)

    jit, mkd = pd.DataFrame(jitter), pd.DataFrame(marker_diff)
    xc, erp = np.mean(xcorr_all, 0), np.mean(erp_all, 0)
    gfp = erp.std(0)
    post = erp_t >= 0
    erp_peak = 1e3 * erp_t[post][np.argmax(gfp[post])] / FS
    xc_peak = 1e3 * lags[np.argmax(xc)] / FS

    print("\n=== 1) trial table vs CSV: within-clip jitter ===")
    print(f"max |deviation| over all subjects/clips: {jit.max_abs_ms.max():.1f} ms; "
          f"median per-clip max: {jit.max_abs_ms.median():.1f} ms")
    print(jit.sort_values("max_abs_ms", ascending=False).head(5).to_string(index=False))
    print("\n=== 2) estimated clip start vs nearest .vmrk marker ===")
    if len(mkd):
        print(f"|diff| median {mkd.diff_ms.abs().median():.1f} ms, max {mkd.diff_ms.abs().max():.1f} ms; "
              f"markers used: {sorted(mkd.marker.unique())[:12]}")
    print("\n=== 3) word-onset ERP ===")
    print(f"global field power peak after onset: {erp_peak:.0f} ms (auditory N1/P2 expected ~100-200 ms)")
    print("\n=== 4) speech envelope vs EEG ===")
    print(f"grand-average peak lag: {xc_peak:.0f} ms (EEG after audio); code shift = "
          f"{1e3 * meta['eeg_shift_s']:.0f} ms; per-subject peaks: "
          f"median {np.median([1e3 * lags[np.argmax(a)] / FS for a in xcorr_all]):.0f} ms")

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].plot(1e3 * erp_t / FS, gfp)
    ax[0].axvline(0, c="k", lw=0.8)
    ax[0].set(title="Word-onset ERP (global field power)", xlabel="ms from word onset")
    ax[1].plot(1e3 * lags / FS, xc)
    ax[1].axvline(0, c="k", lw=0.8)
    ax[1].axvline(1e3 * meta["eeg_shift_s"], c="r", ls="--", lw=0.8, label="code shift")
    ax[1].set(title="Envelope-EEG correlation (RMS over channels)", xlabel="EEG lag (ms)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(ROOT / "alignment_check.png", dpi=120)
    print("\nsaved alignment_check.png")


if __name__ == "__main__":
    main()

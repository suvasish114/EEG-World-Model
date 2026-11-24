"""Prepare the Alice (Brennan & Hale 2019) EEG dataset for paper-style decoding.

Follows Défossez et al. (2023), "Decoding speech perception from non-invasive
brain recordings":

EEG (per subject)
  * 60 recorded scalp channels + implicit reference "29" added back (61 total),
    Easycap M10 montage, common average reference
  * channels rejected by the authors (proc.rejections.final) are interpolated
  * 0.1-40 Hz band-pass, resampled to 120 Hz
  * robust scaling per channel (median / IQR fit on training windows only),
    clamped to +-20

Speech (per audio clip, computed on the whole clip so the model sees context)
  * wav2vec 2.0 (facebook/wav2vec2-large-xlsr-53), mean of the last 4
    transformer layers -> 1024 features
  * log-Mel spectrogram (120 bands) for the paper's baseline
  * both interpolated to 120 Hz and standardized per feature

Samples
  * 3 s windows starting 0.5 s before every word onset (inside one clip);
    the EEG window is shifted 150 ms later than the audio window
  * split by blocks of consecutive sentences (70/20/10); windows whose audio
    touches words of two different splits are dropped; the same sentence is
    in the same split for every subject
  * windows anchored on words rejected by the authors are dropped

Clip alignment (--align marker, default): each audio clip starts at its
BrainVision marker "Stimulus,k" + 51.3 ms. That delay is the median gap between
markers and the authors' trial table (proc.trl) over the 396 clips where the
two agree (runs/realign_markers, job 552891). For 33 subjects both give the same
start. For S02 (trl 2.48 s late) and S24-S36 (trl ~62 s late: its clip k lies on
clip k+1) the trial table is wrong; with markers their envelope decoding goes
from chance (r ~0.00) to r ~0.14. For those subjects the authors' trial
rejections are tied to the wrong clips and are ignored (channel rejections are
kept). A clip without a marker falls back to the trial table only if the
subject's trial table agrees with its markers. --align trl reproduces the old
behaviour (D_T_processed/).

Output (D_T_processed/):
  eeg/Sxx.npy        float16 [61, time] continuous EEG at 120 Hz
  speech_wav2vec.npy float16 [frames, 1024] all clips concatenated, 120 Hz
  speech_mel.npy     float16 [frames, 120]
  windows.csv        one row per (subject, window)
  meta.json          channels, 2-D sensor positions, clip offsets, settings

Usage:  .venv/bin/python prepare_alice_dataset.py
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parent
FS = 120  # Hz, model sample rate (paper)
WIN = 3.0  # s
PRE = 0.5  # s before word onset
SHIFT = 0.15  # s, brain response delay
CLAMP = 20.0
SEED = 42
SENTENCES_PER_BLOCK = 2
W2V_NAME = "facebook/wav2vec2-large-xlsr-53"
SCALP = [str(i) for i in range(1, 62)]  # "29" is the implicit reference
MARKER_DELAY = 0.0513  # s, trial-table clip start minus vmrk marker (median, agreeing clips)
AGREE_TOL = 0.2  # s, trial table "agrees" with markers if within this


# --------------------------------------------------------------------- speech
def load_clip(path: Path) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    return resample_poly(wav, 16000, sr).astype(np.float32)  # 44.1k -> 16k


def to_grid(feat: np.ndarray, times: np.ndarray, n_out: int) -> np.ndarray:
    """Linearly interpolate [frames, F] features onto the 120 Hz grid."""
    grid = np.arange(n_out) / FS
    f = interp1d(times, feat, axis=0, bounds_error=False,
                 fill_value=(feat[0], feat[-1]))
    return f(grid).astype(np.float32)


def speech_features(audio_dir: Path, n_clips: int):
    from transformers import Wav2Vec2Model

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    w2v = Wav2Vec2Model.from_pretrained(W2V_NAME).to(device).eval()
    mel_fn = torchaudio.transforms.MelSpectrogram(
        16000, n_fft=512, hop_length=128, n_mels=120, normalized=True)

    w2v_all, mel_all, durations = [], [], []
    for clip in range(1, n_clips + 1):
        wav = load_clip(audio_dir / f"DownTheRabbitHoleFinal_SoundFile{clip}.wav")
        dur = len(wav) / 16000
        n_out = int(np.floor(dur * FS))
        durations.append(dur)

        x = torch.from_numpy((wav - wav.mean()) / (wav.std() + 1e-7))[None]
        with torch.inference_mode():
            hs = w2v(x.to(device), output_hidden_states=True).hidden_states
            feat = torch.stack(hs[-4:]).mean(0)[0].float().cpu().numpy()
        t = (np.arange(len(feat)) * 320 + 200) / 16000  # frame centres
        w2v_all.append(to_grid(feat, t, n_out))

        mel = torch.log(1e-5 + mel_fn(torch.from_numpy(wav))).T.numpy()
        mel_all.append(to_grid(mel, np.arange(len(mel)) * 128 / 16000, n_out))
        print(f"  clip {clip}: {dur:.1f}s -> {n_out} frames", flush=True)

    out = []
    for arr in (np.concatenate(w2v_all), np.concatenate(mel_all)):
        arr = (arr - arr.mean(0)) / (arr.std(0) + 1e-6)
        out.append(arr.astype(np.float16))
    return out[0], out[1], durations


# ---------------------------------------------------------------- annotations
def assign_splits(words: pd.DataFrame) -> dict[int, str]:
    """Sentence -> split, using shuffled blocks of consecutive sentences."""
    sentences = sorted(words.Sentence.unique())
    blocks = [sentences[i:i + SENTENCES_PER_BLOCK]
              for i in range(0, len(sentences), SENTENCES_PER_BLOCK)]
    order = np.random.default_rng(SEED).permutation(len(blocks))
    n_train, n_val = round(0.7 * len(blocks)), round(0.2 * len(blocks))
    split = {}
    for rank, b in enumerate(order):
        name = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"
        for s in blocks[b]:
            split[s] = name
    return split


def candidate_windows(words: pd.DataFrame, durations: list[float],
                      clip_offset: list[int]) -> pd.DataFrame:
    """One 3 s window per word onset, kept only if it lies in one clip and split."""
    rows = []
    for w in words.itertuples():
        t0, t1 = w.onset - PRE, w.onset - PRE + WIN
        if t0 < 0 or t1 > durations[w.Segment - 1]:
            continue
        inside = words[(words.Segment == w.Segment) & (words.offset > t0) & (words.onset < t1)]
        splits = inside.split.unique()
        if len(splits) != 1:
            continue
        rows.append(dict(
            segment_id=w.Order, word=w.Word, audio_clip=w.Segment, sentence=w.Sentence,
            split=splits[0], audio_t0=t0,
            audio_start=clip_offset[w.Segment - 1] + int(round(t0 * FS)),
            text=" ".join(inside.Word.astype(str))))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ EEG
def vmrk_markers(vmrk: Path, sfreq: float) -> dict[int, float]:
    """{clip number: time in s} of the first 'Stimulus,<k>' marker for k = 1..12."""
    out = {}
    if not vmrk.exists():
        return out
    for typ, desc, pos in re.findall(r"Mk\d+=([^,]*),([^,]*),(\d+),", vmrk.read_text(errors="ignore")):
        d = re.sub(r"\D", "", desc)
        if typ.strip() == "Stimulus" and d and 1 <= int(d) <= 12 and int(d) not in out:
            out[int(d)] = (int(pos) - 1) / sfreq  # 1-based sample position
    return out


def choose_clip_starts(trl_start: dict, markers: dict, align: str):
    """-> (clip starts in s, source label, median trl-minus-marker gap in ms, trl agrees)."""
    mk = {c: t + MARKER_DELAY for c, t in markers.items()}
    gaps = [trl_start[c] - mk[c] for c in mk if c in trl_start]
    gap_ms = 1e3 * float(np.median(gaps)) if gaps else float("nan")
    agree = bool(gaps) and abs(gap_ms) <= 1e3 * AGREE_TOL
    if align == "trl" or not mk:
        return dict(trl_start), "trl", gap_ms, True
    starts = dict(mk)
    if agree:  # fill clips without a marker from the (consistent) trial table
        for c, t in trl_start.items():
            starts.setdefault(c, t)
    return starts, "marker", gap_ms, agree


def load_proc(path: Path):
    p = loadmat(path, squeeze_me=True, struct_as_record=False)["proc"]
    names = list(np.atleast_1d(p.varnames))
    trl = np.asarray(p.trl, dtype=float)
    onset_sample = trl[:, 0] - trl[:, 2]  # begin - offset(-150) = word onset, 500 Hz, 1-based
    final = p.rejections.final
    return dict(
        order=trl[:, 3 + names.index("Order")].astype(int),
        clip=trl[:, 3 + names.index("segment")].astype(int),
        onset_sample=onset_sample - 1,  # 0-based
        kept_trials=set(np.atleast_1d(final.trialpicks).astype(int) - 1),
        kept_chans=set(map(str, np.atleast_1d(final.chanpicks))),
    )


def preprocess_eeg(vhdr: Path, kept_chans: set[str]) -> tuple[np.ndarray, mne.Info, int]:
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="error")
    sfreq = int(round(raw.info["sfreq"]))
    raw.pick([c for c in raw.ch_names if c in SCALP])
    raw = mne.add_reference_channels(raw, "29")
    raw.reorder_channels(SCALP)
    raw.set_montage("easycap-M10", verbose="error")
    raw.info["bads"] = [c for c in SCALP if c != "29" and c not in kept_chans]
    raw.set_eeg_reference("average", verbose="error")
    if raw.info["bads"]:
        raw.interpolate_bads(reset_bads=True, verbose="error")
    raw.filter(0.1, 40.0, verbose="error")
    raw.resample(FS, verbose="error")
    return raw.get_data().astype(np.float32), raw.info, sfreq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=ROOT / "D_T")
    ap.add_argument("--out", type=Path, default=ROOT / "D_T_processed_v2")
    ap.add_argument("--align", choices=["marker", "trl"], default="marker",
                    help="clip starts from vmrk markers (default) or the authors' trial table")
    ap.add_argument("--reuse-speech", type=Path, default=None,
                    help="copy speech_wav2vec.npy / speech_mel.npy from this folder (they do "
                         "not depend on the EEG alignment)")
    args = ap.parse_args()
    (args.out / "eeg").mkdir(parents=True, exist_ok=True)
    if args.reuse_speech:
        for name in ("speech_wav2vec.npy", "speech_mel.npy"):
            if not (args.out / name).exists():
                shutil.copy2(args.reuse_speech / name, args.out / name)

    words = pd.read_csv(args.data / "AliceChapterOne-EEG.csv")
    words["split"] = words.Sentence.map(assign_splits(words))
    n_clips = int(words.Segment.max())

    w2v_path, mel_path = args.out / "speech_wav2vec.npy", args.out / "speech_mel.npy"
    if w2v_path.exists() and mel_path.exists():  # wav2vec is slow: reuse a previous run
        print("Reusing saved speech features", flush=True)
        durations = [sf.info(args.data / "audio" / f"DownTheRabbitHoleFinal_SoundFile{c}.wav").duration
                     for c in range(1, n_clips + 1)]
    else:
        print("Speech features (wav2vec 2.0 + Mel)...", flush=True)
        w2v, mel, durations = speech_features(args.data / "audio", n_clips)
        np.save(w2v_path, w2v)
        np.save(mel_path, mel)
    clip_len = [int(np.floor(d * FS)) for d in durations]
    clip_offset = list(np.cumsum([0] + clip_len[:-1]).astype(int))

    base = candidate_windows(words, durations, clip_offset)
    print(f"{len(base)} candidate windows "
          f"({base.split.value_counts().to_dict()})", flush=True)

    all_rows, excluded, info, alignment = [], {}, None, {}
    subjects = sorted(p.stem for p in (args.data / "eeg").glob("S*.vhdr"))
    for s_idx, subj in enumerate(subjects):
        proc_path = args.data / "proc" / f"{subj}.mat"
        if not proc_path.exists():
            excluded[subj] = "no proc/ trial table"
            continue
        proc = load_proc(proc_path)
        eeg, info_s, sfreq = preprocess_eeg(args.data / "eeg" / f"{subj}.vhdr", proc["kept_chans"])
        info = info or info_s

        # clip start (in EEG seconds) = median over that clip's words of onset - csv onset
        csv_on = words.set_index("Order").onset
        rel = proc["onset_sample"] / sfreq - csv_on.loc[proc["order"]].values
        trl_start = {int(c): float(np.median(rel[proc["clip"] == c])) for c in np.unique(proc["clip"])}
        markers = vmrk_markers(args.data / "eeg" / f"{subj}.vmrk", sfreq)
        clip_start, source, gap_ms, agree = choose_clip_starts(trl_start, markers, args.align)
        if agree:
            kept_orders = {o for i, o in enumerate(proc["order"]) if i in proc["kept_trials"]}
        else:  # trial table misaligned: its trial rejections refer to the wrong clips
            kept_orders = set(words.Order)
        alignment[subj] = dict(source=source, trl_minus_marker_ms=gap_ms, trl_agrees=agree,
                               trial_rejection=agree, clips=sorted(clip_start))

        win = base[base["audio_clip"].isin(clip_start.keys()) & base["segment_id"].isin(kept_orders)].copy()
        win["eeg_start"] = [int(round((clip_start[c] + t0 + SHIFT) * FS))
                            for c, t0 in zip(win["audio_clip"], win["audio_t0"])]
        win = win[(win.eeg_start >= 0) & (win.eeg_start + int(WIN * FS) <= eeg.shape[1])]

        # robust scaling, statistics from training windows only
        train_idx = np.concatenate([np.arange(s, s + int(WIN * FS)) for s in win[win.split == "train"].eeg_start])
        ref = eeg[:, np.unique(train_idx)]
        med = np.median(ref, axis=1, keepdims=True)
        iqr = np.subtract(*np.percentile(ref, [75, 25], axis=1))[:, None]
        eeg = np.clip((eeg - med) / (iqr + 1e-12), -CLAMP, CLAMP)
        np.save(args.out / "eeg" / f"{subj}.npy", eeg.astype(np.float16))

        win.insert(0, "subject", subj)
        all_rows.append(win)
        n_bad = sum(c != "29" and c not in proc["kept_chans"] for c in SCALP)
        print(f"{subj}: {len(win)} windows, {n_bad} bad channels interpolated, clips from {source} "
              f"(trl-marker {gap_ms:+.0f} ms{'' if agree else ', TRL MISALIGNED: no trial rejection'}), "
              f"{len(clip_start)} clips", flush=True)

    windows = pd.concat(all_rows, ignore_index=True)
    windows.insert(1, "subject_idx", windows.subject.astype("category").cat.codes)
    windows.drop(columns="audio_t0").to_csv(args.out / "windows.csv", index=False)

    layout = mne.channels.find_layout(info, ch_type="eeg")
    pos = layout.pos[:, :2]
    pos = (pos - pos.min(0)) / (pos.max(0) - pos.min(0))
    meta = dict(
        fs=FS, window_s=WIN, pre_onset_s=PRE, eeg_shift_s=SHIFT, clamp=CLAMP,
        window_samples=int(WIN * FS), channels=SCALP,
        positions=pos.tolist(), clip_offsets=clip_offset, clip_lengths=clip_len,
        subjects=sorted(windows.subject.unique()), excluded=excluded,
        wav2vec_model=W2V_NAME, sentences_per_block=SENTENCES_PER_BLOCK,
        align=args.align, marker_delay_s=MARKER_DELAY, alignment=alignment,
        counts={k: int(v) for k, v in windows.split.value_counts().items()},
        unique_segments={k: int(v) for k, v in windows.groupby("split").segment_id.nunique().items()},
    )
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1, default=int))  # numpy ints
    print(json.dumps({k: meta[k] for k in ("counts", "unique_segments", "excluded")}, indent=1))


if __name__ == "__main__":
    main()

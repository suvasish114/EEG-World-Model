"""Create aligned, word-level EEG and speech-envelope datasets."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.io import loadmat
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "Dataset"
DEFAULT_OUTPUT = PROJECT_ROOT / "Processed_word_level"
TARGET_FS = 128
LOW_CUT_HZ = 0.1
HIGH_CUT_HZ = 40.0
ENVELOPE_CUT_HZ = 8.0
RANDOM_SEED = 42
STATS_SAMPLE_LIMIT = 100_000


@dataclass(frozen=True)
class Example:
    source: str
    participant: str
    stimulus_group: str
    split: str
    word: str
    order: int
    onset: float
    offset: float
    eeg_start: int
    eeg_stop: int
    envelope_start: int
    envelope_stop: int
    channel_mask: np.ndarray


def _mat_values(value: Any) -> list[Any]:
    result = []
    for item in np.asarray(value).ravel():
        while isinstance(item, np.ndarray) and item.size == 1:
            item = item.item()
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        elif isinstance(item, np.ndarray) and item.dtype.kind in "US":
            item = "".join(item.astype(str).ravel())
        result.append(item)
    return result


def _split_groups(groups: list[str], seed: int = RANDOM_SEED) -> dict[str, str]:
    ordered = sorted(set(groups))
    if len(ordered) < 3:
        raise ValueError(f"At least three stimulus groups are required; got {len(ordered)}")
    rng = np.random.default_rng(seed)
    shuffled = [ordered[i] for i in rng.permutation(len(ordered))]
    train_stop = int(0.7 * len(shuffled))
    validation_stop = int(0.9 * len(shuffled))
    if train_stop == 0 or validation_stop == train_stop or validation_stop == len(shuffled):
        raise ValueError(f"Cannot create train/validation/test splits from {len(ordered)} groups")
    return {
        group: (
            "train" if index < train_stop
            else "validation" if index < validation_stop
            else "test"
        )
        for index, group in enumerate(shuffled)
    }


def _bandpass(data: np.ndarray, sample_rate: int) -> np.ndarray:
    if not 0 < LOW_CUT_HZ < HIGH_CUT_HZ < sample_rate / 2:
        raise ValueError(f"Invalid bandpass for sample rate {sample_rate}")
    sos = butter(
        4,
        [LOW_CUT_HZ, HIGH_CUT_HZ],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    ).astype(np.float32)
    filtered = sosfiltfilt(sos, np.asarray(data, dtype=np.float32), axis=0)
    if not np.isfinite(filtered).all():
        raise ValueError("EEG filtering produced non-finite values")
    return filtered


def _resample(data: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    divisor = gcd(source_rate, target_rate)
    result = resample_poly(
        data,
        target_rate // divisor,
        source_rate // divisor,
        axis=0,
    )
    return np.asarray(result, dtype=np.float32)


def _common_average_reference(data: np.ndarray) -> np.ndarray:
    return data - data.mean(axis=1, keepdims=True)


def _natural_text(path: Path) -> list[tuple[str, float, float]]:
    data = loadmat(path)
    words = [str(value).strip() for value in _mat_values(data["wordVec"])]
    onsets = np.asarray(_mat_values(data["onset_time"]), dtype=np.float64)
    offsets = np.asarray(_mat_values(data["offset_time"]), dtype=np.float64)
    if not (len(words) == len(onsets) == len(offsets)):
        raise ValueError(f"{path}: word and timing counts differ")
    if not np.isfinite(onsets).all() or not np.isfinite(offsets).all():
        raise ValueError(f"{path}: non-finite word timing")
    if any(not word for word in words):
        raise ValueError(f"{path}: empty word token")
    if np.any(offsets <= onsets):
        raise ValueError(f"{path}: word offset must follow onset")
    return list(zip(words, onsets.tolist(), offsets.tolist()))


def _natural_specs(
    dataset_root: Path,
) -> tuple[list[Example], dict[str, tuple[Path, Path]], list[str]]:
    root = dataset_root / "Natural_Speech"
    text_root = root / "Stimuli" / "Text"
    envelope_root = root / "Stimuli" / "Envelopes"
    text_files = {
        int(match.group(1)): path
        for path in text_root.glob("Run*.mat")
        if (match := re.fullmatch(r"Run(\d+)", path.stem))
    }
    if not text_files:
        raise FileNotFoundError(f"No Natural Speech run annotations in {text_root}")
    subjects = sorted(
        (path for path in (root / "EEG").glob("Subject*") if path.is_dir()),
        key=lambda path: int(re.search(r"\d+", path.name).group()),
    )
    if not subjects:
        raise FileNotFoundError(f"No Natural Speech subject folders in {root / 'EEG'}")
    splits = _split_groups([f"Run{run}" for run in text_files])
    words_by_run = {run: _natural_text(path) for run, path in text_files.items()}
    examples: list[Example] = []
    sources: dict[str, tuple[Path, Path]] = {}
    for subject_dir in subjects:
        subject_match = re.fullmatch(r"Subject(\d+)", subject_dir.name)
        if not subject_match:
            continue
        participant = subject_dir.name
        for run in sorted(text_files):
            source = f"{participant}_Run{run}"
            eeg_path = subject_dir / f"{source}.mat"
            envelope_path = envelope_root / f"audio{run}_128Hz.mat"
            if not eeg_path.is_file():
                raise FileNotFoundError(f"Missing EEG recording: {eeg_path}")
            if not envelope_path.is_file():
                raise FileNotFoundError(f"Missing speech envelope: {envelope_path}")
            sources[source] = (eeg_path, envelope_path)
            mask = np.ones(128, dtype=np.bool_)
            for order, (word, onset, offset) in enumerate(words_by_run[run], start=1):
                start = int(np.floor(onset * TARGET_FS))
                stop = int(np.ceil(offset * TARGET_FS))
                examples.append(
                    Example(
                        source=source,
                        participant=participant,
                        stimulus_group=f"Run{run}",
                        split=splits[f"Run{run}"],
                        word=word,
                        order=order,
                        onset=onset,
                        offset=offset,
                        eeg_start=start,
                        eeg_stop=stop,
                        envelope_start=start,
                        envelope_stop=stop,
                        channel_mask=mask,
                    )
                )
    return examples, sources, []


def _alice_annotations(path: Path) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    frame = pd.read_csv(path)
    required = {"Word", "Segment", "onset", "offset", "Order"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns {sorted(missing)}")
    for name in ("Segment", "onset", "offset", "Order"):
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"{path}: missing required annotations")
    if (frame["offset"] <= frame["onset"]).any():
        raise ValueError(f"{path}: word offset must follow onset")
    segments = {
        int(segment): group.sort_values("onset", kind="stable").reset_index(drop=True)
        for segment, group in frame.groupby("Segment", sort=True)
    }
    return frame, segments


def _alice_specs(
    dataset_root: Path,
    source_envelopes: dict[int, np.ndarray],
) -> tuple[list[Example], dict[str, Path], list[dict[str, str]]]:
    root = dataset_root / "D_T"
    annotations_path = root / "AliceChapterOne-EEG.csv"
    _, annotations = _alice_annotations(annotations_path)
    for segment, words in annotations.items():
        if int(np.ceil(words["offset"].max() * TARGET_FS)) > len(source_envelopes[segment]):
            raise ValueError(f"Segment {segment}: annotations exceed the speech audio")
    proc_root = root / "proc"
    eeg_root = root / "eeg"
    proc_paths = {
        match.group(1): path
        for path in proc_root.glob("S*.mat")
        if (match := re.fullmatch(r"(S\d+)", path.stem))
    }
    raw_paths = {path.stem: path for path in eeg_root.glob("S*.vhdr")}
    all_groups = [f"Segment{segment}" for segment in annotations]
    splits = _split_groups(all_groups)
    examples: list[Example] = []
    sources: dict[str, Path] = {}
    exclusions: list[dict[str, str]] = []
    for number in range(1, 50):
        participant = f"S{number:02d}"
        raw_path = raw_paths.get(participant)
        proc_path = proc_paths.get(participant)
        if raw_path is None:
            exclusions.append({"participant": participant, "reason": "missing_brainvision_recording"})
            continue
        if proc_path is None:
            exclusions.append({"participant": participant, "reason": "missing_trial_and_channel_rejection_metadata"})
            continue
        source = participant
        sources[source] = raw_path
        proc = loadmat(proc_path)["proc"][0, 0]
        trials = np.asarray(proc["trl"], dtype=np.float64)
        names = [str(value) for value in _mat_values(proc["varnames"])]
        columns = {name: index + 3 for index, name in enumerate(names)}
        for name in ("segment", "tmin", "Order"):
            if name not in columns or columns[name] >= trials.shape[1]:
                raise ValueError(f"{proc_path}: cannot locate {name} in trial metadata")
        segment_col, onset_col, order_col = (
            columns["segment"],
            columns["tmin"],
            columns["Order"],
        )
        rejections = proc["rejections"][0, 0]
        if "final" not in rejections.dtype.names:
            raise ValueError(f"{proc_path}: final artifact-rejection selections are missing")
        final = rejections["final"][0, 0]
        retained = np.asarray(final["trialpicks"], dtype=np.int64).ravel()
        channel_names = [str(value) for value in _mat_values(final["chanpicks"])]
        channel_mask = np.zeros(61, dtype=np.bool_)
        for channel in channel_names:
            if not channel.isdigit() or not 1 <= int(channel) <= 61:
                raise ValueError(f"{proc_path}: unexpected EEG channel label {channel!r}")
            channel_mask[int(channel) - 1] = True
        header = mne.io.read_raw_brainvision(raw_path, preload=False, verbose="ERROR")
        available_channels = set(header.ch_names)
        channel_mask &= np.asarray(
            [str(index) in available_channels for index in range(1, 62)],
            dtype=np.bool_,
        )
        if not channel_mask.any():
            raise ValueError(f"{proc_path}: no retained EEG channels are present in {raw_path}")
        if not len(retained) or retained.min() < 1 or retained.max() > len(trials):
            raise ValueError(f"{proc_path}: retained trial index is out of range")
        if len(np.unique(retained)) != len(retained):
            raise ValueError(f"{proc_path}: duplicate retained trial indices")
        retained_set = set(retained.tolist())

        per_segment_origins: dict[int, list[int]] = defaultdict(list)
        matched: list[tuple[int, int, pd.Series]] = []
        for trial_index, trial in enumerate(trials, start=1):
            segment = int(round(trial[segment_col]))
            onset = float(trial[onset_col])
            if segment not in annotations:
                raise ValueError(f"{proc_path}: unknown segment {segment}")
            segment_rows = annotations[segment]
            differences = np.abs(segment_rows["onset"].to_numpy(dtype=float) - onset)
            annotation_index = int(np.argmin(differences))
            if differences[annotation_index] > 0.002:
                raise ValueError(
                    f"{proc_path}: trial {trial_index} onset {onset} has no matching annotation"
                )
            annotation = segment_rows.iloc[annotation_index]
            if abs(float(annotation["Order"]) - float(trial[order_col])) > 0.01:
                raise ValueError(
                    f"{proc_path}: trial {trial_index} Order does not match its word annotation"
                )
            if np.count_nonzero(differences <= 0.002) != 1:
                raise ValueError(f"{proc_path}: ambiguous annotation match for trial {trial_index}")
            start_sample_1based = int(round(trial[0]))
            origin_1based = int(round(start_sample_1based - onset * 500 + 150))
            per_segment_origins[segment].append(origin_1based)
            if trial_index in retained_set:
                matched.append((trial_index, segment, annotation))

        for segment, origins in per_segment_origins.items():
            if not origins or max(origins) - min(origins) > 1:
                raise ValueError(
                    f"{proc_path}: inconsistent segment-to-EEG alignment for segment {segment}"
                )

        for trial_index, segment, annotation in matched:
            onset = float(annotation["onset"])
            offset = float(annotation["offset"])
            eeg_onset_1based = int(round(trials[trial_index - 1, 0]))
            eeg_start_source = eeg_onset_1based - 1 + 150
            eeg_start = int(np.floor(eeg_start_source * TARGET_FS / 500))
            envelope_start = int(np.floor(onset * TARGET_FS))
            envelope_stop = int(np.ceil(offset * TARGET_FS))
            length = envelope_stop - envelope_start
            examples.append(
                Example(
                    source=source,
                    participant=participant,
                    stimulus_group=f"Segment{segment}",
                    split=splits[f"Segment{segment}"],
                    word=str(annotation["Word"]).strip(),
                    order=int(round(float(annotation["Order"]))),
                    onset=onset,
                    offset=offset,
                    eeg_start=eeg_start,
                    eeg_stop=eeg_start + length,
                    envelope_start=envelope_start,
                    envelope_stop=envelope_stop,
                    channel_mask=channel_mask,
                )
            )
    return examples, sources, exclusions


def _load_audio_envelopes(dataset_root: Path) -> dict[int, np.ndarray]:
    audio_root = dataset_root / "D_T" / "audio"
    result: dict[int, np.ndarray] = {}
    for segment in range(1, 13):
        path = audio_root / f"DownTheRabbitHoleFinal_SoundFile{segment}.wav"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Alice speech segment: {path}")
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        mono = audio.mean(axis=1)
        analytic = np.abs(hilbert(mono))
        sos = butter(
            4,
            ENVELOPE_CUT_HZ,
            btype="lowpass",
            fs=sample_rate,
            output="sos",
        )
        smooth = sosfiltfilt(sos, analytic)
        divisor = gcd(sample_rate, TARGET_FS)
        envelope = resample_poly(
            smooth,
            TARGET_FS // divisor,
            sample_rate // divisor,
        )
        if not np.isfinite(envelope).all():
            raise ValueError(f"{path}: speech-envelope extraction produced non-finite values")
        result[segment] = np.asarray(envelope, dtype=np.float32)
    return result


def _load_natural_source(
    eeg_path: Path,
    envelope_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    eeg_mat = loadmat(eeg_path)
    eeg = np.asarray(eeg_mat["eegData"], dtype=np.float32)
    sample_rate = int(np.asarray(eeg_mat["fs"]).squeeze())
    if sample_rate != TARGET_FS or eeg.ndim != 2 or eeg.shape[1] != 128:
        raise ValueError(f"{eeg_path}: expected samples x 128 channels at 128 Hz")
    if not np.isfinite(eeg).all():
        raise ValueError(f"{eeg_path}: EEG contains non-finite values")
    eeg = _bandpass(eeg, sample_rate)
    eeg = _common_average_reference(eeg)
    envelope = np.asarray(loadmat(envelope_path)["env"], dtype=np.float32).squeeze()
    if envelope.ndim != 1 or not np.isfinite(envelope).all():
        raise ValueError(f"{envelope_path}: invalid speech envelope")
    return eeg, envelope


def _load_alice_source(
    eeg_path: Path,
    channel_mask: np.ndarray,
) -> np.ndarray:
    raw = mne.io.read_raw_brainvision(eeg_path, preload=True, verbose="ERROR")
    channel_names = [str(index) for index in range(1, 62)]
    missing = [name for name, keep in zip(channel_names, channel_mask) if keep and name not in raw.ch_names]
    if missing:
        raise ValueError(f"{eeg_path}: retained EEG channels are missing: {missing}")
    kept_names = [name for name, keep in zip(channel_names, channel_mask) if keep]
    data = raw.get_data(picks=kept_names).T.astype(np.float32, copy=False)
    if raw.info["sfreq"] != 500:
        raise ValueError(f"{eeg_path}: expected 500 Hz EEG")
    data = _bandpass(data, 500)
    data = _common_average_reference(data)
    data = _resample(data, 500, TARGET_FS)
    result = np.zeros((len(data), 61), dtype=np.float32)
    result[:, channel_mask] = data
    return result


def _training_positions(
    examples: list[Example],
    starts: np.ndarray,
    lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.flatnonzero([example.split == "train" for example in examples])
    train_lengths = lengths[indices].astype(np.int64)
    cumulative = np.cumsum(train_lengths)
    total = int(cumulative[-1])
    count = min(STATS_SAMPLE_LIMIT, total)
    rng = np.random.default_rng(RANDOM_SEED)
    selected = np.sort(rng.choice(total, size=count, replace=False))
    positions_in_examples = np.searchsorted(cumulative, selected, side="right")
    previous = np.concatenate(([0], cumulative[:-1]))
    example_indices = indices[positions_in_examples]
    packed_positions = starts[example_indices] + selected - previous[positions_in_examples]
    return example_indices, packed_positions.astype(np.int64)


def _fit_and_apply_normalization(
    output_dir: Path,
    examples: list[Example],
    starts: np.ndarray,
    lengths: np.ndarray,
    channel_masks: np.ndarray,
) -> dict[str, Any]:
    brain = np.load(output_dir / "brain.npy", mmap_mode="r+")
    envelope = np.load(output_dir / "speech_envelope.npy", mmap_mode="r+")
    sampled_example_indices, sampled_positions = _training_positions(examples, starts, lengths)
    sampled_brain = np.asarray(brain[sampled_positions], dtype=np.float32)
    sampled_masks = channel_masks[sampled_example_indices]
    center = np.zeros(brain.shape[1], dtype=np.float32)
    scale = np.ones(brain.shape[1], dtype=np.float32)
    inactive_channels: list[int] = []
    for channel in range(brain.shape[1]):
        valid = sampled_masks[:, channel]
        if not valid.any():
            inactive_channels.append(channel)
            continue
        values = sampled_brain[valid, channel]
        q25, median, q75 = np.percentile(values, [25, 50, 75])
        center[channel] = median
        spread = (q75 - q25) / 1.349
        scale[channel] = spread if spread > np.finfo(np.float32).eps else 1.0
    sampled_envelope = np.asarray(envelope[sampled_positions], dtype=np.float32)
    envelope_center = float(np.median(sampled_envelope))
    q25, q75 = np.percentile(sampled_envelope, [25, 75])
    envelope_scale = float((q75 - q25) / 1.349)
    if not np.isfinite(envelope_scale) or envelope_scale <= np.finfo(np.float32).eps:
        raise ValueError("Training speech envelope has no usable variation")

    del sampled_brain, sampled_masks, sampled_envelope, brain, envelope
    brain = np.load(output_dir / "brain.npy", mmap_mode="r+")
    envelope = np.load(output_dir / "speech_envelope.npy", mmap_mode="r+")
    chunk_size = 100_000
    for example_index, (start, length) in enumerate(zip(starts, lengths)):
        stop = int(start + length)
        signal = brain[int(start):stop]
        signal -= center
        signal /= scale
        signal[:] = np.clip(signal, -20.0, 20.0)
        signal[:, ~channel_masks[example_index]] = 0.0
        target = envelope[int(start):stop]
        target -= envelope_center
        target /= envelope_scale
        if example_index % 1000 == 0:
            brain.flush()
            envelope.flush()
    brain.flush()
    envelope.flush()
    del brain, envelope

    normalization = {
        "eeg_center": center.tolist(),
        "eeg_scale": scale.tolist(),
        "envelope_center": envelope_center,
        "envelope_scale": envelope_scale,
        "method": "training-only median and IQR/1.349; EEG clipped to [-20, 20]",
        "sampled_training_timepoints": int(len(sampled_positions)),
        "channels_without_training_measurements": inactive_channels,
    }
    np.savez_compressed(
        output_dir / "normalization.npz",
        eeg_center=center,
        eeg_scale=scale,
        envelope_center=np.float32(envelope_center),
        envelope_scale=np.float32(envelope_scale),
    )
    return normalization


def _write_dataset(
    name: str,
    examples: list[Example],
    sources: dict[str, Any],
    output_root: Path,
    exclusions: list[dict[str, str]],
    alice_envelopes: dict[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    if not examples:
        raise ValueError(f"{name}: no aligned word examples were found")
    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=False)
    channels = 128 if name == "Natural_Speech" else 61
    lengths = np.asarray([example.eeg_stop - example.eeg_start for example in examples], dtype=np.int32)
    if np.any(lengths <= 0):
        raise ValueError(f"{name}: non-positive epoch length")
    starts = np.zeros(len(examples), dtype=np.int64)
    starts[1:] = np.cumsum(lengths[:-1], dtype=np.int64)
    total_samples = int(lengths.sum())
    channel_masks = np.stack([example.channel_mask for example in examples])
    brain = np.lib.format.open_memmap(
        output_dir / "brain.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_samples, channels),
    )
    envelope = np.lib.format.open_memmap(
        output_dir / "speech_envelope.npy",
        mode="w+",
        dtype=np.float32,
        shape=(total_samples,),
    )
    records: list[dict[str, Any]] = []
    by_source: dict[str, list[tuple[int, Example]]] = defaultdict(list)
    for index, example in enumerate(examples):
        by_source[example.source].append((index, example))
        records.append(
            {
                "sample_id": index,
                "participant": example.participant,
                "stimulus_group": example.stimulus_group,
                "split": example.split,
                "word": example.word,
                "order": example.order,
                "onset_seconds": example.onset,
                "offset_seconds": example.offset,
                "duration_seconds": example.offset - example.onset,
                "sample_start": int(starts[index]),
                "sample_stop": int(starts[index] + lengths[index]),
                "length_samples": int(lengths[index]),
                "channel_mask_id": index,
            }
        )

    for source, source_examples in by_source.items():
        if name == "Natural_Speech":
            eeg, target = _load_natural_source(*sources[source])
        else:
            eeg = _load_alice_source(sources[source], source_examples[0][1].channel_mask)
            if alice_envelopes is None:
                raise ValueError("Alice speech envelopes were not initialized")

        for index, example in source_examples:
            if name == "AliceChapterOne":
                segment = int(example.stimulus_group.removeprefix("Segment"))
                target = alice_envelopes[segment]
            eeg_start, eeg_stop = example.eeg_start, example.eeg_stop
            target_start, target_stop = example.envelope_start, example.envelope_stop
            if eeg_start < 0 or eeg_stop > len(eeg):
                raise ValueError(
                    f"{source}, word {example.order}: EEG interval [{eeg_start}, {eeg_stop}) "
                    f"exceeds recording length {len(eeg)}"
                )
            if target_start < 0 or target_stop > len(target):
                raise ValueError(
                    f"{source}, word {example.order}: envelope interval "
                    f"[{target_start}, {target_stop}) exceeds length {len(target)}"
                )
            brain_slice = eeg[eeg_start:eeg_stop]
            target_slice = target[target_start:target_stop]
            length = int(lengths[index])
            if len(brain_slice) != length or len(target_slice) != length:
                raise ValueError(f"{source}, word {example.order}: EEG/envelope sample counts differ")
            if not np.isfinite(brain_slice).all() or not np.isfinite(target_slice).all():
                raise ValueError(f"{source}, word {example.order}: non-finite aligned data")
            packed = int(starts[index])
            brain[packed:packed + length] = brain_slice
            envelope[packed:packed + length] = target_slice
        del eeg, target
    brain.flush()
    envelope.flush()
    del brain, envelope

    records_frame = pd.DataFrame.from_records(records)
    records_frame.to_csv(output_dir / "records.csv", index=False)
    np.save(output_dir / "offsets.npy", starts)
    np.save(output_dir / "lengths.npy", lengths)
    np.save(output_dir / "channel_mask.npy", channel_masks)
    normalization = _fit_and_apply_normalization(
        output_dir,
        examples,
        starts,
        lengths,
        channel_masks,
    )

    split_counts = records_frame["split"].value_counts().to_dict()
    if set(split_counts) != {"train", "validation", "test"}:
        raise ValueError(f"{name}: one or more data splits are empty")
    split_groups = {
        split: sorted(records_frame.loc[records_frame["split"] == split, "stimulus_group"].unique().tolist())
        for split in ("train", "validation", "test")
    }
    if any(
        set(split_groups[first]).intersection(split_groups[second])
        for first, second in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError(f"{name}: stimulus groups overlap between splits")
    excluded_channels: dict[str, list[str]] = {}
    if name == "AliceChapterOne":
        mask_by_participant: dict[str, np.ndarray] = {}
        for example in examples:
            previous = mask_by_participant.setdefault(
                example.participant,
                example.channel_mask,
            )
            if not np.array_equal(previous, example.channel_mask):
                raise ValueError(
                    f"{example.participant}: channel mask changes between word examples"
                )
        excluded_channels = {
            participant: [
                str(index + 1)
                for index, available in enumerate(mask)
                if not available
            ]
            for participant, mask in mask_by_participant.items()
            if not mask.all()
        }
    metadata = {
        "dataset": name,
        "sampling_rate_hz": TARGET_FS,
        "channels": channels,
        "channel_names": (
            [f"EEG_{index:03d}" for index in range(1, 129)]
            if name == "Natural_Speech"
            else [str(index) for index in range(1, 62)]
        ),
        "examples": len(examples),
        "packed_timepoints": total_samples,
        "split_examples": {key: int(value) for key, value in split_counts.items()},
        "split_stimulus_groups": split_groups,
        "filter_hz": [LOW_CUT_HZ, HIGH_CUT_HZ],
        "reference": "common average over available EEG channels",
        "speech_envelope": (
            "provided run envelope at 128 Hz"
            if name == "Natural_Speech"
            else "Hilbert amplitude, 8 Hz low-pass, resampled from each WAV"
        ),
        "normalization": normalization,
        "excluded_participants": exclusions,
        "excluded_channels_by_participant": excluded_channels,
        "limitations": [
            "Natural Speech recordings have no event-level artifact rejection metadata.",
            "Alice bad channels are zero-filled and identified by channel_mask.npy; no interpolation is claimed.",
            "Speech envelope and EEG are aligned at the annotated word interval without an extra neural-latency shift.",
            "Word epochs are variable length and are stored packed; batch padding/masking is the model loader's responsibility.",
        ],
        "array_shapes": {
            "brain.npy": [total_samples, channels],
            "speech_envelope.npy": [total_samples],
            "offsets.npy": [len(examples)],
            "lengths.npy": [len(examples)],
            "channel_mask.npy": [len(examples), channels],
        },
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    _validate_saved_dataset(output_dir, metadata, records_frame)
    return metadata


def _validate_saved_dataset(
    output_dir: Path,
    metadata: dict[str, Any],
    records: pd.DataFrame,
) -> None:
    brain = np.load(output_dir / "brain.npy", mmap_mode="r")
    envelope = np.load(output_dir / "speech_envelope.npy", mmap_mode="r")
    offsets = np.load(output_dir / "offsets.npy", mmap_mode="r")
    lengths = np.load(output_dir / "lengths.npy", mmap_mode="r")
    masks = np.load(output_dir / "channel_mask.npy", mmap_mode="r")
    if brain.shape != tuple(metadata["array_shapes"]["brain.npy"]):
        raise ValueError(f"{output_dir}: saved EEG shape does not match metadata")
    if envelope.shape != (metadata["packed_timepoints"],):
        raise ValueError(f"{output_dir}: saved envelope shape does not match metadata")
    if len(offsets) != metadata["examples"] or len(lengths) != metadata["examples"]:
        raise ValueError(f"{output_dir}: saved index arrays do not match example count")
    if masks.shape != (metadata["examples"], metadata["channels"]):
        raise ValueError(f"{output_dir}: saved channel mask shape does not match")
    if not np.isfinite(brain).all() or not np.isfinite(envelope).all():
        raise ValueError(f"{output_dir}: saved arrays contain non-finite values")
    if np.any(lengths <= 0) or np.any(offsets + lengths > len(brain)):
        raise ValueError(f"{output_dir}: saved offsets or lengths are invalid")
    for row in records.itertuples(index=False):
        if row.sample_stop - row.sample_start != row.length_samples:
            raise ValueError(f"{output_dir}: record has inconsistent sample indexes")
    del brain, envelope, offsets, lengths, masks


def run(dataset: str, output_root: Path, dataset_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "target_sampling_rate_hz": TARGET_FS,
        "output_directory": str(output_root),
        "split_seed": RANDOM_SEED,
        "split_policy": (
            "stimulus-grouped: Natural Speech runs and Alice audio segments stay "
            "within one split across all participants"
        ),
        "datasets": {},
    }
    if dataset in ("all", "natural"):
        examples, sources, exclusions = _natural_specs(dataset_root)
        summary["datasets"]["Natural_Speech"] = _write_dataset(
            "Natural_Speech",
            examples,
            sources,
            output_root,
            exclusions,
        )
    if dataset in ("all", "alice"):
        envelopes = _load_audio_envelopes(dataset_root)
        examples, sources, exclusions = _alice_specs(dataset_root, envelopes)
        summary["datasets"]["AliceChapterOne"] = _write_dataset(
            "AliceChapterOne",
            examples,
            sources,
            output_root,
            exclusions,
            envelopes,
        )
    with (output_root / "preprocessing_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("all", "natural", "alice"), default="all")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_root = args.output.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_root}. "
            "Choose a new --output path; existing results are never overwritten."
        )
    summary = run(args.dataset, output_root, args.dataset_root.resolve())
    for name, metadata in summary["datasets"].items():
        print(
            f"{name}: {metadata['examples']:,} word epochs, "
            f"{metadata['packed_timepoints']:,} packed samples; "
            f"train/validation/test="
            f"{metadata['split_examples']['train']:,}/"
            f"{metadata['split_examples']['validation']:,}/"
            f"{metadata['split_examples']['test']:,}"
        )
    print(f"Saved validated outputs to {output_root}")


if __name__ == "__main__":
    main()

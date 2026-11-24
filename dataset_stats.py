"""Read-only statistics of Dataset/ (Alice EEG corpus).

Only opens files for reading; nothing in Dataset/ is modified.
Writes a summary to runs/dataset_stats/stats.json and prints it to the log.

Usage: python dataset_stats.py --data Dataset --out runs/dataset_stats
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def inventory(root: Path) -> dict:
    by_dir = defaultdict(lambda: {"files": 0, "bytes": 0, "ext": Counter()})
    by_ext = defaultdict(lambda: {"files": 0, "bytes": 0})
    total, n = 0, 0
    print("== File tree ==")
    for d, dirs, files in os.walk(root):
        dirs.sort()
        rel = os.path.relpath(d, root)
        for f in sorted(files):
            p = Path(d) / f
            sz = p.stat().st_size
            ext = p.suffix.lower() or "(none)"
            by_dir[rel]["files"] += 1
            by_dir[rel]["bytes"] += sz
            by_dir[rel]["ext"][ext] += 1
            by_ext[ext]["files"] += 1
            by_ext[ext]["bytes"] += sz
            total += sz
            n += 1
            if rel == "." or by_dir[rel]["files"] <= 6:
                print(f"  {os.path.join(rel, f)}  {human(sz)}")
    print(f"Total: {n} files, {human(total)}")
    for k, v in sorted(by_dir.items()):
        print(f"  dir {k}: {v['files']} files, {human(v['bytes'])}, {dict(v['ext'])}")
    return dict(total_files=n, total_bytes=total,
                dirs={k: dict(files=v["files"], bytes=v["bytes"], ext=dict(v["ext"])) for k, v in by_dir.items()},
                ext={k: dict(v) for k, v in by_ext.items()})


def text_files(root: Path) -> dict:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in {".txt", ".md", ".json", ".tsv", ".rtf", ".m", ".py"} \
                and p.stat().st_size < 50_000:
            txt = p.read_text(errors="ignore")
            out[str(p.relative_to(root))] = txt[:3000]
            print(f"\n== {p.relative_to(root)} (first 3000 chars) ==\n{txt[:3000]}")
    return out


def words_csv(root: Path) -> dict:
    import pandas as pd
    out = {}
    for p in sorted(root.rglob("*.csv")):
        df = pd.read_csv(p)
        info = dict(rows=len(df), columns=list(df.columns), head=df.head(5).astype(str).to_dict("records"))
        for c in df.columns:
            if df[c].dtype.kind in "if":
                info.setdefault("numeric", {})[c] = dict(min=float(df[c].min()), max=float(df[c].max()),
                                                         mean=float(df[c].mean()), nunique=int(df[c].nunique()))
            else:
                info.setdefault("text", {})[c] = dict(nunique=int(df[c].nunique()))
        if {"Word", "Sentence", "Segment"} <= set(df.columns):
            info["n_words"] = len(df)
            info["n_unique_words"] = int(df.Word.astype(str).str.lower().nunique())
            info["n_sentences"] = int(df.Sentence.nunique())
            info["n_segments"] = int(df.Segment.nunique())
            info["words_per_segment"] = df.groupby("Segment").size().to_dict()
            info["words_per_sentence"] = dict(mean=float(df.groupby("Sentence").size().mean()),
                                              min=int(df.groupby("Sentence").size().min()),
                                              max=int(df.groupby("Sentence").size().max()))
            if {"onset", "offset"} <= set(df.columns):
                dur = df.offset - df.onset
                info["word_duration_s"] = dict(mean=float(dur.mean()), median=float(dur.median()),
                                               min=float(dur.min()), max=float(dur.max()))
            info["top_words"] = df.Word.astype(str).str.lower().value_counts().head(15).to_dict()
        out[str(p.relative_to(root))] = info
        print(f"\n== CSV {p.relative_to(root)} ==\n{json.dumps(info, indent=1, default=str)[:4000]}")
    return out


def audio(root: Path) -> dict:
    import soundfile as sf
    clips = []
    for p in sorted(root.rglob("*.wav"), key=lambda q: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", q.name)]):
        i = sf.info(p)
        clips.append(dict(file=p.name, duration_s=i.duration, samplerate=i.samplerate,
                          channels=i.channels, subtype=i.subtype))
    tot = sum(c["duration_s"] for c in clips)
    print(f"\n== Audio: {len(clips)} wav, total {tot:.1f} s ({tot/60:.2f} min) ==")
    for c in clips:
        print(f"  {c}")
    return dict(n=len(clips), total_s=tot, clips=clips)


def eeg(root: Path) -> dict:
    import mne
    subs = []
    for vhdr in sorted(root.rglob("*.vhdr")):
        try:
            raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose="error")
            ann = Counter(raw.annotations.description)
            subs.append(dict(subject=vhdr.stem, n_channels=len(raw.ch_names), sfreq=raw.info["sfreq"],
                             duration_s=raw.n_times / raw.info["sfreq"], n_samples=int(raw.n_times),
                             ch_names=raw.ch_names, ch_types=dict(Counter(raw.get_channel_types())),
                             n_markers=int(sum(ann.values())),
                             stim_markers=int(sum(v for k, v in ann.items() if k.startswith("Stimulus"))),
                             highpass=raw.info["highpass"], lowpass=raw.info["lowpass"]))
        except Exception as e:  # noqa: BLE001
            subs.append(dict(subject=vhdr.stem, error=repr(e)))
    ok = [s for s in subs if "error" not in s]
    print(f"\n== EEG: {len(subs)} .vhdr headers ({len(ok)} readable) ==")
    for s in subs:
        print("  " + json.dumps({k: v for k, v in s.items() if k != "ch_names"}, default=str))
    if ok:
        print(f"  channel names (first subject): {ok[0]['ch_names']}")
    summ = {}
    if ok:
        d = np.array([s["duration_s"] for s in ok])
        summ = dict(n_subjects=len(ok), channels=dict(Counter(s["n_channels"] for s in ok)),
                    sfreq=dict(Counter(s["sfreq"] for s in ok)),
                    duration_s=dict(mean=float(d.mean()), min=float(d.min()), max=float(d.max()),
                                    total_h=float(d.sum() / 3600)))
    return dict(summary=summ, subjects=subs)


def proc(root: Path) -> dict:
    from scipy.io import loadmat
    out = []
    for p in sorted(root.rglob("*.mat")):
        try:
            m = loadmat(p, squeeze_me=True, struct_as_record=False)
            keys = [k for k in m if not k.startswith("__")]
            rec = dict(file=str(p.relative_to(root)), keys=keys)
            if "proc" in m:
                pr = m["proc"]
                rec["fields"] = list(pr._fieldnames)
                trl = np.asarray(pr.trl)
                rec["n_trials"] = int(trl.shape[0])
                fin = pr.rejections.final
                rec["kept_trials"] = int(np.atleast_1d(fin.trialpicks).size)
                rec["kept_chans"] = int(np.atleast_1d(fin.chanpicks).size)
                rec["varnames"] = [str(v) for v in np.atleast_1d(pr.varnames)]
            out.append(rec)
        except Exception as e:  # noqa: BLE001
            out.append(dict(file=str(p.relative_to(root)), error=repr(e)))
    print(f"\n== MAT files: {len(out)} ==")
    for r in out:
        print("  " + json.dumps(r, default=str))
    return dict(n=len(out), files=out)


# =========================================================== EEG signal stats
SCALP_REC = [str(i) for i in range(1, 62) if i != 29]  # recorded scalp channels ("29" = reference)
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
MARKER_DELAY = 0.0513  # s, clip start = "Stimulus,k" marker + 51.3 ms (see prepare_alice_dataset.py)


def vmrk_markers(vmrk: Path, sfreq: float) -> dict[int, float]:
    """{clip k: time in s} of the first 'Stimulus,<k>' marker (same parser as prepare_alice_dataset.py)."""
    out = {}
    if not vmrk.exists():
        return out
    for typ, desc, pos in re.findall(r"Mk\d+=([^,]*),([^,]*),(\d+),", vmrk.read_text(errors="ignore")):
        d = re.sub(r"\D", "", desc)
        if typ.strip() == "Stimulus" and d and 1 <= int(d) <= 12 and int(d) not in out:
            out[int(d)] = (int(pos) - 1) / sfreq
    return out


def proc_bad_channels(path: Path) -> set[str] | None:
    """Scalp channels the authors rejected (None if no proc file)."""
    from scipy.io import loadmat
    if not path.exists():
        return None
    p = loadmat(path, squeeze_me=True, struct_as_record=False)["proc"]
    kept = set(map(str, np.atleast_1d(p.rejections.final.chanpicks)))
    return {c for c in SCALP_REC if c not in kept}


def read_eeg(vhdr: Path):
    import mne
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="error")
    # scalp = numbered electrodes; VEOG = eog; anything else (Aux5, AUD audio channel, ...) = misc
    types = {c: "eeg" if c in SCALP_REC else "eog" if c.upper().startswith("VEOG") else "misc"
             for c in raw.ch_names}
    raw.set_channel_types(types, verbose="error")
    return raw


def band_power(f, P, lo, hi):
    from scipy.integrate import trapezoid
    m = (f >= lo) & (f < hi)
    return trapezoid(P[..., m], f[m], axis=-1)


def channel_stats(raw, bad: set[str] | None) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Per-channel statistics. Signals in the recording reference (ch 29), in µV.

    Amplitude/shape stats are computed after a 0.1 Hz high-pass (removes the DC offset);
    the raw DC offset and drift are reported separately.
    """
    from scipy import signal as sg
    from scipy import stats as st

    sf = raw.info["sfreq"]
    names = raw.ch_names
    types = raw.get_channel_types()
    x_raw = raw.get_data(picks="all") * 1e6
    dc = np.median(x_raw, axis=1)
    drift = np.median(x_raw[:, -int(10 * sf):], axis=1) - np.median(x_raw[:, :int(10 * sf)], axis=1)
    flat = np.mean(np.diff(x_raw, axis=1) == 0, axis=1)
    del x_raw

    x = raw.copy().filter(0.1, None, picks="all", n_jobs=4, verbose="error").get_data(picks="all") * 1e6
    f, P = sg.welch(x, fs=sf, nperseg=int(2 * sf), axis=-1)  # µV²/Hz, 0.5 Hz resolution

    tot = band_power(f, P, 1, 45)
    bp = {b: band_power(f, P, lo, hi) for b, (lo, hi) in BANDS.items()}
    line = band_power(f, P, 59, 61) / band_power(f, P, 1, 100)
    hf = band_power(f, P, 45, 100) / band_power(f, P, 1, 100)  # muscle / high-frequency noise
    m = (f >= 1) & (f < 45)
    pn = P[:, m] / P[:, m].sum(1, keepdims=True)
    sent = -(pn * np.log2(pn + 1e-30)).sum(1) / np.log2(m.sum())
    sef95 = f[m][np.argmax(np.cumsum(pn, 1) >= 0.95, axis=1)]
    ma = (f >= 7) & (f <= 14)
    paf = f[ma][np.argmax(P[:, ma], axis=1)]
    ms = (f >= 2) & (f <= 40) & ~ma
    slope = np.array([np.polyfit(np.log10(f[ms]), np.log10(p[ms] + 1e-30), 1)[0] for p in P])

    d1 = np.diff(x, axis=1)
    d2 = np.diff(d1, axis=1)
    v0, v1, v2 = x.var(1), d1.var(1), d2.var(1)
    mob = np.sqrt(v1 / v0)
    comp = np.sqrt(v2 / v1) / mob
    del d1, d2
    zcr = np.sum(np.diff(np.signbit(x), axis=1), axis=1) / (x.shape[1] / sf)

    scalp_idx = [i for i, t in enumerate(types) if t == "eeg"]
    C = np.corrcoef(x[scalp_idx, ::5])
    np.fill_diagonal(C, np.nan)
    mean_abs_corr = np.full(len(names), np.nan)
    mean_abs_corr[scalp_idx] = np.nanmean(np.abs(C), axis=1)

    q1, q99 = np.percentile(x, [1, 99], axis=1)
    q25, q75 = np.percentile(x, [25, 75], axis=1)
    rms = np.sqrt(np.mean(x ** 2, axis=1))
    rows = []
    for i, ch in enumerate(names):
        r = dict(channel=ch, type=types[i], dc_offset_uv=dc[i], drift_uv=drift[i], flat_frac=flat[i],
                 mean_uv=x[i].mean(), std_uv=x[i].std(), rms_uv=rms[i], min_uv=x[i].min(), max_uv=x[i].max(),
                 ptp_uv=np.ptp(x[i]), p1_uv=q1[i], p99_uv=q99[i], iqr_uv=q75[i] - q25[i],
                 mad_uv=np.median(np.abs(x[i] - np.median(x[i]))),
                 skew=st.skew(x[i]), kurtosis=st.kurtosis(x[i]),
                 frac_abs_gt_100uv=np.mean(np.abs(x[i]) > 100),
                 hjorth_activity=v0[i], hjorth_mobility=mob[i], hjorth_complexity=comp[i],
                 zero_cross_per_s=zcr[i], power_1_45=tot[i], line60_ratio=line[i], hf45_100_ratio=hf[i],
                 spectral_entropy=sent[i], sef95_hz=sef95[i], peak_alpha_hz=paf[i], aperiodic_slope=slope[i],
                 mean_abs_corr=mean_abs_corr[i],
                 proc_bad=(ch in bad) if bad is not None else None)
        for b in BANDS:
            r[f"{b}_abs"] = bp[b][i]
            r[f"{b}_rel"] = bp[b][i] / tot[i]
        rows.append(r)

    # own outlier flag: robust z of log RMS over scalp channels
    lr = np.log(rms[scalp_idx])
    z = (lr - np.median(lr)) / (1.4826 * np.median(np.abs(lr - np.median(lr))) + 1e-12)
    for j, i in enumerate(scalp_idx):
        rows[i]["rms_robust_z"] = z[j]
        rows[i]["flag_outlier"] = bool(abs(z[j]) > 3 or flat[i] > 0.5)
    good = [i for i in scalp_idx if not rows[i]["flag_outlier"] and (bad is None or names[i] not in bad)]
    return rows, f, P[good].mean(0)


def scalp_clean(raw, bad: set[str] | None):
    """Scalp EEG: ref 29 added back, rejected channels dropped, average reference, 1-40 Hz."""
    import mne
    r = raw.copy().pick([c for c in raw.ch_names if c in SCALP_REC])
    r = mne.add_reference_channels(r, "29")
    r.set_montage("easycap-M10", verbose="error")
    if bad:
        r.drop_channels(sorted(bad & set(r.ch_names)))
    r.set_eeg_reference("average", verbose="error")
    r.filter(1.0, 40.0, n_jobs=4, verbose="error")
    return r


def subject_summary(subj, raw, rows, bad, markers) -> dict:
    from scipy.signal import find_peaks
    sf = raw.info["sfreq"]
    scalp = [r for r in rows if r["type"] == "eeg"]
    flagged = {r["channel"] for r in scalp if r.get("flag_outlier")}
    s = dict(subject=subj, duration_s=raw.n_times / sf, n_channels=len(raw.ch_names), n_markers=len(markers),
             n_proc_bad=None if bad is None else len(bad), n_flagged=len(flagged),
             flag_vs_proc_jaccard=None if not bad and not flagged or bad is None
             else len(flagged & bad) / max(1, len(flagged | bad)))
    for k in ("rms_uv", "kurtosis", "line60_ratio", "hf45_100_ratio", "mean_abs_corr", "peak_alpha_hz",
              "aperiodic_slope", "spectral_entropy") + tuple(f"{b}_rel" for b in BANDS):
        s[f"median_{k}"] = float(np.median([r[k] for r in scalp]))
    r = scalp_clean(raw, bad)
    y = r.get_data() * 1e6
    gfp = y.std(0)
    s.update(gfp_mean_uv=float(gfp.mean()), gfp_p95_uv=float(np.percentile(gfp, 95)))
    eog = [c for c in raw.ch_names if c.upper().startswith("VEOG")]
    if eog:
        v = raw.copy().pick(eog).filter(1, 10, picks="all", verbose="error").get_data()[0] * 1e6
        thr = max(50.0, np.median(np.abs(v)) * 8)
        pk, _ = find_peaks(np.abs(v), height=thr, distance=int(0.3 * sf))
        s["blinks_per_min"] = len(pk) / (raw.n_times / sf / 60)
    return s


def plot_subject(subj, raw, rows, bad, markers, words, figdir: Path) -> dict:
    """3-D figures of the most significant signals of one participant."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from scipy import signal as sg

    figdir.mkdir(parents=True, exist_ok=True)
    r = scalp_clean(raw, bad)
    sf = r.info["sfreq"]
    names = r.ch_names
    y = r.get_data() * 1e6

    # significance = share of each channel in the variance of the top-3 principal components
    yd = y[:, ::5] - y[:, ::5].mean(1, keepdims=True)
    U, S, Vt = np.linalg.svd(yd, full_matrices=False)
    expl = S ** 2 / np.sum(S ** 2)
    score = ((U[:, :3] * S[:3]) ** 2).sum(1) / np.sum(S ** 2)
    order = np.argsort(score)[::-1]
    top = order[:8]
    by_name = {rw["channel"]: rw for rw in rows}
    details = []
    for rank, i in enumerate(top, 1):
        rw = by_name.get(names[i], {})
        details.append(dict(rank=rank, channel=names[i], pca_top3_share=float(score[i]),
                            rms_uv_clean=float(np.sqrt(np.mean(y[i] ** 2))),
                            **{k: rw.get(k) for k in ("rms_uv", "kurtosis", "peak_alpha_hz", "alpha_rel",
                                                       "theta_rel", "delta_rel", "beta_rel", "line60_ratio")}))
    import pandas as pd
    pd.DataFrame(details).to_csv(figdir / "top_channels.csv", index=False)
    print(f"\n== {subj}: top channels (share of top-3 PC variance; PC1-3 explain "
          f"{100 * expl[:3].sum():.1f}%) ==\n{pd.DataFrame(details).to_string(index=False)}")

    t0 = (markers.get(1, markers[min(markers)]) if markers else 30.0) + MARKER_DELAY + 5.0
    i0, i1 = int(t0 * sf), int((t0 + 10) * sf)
    tt = np.arange(i1 - i0) / sf

    # 1) 3-D waterfall of the top-8 channels
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("viridis")
    for k, i in enumerate(top[::-1]):
        ax.plot(tt, np.full_like(tt, k), y[i, i0:i1], lw=0.7, color=cmap(k / len(top)))
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"ch {names[i]}" for i in top[::-1]])
    ax.set_xlabel("time (s)")
    ax.set_zlabel("µV")
    ax.set_title(f"{subj}: 8 most significant channels, 10 s of story (1-40 Hz, avg ref)")
    ax.view_init(28, -65)
    fig.tight_layout()
    fig.savefig(figdir / "01_waterfall_top8_3d.png", dpi=150)
    plt.close(fig)

    # 2) 3-D scalp: electrode positions coloured by significance
    pos = mne.channels.make_standard_montage("easycap-M10").get_positions()["ch_pos"]
    xyz = np.array([pos[c] for c in names]) * 100
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(*xyz.T, c=score, s=40 + 900 * score / score.max(), cmap="magma_r", edgecolor="k", lw=0.4)
    for i in top:
        ax.text(*xyz[i], f" {names[i]}", fontsize=9, weight="bold")
    if bad:
        bxyz = np.array([pos[c] for c in sorted(bad)]) * 100
        ax.scatter(*bxyz.T, marker="x", color="grey", s=40, label="rejected (proc)")
        ax.legend(loc="upper left")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="share of top-3 PC variance")
    ax.set_xlabel("x (cm, right)")
    ax.set_ylabel("y (cm, front)")
    ax.set_zlabel("z (cm)")
    ax.set_title(f"{subj}: channel significance on the scalp")
    ax.view_init(35, -60)
    fig.tight_layout()
    fig.savefig(figdir / "02_scalp_significance_3d.png", dpi=150)
    plt.close(fig)

    # 3) 3-D spectrogram of the most significant channel (first 60 s of story)
    j = top[0]
    seg = y[j, int((t0 - 5) * sf): int((t0 + 55) * sf)]
    fs_, ts_, Sxx = sg.spectrogram(seg, fs=sf, nperseg=int(2 * sf), noverlap=int(1.5 * sf))
    keep = fs_ <= 40
    T, F = np.meshgrid(ts_, fs_[keep])
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(T, F, 10 * np.log10(Sxx[keep] + 1e-12), cmap="inferno", linewidth=0, antialiased=True)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_zlabel("power (dB µV²/Hz)")
    ax.set_title(f"{subj}: spectrogram of channel {names[j]} (most significant)")
    ax.view_init(35, -120)
    fig.tight_layout()
    fig.savefig(figdir / "03_spectrogram_top1_3d.png", dpi=150)
    plt.close(fig)

    # 4) 3-D trajectory of the first three principal components (10 s)
    pcs = (U[:, :3].T @ (y[:, i0:i1] - y[:, i0:i1].mean(1, keepdims=True)))
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(*pcs, c=tt, cmap="viridis", s=2)
    ax.plot(*pcs, lw=0.3, color="grey", alpha=0.5)
    fig.colorbar(p, ax=ax, shrink=0.6, label="time (s)")
    ax.set_xlabel(f"PC1 ({100 * expl[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100 * expl[1]:.1f}%)")
    ax.set_zlabel(f"PC3 ({100 * expl[2]:.1f}%)")
    ax.set_title(f"{subj}: EEG state-space trajectory (top-3 PCs)")
    fig.tight_layout()
    fig.savefig(figdir / "04_pca_trajectory_3d.png", dpi=150)
    plt.close(fig)

    # 5) 3-D word-onset ERP surface (16 most significant channels), marker-aligned
    erp_info = {}
    if markers:
        sel = order[:16]
        pre, post = int(0.2 * sf), int(0.8 * sf)
        acc, n, n_rej = np.zeros((len(sel), pre + post)), 0, 0
        for w in words.itertuples():
            if w.Segment not in markers:
                continue
            c = int(round((markers[w.Segment] + MARKER_DELAY + w.onset) * sf))
            if c - pre < 0 or c + post > y.shape[1]:
                continue
            ep = y[sel, c - pre:c + post]
            ep = ep - ep[:, :pre].mean(1, keepdims=True)
            if np.ptp(ep, axis=1).max() > 150:
                n_rej += 1
                continue
            acc += ep
            n += 1
        if n:
            erp = acc / n
            te = (np.arange(pre + post) - pre) / sf * 1000
            Tm, Cm = np.meshgrid(te, np.arange(len(sel)))
            fig = plt.figure(figsize=(12, 7))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(Tm, Cm, erp, cmap="coolwarm", linewidth=0, antialiased=True)
            ax.set_yticks(range(len(sel)))
            ax.set_yticklabels([names[i] for i in sel], fontsize=7)
            ax.set_xlabel("time from word onset (ms)")
            ax.set_ylabel("channel")
            ax.set_zlabel("µV")
            ax.set_title(f"{subj}: word-onset ERP, {n} words ({n_rej} rejected >150 µV)")
            ax.view_init(30, -60)
            fig.tight_layout()
            fig.savefig(figdir / "05_word_erp_surface_3d.png", dpi=150)
            plt.close(fig)
            gfp = erp.std(0)
            erp_info = dict(n_epochs=n, n_rejected=n_rej, gfp_peak_ms=float(te[np.argmax(gfp)]),
                            gfp_peak_uv=float(gfp.max()))

    # 6) 2-D PSD of all clean channels, top-8 highlighted
    f, P = sg.welch(y, fs=sf, nperseg=int(2 * sf), axis=-1)
    m = f <= 45
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(len(names)):
        ax.semilogy(f[m], P[i, m], color="lightgrey", lw=0.6)
    for k, i in enumerate(top):
        ax.semilogy(f[m], P[i, m], lw=1.2, color=cmap(k / len(top)), label=f"ch {names[i]}")
    for b, (lo, hi) in BANDS.items():
        ax.axvspan(lo, hi, alpha=0.06, color="C0" if b in ("delta", "alpha", "gamma") else "C1")
        ax.text((lo + hi) / 2, ax.get_ylim()[1], b, ha="center", va="top", fontsize=8)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD (µV²/Hz)")
    ax.set_title(f"{subj}: power spectra (grey = all channels)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(figdir / "06_psd_all_channels.png", dpi=150)
    plt.close(fig)

    return dict(subject=subj, figdir=str(figdir), pca_explained_top3=[float(e) for e in expl[:3]],
                window_s=[t0, t0 + 10], top_channels=details, erp=erp_info)


def group_plots(ch_df, subj_df, psd_f, psd_by_subj: dict, figdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    figdir.mkdir(parents=True, exist_ok=True)
    sc = ch_df[ch_df.type == "eeg"]
    piv = sc.pivot(index="subject", columns="channel", values="rms_uv")
    piv = piv[sorted(piv.columns, key=int)]
    fig, ax = plt.subplots(figsize=(14, 9))
    im = ax.imshow(np.log10(piv.values), aspect="auto", cmap="viridis")
    bad = sc.pivot(index="subject", columns="channel", values="proc_bad")[piv.columns]
    yy, xx = np.where(bad.values == True)  # noqa: E712
    ax.scatter(xx, yy, marker="x", color="red", s=12, label="rejected by authors")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, fontsize=6)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=7)
    fig.colorbar(im, ax=ax, label="log10 RMS (µV, 0.1 Hz high-pass, ref 29)")
    ax.legend(loc="upper right")
    ax.set_title("Channel RMS per subject")
    fig.tight_layout()
    fig.savefig(figdir / "group_rms_heatmap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot([subj_df[f"median_{b}_rel"] for b in BANDS], labels=list(BANDS))
    ax.set_ylabel("relative power (of 1-45 Hz)")
    ax.set_title("Relative band power (median over scalp channels), one point per subject")
    fig.tight_layout()
    fig.savefig(figdir / "group_band_power.png", dpi=150)
    plt.close(fig)

    # 3-D: mean PSD of every subject stacked along the subject axis
    subs = sorted(psd_by_subj)
    m = (psd_f >= 1) & (psd_f <= 45)
    Z = np.log10(np.array([psd_by_subj[s][m] for s in subs]))
    Fm, Sm = np.meshgrid(psd_f[m], np.arange(len(subs)))
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(Fm, Sm, Z, cmap="viridis", linewidth=0)
    ax.set_yticks(range(0, len(subs), 4))
    ax.set_yticklabels(subs[::4], fontsize=7)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("subject")
    ax.set_zlabel("log10 PSD (µV²/Hz)")
    ax.set_title("Mean scalp power spectrum per subject")
    ax.view_init(30, -120)
    fig.tight_layout()
    fig.savefig(figdir / "group_psd_surface_3d.png", dpi=150)
    plt.close(fig)


def eeg_signals(root: Path, out: Path, figroot: Path, subject: str | None, seed: int | None) -> dict:
    import pandas as pd
    words = pd.read_csv(root / "AliceChapterOne-EEG.csv")
    subs = sorted(p.stem for p in (root / "eeg").glob("S*.vhdr"))
    usable = [s for s in subs if (root / "proc" / f"{s}.mat").exists()]
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % 2 ** 31)
    pick = subject or str(np.random.default_rng(seed).choice(usable))
    print(f"\n== EEG signal statistics: {len(subs)} subjects; random participant = {pick} (seed {seed}) ==")

    ch_rows, subj_rows, psd_by, plot_info, psd_f = [], [], {}, {}, None
    for s in subs:
        try:
            raw = read_eeg(root / "eeg" / f"{s}.vhdr")
            bad = proc_bad_channels(root / "proc" / f"{s}.mat")
            mk = vmrk_markers(root / "eeg" / f"{s}.vmrk", raw.info["sfreq"])
            rows, psd_f, psd = channel_stats(raw, bad)
            for r in rows:
                r["subject"] = s
            ch_rows += rows
            psd_by[s] = psd
            subj_rows.append(subject_summary(s, raw, rows, bad, mk))
            print(f"  {s}: done ({len(rows)} ch, flagged {subj_rows[-1]['n_flagged']}, "
                  f"proc bad {subj_rows[-1]['n_proc_bad']})", flush=True)
            if s == pick:
                plot_info = plot_subject(s, raw, rows, bad, mk, words, figroot / s)
            del raw
        except Exception as e:  # noqa: BLE001
            print(f"  !! {s}: {e!r}", flush=True)

    ch_df = pd.DataFrame(ch_rows)
    ch_df.insert(0, "subject", ch_df.pop("subject"))
    subj_df = pd.DataFrame(subj_rows)
    out.mkdir(parents=True, exist_ok=True)
    ch_df.to_csv(out / "eeg_channel_stats.csv", index=False)
    subj_df.to_csv(out / "eeg_subject_stats.csv", index=False)
    try:
        group_plots(ch_df, subj_df, psd_f, psd_by, figroot / "group")
    except Exception as e:  # noqa: BLE001
        print(f"  !! group plots: {e!r}")

    sc = ch_df[ch_df.type == "eeg"]
    num = [c for c in sc.columns if sc[c].dtype.kind in "if"]
    by_type = ch_df.groupby("type")[["rms_uv", "kurtosis", "line60_ratio"]].median()
    pd.set_option("display.width", 200)
    print("\n== Scalp channel statistics over all subjects x channels ==")
    print(sc[num].describe(percentiles=[0.05, 0.5, 0.95]).T.to_string())
    print("\n== Per-subject summary ==")
    print(subj_df.to_string(index=False))
    print("\n== Median by channel type ==")
    print(by_type.to_string())
    if "proc_bad" in sc:
        cmp = sc.dropna(subset=["proc_bad"]).groupby("proc_bad")[["rms_uv", "kurtosis", "hf45_100_ratio",
                                                                  "line60_ratio", "mean_abs_corr"]].median()
        print("\n== Authors-rejected vs kept channels (medians) ==")
        print(cmp.to_string())
    worst = (sc.dropna(subset=["proc_bad"]).assign(pb=lambda d: d.proc_bad.astype(float))
             .groupby("channel").pb.mean().sort_values(ascending=False).head(10))
    print("\n== Channels most often rejected by the authors (fraction of subjects) ==")
    print(worst.to_string())
    return dict(random_subject=pick, seed=seed, plots=plot_info,
                scalp_describe=sc[num].describe().to_dict(),
                subjects=subj_df.to_dict("records"),
                rejected_vs_kept=cmp.to_dict() if "proc_bad" in sc else None,
                most_rejected_channels=worst.to_dict())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("Dataset"))
    ap.add_argument("--out", type=Path, default=Path("runs/dataset_stats"))
    ap.add_argument("--figs", type=Path, default=Path("figures/eeg_stats"))
    ap.add_argument("--subject", default=None, help="participant to plot (default: random usable one)")
    ap.add_argument("--seed", type=int, default=None, help="seed for the random participant (logged)")
    ap.add_argument("--skip-basic", action="store_true", help="only run the EEG signal statistics")
    a = ap.parse_args()
    res = {}
    basic = [] if a.skip_basic else [("inventory", inventory), ("text_files", text_files), ("csv", words_csv),
                                     ("audio", audio), ("eeg", eeg), ("mat", proc)]
    for name, fn in basic:
        try:
            res[name] = fn(a.data)
        except Exception as e:  # noqa: BLE001
            res[name] = dict(error=repr(e))
            print(f"!! {name} failed: {e!r}")
    try:
        res["eeg_signals"] = eeg_signals(a.data, a.out, a.figs, a.subject, a.seed)
    except Exception as e:  # noqa: BLE001
        res["eeg_signals"] = dict(error=repr(e))
        print(f"!! eeg_signals failed: {e!r}")
    a.out.mkdir(parents=True, exist_ok=True)
    name = "eeg_signal_stats.json" if a.skip_basic else "stats.json"
    (a.out / name).write_text(json.dumps(res, indent=1, default=str))
    print(f"\nWrote {a.out / name}")


if __name__ == "__main__":
    main()

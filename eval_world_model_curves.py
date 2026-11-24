"""Train / val / test curves over training steps for world model v1 and v3 (one figure per model).

The training scripts only log val metrics. This scores every checkpoint saved with
--save-every (run/ckpt/step_N.pt) on four splits, the way mtrf_baseline_v2 is
plotted against ridge lambda:
  train        seen subjects, training sentences   (word-onset windows, --per-subject-train per subject)
  val          seen subjects, val sentences        (same windows as training-time validation)
  test_seen    seen subjects, test sentences       (--per-subject-test per subject)
  test_unseen  held-out subjects, test sentences   (all windows)
Metrics are computed per subject, then mean +- SEM over subjects.

  v1 (world_model_alice.py): mel_loss = L1 + (1 - Pearson) of the IDM-decoded Mel (no input noise),
     dyn_loss = dynamics latent MSE with true audio; r_env, r_above_null, mm_acc, wm_mm_acc
  v3 (world_model_v3.py): tf_loss = next-patch L1 + (1 - Pearson) teacher forced (the training
     objective without noise), eeg_only_loss = same for the EEG-only decoding; eeg_only / rollout
     r_env, mm_acc, eeg_gain, wm_mm_acc

Usage
  python eval_world_model_curves.py eval --model v1 --run runs/world_model_curves_v1   (GPU)
  python eval_world_model_curves.py plot --model v1 --run runs/world_model_curves_v1   (CPU)
  python eval_world_model_curves.py mel  --model v1 --run runs/world_model             (CPU ok)
Outputs in --run: per_subject_curves.csv, curves.csv, eval_curves.png
mel: true vs predicted 120-band Mel, 8 coarse bands and envelope (+ mTRF envelope) of best.pt for the
     windows of mtrf_mel_plots.py, like figures/mtrf_mel/<subject>_w<k>.png
     -> figures/world_model_mel/<model>/<subject>_w<k>[_eeg_only|_rollout].png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SPLITS = ["train", "val", "test_seen", "test_unseen"]


# ============================================================================== eval
def mel_loss(p, y):
    """Per-window L1 + (1 - mean Pearson over bands), as pearson_loss in world_model_alice.py. [n,T,K] -> [n]."""
    import torch
    pc, yc = p - p.mean(1, keepdim=True), y - y.mean(1, keepdim=True)
    r = (pc * yc).sum(1) / (pc.norm(dim=1) * yc.norm(dim=1) + 1e-6)
    return ((p - y).abs().mean((1, 2)) + 1 - r.mean(1)).float()


def eval_v1(ck, d, w, a, dev):
    import torch
    import world_model_alice as wm
    model, tgt = wm.build(a, d, dev)
    model.load_state_dict(ck["model"])
    tgt.load_state_dict(ck["tgt"])
    P, Y, subj, starts, df, _ = wm.run_windows(model, tgt, d, w, a, dev)
    m = wm.summarize(P, Y, subj, starts, df, d.T)
    ml = np.concatenate([mel_loss(torch.from_numpy(P[i:i + 512]), torch.from_numpy(Y[i:i + 512])).numpy()
                         for i in range(0, len(P), 512)])
    per = pd.DataFrame(dict(subject=subj, mel_loss=ml, dyn_loss=df.err_true.to_numpy())).groupby("subject").mean()
    return m.join(per, on="subject")


def eval_v3(ck, d, w, a, dev):
    import torch
    import world_model_v3 as wm
    model = wm.build(a, d, dev)
    model.load_state_dict(ck["model"])
    model.eval()
    C0, ctx, L = a.ctx_patches, d.ctx, {"tf_loss": [], "eeg_only_loss": []}
    with torch.no_grad():
        for i in range(0, len(w), a.eval_batch):
            r = w.iloc[i:i + a.eval_batch]
            b = d.assemble(r.subject.tolist(), (r.audio_start - ctx).tolist(), (r.eeg_start - ctx).tolist(), dev)
            y = b["mel"][:, ctx:]
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                e = model.eeg(b["x"], b["sid"]).float()
                s = model.aud(b["mel"]).float()
                for mode in ("tf", "eeg_only"):
                    L[f"{mode}_loss"].append(mel_loss(wm.decode(model, s, e, C0, mode).float(), y).cpu().numpy())
    R, subj, starts = wm.run_windows(model, d, w, a, dev)
    m = wm.summarize(R, subj, starts, d.T)
    per = pd.DataFrame(dict(subject=subj, **{k: np.concatenate(v) for k, v in L.items()})).groupby("subject").mean()
    return m.join(per, on="subject")


def evaluate(o):
    import torch
    run = Path(o.run)
    cks = sorted((run / "ckpt").glob("step_*.pt"))
    assert cks, f"no checkpoints in {run / 'ckpt'} (train with --save-every)"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = argparse.Namespace(**torch.load(cks[0], map_location="cpu")["args"])
    a.eval_batch = o.eval_batch
    import world_model_alice as wm1
    d = wm1.AliceSeq(a)
    wins = {"train": d.eval_windows("train", d.seen, o.per_subject_train),
            "val": d.eval_windows("val", d.seen, a.val_per_subject),
            "test_seen": d.eval_windows("test", d.seen, o.per_subject_test),
            "test_unseen": d.eval_windows("test", d.heldout)}
    print("windows: " + ", ".join(f"{k} {len(v)}" for k, v in wins.items()), flush=True)
    fn = eval_v1 if o.model == "v1" else eval_v3
    rows = []
    for c in cks:
        ck = torch.load(c, map_location=dev)
        for split, w in wins.items():
            if not len(w):
                continue
            m = fn(ck, d, w, a, dev)
            m.insert(0, "split", split)
            m.insert(0, "step", int(ck["step"]))
            rows.append(m)
        last = pd.concat(rows[-len(wins):])
        print(f"step {ck['step']}: " + " | ".join(
            f"{s} " + " ".join(f"{k} {last[last.split == s][k].mean():.4f}" for k in KEY[o.model])
            for s in SPLITS), flush=True)
        pd.concat(rows).to_csv(run / "per_subject_curves.csv", index=False)  # after every checkpoint
    per = pd.concat(rows)
    num = [c for c in per.columns if c not in ("step", "split", "subject")]
    g = per.groupby(["step", "split"])[num]
    curves = g.mean().add_suffix("_mean").join(g.sem().add_suffix("_sem")).reset_index()
    curves.to_csv(run / "curves.csv", index=False)
    print(f"saved {run / 'curves.csv'}")


KEY = {"v1": ["mel_loss", "r_env", "mm_acc"], "v3": ["tf_loss", "eeg_only_r_env", "eeg_only_mm_acc"]}


# ============================================================================== plot
COL = {"train": "#2a78d6", "val": "#eb6834", "test_seen": "#1baf7a", "test_unseen": "#eda100"}
MK = {"train": "o", "val": "s", "test_seen": "^", "test_unseen": "v"}
LBL = {"train": "train (seen)", "val": "val (seen)", "test_seen": "test seen", "test_unseen": "test unseen"}
PANELS = {
    "v1": [("mel_loss", "Loss: IDM Mel decoding (L1 + 1 - r)", "loss", None),
           ("dyn_loss", "Loss: dynamics, latent MSE (true audio)", "loss", None),
           ("r_env", "Accuracy: envelope reconstruction r", "envelope Pearson r", "r_env"),
           ("r_above_null", "Envelope r above null (r_env - r_env_null)", "Δ Pearson r", "r_above_null"),
           ("mm_acc", "Accuracy: 3 s match-mismatch (chance 0.5)", "match-mismatch accuracy", "mm_acc"),
           ("wm_mm_acc", "World model: true beats shuffled audio (chance 0.5)", "accuracy", None)],
    "v3": [("tf_loss", "Loss: next-patch Mel, teacher forced (L1 + 1 - r)", "loss", None),
           ("eeg_only_loss", "Loss: EEG-only Mel decoding (L1 + 1 - r)", "loss", None),
           ("eeg_only_r_env", "Accuracy: envelope r, EEG only (fair vs mTRF)", "envelope Pearson r", "r_env"),
           ("eeg_only_mm_acc", "Accuracy: 3 s match-mismatch, EEG only", "match-mismatch accuracy", "mm_acc"),
           ("rollout_mm_acc", "Match-mismatch, rollout (sees 1 s true audio)", "match-mismatch accuracy", "mm_acc"),
           ("eeg_only_eeg_gain", "EEG gain, EEG only: r(true EEG) - r(shuffled EEG)", "Δ Pearson r", None)],
}
TITLE = {"v1": "World model v1 (EEG = state, audio = action, IDM decoder)",
         "v3": "World model v3 (audio = state, EEG = action)"}


def plot(o):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(o.run)
    c = pd.read_csv(run / "curves.csv")
    n_subj = pd.read_csv(run / "per_subject_curves.csv").groupby("split").subject.nunique().to_dict()
    best = json.loads((run / "metrics.json").read_text())["checkpoint_step"] if (run / "metrics.json").exists() else None
    base = None
    if Path(o.baseline).exists():
        b = pd.read_csv(o.baseline).assign(r_above_null=lambda x: x.r_env - x.r_env_null)
        per = pd.read_csv(run / "per_subject_curves.csv")
        seen = per[per.split == "test_seen"].subject.unique()
        unseen = per[per.split == "test_unseen"].subject.unique()
        base = (b[(b.model == "subject") & b.subject.isin(seen)], b[(b.model == "pooled") & b.subject.isin(unseen)])

    fig, axs = plt.subplots(2, 3, figsize=(17, 9.5))
    for ax, (k, title, ylabel, bk) in zip(axs.flat, PANELS[o.model]):
        for s in SPLITS:
            q = c[c.split == s].sort_values("step")
            if not len(q) or f"{k}_mean" not in q:
                continue
            ax.errorbar(q.step, q[f"{k}_mean"], yerr=q[f"{k}_sem"], color=COL[s], marker=MK[s], ms=5, lw=2,
                        capsize=3, ls=":" if s == "test_unseen" else "-",
                        label=f"{LBL[s]} (n={n_subj.get(s, '?')})")
        if base is not None and bk:
            ax.axhline(base[0][bk].mean(), color="#8a8985", ls="--", lw=1.2, label="mTRF subject, test seen")
            ax.axhline(base[1][bk].mean(), color="#8a8985", ls="-.", lw=1.2, label="mTRF pooled, test unseen")
        if "mm_acc" in k:
            ax.axhline(0.5, color="#52514e", lw=0.8)
        if k.endswith("gain"):
            ax.axhline(0, color="#52514e", lw=0.8)
        if best:
            ax.axvline(best, color="#0b0b0b", ls=":", lw=1, label=f"best.pt (step {best})")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#e6e5e1", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    h, l = axs.flat[2].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=9)
    fig.suptitle(f"{TITLE[o.model]}: train / val / test over training steps (mean ± SEM over subjects)", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    out = run / "eval_curves.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


# ===================================================================== mel examples
def plot_window(path, title, label, true120, pred120, fs, env_mtrf=None):
    """Same layout as mtrf_mel_plots.py: true / predicted 120-band Mel, true / predicted 8 bands, envelope."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mtrf_baseline import corr_rows

    T = true120.shape[0]
    true8, pred8 = (m.reshape(T, 8, -1).mean(2) for m in (true120, pred120))
    env_t, env_p = true120.mean(1), pred120.mean(1)
    r120 = corr_rows(pred120[None], true120[None])[0]
    r8 = corr_rows(pred8[None], true8[None])[0]
    rc = lambda p: float(corr_rows(p[None, :, None], env_t[None, :, None])[0, 0])
    fig, ax = plt.subplots(5, 1, figsize=(11, 13), gridspec_kw=dict(height_ratios=[3, 3, 1.4, 1.4, 1.6]))

    def show(a, data, ylabel, nb):
        lim = np.percentile(np.abs(data), 98) + 1e-6
        im = a.imshow(data.T, origin="lower", aspect="auto", cmap="magma", vmin=-lim, vmax=lim,
                      extent=[0, T / fs, 0, nb])
        a.set_ylabel(ylabel)
        fig.colorbar(im, ax=a, pad=0.01, label="z")

    show(ax[0], true120, "Mel band\n(true)", 120)
    ax[0].set_title("True log-Mel spectrogram (120 bands, standardized)")
    show(ax[1], pred120, "Mel band\n(predicted)", 120)
    ax[1].set_title(f"{label} predicted log-Mel (own colour scale; mean band r = {r120.mean():.3f})")
    show(ax[2], true8, "band\n(true)", 8)
    ax[2].set_title("True 8 coarse Mel bands")
    show(ax[3], pred8, "band\n(pred)", 8)
    ax[3].set_title(f"{label} predicted 8 coarse bands (means of its 120 bands; mean band r = {r8.mean():.3f})")
    z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    t = np.arange(T) / fs
    ax[4].plot(t, z(env_t), label="true envelope", lw=1.2, color="#2a78d6")
    ax[4].plot(t, z(env_p), label=f"{label} envelope (r = {rc(env_p):.3f})", lw=1.2, color="#eb6834")
    if env_mtrf is not None:
        ax[4].plot(t, z(env_mtrf), label=f"mTRF subject model (r = {rc(env_mtrf):.3f})", lw=1, ls="--",
                   color="#8a8985")
    ax[4].set(xlabel="time in window (s)", ylabel="z", xlim=(0, T / fs))
    ax[4].legend(fontsize=8, loc="upper right")
    for a in ax[:4]:
        a.set_xticklabels([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return rc(env_p), float(r8.mean()), float(r120.mean())


def mel_examples(o):
    """Decode the windows of figures/mtrf_mel/windows_plotted.csv with best.pt and plot them like mtrf_mel_plots.py."""
    import torch
    import world_model_alice as wm1
    run, out = Path(o.run), Path(o.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(run / "best.pt", map_location="cpu")
    a = argparse.Namespace(**ck["args"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = wm1.AliceSeq(a)
    fs, ctx = d.meta["fs"], d.ctx
    if o.model == "v1":
        model, tgt = wm1.build(a, d, dev)
        tgt.load_state_dict(ck["tgt"])
    else:
        import world_model_v3 as wm3
        model = wm3.build(a, d, dev)
    model.load_state_dict(ck["model"])
    model.eval()

    sel = pd.read_csv(o.windows).merge(d.windows[["subject", "audio_start", "eeg_start", "audio_clip"]]
                                       .drop_duplicates(["subject", "audio_start"]), on=["subject", "audio_start"])
    n_eeg = sel.subject.map(lambda s: d.eegs[s].shape[1])
    ok = (sel.eeg_start - ctx >= 0) & (sel.eeg_start + d.T <= n_eeg)
    for r in sel[~ok].itertuples():
        print(f"skip {r.subject} w{r.window}: not enough EEG before the window for {ctx} context frames")
    sel = sel[ok].reset_index(drop=True)
    b = d.assemble(sel.subject.tolist(), (sel.audio_start - ctx).tolist(), (sel.eeg_start - ctx).tolist(), dev)
    with torch.no_grad():
        if o.model == "v1":
            preds = {"": ("world model v1 (IDM)", model.idm(model.enc(b["x"], b["sid"]))[1][:, ctx:])}
        else:
            e, s = model.eeg(b["x"], b["sid"]), model.aud(b["mel"])
            preds = {"_eeg_only": ("world model v3, EEG only", wm3.decode(model, s, e, a.ctx_patches, "eeg_only")),
                     "_rollout": ("world model v3, rollout",
                                  wm3.decode(model, s, e, a.ctx_patches, "rollout"))}
    true = b["mel"][:, ctx:].float().cpu().numpy()
    clip_start = np.array(d.meta["clip_offsets"])[sel.audio_clip.to_numpy() - 1]
    rows = []
    for i, r in enumerate(sel.itertuples()):
        env_m = None
        npz = Path(o.mtrf_dir) / f"predictions_{r.subject}.npz"
        if npz.exists():
            z = np.load(npz)
            k = np.flatnonzero(z["audio_start"] == r.audio_start)
            env_m = z["pred_env"][k[0]] if len(k) else None
        group = "seen" if r.subject in d.sid else "unseen"
        cross = r.audio_start - ctx < clip_start[i]
        t0 = (r.audio_start - clip_start[i]) / fs
        for sfx, (label, p) in preds.items():
            title = (f"{r.subject} ({group} subject) | test window {r.window}: clip {r.audio_clip}, "
                     f"{t0:.1f}-{t0 + d.T / fs:.1f} s | {label}, best.pt step {ck['step']}"
                     + (" (decoding starts from 1 s of true audio context)" if sfx == "_rollout" else "")
                     + (" | context starts in the previous clip" if cross and sfx == "_rollout" else "")
                     + f"\n\"{r.text}\"")
            re, r8, r120 = plot_window(out / f"{r.subject}_w{r.window}{sfx}.png", title, label, true[i],
                                       p[i].float().cpu().numpy(), fs, env_m)
            rows.append(dict(subject=r.subject, group=group, window=r.window, mode=sfx.strip("_") or "idm",
                             r_env=re, mean_r_8bands=r8, mean_r_120bands=r120, mtrf_r_env=r.r_env))
    df = pd.DataFrame(rows)
    df.to_csv(out / "windows_plotted.csv", index=False)
    print(df.to_string(index=False))
    print(f"saved figures to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("cmd", choices=["eval", "plot", "mel"])
    p.add_argument("--out", default=None, help="mel: figure folder (default figures/world_model_mel/<model>)")
    p.add_argument("--windows", default=str(ROOT / "figures" / "mtrf_mel" / "windows_plotted.csv"),
                   help="mel: windows to plot (default: the ones of mtrf_mel_plots.py)")
    p.add_argument("--mtrf-dir", default=str(ROOT / "figures" / "mtrf_mel"),
                   help="mel: predictions_<subject>.npz of mtrf_mel_plots.py for the mTRF envelope overlay")
    p.add_argument("--model", choices=["v1", "v3"], required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--baseline", default=str(ROOT / "runs" / "mtrf_baseline_v2" / "results.csv"))
    p.add_argument("--per-subject-train", type=int, default=100)
    p.add_argument("--per-subject-test", type=int, default=100)
    p.add_argument("--eval-batch", type=int, default=128)
    o = p.parse_args()
    o.out = o.out or str(ROOT / "figures" / "world_model_mel" / o.model)
    {"eval": evaluate, "plot": plot, "mel": mel_examples}[o.cmd](o)

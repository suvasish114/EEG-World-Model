"""Train / val / test curves over training steps for the SENSE-style model (sense_alice.py),
plus true vs predicted Mel examples, laid out like the mTRF figures.

sense_alice.py only logs val metrics. This scores every checkpoint saved with --save-every
(run/ckpt/step_N.pt) on four splits, no augmentation (no jitter / channel dropout / noise):
  train        seen subjects, training sentences   (--per-subject-train windows per subject)
  val          seen subjects, val sentences        (same windows as training-time validation)
  test_seen    seen subjects, test sentences       (--per-subject-test windows per subject)
  test_unseen  held-out subjects, test sentences   (all windows)
Per subject, then mean +- SEM over subjects:
  losses   : mel_loss = L1 + (1 - Pearson per band), recon (cosine), ctc (per phoneme), esc (1 - cos to
             CLIP text, low-surprisal windows only), total = the training objective with the run's weights
  accuracy : r_env, r_env_null, r_above_null, r_bands, mm_acc (mtrf_baseline.evaluate, as in sense_alice.py)

Usage
  python eval_sense_curves.py eval --run runs/sense_curves    (GPU)
  python eval_sense_curves.py plot --run runs/sense_curves    (CPU)
  python eval_sense_curves.py mel  --run runs/sense_curves    (CPU ok)
Outputs in --run: per_subject_curves.csv, curves.csv, eval_curves.png, aux_losses.png
mel: true vs predicted 120-band Mel, 8 coarse bands and envelope (+ mTRF envelope) of best.pt for the windows
     of figures/mtrf_mel/windows_plotted.csv -> figures/sense_mel/<subject>_w<k>.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval_world_model_curves import COL, LBL, MK, SPLITS, plot_window

ROOT = Path(__file__).resolve().parent


# ============================================================================== eval
def score(model, ds, T, dev, a, batch):
    """Per-subject accuracy + loss table for one checkpoint on one set of windows."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    import sense_alice as sa
    from mtrf_baseline import evaluate as window_metrics

    model.eval()
    acc = {k: [] for k in ("Pe", "Ye", "subj", "starts", "l1", "corr", "recon", "ctc", "esc_num", "esc_w")}
    with torch.no_grad():
        for b in DataLoader(ds, batch, collate_fn=sa.collate, num_workers=4):
            x, y = b["eeg"].to(dev), b["mel"].to(dev)
            o = model(x, T)
            p = o["mel"].float()
            pc, yc = p - p.mean(1, keepdim=True), y - y.mean(1, keepdim=True)
            r = (pc * yc).sum(1) / (pc.norm(dim=1) * yc.norm(dim=1) + 1e-6)
            xf, xh = x.flatten(1), o["xh"].float().flatten(1)
            logp = o["ctc"].float().log_softmax(-1).transpose(0, 1)
            pl = b["phon_len"].to(dev)
            ctc = F.ctc_loss(logp, b["phon"].to(dev), o["lens3"], pl, blank=0, reduction="none",
                             zero_infinity=True) / pl.clamp_min(1)
            w = b["esc_w"].to(dev).float()
            cos = F.cosine_similarity(o["g"].float(), b["clip"].to(dev), dim=-1)
            for k, v in (("l1", (p - y).abs().mean((1, 2))), ("corr", 1 - r.mean(1)),
                         ("recon", 1 - (xf * xh).sum(1) / (xf.norm(dim=1) * xh.norm(dim=1) + 1e-8)),
                         ("ctc", ctc), ("esc_num", (1 - cos) * w), ("esc_w", w)):
                acc[k].append(v.cpu().numpy())
            acc["Pe"].append(sa.env_bands(p.cpu().numpy()))
            acc["Ye"].append(sa.env_bands(b["mel"].numpy()))
            acc["subj"] += b["subject"]
            acc["starts"].append(b["audio_start"])
    cat = {k: (np.array(v) if k == "subj" else np.concatenate(v)) for k, v in acc.items()}
    rows = []
    for s in sorted(set(cat["subj"])):
        k = cat["subj"] == s
        m = window_metrics(cat["Pe"][k], cat["Ye"][k], cat["starts"][k], T)
        L = {n: float(cat[n][k].mean()) for n in ("l1", "corr", "recon", "ctc")}
        L["esc"] = float(cat["esc_num"][k].sum() / max(cat["esc_w"][k].sum(), 1.0))
        L["mel_loss"] = L["l1"] + L["corr"]
        L["total_loss"] = (L["l1"] + a.l_corr * L["corr"] + a.l_recon * L["recon"] + a.l_ctc * L["ctc"]
                           + a.l_esc * L["esc"])
        rows.append(dict(subject=s, n_windows=int(k.sum()), r_above_null=m["r_env"] - m["r_env_null"], **m, **L))
    return pd.DataFrame(rows)


def per_subject(df, n, seed=0):
    return df.sample(frac=1.0, random_state=seed).groupby("subject").head(n) if n else df


def evaluate(o):
    import torch

    import sense_alice as sa
    run = Path(o.run)
    cks = sorted((run / "ckpt").glob("step_*.pt"))
    assert cks, f"no checkpoints in {run / 'ckpt'} (train with --save-every)"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = argparse.Namespace(**torch.load(cks[0], map_location="cpu")["args"])
    d = sa.load_data(a)
    T, W = d["T"], d["windows"]
    seen, held = W.subject.isin(d["seen"]), W.subject.isin(d["heldout"])
    wins = {"train": per_subject(W[(W.split == "train") & seen], o.per_subject_train),
            "val": per_subject(W[(W.split == "val") & seen], a.val_per_subject),  # as in sense_alice.train
            "test_seen": per_subject(W[(W.split == "test") & seen], o.per_subject_test),
            "test_unseen": W[(W.split == "test") & held]}
    print("windows: " + ", ".join(f"{k} {len(v)}" for k, v in wins.items()), flush=True)
    model = sa.build(d, dev)
    rows = []
    for c in cks:
        ck = torch.load(c, map_location=dev)
        model.load_state_dict(ck["model"])
        for split, w in wins.items():
            if not len(w):
                continue
            m = score(model, d["mk"](w), T, dev, a, o.eval_batch)
            m.insert(0, "split", split)
            m.insert(0, "step", int(ck["step"]))
            rows.append(m)
        per = pd.concat(rows)
        last = per[per.step == int(ck["step"])]
        print(f"step {ck['step']}: " + " | ".join(
            f"{s} " + " ".join(f"{k} {last[last.split == s][k].mean():.4f}" for k in ("total_loss", "r_env", "mm_acc"))
            for s in SPLITS), flush=True)
        per.to_csv(run / "per_subject_curves.csv", index=False)  # after every checkpoint
    num = [c for c in per.columns if c not in ("step", "split", "subject")]
    g = per.groupby(["step", "split"])[num]
    curves = g.mean().add_suffix("_mean").join(g.sem().add_suffix("_sem")).reset_index()
    curves.to_csv(run / "curves.csv", index=False)
    print(f"saved {run / 'curves.csv'}")


# ============================================================================== plot
PANELS = [("total_loss", "Loss: total training objective (Mel + aux)", "loss", None),
          ("mel_loss", "Loss: Mel decoding (L1 + 1 - r)", "loss", None),
          ("r_env", "Accuracy: envelope reconstruction r", "envelope Pearson r", "r_env"),
          ("r_above_null", "Envelope r above null (r_env - r_env_null)", "Δ Pearson r", "r_above_null"),
          ("mm_acc", "Accuracy: 3 s match-mismatch (chance 0.5)", "match-mismatch accuracy", "mm_acc"),
          ("r_bands", "Accuracy: 8 coarse Mel bands r", "mean band Pearson r", "r_bands")]
AUX = [("recon", "Aux loss: EEG self-reconstruction (1 - cos)", "loss"),
       ("ctc", "Aux loss: CTC phonemes (per phoneme)", "loss"),
       ("esc", "Aux loss: ESC, 1 - cos(g_eeg, CLIP text)", "loss")]


def plot(o):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(o.run)
    c = pd.read_csv(run / "curves.csv")
    per = pd.read_csv(run / "per_subject_curves.csv")
    n_subj = per.groupby("split").subject.nunique().to_dict()
    best = json.loads((run / "metrics.json").read_text())["checkpoint_step"] if (run / "metrics.json").exists() else None
    base = None
    if Path(o.baseline).exists():
        b = pd.read_csv(o.baseline).assign(r_above_null=lambda x: x.r_env - x.r_env_null)
        base = (b[(b.model == "subject") & b.subject.isin(per[per.split == "test_seen"].subject.unique())],
                b[(b.model == "pooled") & b.subject.isin(per[per.split == "test_unseen"].subject.unique())])

    def draw(ax, k, title, ylabel, bk=None):
        for s in SPLITS:
            q = c[c.split == s].sort_values("step")
            if not len(q) or f"{k}_mean" not in q:
                continue
            ax.errorbar(q.step, q[f"{k}_mean"], yerr=q[f"{k}_sem"], color=COL[s], marker=MK[s], ms=5, lw=2,
                        capsize=3, ls=":" if s == "test_unseen" else "-",
                        label=f"{LBL[s]} (n={n_subj.get(s, '?')})")
        if base is not None and bk and bk in base[0]:
            ax.axhline(base[0][bk].mean(), color="#8a8985", ls="--", lw=1.2, label="mTRF subject, test seen")
            ax.axhline(base[1][bk].mean(), color="#8a8985", ls="-.", lw=1.2, label="mTRF pooled, test unseen")
        if k == "mm_acc":
            ax.axhline(0.5, color="#52514e", lw=0.8)
        if best:
            ax.axvline(best, color="#0b0b0b", ls=":", lw=1, label=f"best.pt (step {best})")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#e6e5e1", lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    fig, axs = plt.subplots(2, 3, figsize=(17, 9.5))
    for ax, (k, title, ylabel, bk) in zip(axs.flat, PANELS):
        draw(ax, k, title, ylabel, bk)
    h, l = axs.flat[2].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=9)
    fig.suptitle("SENSE-style model (sense_alice.py): train / val / test over training steps "
                 "(mean ± SEM over subjects)", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(run / "eval_curves.png", dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, (k, title, ylabel) in zip(axs, AUX):
        draw(ax, k, title, ylabel)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=9)
    fig.suptitle("SENSE-style model: auxiliary losses over training steps (mean ± SEM over subjects)", fontsize=13)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    fig.savefig(run / "aux_losses.png", dpi=130)
    plt.close(fig)
    print(f"saved {run / 'eval_curves.png'} and {run / 'aux_losses.png'}")


# ===================================================================== mel examples
def mel_examples(o):
    """Decode the mtrf_mel_plots.py windows with best.pt; same layout as figures/mtrf_mel/<subject>_w<k>.png."""
    import torch

    import sense_alice as sa
    run, out = Path(o.run), Path(o.out)
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(run / o.ckpt, map_location="cpu")
    a = argparse.Namespace(**ck["args"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = sa.load_data(a)
    T, fs, meta = d["T"], d["meta"]["fs"], d["meta"]
    model = sa.build(d, dev)
    model.load_state_dict(ck["model"])

    sel = pd.read_csv(o.windows)[["subject", "window", "audio_start", "text"]]
    if o.subjects:
        sel = sel[sel.subject.isin(o.subjects)]
    W = d["windows"].drop_duplicates(["subject", "audio_start"])
    m = sel.merge(W[["subject", "audio_start", "eeg_start", "segment_id", "audio_clip"]],
                  on=["subject", "audio_start"], how="left")
    for r in m[m.eeg_start.isna()].itertuples():
        print(f"skip {r.subject} w{r.window}: window not usable (too close to the recording edge or excluded)")
    m = m.dropna(subset=["eeg_start"]).astype({"eeg_start": int, "segment_id": int, "audio_clip": int})
    m = m.reset_index(drop=True)
    P, Y, subj, starts = sa.predict(model, d["mk"](m), T, dev)
    rows = []
    for i, r in enumerate(m.itertuples()):
        env_m = None
        npz = Path(o.mtrf_dir) / f"predictions_{r.subject}.npz"
        if npz.exists():
            z = np.load(npz)
            k = np.flatnonzero(z["audio_start"] == r.audio_start)
            env_m = z["pred_env"][k[0]] if len(k) else None
        group = "seen" if r.subject in d["seen"] else "unseen"
        t0 = (r.audio_start - meta["clip_offsets"][r.audio_clip - 1]) / fs
        title = (f"{r.subject} ({group} subject) | test window {r.window}: clip {r.audio_clip}, "
                 f"{t0:.1f}-{t0 + T / fs:.1f} s | SENSE-style model, {o.ckpt} step {ck['step']}\n\"{r.text}\"")
        re, r8, r120 = plot_window(out / f"{r.subject}_w{r.window}.png", title, "SENSE", Y[i], P[i], fs, env_m)
        rows.append(dict(subject=r.subject, group=group, window=r.window, r_env=re, mean_r_8bands=r8,
                         mean_r_120bands=r120))
    df = pd.DataFrame(rows)
    df.to_csv(out / "windows_plotted.csv", index=False)
    print(df.to_string(index=False))
    print(f"saved figures to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("cmd", choices=["eval", "plot", "mel"])
    p.add_argument("--run", required=True)
    p.add_argument("--baseline", default=str(ROOT / "runs" / "mtrf_baseline_v2" / "results.csv"))
    p.add_argument("--per-subject-train", type=int, default=100)
    p.add_argument("--per-subject-test", type=int, default=100)
    p.add_argument("--eval-batch", type=int, default=128)
    p.add_argument("--out", default=str(ROOT / "figures" / "sense_mel"))
    p.add_argument("--ckpt", default="best.pt", help="mel: checkpoint file inside --run")
    p.add_argument("--subjects", nargs="*", default=None, help="mel: only these subjects (default: all in --windows)")
    p.add_argument("--windows", default=str(ROOT / "figures" / "mtrf_mel" / "windows_plotted.csv"))
    p.add_argument("--mtrf-dir", default=str(ROOT / "figures" / "mtrf_mel"))
    o = p.parse_args()
    {"eval": evaluate, "plot": plot, "mel": mel_examples}[o.cmd](o)

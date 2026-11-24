"""Brain-module ablations from Défossez et al. (2023), Extended Data Fig. 2.

Each ablation removes one component and retrains the model from scratch with
the same data, loss (CLIP on wav2vec 2.0), optimizer, seed and stopping rule as
the reference run. The reference ("full") row is the existing checkpoint in
runs/wav2vec_clip/best_model.pt, re-evaluated on the test set.

  no_sa_dropout       - Spatial attention dropout   (drop radius 0)
  relu                - GELU + ReLU                  (every GELU -> ReLU)
  no_final_convs      - Final convs                  (two 1x1 convs -> one linear 1x1 projection)
  no_glu              - Non-residual GLU conv        (third, GLU conv of each block removed)
  no_skip             - Skip connections             (no residual adds in the blocks)
  no_initial_conv     - Initial 1x1 conv             (conv after spatial attention removed)
  no_spatial_attention- Spatial attention            (replaced by a learned 1x1 conv, 61 -> 270)
  no_subject_layer    - Subj layer                   (participant layer removed)
  no_clamping         - Clamping                     (EEG re-prepared without the +-20 clamp)

Outputs (in runs/wav2vec_clip/ablations/):
  <name>/best_model.pt, history.csv, results.json, test_per_subject.csv
  ../ablations.csv, ../ablations.json   summary table with the paper's values,
                                        delta vs full model and Wilcoxon p-values

Usage:
  python ablate.py                     # all ablations (skips finished ones)
  python ablate.py --only relu no_skip # a subset
  python ablate.py --smoke             # quick CPU check: build + one step per variant
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon
from torch import nn

from train_clip_decoder import (AliceData, BrainModule, SpatialAttention, SubjectLayer,
                                batches, evaluate_loss, evaluate_retrieval, forward,
                                loss_fn, to_dev)

ROOT = Path(__file__).resolve().parent

# Paper, Extended Data Fig. 2, Brennan (EEG) column: segment-level top-10 accuracy (%)
PAPER_TOP10 = {"full": 25.7, "no_sa_dropout": 26.8, "relu": 24.6, "no_final_convs": 19.0,
               "no_glu": 6.0, "no_skip": 24.2, "no_initial_conv": 22.1,
               "no_spatial_attention": 20.6, "no_subject_layer": 20.2, "no_clamping": 14.1}
LABELS = {"full": "Our model", "no_sa_dropout": "- Spatial attention dropout",
          "relu": "- GELU + ReLU", "no_final_convs": "- Final convs",
          "no_glu": "- Non-residual GLU conv", "no_skip": "- Skip connections",
          "no_initial_conv": "- Initial 1x1 conv", "no_spatial_attention": "- Spatial attention",
          "no_subject_layer": "- Subj layer", "no_clamping": "- Clamping"}


@dataclass(frozen=True)
class Cfg:
    spatial_attention: bool = True
    sa_dropout: bool = True
    initial_conv: bool = True
    subject_layer: bool = True
    skip: bool = True
    glu: bool = True
    final_convs: bool = True
    relu: bool = False
    clamp: bool = True


FULL = Cfg()
VARIANTS = {
    "no_sa_dropout": replace(FULL, sa_dropout=False),
    "relu": replace(FULL, relu=True),
    "no_final_convs": replace(FULL, final_convs=False),
    "no_glu": replace(FULL, glu=False),
    "no_skip": replace(FULL, skip=False),
    "no_initial_conv": replace(FULL, initial_conv=False),
    "no_spatial_attention": replace(FULL, spatial_attention=False),
    "no_subject_layer": replace(FULL, subject_layer=False),
    "no_clamping": replace(FULL, clamp=False),
}


# ------------------------------------------------------------------ model
class Block(nn.Module):
    """ConvBlock of train_clip_decoder.py with switchable skip / GLU / activation."""

    def __init__(self, d_in, d, k, act, skip=True, glu=True):
        super().__init__()
        dil = (2 ** ((2 * k) % 5), 2 ** ((2 * k + 1) % 5))
        self.c1 = nn.Conv1d(d_in, d, 3, padding=dil[0], dilation=dil[0])
        self.c2 = nn.Conv1d(d, d, 3, padding=dil[1], dilation=dil[1])
        self.c3 = nn.Conv1d(d, 2 * d, 3, padding=1) if glu else None
        self.n1, self.n2 = nn.BatchNorm1d(d), nn.BatchNorm1d(d)
        self.act, self.skip, self.res1 = act, skip, skip and d_in == d

    def forward(self, x):
        h = self.act(self.n1(self.c1(x)))
        x = x + h if self.res1 else h
        h = self.act(self.n2(self.c2(x)))
        x = x + h if self.skip else h
        return F.glu(self.c3(x), dim=1) if self.c3 is not None else x


class AblatedBrainModule(nn.Module):
    def __init__(self, positions, n_subjects, n_chans, n_out, cfg: Cfg, d1=270, d2=320, n_blocks=5):
        super().__init__()
        act_fn, act_mod = (F.relu, nn.ReLU) if cfg.relu else (F.gelu, nn.GELU)
        self.sa = (SpatialAttention(positions, d1, drop_radius=0.2 if cfg.sa_dropout else 0.0)
                   if cfg.spatial_attention else nn.Conv1d(n_chans, d1, 1))
        self.merge = nn.Conv1d(d1, d1, 1) if cfg.initial_conv else nn.Identity()
        self.subject = SubjectLayer(n_subjects, d1) if cfg.subject_layer else None
        self.blocks = nn.Sequential(*[Block(d1 if k == 0 else d2, d2, k, act_fn, cfg.skip, cfg.glu)
                                      for k in range(n_blocks)])
        self.head = (nn.Sequential(nn.Conv1d(d2, 2 * d2, 1), act_mod(), nn.Conv1d(2 * d2, n_out, 1))
                     if cfg.final_convs else nn.Conv1d(d2, n_out, 1))

    def forward(self, x, subj):
        x = self.merge(self.sa(x))
        if self.subject is not None:
            x = self.subject(x, subj)
        return self.head(self.blocks(x))


# ------------------------------------------------------------------- data
def unclamped_data(src: Path, dst: Path, raw: Path) -> Path:
    """Re-run the EEG part of prepare_alice_dataset.py without the clamp (speech reused)."""
    if (dst / "meta.json").exists():
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("speech_wav2vec.npy", "speech_mel.npy"):
        if not (dst / f).exists():
            (dst / f).symlink_to((src / f).resolve())
    import prepare_alice_dataset as prep
    prep.CLAMP = 6e4  # effectively no clamp; keeps values inside float16 range
    argv, sys.argv = sys.argv, ["prepare_alice_dataset.py", "--data", str(raw), "--out", str(dst)]
    try:
        prep.main()
    finally:
        sys.argv = argv
    return dst


# ------------------------------------------------------------------ train
def train_variant(name, cfg, data, args, seed, out: Path, dev):
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train, val, test = data.split("train"), data.split("val"), data.split("test")
    if len(val) > args.max_val:
        val = val.sample(args.max_val, random_state=0).reset_index(drop=True)
    model = AblatedBrainModule(np.array(data.meta["positions"]), data.n_subjects,
                               len(data.meta["channels"]), data.speech.shape[1], cfg).to(dev)
    print(f"[{name} seed {seed}] {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def stream():
        while True:
            yield from batches(train, args.batch, rng)

    it, best, wait, hist = stream(), float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for _ in range(args.steps_per_epoch):
            rows = next(it)
            x, y, s, seg = to_dev(data, rows, dev)
            loss = loss_fn("clip", forward(model, x, s), y, seg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, n = tot + loss.item() * len(rows), n + len(rows)
        v = evaluate_loss(model, data, val, "clip", args.batch, dev)
        hist.append(dict(epoch=epoch, train_loss=tot / n, val_loss=v, seconds=time.time() - t0))
        print(f"[{name}] epoch {epoch:3d}  train {tot / n:.4f}  val {v:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        if v < best:
            best, wait = v, 0
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "seed": seed}, out / "best_model.pt")
        else:
            wait += 1
            if wait >= args.patience:
                break

    model.load_state_dict(torch.load(out / "best_model.pt", map_location=dev)["model"])
    res, _, per_subj = evaluate_retrieval(model, data, test, args.batch, dev)
    res |= dict(variant=name, seed=seed, best_val_loss=best, epochs_run=len(hist), cfg=asdict(cfg))
    per_subj.to_csv(out / "test_per_subject.csv")
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(f"[{name}] top1 {res['top1']:.4f}  top10 {res['top10']:.4f}", flush=True)
    return res, per_subj


def evaluate_full(run: Path, data, args, dev):
    ck = torch.load(run / "best_model.pt", map_location=dev)
    model = BrainModule(np.array(data.meta["positions"]), data.n_subjects, data.speech.shape[1]).to(dev)
    model.load_state_dict(ck["model"])
    res, _, per_subj = evaluate_retrieval(model, data, data.split("test"), args.batch, dev)
    return res, per_subj


# ------------------------------------------------------------------ smoke
def smoke(data):
    rows = data.split("train").iloc[:8]
    x, y, s, seg = to_dev(data, rows, "cpu")
    for name, cfg in {"full": FULL, **VARIANTS}.items():
        m = AblatedBrainModule(np.array(data.meta["positions"]), data.n_subjects,
                               len(data.meta["channels"]), data.speech.shape[1], cfg)
        z = m(x, s)
        loss = loss_fn("clip", z, y, seg)
        loss.backward()
        print(f"{name:22s} out {tuple(z.shape)}  loss {loss.item():.3f}  "
              f"params {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M")
    ref = BrainModule(np.array(data.meta["positions"]), data.n_subjects, data.speech.shape[1])
    abl = AblatedBrainModule(np.array(data.meta["positions"]), data.n_subjects,
                             len(data.meta["channels"]), data.speech.shape[1], FULL)
    same = sum(p.numel() for p in ref.parameters()) == sum(p.numel() for p in abl.parameters())
    print(f"full ablation module has the same parameter count as BrainModule: {same}")


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "D_T_processed")
    ap.add_argument("--raw", type=Path, default=ROOT / "Dataset")
    ap.add_argument("--run", type=Path, default=ROOT / "runs" / "wav2vec_clip")
    ap.add_argument("--only", nargs="*", choices=list(VARIANTS), default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    # same settings as the reference run; patience 3 stops at the same best epoch sooner
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps-per-epoch", type=int, default=1200)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--max-val", type=int, default=4000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        smoke(AliceData(args.data, "wav2vec"))
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = args.run / "ablations"
    data = AliceData(args.data, "wav2vec")
    print(f"device={dev}  subjects={data.n_subjects}", flush=True)

    full_res, full_ps = evaluate_full(args.run, data, args, dev)
    out_root.mkdir(parents=True, exist_ok=True)
    full_ps.to_csv(out_root / "full_test_per_subject.csv")
    print(f"[full] top1 {full_res['top1']:.4f}  top10 {full_res['top10']:.4f}", flush=True)

    names = args.only or list(VARIANTS)
    results = {"full": [(full_res, full_ps)]}
    for name in names:
        cfg = VARIANTS[name]
        vdata = data
        if not cfg.clamp:
            vdata = AliceData(unclamped_data(args.data, ROOT / "D_T_processed_noclamp", args.raw), "wav2vec")
        for seed in args.seeds:
            out = out_root / (name if len(args.seeds) == 1 else f"{name}_seed{seed}")
            if (out / "results.json").exists():  # resume
                res = json.loads((out / "results.json").read_text())
                ps = pd.read_csv(out / "test_per_subject.csv", index_col=0)
                print(f"[{name}] reusing finished run", flush=True)
            else:
                res, ps = train_variant(name, cfg, vdata, args, seed, out, dev)
            results.setdefault(name, []).append((res, ps))
        if vdata is not data:
            del vdata

    # ------------------------------------------------------------- summary
    rows = []
    full_top10 = full_ps.top10
    for name, runs in results.items():
        top10s = [r["top10"] for r, _ in runs]
        top1s = [r["top1"] for r, _ in runs]
        ps = pd.concat([p.top10 for _, p in runs], axis=1).mean(1)
        p = np.nan
        if name != "full":
            common = full_top10.index.intersection(ps.index)
            try:
                p = float(wilcoxon(full_top10.loc[common], ps.loc[common]).pvalue)
            except ValueError:
                pass
        rows.append(dict(
            variant=name, label=LABELS[name], n_runs=len(runs),
            top1=100 * np.mean(top1s), top10=100 * np.mean(top10s),
            top10_sem=100 * (np.std(top10s, ddof=1) / np.sqrt(len(top10s)) if len(top10s) > 1 else np.nan),
            delta_top10=100 * (full_res["top10"] - np.mean(top10s)) if name != "full" else 0.0,
            wilcoxon_p=p, paper_top10=PAPER_TOP10[name],
            paper_delta=PAPER_TOP10["full"] - PAPER_TOP10[name],
            epochs_run=np.mean([r.get("epochs_run", np.nan) for r, _ in runs]),
        ))
    table = pd.DataFrame(rows)
    table.to_csv(args.run / "ablations.csv", index=False)
    (args.run / "ablations.json").write_text(json.dumps(table.to_dict(orient="records"), indent=1, default=float))
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

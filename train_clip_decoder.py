"""Paper method: brain module + CLIP loss against wav2vec 2.0 speech features.

Défossez et al. (2023). A convolutional "brain module" maps a 3 s EEG window
(61 channels x 360 samples) to a representation Z with the shape of the speech
features Y (1024 x 360). It is trained with a contrastive (CLIP) loss so that
Z is most similar to the Y of the audio actually heard. At test time every
EEG window is compared with all unique test speech segments, and the
segments are ranked: the top one is the decoded ("predicted") speech.

Brain module: spatial attention over sensor positions (270 ch, Fourier
parameterised, spatial dropout) -> 1x1 conv -> per-subject 270x270 layer ->
5 residual dilated conv blocks (320 ch, BN, GELU, GLU) -> 1x1 convs -> F.

Also runs the paper's baseline variants with flags:
  --target mel --loss mse   "Base model" (Mel regression)
  --target mel --loss clip  "+ Contrastive"
  --target wav2vec --loss clip (default) "+ wav2vec 2.0"  (the full method)

Usage:  .venv/bin/python train_clip_decoder.py [--target ...] [--loss ...]
Outputs in runs/<target>_<loss>/: best_model.pt, history.csv,
test_predictions.csv, results.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parent


# ------------------------------------------------------------------- data
class AliceData:
    def __init__(self, folder: Path, target: str):
        self.meta = json.loads((folder / "meta.json").read_text())
        self.win = self.meta["window_samples"]
        self.base = int(0.5 * self.meta["fs"])  # baseline: first 0.5 s
        self.rows = pd.read_csv(folder / "windows.csv")
        self.eeg = {s: np.load(folder / "eeg" / f"{s}.npy") for s in self.meta["subjects"]}
        self.speech = np.load(folder / f"speech_{target}.npy", mmap_mode="r")
        self.speech = np.ascontiguousarray(self.speech)
        self.n_subjects = int(self.rows.subject_idx.max()) + 1

    def split(self, name: str) -> pd.DataFrame:
        return self.rows[self.rows.split == name].reset_index(drop=True)

    def eeg_batch(self, rows: pd.DataFrame) -> torch.Tensor:
        x = np.stack([self.eeg[s][:, e:e + self.win] for s, e in zip(rows.subject, rows.eeg_start)])
        x = x.astype(np.float32)
        x -= x[:, :, :self.base].mean(-1, keepdims=True)  # baseline correction
        return torch.from_numpy(x)

    def speech_batch(self, starts) -> torch.Tensor:
        y = np.stack([self.speech[a:a + self.win] for a in starts]).astype(np.float32)
        return torch.from_numpy(y).transpose(1, 2)  # [B, F, T]


# ------------------------------------------------------------------ model
class SpatialAttention(nn.Module):
    def __init__(self, positions: np.ndarray, d_out=270, k=32, drop_radius=0.2):
        super().__init__()
        pos = torch.tensor(positions, dtype=torch.float32) * 0.8 + 0.1  # keep a 0.1 margin
        self.register_buffer("pos", pos)
        kk = torch.arange(k, dtype=torch.float32)
        phase = 2 * math.pi * (kk[:, None] * pos[:, 0, None, None] + kk[None, :] * pos[:, 1, None, None])
        self.register_buffer("cos", phase.cos())  # [C, K, K]
        self.register_buffer("sin", phase.sin())
        self.z_re = nn.Parameter(torch.randn(d_out, k, k) / k)
        self.z_im = nn.Parameter(torch.randn(d_out, k, k) / k)
        self.drop_radius = drop_radius

    def forward(self, x):  # x [B, C, T]
        a = torch.einsum("dkl,ckl->dc", self.z_re, self.cos) + torch.einsum("dkl,ckl->dc", self.z_im, self.sin)
        if self.training and self.drop_radius > 0:  # spatial dropout
            centre = torch.rand(2, device=x.device) * 0.8 + 0.1
            dropped = (self.pos - centre).norm(dim=1) < self.drop_radius
            a = a.masked_fill(dropped[None], float("-inf"))
        return torch.einsum("dc,bct->bdt", a.softmax(-1), x)


class SubjectLayer(nn.Module):
    def __init__(self, n_subjects: int, d: int):
        super().__init__()
        self.w = nn.Parameter(torch.eye(d).repeat(n_subjects, 1, 1))

    def forward(self, x, subj):
        return torch.einsum("bct,bcd->bdt", x, self.w[subj])


class ConvBlock(nn.Module):
    def __init__(self, d_in: int, d: int, k: int):
        super().__init__()
        dil = (2 ** ((2 * k) % 5), 2 ** ((2 * k + 1) % 5))
        self.c1 = nn.Conv1d(d_in, d, 3, padding=dil[0], dilation=dil[0])
        self.c2 = nn.Conv1d(d, d, 3, padding=dil[1], dilation=dil[1])
        self.c3 = nn.Conv1d(d, 2 * d, 3, padding=1)
        self.n1, self.n2 = nn.BatchNorm1d(d), nn.BatchNorm1d(d)
        self.res1 = d_in == d

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        x = x + h if self.res1 else h
        x = x + F.gelu(self.n2(self.c2(x)))
        return F.glu(self.c3(x), dim=1)


class BrainModule(nn.Module):
    def __init__(self, positions, n_subjects, n_out, d1=270, d2=320, n_blocks=5):
        super().__init__()
        self.sa = SpatialAttention(positions, d1)
        self.merge = nn.Conv1d(d1, d1, 1)
        self.subject = SubjectLayer(n_subjects, d1)
        self.blocks = nn.Sequential(*[ConvBlock(d1 if k == 0 else d2, d2, k) for k in range(n_blocks)])
        self.head = nn.Sequential(nn.Conv1d(d2, 2 * d2, 1), nn.GELU(), nn.Conv1d(2 * d2, n_out, 1))

    def forward(self, x, subj):
        x = self.subject(self.merge(self.sa(x)), subj)
        return self.head(self.blocks(x))


# ------------------------------------------------------------------- loss
def scores(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """<Z_i, Y_j / |Y_j|> over channels and time (candidates normalised, as in the paper's code)."""
    y = y / (y.flatten(1).norm(dim=1)[:, None, None] + 1e-8)
    return z.flatten(1) @ y.flatten(1).T


def clip_loss(z, y, seg):
    logits = scores(z, y)
    same = seg[:, None] == seg[None, :]  # same audio heard by another subject: not a negative
    logits = logits.masked_fill(same & ~torch.eye(len(seg), dtype=torch.bool, device=z.device), float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(seg), device=z.device))


def loss_fn(kind, z, y, seg):
    return clip_loss(z, y, seg) if kind == "clip" else F.mse_loss(z, y)


# ------------------------------------------------------------------ train
def batches(df: pd.DataFrame, bs: int, rng=None):
    idx = rng.permutation(len(df)) if rng is not None else np.arange(len(df))
    for i in range(0, len(idx) - (bs // 2 if rng is not None else 0), bs):
        yield df.iloc[idx[i:i + bs]]


def forward(model, x, s):
    with torch.autocast(x.device.type, dtype=torch.float16, enabled=x.device.type != "cpu"):
        return model(x, s).float()


def to_dev(data, rows, dev):
    return (data.eeg_batch(rows).to(dev), data.speech_batch(rows.audio_start).to(dev),
            torch.tensor(rows.subject_idx.values, device=dev),
            torch.tensor(rows.segment_id.values, device=dev))


@torch.no_grad()
def evaluate_loss(model, data, df, kind, bs, dev):
    model.eval()
    tot, n = 0.0, 0
    for rows in batches(df, bs, np.random.default_rng(0)):
        x, y, s, seg = to_dev(data, rows, dev)
        tot += loss_fn(kind, forward(model, x, s), y, seg).item() * len(rows)
        n += len(rows)
    return tot / n


@torch.no_grad()
def evaluate_retrieval(model, data, df, bs, dev):
    """Rank every unique test segment for each test EEG window (zero-shot)."""
    model.eval()
    cand = df.drop_duplicates("segment_id").reset_index(drop=True)
    y_c = data.speech_batch(cand.audio_start).to(dev)
    seg_pos = {s: i for i, s in enumerate(cand.segment_id)}
    ranks, top = [], []
    for rows in batches(df, bs):
        x, _, s, _ = to_dev(data, rows, dev)
        sc = scores(forward(model, x, s), y_c).cpu()
        true = torch.tensor([seg_pos[v] for v in rows.segment_id])
        ranks.append((sc > sc[torch.arange(len(true)), true][:, None]).sum(1))
        top.append(sc.topk(5, dim=1).indices)
    out = df.copy()
    out["rank"] = torch.cat(ranks).numpy()
    t5 = torch.cat(top).numpy()
    out["predicted_text"] = cand.text.values[t5[:, 0]]
    out["top5_segment_ids"] = [" ".join(map(str, cand.segment_id.values[r])) for r in t5]
    per_subj = out.groupby("subject").agg(top1=("rank", lambda r: (r < 1).mean()),
                                          top10=("rank", lambda r: (r < 10).mean()))
    res = dict(n_candidates=len(cand), n_test_windows=len(out),
               top1=float(per_subj.top1.mean()), top10=float(per_subj.top10.mean()),
               top10_best_subject=float(per_subj.top10.max()),
               chance_top1=1 / len(cand), chance_top10=min(1.0, 10 / len(cand)))
    return res, out, per_subj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "D_T_processed")
    ap.add_argument("--target", choices=["wav2vec", "mel"], default="wav2vec")
    ap.add_argument("--loss", choices=["clip", "mse"], default="clip")
    # paper: batch 256, epoch = 1200 updates, patience 10. Defaults are scaled
    # down to fit an 8 GB Apple M1 in a few hours.
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--steps-per-epoch", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--max-val", type=int, default=4000, help="fixed random subset of val windows")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = args.out or ROOT / "runs" / f"{args.target}_{args.loss}"
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    data = AliceData(args.data, args.target)
    train, val, test = data.split("train"), data.split("val"), data.split("test")
    if len(val) > args.max_val:
        val = val.sample(args.max_val, random_state=0).reset_index(drop=True)
    print(f"device={dev} train={len(train)} val={len(val)} test={len(test)} "
          f"subjects={data.n_subjects} F={data.speech.shape[1]}", flush=True)

    model = BrainModule(np.array(data.meta["positions"]), data.n_subjects, data.speech.shape[1]).to(dev)
    print(f"brain module: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def train_batches():  # endless shuffled stream; an epoch = fixed number of updates
        while True:
            yield from batches(train, args.batch, rng)

    stream = train_batches()
    best, wait, hist = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for _ in range(args.steps_per_epoch):
            rows = next(stream)
            x, y, s, seg = to_dev(data, rows, dev)
            loss = loss_fn(args.loss, forward(model, x, s), y, seg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, n = tot + loss.item() * len(rows), n + len(rows)
        v = evaluate_loss(model, data, val, args.loss, args.batch, dev)
        hist.append(dict(epoch=epoch, train_loss=tot / n, val_loss=v, seconds=time.time() - t0))
        print(f"epoch {epoch:3d}  train {tot / n:.4f}  val {v:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        if v < best:
            best, wait = v, 0
            torch.save({"model": model.state_dict(), "args": vars(args) | {"data": str(args.data), "out": str(out)}},
                       out / "best_model.pt")
        else:
            wait += 1
            if wait >= args.patience:
                print("early stopping", flush=True)
                break

    model.load_state_dict(torch.load(out / "best_model.pt", map_location=dev)["model"])
    res, preds, per_subj = evaluate_retrieval(model, data, test, args.batch, dev)
    res |= dict(target=args.target, loss=args.loss, best_val_loss=best, epochs_run=len(hist))
    preds.rename(columns={"text": "true_text"}).to_csv(out / "test_predictions.csv", index=False)
    per_subj.to_csv(out / "test_per_subject.csv")
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

"""World-model objective: masked / future prediction of speech latents from EEG.

Instead of only ranking candidate segments (CLIP), the model *predicts* the
wav2vec 2.0 latents of hidden parts of the 3 s audio window, given the EEG
and (optionally) the visible part of the audio.

Tokens (3 s window, 120 Hz, patch = 12 samples = 100 ms -> N = 30 tokens)
  EEG tokens   : BrainModule(EEG) -> [1024, 360] -> mean-pool per patch -> Linear
  audio tokens : wav2vec latents   -> mean-pool per patch -> Linear,
                 hidden positions replaced by a learned [MASK] token
  predictor    : Transformer over [EEG tokens ; audio tokens], shared
                 time-position embedding (EEG is already shifted +150 ms)
  output       : predicted pooled wav2vec latent (1024) at every audio token

Masking modes (sampled per batch during training)
  all     every audio token hidden          -> pure EEG -> audio prediction
  future  a visible prefix, the rest hidden  -> "what comes next" (world model)
  span    random 300 ms blocks hidden        -> JEPA-style infilling

Loss
  masked-token InfoNCE (cosine / tau) between predicted and true latents, over
  all hidden tokens in the batch. Tokens whose audio overlaps the anchor's
  audio (neighbouring-word windows share ~2.5 s) are NOT used as negatives.
  + lambda * CLIP loss on the BrainModule output (keeps the retrieval head).

EEG dropout: in future/span batches, a sample's EEG is hidden with prob p, so
the same model also learns the audio-only prior. --eeg-horizon past also
hides EEG tokens after the visible prefix in future mode (strict "predict the
future from the past" world model).

Evaluation on the test split (same 239-candidate protocol as the CLIP run)
  for each mode in {all, future, span} and condition in
    eeg           real EEG
    no_eeg        EEG hidden (audio-only prior)
    shuffled_eeg  EEG of a test window whose audio does not overlap
  score(candidate) = mean cosine between predicted and candidate latents on
  the hidden positions only; top-1 / top-10 averaged over subjects, plus the
  mean token cosine to the true latents.
  The EEG contribution is (eeg - no_eeg); shuffled_eeg should match no_eeg.
  The CLIP head of the BrainModule is also evaluated as a reference.

Usage:  python train_world_model.py [--init-brain runs/wav2vec_clip/best_model.pt]
Outputs in runs/world_model_<eeg-horizon>/ (never overwrites a finished run):
  best_model.pt, history.csv, results.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from train_clip_decoder import AliceData, BrainModule, batches, clip_loss, evaluate_retrieval, to_dev

ROOT = Path(__file__).resolve().parent
MODES = ("all", "future", "span")
CONDITIONS = ("eeg", "no_eeg", "shuffled_eeg")
SPAN = 3  # tokens per masked block in span mode (300 ms)


# ------------------------------------------------------------------ model
class WorldModel(nn.Module):
    def __init__(self, positions, n_subjects, n_feat, n_tok, patch,
                 d=384, layers=4, heads=6, dropout=0.1):
        super().__init__()
        self.n_tok, self.patch = n_tok, patch
        self.brain = BrainModule(positions, n_subjects, n_feat)
        self.eeg_in = nn.Sequential(nn.LayerNorm(n_feat), nn.Linear(n_feat, d))
        self.audio_in = nn.Sequential(nn.LayerNorm(n_feat), nn.Linear(n_feat, d))
        self.mask_tok = nn.Parameter(torch.zeros(d))
        self.pos = nn.Parameter(torch.randn(n_tok, d) * 0.02)
        self.modality = nn.Parameter(torch.randn(2, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.predictor = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_feat))

    def pool(self, x):
        """[B, F, T] -> [B, N, F]: mean over each 100 ms patch."""
        b, f, _ = x.shape
        x = x[..., :self.n_tok * self.patch].reshape(b, f, self.n_tok, self.patch)
        return x.mean(-1).transpose(1, 2)

    def forward(self, x, subj, a_tok, mask, eeg_pad):
        """x EEG [B,C,T]; a_tok true audio tokens [B,N,F]; mask/eeg_pad [B,N] bool (True = hidden)."""
        z = self.brain(x, subj)  # [B, F, T], also fed to the auxiliary CLIP loss
        e = self.eeg_in(self.pool(z)) + self.pos + self.modality[0]
        a = self.audio_in(a_tok)
        a = torch.where(mask[..., None], self.mask_tok.to(a.dtype), a) + self.pos + self.modality[1]
        pad = torch.cat([eeg_pad, torch.zeros_like(mask)], 1)  # audio tokens are always keys
        h = self.predictor(torch.cat([e, a], 1), src_key_padding_mask=pad)
        return self.out(h[:, self.n_tok:]), z


def run(model, x, s, a_tok, mask, eeg_pad):
    with torch.autocast(x.device.type, dtype=torch.float16, enabled=x.device.type == "cuda"):
        pred, z = model(x, s, a_tok, mask, eeg_pad)
    return pred.float(), z.float()


# ---------------------------------------------------------------- masking
def sample_mask(mode: str, b: int, n: int, rng: np.random.Generator, train: bool) -> np.ndarray:
    if mode == "all":
        return np.ones((b, n), bool)
    if mode == "future":
        k = rng.integers(n // 4, 3 * n // 4 + 1, b) if train else np.full(b, n // 2)
        return np.arange(n)[None] >= k[:, None]
    nb = n // SPAN
    ratio = rng.uniform(0.4, 0.75, b) if train else np.full(b, 0.5)
    m = np.zeros((b, n), bool)
    for i in range(b):
        for j in rng.choice(nb, max(1, int(round(ratio[i] * nb))), replace=False):
            m[i, j * SPAN:(j + 1) * SPAN] = True
    return m


def eeg_padding(mode, mask, horizon, drop=None):
    pad = torch.zeros_like(mask)
    if mode == "future" and horizon == "past":
        pad = mask.clone()  # EEG token t visible only while audio token t is visible
    if drop is not None:
        pad = pad | drop[:, None]
    return pad


# ------------------------------------------------------------------- loss
def masked_nce(pred, tgt, mask, abs_pos, patch, tau):
    p = F.normalize(pred[mask], dim=-1)
    t = F.normalize(tgt[mask], dim=-1)
    pos = abs_pos[mask]
    logits = p @ t.T / tau
    eye = torch.eye(len(pos), dtype=torch.bool, device=p.device)
    overlap = (pos[:, None] - pos[None]).abs() < patch  # same audio -> not a negative
    logits = logits.masked_fill(overlap & ~eye, float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(pos), device=p.device)), (p * t).sum(-1).mean()


def step_loss(model, data, rows, mode, mask_np, drop_np, args, dev):
    x, y, s, seg = to_dev(data, rows, dev)
    tgt = model.pool(y)
    mask = torch.from_numpy(mask_np).to(dev)
    drop = torch.from_numpy(drop_np).to(dev) if drop_np is not None else None
    pred, z = run(model, x, s, tgt, mask, eeg_padding(mode, mask, args.eeg_horizon, drop))
    abs_pos = torch.tensor(rows.audio_start.values, device=dev)[:, None] \
        + torch.arange(model.n_tok, device=dev)[None] * model.patch
    l_wm, cos = masked_nce(pred, tgt, mask, abs_pos, model.patch, args.tau)
    l_clip = clip_loss(z, y, seg) if args.clip_weight > 0 else torch.zeros((), device=dev)
    return l_wm + args.clip_weight * l_clip, l_wm, l_clip, cos


@torch.no_grad()
def evaluate_loss(model, data, df, args, dev):
    model.eval()
    rng = np.random.default_rng(0)
    tot = dict(loss=0.0, wm=0.0, clip=0.0, cos=0.0)
    n = 0
    for i, rows in enumerate(batches(df, args.batch, np.random.default_rng(0))):
        mode = MODES[i % len(MODES)]
        m = sample_mask(mode, len(rows), model.n_tok, rng, train=False)
        out = step_loss(model, data, rows, mode, m, None, args, dev)
        for k, v in zip(tot, out):
            tot[k] += v.item() * len(rows)
        n += len(rows)
    return {k: v / n for k, v in tot.items()}


# -------------------------------------------------------------- retrieval
def shuffled_partners(df: pd.DataFrame, win: int, rng) -> np.ndarray:
    """For each row, a random row whose audio does not overlap it."""
    a = df.audio_start.values
    partner = np.empty(len(df), int)
    for i in range(len(df)):
        while True:
            j = rng.integers(len(df))
            if abs(a[j] - a[i]) >= win:
                partner[i] = j
                break
    return partner


@torch.no_grad()
def evaluate_wm(model, data, df, mode, cond, args, dev):
    model.eval()
    rng = np.random.default_rng(args.seed + 1)
    partner = shuffled_partners(df, data.win, np.random.default_rng(args.seed + 2))
    cand = df.drop_duplicates("segment_id").reset_index(drop=True)
    y_c = F.normalize(model.pool(data.speech_batch(cand.audio_start).to(dev)), dim=-1).flatten(1)
    seg_pos = {s: i for i, s in enumerate(cand.segment_id)}
    ranks, coss = [], []
    for i in range(0, len(df), args.batch):
        rows = df.iloc[i:i + args.batch]
        eeg_rows = df.iloc[partner[i:i + args.batch]] if cond == "shuffled_eeg" else rows
        x = data.eeg_batch(eeg_rows).to(dev)
        s = torch.tensor(eeg_rows.subject_idx.values, device=dev)
        tgt = model.pool(data.speech_batch(rows.audio_start).to(dev))
        mask = torch.from_numpy(sample_mask(mode, len(rows), model.n_tok, rng, train=False)).to(dev)
        pad = torch.ones_like(mask) if cond == "no_eeg" else eeg_padding(mode, mask, args.eeg_horizon)
        pred, _ = run(model, x, s, tgt, mask, pad)
        p = F.normalize(pred, dim=-1) * mask[..., None]
        cnt = mask.sum(1, keepdim=True).float()
        sc = (p.flatten(1) @ y_c.T / cnt).cpu()  # mean cosine on hidden positions
        true = torch.tensor([seg_pos[v] for v in rows.segment_id])
        ranks.append((sc > sc[torch.arange(len(true)), true][:, None]).sum(1))
        coss.append(((p * F.normalize(tgt, dim=-1)).sum(-1).sum(1, keepdim=True) / cnt)[:, 0].cpu())
    out = pd.DataFrame(dict(subject=df.subject.values, rank=torch.cat(ranks).numpy(),
                            cos=torch.cat(coss).numpy()))
    per = out.groupby("subject").agg(top1=("rank", lambda r: (r < 1).mean()),
                                     top10=("rank", lambda r: (r < 10).mean()),
                                     cos=("cos", "mean"))
    return dict(top1=float(per.top1.mean()), top10=float(per.top10.mean()),
                top10_best_subject=float(per.top10.max()), token_cos=float(per.cos.mean()))


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "D_T_processed")
    ap.add_argument("--init-brain", type=Path, default=ROOT / "runs" / "wav2vec_clip" / "best_model.pt",
                    help="CLIP checkpoint to start the BrainModule from ('none' = scratch)")
    ap.add_argument("--eeg-horizon", choices=["full", "past"], default="full")
    ap.add_argument("--mode-probs", type=float, nargs=3, default=[0.25, 0.375, 0.375],
                    metavar=("ALL", "FUTURE", "SPAN"))
    ap.add_argument("--eeg-drop", type=float, default=0.2, help="P(hide EEG) in future/span batches")
    ap.add_argument("--clip-weight", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--patch", type=int, default=12, help="samples per token (12 = 100 ms)")
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--brain-lr-mult", type=float, default=0.3)
    ap.add_argument("--steps-per-epoch", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max-val", type=int, default=4000)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the code path")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.steps_per_epoch, args.batch, args.max_val = 1, 2, 16, 64
    out = args.out or ROOT / "runs" / f"world_model_{args.eeg_horizon}{'_smoke' if args.smoke else ''}"
    if (out / "results.json").exists() and not args.smoke:
        raise SystemExit(f"{out}/results.json exists - pass a new --out instead of overwriting it")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    data = AliceData(args.data, "wav2vec")
    train, val, test = data.split("train"), data.split("val"), data.split("test")
    if len(val) > args.max_val:
        val = val.sample(args.max_val, random_state=0).reset_index(drop=True)
    if args.smoke:
        test = test.iloc[:128].reset_index(drop=True)
    n_tok = data.win // args.patch
    print(f"device={dev} train={len(train)} val={len(val)} test={len(test)} tokens={n_tok}", flush=True)

    model = WorldModel(np.array(data.meta["positions"]), data.n_subjects, data.speech.shape[1],
                       n_tok, args.patch, args.d_model, args.layers, args.heads).to(dev)
    if str(args.init_brain).lower() != "none":
        model.brain.load_state_dict(torch.load(args.init_brain, map_location=dev)["model"])
        print(f"BrainModule initialised from {args.init_brain}", flush=True)
    print(f"world model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters", flush=True)
    brain_p = list(model.brain.parameters())
    other_p = [p for n, p in model.named_parameters() if not n.startswith("brain.")]
    opt = torch.optim.AdamW([dict(params=brain_p, lr=args.lr * args.brain_lr_mult, weight_decay=0.0),
                             dict(params=other_p, lr=args.lr, weight_decay=0.01)])
    probs = np.array(args.mode_probs) / sum(args.mode_probs)

    def stream():
        while True:
            yield from batches(train, args.batch, rng)

    it = stream()
    best, wait, hist = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, acc, n = time.time(), dict(loss=0.0, wm=0.0, clip=0.0, cos=0.0), 0
        for _ in range(args.steps_per_epoch):
            rows = next(it)
            mode = MODES[rng.choice(len(MODES), p=probs)]
            m = sample_mask(mode, len(rows), n_tok, rng, train=True)
            drop = rng.random(len(rows)) < args.eeg_drop if mode != "all" else None
            vals = step_loss(model, data, rows, mode, m, drop, args, dev)
            opt.zero_grad()
            vals[0].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in zip(acc, vals):
                acc[k] += v.item() * len(rows)
            n += len(rows)
        v = evaluate_loss(model, data, val, args, dev)
        rec = dict(epoch=epoch, **{f"train_{k}": a / n for k, a in acc.items()},
                   **{f"val_{k}": x for k, x in v.items()}, seconds=time.time() - t0)
        hist.append(rec)
        print(f"epoch {epoch:3d}  train wm {rec['train_wm']:.3f} clip {rec['train_clip']:.3f} "
              f"cos {rec['train_cos']:.3f} | val wm {v['wm']:.3f} clip {v['clip']:.3f} "
              f"cos {v['cos']:.3f}  ({rec['seconds']:.0f}s)", flush=True)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        if v["loss"] < best:
            best, wait = v["loss"], 0
            torch.save({"model": model.state_dict(), "args": {k: str(x) for k, x in vars(args).items()}},
                       out / "best_model.pt")
        else:
            wait += 1
            if wait >= args.patience:
                print("early stopping", flush=True)
                break

    model.load_state_dict(torch.load(out / "best_model.pt", map_location=dev)["model"])
    n_cand = test.segment_id.nunique()
    res = dict(n_candidates=n_cand, n_test_windows=len(test), chance_top1=1 / n_cand,
               chance_top10=min(1.0, 10 / n_cand), eeg_horizon=args.eeg_horizon,
               best_val_loss=best, epochs_run=len(hist), stopped_by="patience" if wait >= args.patience else "epochs")
    clip_res, _, _ = evaluate_retrieval(model.brain, data, test, args.batch, dev)
    res["clip_head"] = {k: clip_res[k] for k in ("top1", "top10", "top10_best_subject")}
    for mode in MODES:
        res[mode] = {c: evaluate_wm(model, data, test, mode, c, args, dev) for c in CONDITIONS}
        res[mode]["eeg_gain_top10"] = res[mode]["eeg"]["top10"] - res[mode]["no_eeg"]["top10"]
        print(mode, json.dumps(res[mode]), flush=True)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

"""Brain-as-world model v2: audio -> EEG state, without the subject/timing shortcut.

v1 (train_brain_world_model.py) did not collapse (SIGReg kept the EEG latent at
rank 124/128), but the latent became a code for subject identity and position
in the window: it was ~64% predictable with no audio at all, and real,
missing and unrelated audio gave the same error (retrieval at chance).

v2 keeps the v1 architecture and evaluation and changes three things:

1. Residual latent. The EEG state is s - mu[subject, position], where mu is a
   running (EMA) mean of the state per subject and patch position, updated in
   training and frozen at test time. Whatever subject and onset timing explain
   is removed before prediction, and SIGReg is applied to this residual, so
   the encoder must put unit variance into what differs *between windows*.

2. Audio-contrastive loss instead of MSE. For each EEG window, its states are
   predicted from its own audio and from K-1 other audios in the batch (whose
   audio does not overlap it), all with the same subject, positions and EEG
   context. The loss is cross-entropy over the K candidates with
   logit = -scale * MSE(predicted, observed state). Anything shared by all
   candidates (subject, timing, EEG autocorrelation in dynamics mode) cancels,
   so only audio-specific information lowers the loss. This is the training
   version of the test-time ranking.

3. Checkpoint selection on the validation contrastive loss (a ranking
   criterion), not on prediction error.

Loss = (1 - lambda) * CE_audio + lambda * SIGReg(residual) + raw_weight * MSE_raw
(raw EEG head on the true audio only, small weight, keeps the latent grounded).

Evaluation: identical to v1 (evaluate_test), computed on the residual state.
v2 trains without audio dropout by default, so the "no_audio" condition is
outside training; read mismatch_gap_latent_mse and the retrieval numbers.

Usage:  python train_brain_world_model_v2.py [--smoke]
Outputs in runs/brain_world_model_v2/ (never overwrites a finished run):
  best_model.pt, history.csv, results.json
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

from train_clip_decoder import AliceData, batches, to_dev
from train_brain_world_model import (MODES, BrainWorldModel, autocast, evaluate_test, make_mask,
                                     masked_mse, sigreg)

ROOT = Path(__file__).resolve().parent


# ------------------------------------------------------------------ model
class BrainWorldModelV2(BrainWorldModel):
    def __init__(self, *a, momentum=0.99, **kw):
        super().__init__(*a, **kw)
        n_subjects = self.brain.subject.w.shape[0]
        self.register_buffer("mu", torch.zeros(n_subjects, self.n_tok, self.d_lat))
        self.register_buffer("seen", torch.zeros(n_subjects, dtype=torch.bool))
        self.momentum = momentum
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def encode(self, x, subj):
        """Residual EEG state: s - mu[subject, position] -> [B, N, d_lat]."""
        s = super().encode(x, subj)
        if self.training:
            self.update_mu(s.detach().float(), subj)
        return s - self.mu[subj].to(s.dtype)

    @torch.no_grad()
    def update_mu(self, s, subj):
        n_s = self.mu.shape[0]
        sums = torch.zeros_like(self.mu).index_add_(0, subj, s)
        cnt = torch.zeros(n_s, device=s.device).index_add_(0, subj, torch.ones(len(subj), device=s.device))
        present = cnt > 0
        mean = sums[present] / cnt[present][:, None, None]
        m = torch.where(self.seen[present], self.momentum, 0.0)[:, None, None]
        self.mu[present] = m * self.mu[present] + (1 - m) * mean
        self.seen |= present


# ------------------------------------------------------------------- loss
def sample_candidates(audio_start: np.ndarray, k: int, rng, win: int):
    """Column 0 = own audio; columns 1.. = other windows of the batch. valid = no audio overlap."""
    b = len(audio_start)
    idx = rng.integers(0, b, (b, k - 1))
    valid = np.abs(audio_start[idx] - audio_start[:, None]) >= win
    cand = np.concatenate([np.arange(b)[:, None], idx], 1)
    return cand, np.concatenate([np.ones((b, 1), bool), valid], 1)


def step(model, data, rows, mode, mask_np, cand_np, valid_np, seed, args, dev):
    x, y, subj, _ = to_dev(data, rows, dev)
    a_tok, raw = model.pool(y), model.patchify(x)
    b, k, n = len(rows), cand_np.shape[1], model.n_tok
    mask = torch.from_numpy(mask_np).to(dev)
    cand = torch.from_numpy(cand_np).to(dev)
    valid = torch.from_numpy(valid_np).to(dev)
    m_rep = mask.repeat_interleave(k, 0)
    with autocast(dev):
        s = model.encode(x, subj)
        pl, pr = model.predict(a_tok[cand.flatten()], s.repeat_interleave(k, 0), m_rep,
                               subj.repeat_interleave(k), torch.zeros(b * k, dtype=torch.bool, device=dev))
    s, pl, pr = s.float(), pl.float(), pr.float()
    err = masked_mse(pl, s.repeat_interleave(k, 0), m_rep).view(b, k)
    logits = (-err * model.logit_scale.exp().clamp(max=100.0)).masked_fill(~valid, float("-inf"))
    target = torch.zeros(b, dtype=torch.long, device=dev)
    ce = F.cross_entropy(logits, target)
    acc = (logits.argmax(1) == 0).float().mean()
    l_sig = sigreg(s.flatten(0, 1), seed)
    l_raw = masked_mse(pr.view(b, k, n, -1)[:, 0], raw, mask).mean()
    loss = (1 - args.sigreg_weight) * ce + args.sigreg_weight * l_sig + args.raw_weight * l_raw
    return loss, ce, acc, l_sig, l_raw


KEYS = ("loss", "ce", "acc", "sigreg", "raw")


@torch.no_grad()
def evaluate_val(model, data, df, args, dev):
    model.eval()
    rng = np.random.default_rng(0)
    tot, n = np.zeros(len(KEYS)), 0
    for i, rows in enumerate(batches(df, args.batch, np.random.default_rng(0))):
        mode = MODES[i % len(MODES)]
        m = make_mask(mode, len(rows), model.n_tok, None, train=False)
        cand, valid = sample_candidates(rows.audio_start.values, args.k, rng, data.win)
        out = step(model, data, rows, mode, m, cand, valid, i, args, dev)
        tot += np.array([v.item() for v in out]) * len(rows)
        n += len(rows)
    return dict(zip(KEYS, tot / n))


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "D_T_processed")
    ap.add_argument("--init-brain", type=Path, default=ROOT / "runs" / "wav2vec_clip" / "best_model.pt",
                    help="CLIP checkpoint to start the BrainModule from ('none' = scratch)")
    ap.add_argument("--mode-probs", type=float, nargs=2, default=[0.5, 0.5], metavar=("ENCODING", "DYNAMICS"))
    ap.add_argument("--k", type=int, default=16, help="candidate audios per EEG window (1 true + K-1 negatives)")
    ap.add_argument("--mu-momentum", type=float, default=0.99)
    ap.add_argument("--sigreg-weight", type=float, default=0.05, help="lambda in LeJEPA")
    ap.add_argument("--raw-weight", type=float, default=0.1)
    ap.add_argument("--d-lat", type=int, default=128)
    ap.add_argument("--patch", type=int, default=12, help="samples per token (12 = 100 ms)")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--brain-lr-mult", type=float, default=0.3)
    ap.add_argument("--steps-per-epoch", type=int, default=1200)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--max-val", type=int, default=4000)
    ap.add_argument("--eval-group", type=int, default=16, help="test windows scored against all candidates at once")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the code path")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.steps_per_epoch, args.batch, args.max_val, args.k = 1, 2, 16, 64, 8
    out = args.out or ROOT / "runs" / f"brain_world_model_v2{'_smoke' if args.smoke else ''}"
    if (out / "results.json").exists() and not args.smoke:
        raise SystemExit(f"{out}/results.json exists - pass a new --out instead of overwriting it")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = AliceData(args.data, "wav2vec")
    train, val, test = data.split("train"), data.split("val"), data.split("test")
    if len(val) > args.max_val:
        val = val.sample(args.max_val, random_state=0).reset_index(drop=True)
    if args.smoke:
        test = test.iloc[:64].reset_index(drop=True)
    n_tok, n_chan = data.win // args.patch, len(data.meta["channels"])
    print(f"device={dev} train={len(train)} val={len(val)} test={len(test)} tokens={n_tok} K={args.k}", flush=True)

    model = BrainWorldModelV2(np.array(data.meta["positions"]), data.n_subjects, data.speech.shape[1], n_chan,
                              n_tok, args.patch, args.d_lat, args.d_model, args.layers, args.heads,
                              args.dropout, momentum=args.mu_momentum).to(dev)
    if str(args.init_brain).lower() != "none":
        model.brain.load_state_dict(torch.load(args.init_brain, map_location=dev)["model"])
        print(f"BrainModule initialised from {args.init_brain}", flush=True)
    print(f"brain world model v2: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters", flush=True)
    other = [p for k, p in model.named_parameters() if not k.startswith("brain.")]
    opt = torch.optim.AdamW([dict(params=model.brain.parameters(), lr=args.lr * args.brain_lr_mult, weight_decay=0.0),
                             dict(params=other, lr=args.lr, weight_decay=0.01)])
    probs = np.array(args.mode_probs) / sum(args.mode_probs)

    def stream():
        while True:
            yield from batches(train, args.batch, rng)

    it, gstep = stream(), 0
    best, wait, hist = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, acc, n = time.time(), np.zeros(len(KEYS)), 0
        for _ in range(args.steps_per_epoch):
            rows = next(it)
            mode = MODES[rng.choice(len(MODES), p=probs)]
            m = make_mask(mode, len(rows), n_tok, rng, train=True)
            cand, valid = sample_candidates(rows.audio_start.values, args.k, rng, data.win)
            vals = step(model, data, rows, mode, m, cand, valid, gstep, args, dev)
            opt.zero_grad()
            vals[0].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            acc += np.array([v.item() for v in vals]) * len(rows)
            n, gstep = n + len(rows), gstep + 1
        v = evaluate_val(model, data, val, args, dev)
        tr = dict(zip(KEYS, acc / n))
        rec = dict(epoch=epoch, **{f"train_{k}": x for k, x in tr.items()},
                   **{f"val_{k}": x for k, x in v.items()}, logit_scale=float(model.logit_scale.exp()),
                   seconds=time.time() - t0)
        hist.append(rec)
        print(f"epoch {epoch:3d}  train ce {tr['ce']:.4f} acc@{args.k} {tr['acc']:.3f} sigreg {tr['sigreg']:.3f} "
              f"raw {tr['raw']:.4f} | val ce {v['ce']:.4f} acc@{args.k} {v['acc']:.3f} sigreg {v['sigreg']:.3f} "
              f"raw {v['raw']:.4f}  scale {rec['logit_scale']:.1f}  ({rec['seconds']:.0f}s)", flush=True)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        if v["ce"] < best:
            best, wait = v["ce"], 0
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
               chance_top10=min(1.0, 10 / n_cand), chance_val_acc=1 / args.k, best_val_ce=best,
               epochs_run=len(hist), stopped_by="patience" if wait >= args.patience else "epochs")
    for mode in MODES:
        res[mode] = evaluate_test(model, data, test, mode, args, dev)
        print(mode, json.dumps(res[mode]), flush=True)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

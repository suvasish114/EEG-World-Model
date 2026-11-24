"""seq-JEPA for EEG -> speech: predict the audio latent from a sequence of EEG views.

Adapted from seq-JEPA (Ghaemi, Muller & Bakhtiari, NeurIPS 2025), with the
EMA target / stop-gradient replaced by SIGReg (LeJEPA, Balestriero & LeCun
2025) and the target made cross-modal (speech instead of the next EEG view).

seq-JEPA                                  here (EEG -> speech)
---------------------------------------   ---------------------------------------
view x_i of an image (crop / saccade)     EEG chunk i of the 3 s window (L samples,
                                          default 30 = 250 ms -> N = 12 chunks)
backbone encoder f                        BrainModule (CLIP-pretrained) applied to
                                          each chunk independently -> mean over
                                          time -> Linear -> z_i (d_lat)
action a_i = relative transformation      a_i = time offset to the next view
  (learnable linear projection)             (Fourier features of dt -> Linear)
transformer aggregator g + [AGG] token    same: g((z_1,a_1),...,(z_{M-1},a_{M-1}),z_M)
                                          -> z_AGG  (sequence of M EEG views)
MLP predictor h(z_AGG, a_M)               same, with a_M = offset from the last
                                          EEG view to the target position; one
                                          prediction per target position (all N)
target = EMA encoder, stop-grad           target = audio encoder on wav2vec 2.0
  (to avoid collapse)                       features of the chunk, trained jointly
                                            (no EMA, no stop-grad); collapse is
                                            prevented by SIGReg on the audio
                                            latents and on the EEG view latents
cosine loss                               MSE (LeJEPA) or cosine (--pred-loss)

Loss = (1 - lambda) * L_pred + lambda * SIGReg + nce_weight * L_nce
  L_pred  MSE between h(z_AGG, a_t) and the audio latent of chunk t, all t
  SIGReg  0.5 * (SIGReg(audio latents) + SIGReg(EEG view latents))
  L_nce   optional (default off): in-batch InfoNCE with logit = -scale * MSE
          between a window's predicted audio latents and every other window's
          audio latents (overlapping audio is not a negative). The in-batch
          accuracy is always logged as a monitor.

Training: the number of EEG views M is sampled per batch in [views_min,
views_max] and their positions at random (sorted in time), as seq-JEPA does
with M_tr. Evaluation: M_val evenly spaced views (default 4 and all N).

Evaluation on the test split (same protocol as train_clip_decoder.py): for
each EEG window predict the audio latents of all N chunks, encode every unique
test speech segment (239) with the audio encoder, rank candidates by -MSE and
by mean cosine; top-1 / top-10 averaged over subjects. Also: prediction R^2,
MSE to the true vs. the average candidate, and effective rank of the audio,
EEG-view, aggregate and predicted latents (collapse check).

Usage:  python train_eeg2speech_seqjepa.py [--smoke] [--nce-weight 0.5] [--no-actions]
Outputs in runs/eeg2speech_seqjepa/ (never overwrites a finished run):
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

from train_clip_decoder import AliceData, BrainModule, batches, to_dev
from train_brain_world_model import CovAcc, autocast, sigreg

ROOT = Path(__file__).resolve().parent


# ------------------------------------------------------------------ model
class SeqJEPA(nn.Module):
    def __init__(self, positions, n_subjects, n_feat, n_chunks, chunk, fs, d_lat=256, d_act=128,
                 d_model=256, layers=3, heads=4, pred_hidden=1024, sub=3, n_freq=16,
                 use_actions=True, dropout=0.1):
        super().__init__()
        assert chunk % sub == 0, "--chunk must be divisible by --audio-sub"
        self.N, self.L, self.fs, self.sub = n_chunks, chunk, fs, sub
        self.d_act, self.use_actions = d_act, use_actions
        # backbone encoder f (EEG views)
        self.brain = BrainModule(positions, n_subjects, n_feat)
        self.eeg_out = nn.Sequential(nn.LayerNorm(n_feat), nn.Linear(n_feat, d_lat))
        # target encoder (speech): sub-pooled wav2vec chunk -> MLP
        self.audio_enc = nn.Sequential(nn.LayerNorm(sub * n_feat), nn.Linear(sub * n_feat, pred_hidden),
                                       nn.GELU(), nn.Linear(pred_hidden, d_lat))
        # actions: time offsets in seconds -> Fourier features -> learnable linear projection
        self.register_buffer("freqs", torch.logspace(math.log10(0.1), math.log10(4.0), n_freq))
        self.act = nn.Linear(2 * n_freq + 1, d_act)
        self.no_act = nn.Parameter(torch.zeros(d_act))  # placeholder action for the last view z_M
        # sequence aggregator g with [AGG] token
        self.tok_in = nn.Linear(d_lat + d_act, d_model)
        self.agg_tok = nn.Parameter(torch.randn(d_model) * 0.02)
        self.seq_pos = nn.Parameter(torch.randn(n_chunks + 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.aggregator = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.agg_norm = nn.LayerNorm(d_model)
        # predictor h: 2-layer MLP, ReLU (as in the paper)
        self.predictor = nn.Sequential(nn.Linear(d_model + d_act, pred_hidden), nn.ReLU(),
                                       nn.Linear(pred_hidden, d_lat))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))  # only for the optional InfoNCE term

    def chunks(self, x):
        """[B, C, T] -> [B, N, C, L]."""
        b, c, _ = x.shape
        return x[..., :self.N * self.L].reshape(b, c, self.N, self.L).transpose(1, 2)

    def encode_eeg(self, x, subj):
        """Each EEG chunk encoded on its own -> [B, N, d_lat]."""
        b = len(x)
        h = self.brain(self.chunks(x).flatten(0, 1), subj.repeat_interleave(self.N)).mean(-1)
        return self.eeg_out(h).view(b, self.N, -1)

    def encode_audio(self, y):
        """Speech features [B, F, T] -> [B, N, d_lat]."""
        b, f, _ = y.shape
        yc = self.chunks(y).reshape(b, self.N, f, self.sub, self.L // self.sub).mean(-1)
        return self.audio_enc(yc.flatten(2))

    def action(self, d_chunks):
        """Offsets in chunks [...] -> action embeddings [..., d_act]."""
        if not self.use_actions:
            return torch.zeros(*d_chunks.shape, self.d_act, device=d_chunks.device)
        dt = d_chunks.float() * self.L / self.fs
        ang = 2 * math.pi * dt[..., None] * self.freqs
        return self.act(torch.cat([dt[..., None], ang.sin(), ang.cos()], -1))

    def aggregate(self, z_views, pos):
        """z_views [B, M, d_lat] at chunk positions pos [B, M] (sorted) -> z_AGG [B, d_model]."""
        b, m, _ = z_views.shape
        a = self.action(pos[:, 1:] - pos[:, :-1])
        a = torch.cat([a, self.no_act.expand(b, 1, -1).to(a.dtype)], 1)
        tok = self.tok_in(torch.cat([z_views, a.to(z_views.dtype)], -1))
        tok = torch.cat([self.agg_tok.expand(b, 1, -1).to(tok.dtype), tok], 1) + self.seq_pos[:m + 1]
        return self.agg_norm(self.aggregator(tok)[:, 0])

    def predict(self, z_agg, pos_last, targets):
        """Audio latents at chunk positions targets [B, K] -> [B, K, d_lat]."""
        a = self.action(targets - pos_last[:, None])
        z = z_agg[:, None].expand(-1, targets.shape[1], -1)
        return self.predictor(torch.cat([z, a.to(z.dtype)], -1))

    def forward(self, x, y, subj, pos):
        b = len(x)
        z_eeg = self.encode_eeg(x, subj)
        z_aud = self.encode_audio(y)
        views = torch.gather(z_eeg, 1, pos[..., None].expand(-1, -1, z_eeg.shape[-1]))
        z_agg = self.aggregate(views, pos)
        targets = torch.arange(self.N, device=x.device).expand(b, -1)
        return self.predict(z_agg, pos[:, -1], targets), z_aud, z_eeg, z_agg


# ------------------------------------------------------------------- loss
def sample_views(b, n, m, rng):
    """m random chunk positions per window, sorted in time -> [b, m]."""
    return np.sort(rng.random((b, n)).argsort(1)[:, :m], 1)


def pair_mse(pred, tgt):
    """Mean squared distance between every predicted window and every target window -> [B, C]."""
    p, t = pred.flatten(1), tgt.flatten(1)
    return (p.pow(2).sum(1)[:, None] + t.pow(2).sum(1)[None] - 2 * p @ t.T) / p.shape[1]


def step(model, data, rows, pos_np, seed, args, dev):
    x, y, subj, _ = to_dev(data, rows, dev)
    pos = torch.from_numpy(pos_np).to(dev)
    with autocast(dev):
        pred, z_aud, z_eeg, _ = model(x, y, subj, pos)
    pred, z_aud, z_eeg = pred.float(), z_aud.float(), z_eeg.float()
    if args.pred_loss == "mse":
        l_pred = F.mse_loss(pred, z_aud)
    else:
        l_pred = (1 - F.cosine_similarity(pred, z_aud, dim=-1)).mean()
    l_sig = 0.5 * (sigreg(z_aud.flatten(0, 1), seed) + sigreg(z_eeg.flatten(0, 1), seed + 7919))
    # in-batch InfoNCE (loss only if nce_weight > 0; accuracy always logged)
    a = torch.as_tensor(rows.audio_start.values, device=dev)
    b = len(rows)
    eye = torch.eye(b, dtype=torch.bool, device=dev)
    overlap = ((a[:, None] - a[None]).abs() < data.win) & ~eye
    logits = (-pair_mse(pred, z_aud) * model.logit_scale.exp().clamp(max=100.0)).masked_fill(overlap, float("-inf"))
    target = torch.arange(b, device=dev)
    l_nce = F.cross_entropy(logits, target)
    acc = (logits.argmax(1) == target).float().mean()
    loss = (1 - args.sigreg_weight) * l_pred + args.sigreg_weight * l_sig + args.nce_weight * l_nce
    return loss, l_pred, l_sig, l_nce, acc


KEYS = ("loss", "pred", "sigreg", "nce", "acc")


@torch.no_grad()
def evaluate_loss(model, data, df, args, dev):
    model.eval()
    tot, n = np.zeros(len(KEYS)), 0
    for i, rows in enumerate(batches(df, args.batch, np.random.default_rng(0))):
        pos = np.tile(np.arange(model.N), (len(rows), 1))
        out = step(model, data, rows, pos, i, args, dev)
        tot += np.array([v.item() for v in out]) * len(rows)
        n += len(rows)
    return dict(zip(KEYS, tot / n))


@torch.no_grad()
def evaluate_retrieval(model, data, df, m_views, args, dev):
    """Rank every unique speech segment of df for each EEG window, from M evenly spaced EEG views."""
    model.eval()
    n = model.N
    pos1 = torch.as_tensor(np.unique(np.linspace(0, n - 1, m_views).round().astype(int)), device=dev)
    cand = df.drop_duplicates("segment_id").reset_index(drop=True)
    tc = []
    for i in range(0, len(cand), 64):
        with autocast(dev):
            tc.append(model.encode_audio(data.speech_batch(cand.audio_start.iloc[i:i + 64]).to(dev)).float())
    tc = torch.cat(tc)  # [C, N, d]
    tc_unit = F.normalize(tc, dim=-1)
    seg_pos = {s: i for i, s in enumerate(cand.segment_id)}
    ranks = dict(mse=[], cos=[])
    mse_true, mse_cand = [], []
    cov = dict(audio=CovAcc(), eeg=CovAcc(), agg=CovAcc(), pred=CovAcc())
    for rows in batches(df, args.batch):
        x, y, subj, _ = to_dev(data, rows, dev)
        b = len(rows)
        with autocast(dev):
            pred, z_aud, z_eeg, z_agg = model(x, y, subj, pos1.expand(b, -1))
        pred, z_aud = pred.float(), z_aud.float()
        d = pair_mse(pred, tc)
        cos = torch.einsum("bnd,cnd->bc", F.normalize(pred, dim=-1), tc_unit) / n
        true = torch.tensor([seg_pos[v] for v in rows.segment_id], device=dev)
        for key, sc in (("mse", -d), ("cos", cos)):
            ranks[key].append((sc > sc[torch.arange(b, device=dev), true][:, None]).sum(1).cpu())
        mse_true.append(((pred - z_aud) ** 2).mean((1, 2)).cpu())
        mse_cand.append(d.mean(1).cpu())
        cov["audio"].add(z_aud.flatten(0, 1))
        cov["eeg"].add(z_eeg.float().flatten(0, 1))
        cov["agg"].add(z_agg.float())
        cov["pred"].add(pred.flatten(0, 1))
    ca = cov["audio"]
    mu = ca.s / ca.n
    tvar = float((ca.ss.diagonal() / ca.n - mu ** 2).mean())
    m_true, m_cand = float(torch.cat(mse_true).mean()), float(torch.cat(mse_cand).mean())
    res = dict(m_views=int(len(pos1)), n_candidates=len(cand), pred_mse=m_true, pred_r2=1 - m_true / tvar,
               mse_avg_candidate=m_cand, mismatch_gap=m_cand - m_true)
    for key, r in ranks.items():
        out = pd.DataFrame(dict(subject=df.subject.values, rank=torch.cat(r).numpy()))
        per = out.groupby("subject").agg(top1=("rank", lambda v: (v < 1).mean()),
                                         top10=("rank", lambda v: (v < 10).mean()))
        res[f"retrieval_{key}"] = dict(top1=float(per.top1.mean()), top10=float(per.top10.mean()),
                                       top10_best_subject=float(per.top10.max()))
    res["collapse"] = {k: c.stats() for k, c in cov.items()}
    return res


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "D_T_processed")
    ap.add_argument("--target", choices=["wav2vec", "mel"], default="wav2vec")
    ap.add_argument("--init-brain", type=Path, default=ROOT / "runs" / "wav2vec_clip" / "best_model.pt",
                    help="CLIP checkpoint to start the BrainModule from ('none' = scratch; wav2vec only)")
    ap.add_argument("--chunk", type=int, default=30, help="samples per EEG view / audio target (30 = 250 ms)")
    ap.add_argument("--audio-sub", type=int, default=3, help="sub-frames kept per audio chunk")
    ap.add_argument("--views-min", type=int, default=4, help="M_tr is sampled in [views-min, views-max]")
    ap.add_argument("--views-max", type=int, default=0, help="0 = all chunks")
    ap.add_argument("--eval-views", type=int, nargs="+", default=[4, 0], help="M_val values (0 = all chunks)")
    ap.add_argument("--no-actions", action="store_true", help="ablation: no action conditioning")
    ap.add_argument("--pred-loss", choices=["mse", "cos"], default="mse")
    ap.add_argument("--sigreg-weight", type=float, default=0.05, help="lambda in LeJEPA")
    ap.add_argument("--nce-weight", type=float, default=0.0, help="optional in-batch InfoNCE term")
    ap.add_argument("--d-lat", type=int, default=256)
    ap.add_argument("--d-act", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--pred-hidden", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--brain-lr-mult", type=float, default=0.3)
    ap.add_argument("--steps-per-epoch", type=int, default=1200)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--max-val", type=int, default=4000)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the code path")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.steps_per_epoch, args.batch, args.max_val = 1, 2, 16, 64
    out = args.out or ROOT / "runs" / f"eeg2speech_seqjepa{'_smoke' if args.smoke else ''}"
    if (out / "results.json").exists() and not args.smoke:
        raise SystemExit(f"{out}/results.json exists - pass a new --out instead of overwriting it")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = AliceData(args.data, args.target)
    train, val, test = data.split("train"), data.split("val"), data.split("test")
    if len(val) > args.max_val:
        val = val.sample(args.max_val, random_state=0).reset_index(drop=True)
    if args.smoke:
        test = test.iloc[:64].reset_index(drop=True)
    n_chunks = data.win // args.chunk
    v_max = args.views_max or n_chunks
    assert 1 <= args.views_min <= v_max <= n_chunks
    eval_views = [m or n_chunks for m in args.eval_views]
    print(f"device={dev} train={len(train)} val={len(val)} test={len(test)} chunks={n_chunks} "
          f"M_tr=[{args.views_min},{v_max}] M_val={eval_views}", flush=True)

    model = SeqJEPA(np.array(data.meta["positions"]), data.n_subjects, data.speech.shape[1], n_chunks,
                    args.chunk, data.meta["fs"], args.d_lat, args.d_act, args.d_model, args.layers, args.heads,
                    args.pred_hidden, args.audio_sub, use_actions=not args.no_actions,
                    dropout=args.dropout).to(dev)
    if str(args.init_brain).lower() != "none":
        model.brain.load_state_dict(torch.load(args.init_brain, map_location=dev)["model"])
        print(f"BrainModule initialised from {args.init_brain}", flush=True)
    print(f"seq-JEPA EEG->speech: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters", flush=True)
    other = [p for k, p in model.named_parameters() if not k.startswith("brain.")]
    opt = torch.optim.AdamW([dict(params=model.brain.parameters(), lr=args.lr * args.brain_lr_mult, weight_decay=0.0),
                             dict(params=other, lr=args.lr, weight_decay=0.01)])

    def stream():
        while True:
            yield from batches(train, args.batch, rng)

    it, gstep = stream(), 0
    best, wait, hist = -float("inf"), 0, []
    sel_key = f"retrieval_{args.pred_loss}"
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, acc, n = time.time(), np.zeros(len(KEYS)), 0
        for _ in range(args.steps_per_epoch):
            rows = next(it)
            m = int(rng.integers(args.views_min, v_max + 1))
            vals = step(model, data, rows, sample_views(len(rows), n_chunks, m, rng), gstep, args, dev)
            opt.zero_grad()
            vals[0].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            acc += np.array([v.item() for v in vals]) * len(rows)
            n, gstep = n + len(rows), gstep + 1
        v = evaluate_loss(model, data, val, args, dev)
        vr = evaluate_retrieval(model, data, val, n_chunks, args, dev)
        tr = dict(zip(KEYS, acc / n))
        rec = dict(epoch=epoch, **{f"train_{k}": x for k, x in tr.items()}, **{f"val_{k}": x for k, x in v.items()},
                   val_top1=vr[sel_key]["top1"], val_top10=vr[sel_key]["top10"], val_r2=vr["pred_r2"],
                   val_erank_audio=vr["collapse"]["audio"]["effective_rank"],
                   val_erank_pred=vr["collapse"]["pred"]["effective_rank"], seconds=time.time() - t0)
        hist.append(rec)
        print(f"epoch {epoch:3d}  train pred {tr['pred']:.4f} sigreg {tr['sigreg']:.3f} nce {tr['nce']:.3f} "
              f"acc {tr['acc']:.3f} | val pred {v['pred']:.4f} sigreg {v['sigreg']:.3f} R2 {vr['pred_r2']:.3f} "
              f"top1 {rec['val_top1']:.3f} top10 {rec['val_top10']:.3f} (of {vr['n_candidates']}) "
              f"erank audio {rec['val_erank_audio']:.0f} pred {rec['val_erank_pred']:.0f}  "
              f"({rec['seconds']:.0f}s)", flush=True)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        if rec["val_top10"] > best:
            best, wait = rec["val_top10"], 0
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
               chance_top10=min(1.0, 10 / n_cand), best_val_top10=best, epochs_run=len(hist),
               stopped_by="patience" if wait >= args.patience else "epochs",
               args={k: str(x) for k, x in vars(args).items()})
    for m in eval_views:
        res[f"M_val={m}"] = evaluate_retrieval(model, data, test, m, args, dev)
        print(f"M_val={m}", json.dumps(res[f"M_val={m}"]), flush=True)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

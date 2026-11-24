"""Brain-as-world model: predict EEG states from the audio (the "action").

World = the brain. State = a learned latent of the EEG; action = the speech
heard. A JEPA-style model predicts the EEG latent state from the audio (and,
in dynamics mode, from the past EEG states), regularised with SIGReg
(LeJEPA, Balestriero & LeCun 2025) so the learned EEG latent cannot collapse.

Tokens (3 s window, 120 Hz, patch = 12 samples = 100 ms -> N = 30 tokens)
  state encoder : BrainModule(EEG) -> mean-pool per patch -> Linear -> s_t (d_lat)
                  BrainModule starts from the CLIP checkpoint and is fine-tuned
  action tokens : wav2vec 2.0 features -> mean-pool per patch -> Linear
  predictor     : Transformer over [audio tokens ; EEG-state tokens]; states to
                  predict are replaced by a [MASK] token (+ subject embedding)
  heads         : latent head  -> predicted s_t
                  raw head     -> predicted raw EEG patch (61 ch x 12 samples),
                                  which keeps the latent grounded in the signal

Modes (sampled per batch)
  encoding   no EEG visible: EEG states predicted from the audio alone
             (classic encoding model, cannot take an EEG -> EEG shortcut)
  dynamics   the first k EEG states visible, the rest predicted
             (world model: next brain state given past states + action)

Loss
  (1 - lambda) * MSE(pred latent, s) on predicted tokens
  + lambda * SIGReg(s)            (s pushed towards an isotropic Gaussian)
  + raw_weight * MSE(pred raw EEG, raw EEG) on predicted tokens
  Audio is dropped for a whole sample with prob --audio-drop so the same model
  also learns the no-audio prior.

Evaluation on the test split, per mode
  conditions  audio / no_audio / shuffled_audio (audio of a non-overlapping
              window): latent MSE and raw-EEG Pearson r (mean over channels)
              audio_gain = no_audio MSE - audio MSE  (what the action adds)
  retrieval   for each EEG window, predict its states from each of the 239
              candidate audios and rank candidates by -MSE to the observed
              state (latent) and to the observed raw EEG (raw); top-1 / top-10
              averaged over subjects (same protocol as the CLIP run)
  collapse    effective rank and mean per-dimension std of s and of the
              predicted latents (d_lat = full rank, ~0 std = collapse)

Usage:  python train_brain_world_model.py [--smoke]
Outputs in runs/brain_world_model/ (never overwrites a finished run):
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

from train_clip_decoder import AliceData, BrainModule, batches, to_dev
from train_world_model import shuffled_partners

ROOT = Path(__file__).resolve().parent
MODES = ("encoding", "dynamics")
CONDITIONS = ("audio", "no_audio", "shuffled_audio")


# ------------------------------------------------------------------ model
class BrainWorldModel(nn.Module):
    def __init__(self, positions, n_subjects, n_feat, n_chan, n_tok, patch,
                 d_lat=128, d=256, layers=4, heads=4, dropout=0.1):
        super().__init__()
        self.n_tok, self.patch, self.n_chan, self.d_lat = n_tok, patch, n_chan, d_lat
        self.brain = BrainModule(positions, n_subjects, n_feat)
        self.state_proj = nn.Linear(n_feat, d_lat)
        self.audio_in = nn.Sequential(nn.LayerNorm(n_feat), nn.Linear(n_feat, d))
        self.state_in = nn.Linear(d_lat, d)
        self.mask_tok = nn.Parameter(torch.zeros(d))
        self.subj_emb = nn.Embedding(n_subjects, d)
        self.pos = nn.Parameter(torch.randn(n_tok, d) * 0.02)
        self.modality = nn.Parameter(torch.randn(2, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.predictor = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.lat_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d_lat))
        self.raw_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_chan * patch))

    def pool(self, x):
        """[B, F, T] -> [B, N, F]: mean over each 100 ms patch."""
        b, f, _ = x.shape
        return x[..., :self.n_tok * self.patch].reshape(b, f, self.n_tok, self.patch).mean(-1).transpose(1, 2)

    def patchify(self, x):
        """Raw EEG [B, C, T] -> [B, N, C*P]."""
        b, c, _ = x.shape
        x = x[..., :self.n_tok * self.patch].reshape(b, c, self.n_tok, self.patch)
        return x.permute(0, 2, 1, 3).reshape(b, self.n_tok, c * self.patch)

    def encode(self, x, subj):
        """EEG -> latent brain states [B, N, d_lat]."""
        return self.state_proj(self.pool(self.brain(x, subj)))

    def predict(self, a_tok, s_ctx, mask, subj, audio_pad):
        """a_tok [B,N,F] audio; s_ctx [B,N,d_lat] states; mask [B,N] True = predict; audio_pad [B]."""
        n = self.n_tok
        a = self.audio_in(a_tok) + self.pos + self.modality[0]
        e = self.state_in(s_ctx)
        e = torch.where(mask[..., None], self.mask_tok.to(e.dtype), e)
        e = e + self.pos + self.modality[1] + self.subj_emb(subj)[:, None]
        pad = torch.cat([audio_pad[:, None].expand(-1, n), torch.zeros_like(mask)], 1)
        h = self.predictor(torch.cat([a, e], 1), src_key_padding_mask=pad)[:, n:]
        return self.lat_out(h), self.raw_out(h)


def autocast(dev):
    return torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda")


# ------------------------------------------------------------- regulariser
def sigreg(z, seed, n_proj=256, n_t=17):
    """SIGReg (LeJEPA): Epps-Pulley test of random 1-D projections of z [N, D] against N(0, 1)."""
    g = torch.Generator(device=z.device)
    g.manual_seed(int(seed))
    a = torch.randn(z.shape[1], n_proj, generator=g, device=z.device)
    a = a / a.norm(dim=0, keepdim=True)
    t = torch.linspace(-5, 5, n_t, device=z.device)
    w = torch.exp(-0.5 * t ** 2)  # characteristic function of N(0, 1)
    xt = (z @ a).unsqueeze(-1) * t  # [N, P, T]
    err = (torch.cos(xt).mean(0) - w) ** 2 + torch.sin(xt).mean(0) ** 2
    return (torch.trapezoid(err * w, t, dim=-1) * z.shape[0]).mean()


# ---------------------------------------------------------------- helpers
def make_mask(mode, b, n, rng, train):
    if mode == "encoding":
        return np.ones((b, n), bool)
    k = rng.integers(n // 4, 3 * n // 4 + 1, b) if train else np.full(b, n // 2)
    return np.arange(n)[None] >= k[:, None]


def masked_mse(pred, tgt, mask):
    """Per-sample mean squared error over the predicted tokens -> [B]."""
    e = ((pred - tgt) ** 2).mean(-1) * mask
    return e.sum(1) / mask.sum(1).clamp(min=1)


def step(model, data, rows, mode, mask_np, drop_np, seed, args, dev):
    x, y, subj, _ = to_dev(data, rows, dev)
    a_tok, raw = model.pool(y), model.patchify(x)
    mask = torch.from_numpy(mask_np).to(dev)
    apad = torch.from_numpy(drop_np).to(dev) if drop_np is not None else torch.zeros(len(rows), dtype=torch.bool, device=dev)
    with autocast(dev):
        s = model.encode(x, subj)
        pl, pr = model.predict(a_tok, s, mask, subj, apad)
    s, pl, pr = s.float(), pl.float(), pr.float()
    l_pred = masked_mse(pl, s, mask).mean()
    l_sig = sigreg(s.flatten(0, 1), seed)
    l_raw = masked_mse(pr, raw, mask).mean()
    loss = (1 - args.sigreg_weight) * l_pred + args.sigreg_weight * l_sig + args.raw_weight * l_raw
    return loss, l_pred, l_sig, l_raw


@torch.no_grad()
def evaluate_loss(model, data, df, args, dev):
    model.eval()
    keys, tot, n = ("loss", "pred", "sigreg", "raw"), np.zeros(4), 0
    for i, rows in enumerate(batches(df, args.batch, np.random.default_rng(0))):
        mode = MODES[i % len(MODES)]
        m = make_mask(mode, len(rows), model.n_tok, None, train=False)
        out = step(model, data, rows, mode, m, None, i, args, dev)
        tot += np.array([v.item() for v in out]) * len(rows)
        n += len(rows)
    return dict(zip(keys, tot / n))


class CorrAcc:
    """Streaming per-channel Pearson r between predicted and observed raw EEG."""

    def __init__(self):
        self.s, self.n = None, 0

    def add(self, pr, raw, mask, c, p):
        a = pr[mask].view(-1, c, p).transpose(0, 1).reshape(c, -1).double()
        b = raw[mask].view(-1, c, p).transpose(0, 1).reshape(c, -1).double()
        v = torch.stack([a.sum(1), b.sum(1), (a * a).sum(1), (b * b).sum(1), (a * b).sum(1)])
        self.s = v if self.s is None else self.s + v
        self.n += a.shape[1]

    def r(self):
        sa, sb, saa, sbb, sab = self.s / self.n
        cov, va, vb = sab - sa * sb, saa - sa ** 2, sbb - sb ** 2
        return float((cov / (va * vb).clamp(min=1e-12).sqrt()).mean())


class CovAcc:
    """Streaming covariance -> effective rank and mean per-dimension std."""

    def __init__(self):
        self.s = self.ss = None
        self.n = 0

    def add(self, z):
        z = z.double()
        s, ss = z.sum(0), z.T @ z
        self.s, self.ss = (s, ss) if self.s is None else (self.s + s, self.ss + ss)
        self.n += len(z)

    def stats(self):
        mu = self.s / self.n
        cov = self.ss / self.n - mu[:, None] * mu[None]
        ev = torch.linalg.eigvalsh(cov).clamp(min=0)
        p = ev / ev.sum().clamp(min=1e-12)
        erank = torch.exp(-(p * (p + 1e-12).log()).sum())
        return dict(effective_rank=float(erank), dim=int(len(ev)),
                    mean_std=float(cov.diagonal().clamp(min=0).sqrt().mean()))


@torch.no_grad()
def evaluate_test(model, data, df, mode, args, dev):
    model.eval()
    n, c, p = model.n_tok, model.n_chan, model.patch
    cand = df.drop_duplicates("segment_id").reset_index(drop=True)
    a_c = model.pool(data.speech_batch(cand.audio_start).to(dev))  # [C, N, F]
    n_c = len(cand)
    seg_pos = {s: i for i, s in enumerate(cand.segment_id)}
    partner = shuffled_partners(df, data.win, np.random.default_rng(args.seed + 2))
    mse = {k: [] for k in CONDITIONS}
    corr = {k: CorrAcc() for k in CONDITIONS}
    cov_s, cov_p = CovAcc(), CovAcc()
    ranks = dict(latent=[], raw=[])
    g = args.eval_group
    for i in range(0, len(df), g):
        rows = df.iloc[i:i + g]
        b = len(rows)
        x, y, subj, _ = to_dev(data, rows, dev)
        a_tok, raw = model.pool(y), model.patchify(x)
        a_sh = model.pool(data.speech_batch(df.iloc[partner[i:i + g]].audio_start).to(dev))
        mask = torch.from_numpy(make_mask(mode, b, n, None, train=False)).to(dev)
        with autocast(dev):
            s = model.encode(x, subj).float()
        cov_s.add(s.flatten(0, 1))
        for cond, a in (("audio", a_tok), ("no_audio", a_tok), ("shuffled_audio", a_sh)):
            apad = torch.full((b,), cond == "no_audio", device=dev)
            with autocast(dev):
                pl, pr = model.predict(a, s, mask, subj, apad)
            pl, pr = pl.float(), pr.float()
            mse[cond].append(masked_mse(pl, s, mask).cpu())
            corr[cond].add(pr, raw, mask, c, p)
            if cond == "audio":
                cov_p.add(pl[mask])
        # retrieval: predict this window's states from every candidate audio
        m_rep = mask.repeat_interleave(n_c, 0)
        s_rep = s.repeat_interleave(n_c, 0)
        with autocast(dev):
            pl, pr = model.predict(a_c.repeat(b, 1, 1), s_rep, m_rep, subj.repeat_interleave(n_c),
                                   torch.zeros(b * n_c, dtype=torch.bool, device=dev))
        true = torch.tensor([seg_pos[v] for v in rows.segment_id], device=dev)
        for key, sc in (("latent", -masked_mse(pl.float(), s_rep, m_rep).view(b, n_c)),
                        ("raw", -masked_mse(pr.float(), raw.repeat_interleave(n_c, 0), m_rep).view(b, n_c))):
            ranks[key].append((sc > sc[torch.arange(b, device=dev), true][:, None]).sum(1).cpu())

    res = {}
    for cond in CONDITIONS:
        res[cond] = dict(latent_mse=float(torch.cat(mse[cond]).mean()), raw_r=corr[cond].r())
    res["audio_gain_latent_mse"] = res["no_audio"]["latent_mse"] - res["audio"]["latent_mse"]
    res["mismatch_gap_latent_mse"] = res["shuffled_audio"]["latent_mse"] - res["audio"]["latent_mse"]
    res["audio_gain_raw_r"] = res["audio"]["raw_r"] - res["no_audio"]["raw_r"]
    for key, r in ranks.items():
        out = pd.DataFrame(dict(subject=df.subject.values, rank=torch.cat(r).numpy()))
        per = out.groupby("subject").agg(top1=("rank", lambda v: (v < 1).mean()),
                                         top10=("rank", lambda v: (v < 10).mean()))
        res[f"retrieval_{key}"] = dict(top1=float(per.top1.mean()), top10=float(per.top10.mean()),
                                       top10_best_subject=float(per.top10.max()))
    res["collapse_state"] = cov_s.stats()
    res["collapse_pred"] = cov_p.stats()
    return res


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "D_T_processed")
    ap.add_argument("--init-brain", type=Path, default=ROOT / "runs" / "wav2vec_clip" / "best_model.pt",
                    help="CLIP checkpoint to start the BrainModule from ('none' = scratch)")
    ap.add_argument("--mode-probs", type=float, nargs=2, default=[0.5, 0.5], metavar=("ENCODING", "DYNAMICS"))
    ap.add_argument("--audio-drop", type=float, default=0.2, help="P(hide the audio of a sample)")
    ap.add_argument("--sigreg-weight", type=float, default=0.05, help="lambda in LeJEPA")
    ap.add_argument("--raw-weight", type=float, default=0.5)
    ap.add_argument("--d-lat", type=int, default=128)
    ap.add_argument("--patch", type=int, default=12, help="samples per token (12 = 100 ms)")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--brain-lr-mult", type=float, default=0.3)
    ap.add_argument("--steps-per-epoch", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--max-val", type=int, default=4000)
    ap.add_argument("--eval-group", type=int, default=16, help="test windows scored against all candidates at once")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the code path")
    args = ap.parse_args()
    if args.smoke:
        args.epochs, args.steps_per_epoch, args.batch, args.max_val = 1, 2, 16, 64
    out = args.out or ROOT / "runs" / f"brain_world_model{'_smoke' if args.smoke else ''}"
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
    print(f"device={dev} train={len(train)} val={len(val)} test={len(test)} tokens={n_tok}", flush=True)

    model = BrainWorldModel(np.array(data.meta["positions"]), data.n_subjects, data.speech.shape[1], n_chan,
                            n_tok, args.patch, args.d_lat, args.d_model, args.layers, args.heads,
                            args.dropout).to(dev)
    if str(args.init_brain).lower() != "none":
        model.brain.load_state_dict(torch.load(args.init_brain, map_location=dev)["model"])
        print(f"BrainModule initialised from {args.init_brain}", flush=True)
    print(f"brain world model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters", flush=True)
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
        t0, acc, n = time.time(), np.zeros(4), 0
        for _ in range(args.steps_per_epoch):
            rows = next(it)
            mode = MODES[rng.choice(len(MODES), p=probs)]
            m = make_mask(mode, len(rows), n_tok, rng, train=True)
            drop = rng.random(len(rows)) < args.audio_drop
            vals = step(model, data, rows, mode, m, drop, gstep, args, dev)
            opt.zero_grad()
            vals[0].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            acc += np.array([v.item() for v in vals]) * len(rows)
            n, gstep = n + len(rows), gstep + 1
        v = evaluate_loss(model, data, val, args, dev)
        tr = dict(zip(("loss", "pred", "sigreg", "raw"), acc / n))
        rec = dict(epoch=epoch, **{f"train_{k}": x for k, x in tr.items()},
                   **{f"val_{k}": x for k, x in v.items()}, seconds=time.time() - t0)
        hist.append(rec)
        print(f"epoch {epoch:3d}  train pred {tr['pred']:.4f} sigreg {tr['sigreg']:.3f} raw {tr['raw']:.4f} | "
              f"val pred {v['pred']:.4f} sigreg {v['sigreg']:.3f} raw {v['raw']:.4f} loss {v['loss']:.4f}  "
              f"({rec['seconds']:.0f}s)", flush=True)
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
               chance_top10=min(1.0, 10 / n_cand), best_val_loss=best, epochs_run=len(hist),
               stopped_by="patience" if wait >= args.patience else "epochs")
    for mode in MODES:
        res[mode] = evaluate_test(model, data, test, mode, args, dev)
        print(mode, json.dumps(res[mode]), flush=True)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

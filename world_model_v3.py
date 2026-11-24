"""World model v3 for the Alice data (D_T_processed_v2): EEG is the action, audio is the state.

Framing (the reverse of world_model_alice.py)
  environment = the speech stream, state s_t = audio (log-Mel) patch, action a_t = EEG patch.
  The world model predicts the NEXT audio patch from the audio so far plus the EEG:
        mel_hat_{t+1} = f(s_<=t, a_<=t+1+lookahead)
  Patches are P frames at 120 Hz (default 30 = 250 ms). EEG is already 150 ms later than the
  audio (prepare_alice_dataset.py), so EEG patch t+1 is the brain response to audio patch t+1;
  --lookahead 1 also lets the model see the following EEG patch (responses last ~400 ms).

  sequence = ctx context patches (default 4 = 1 s) + the 3 s target window (12 patches)

Model (deliberately simple)
  E   EEG patch encoder (spatial attention + per-subject layer + per-patch conv, as in v1)
  A   audio patch encoder (per-frame Linear -> per-patch Linear, ActionEncoder of v1)
  f   causal transformer; token t = [A(s_t), E(a_{t+1}), .., E(a_{t+1+lookahead})]
      -> Linear head -> the P x 120 Mel frames of patch t+1
  Learned null tokens replace missing audio (dropped / not yet decoded) and missing EEG.

Losses
  L_pred  L1 + (1 - Pearson) between predicted and true Mel of every next patch
  L_anc   action-contrastive: the TRUE EEG must predict the next audio better than K shuffled
          EEG sequences (stops f from ignoring the EEG, since audio history alone predicts a lot)
  Training noise so that decoding works without the true audio:
    --state-drop / --state-drop-seq / --state-drop-all  null audio input patches / the whole
        target span / the whole sequence (the last makes the EEG-only mode in-distribution)
    --self-feed  replace the target-span audio inputs with the model's own (detached)
        one-step predictions (scheduled sampling for the open-loop rollout)
    --act-drop / --act-drop-seq  null EEG patches (makes the null-EEG diagnostic meaningful)

Decoding the 3 s target window (all scored with the mtrf_baseline metrics: r_env, mm_acc)
  rollout  : true context audio (1 s before the window), then open loop: each predicted
             patch is fed back as the next state. This is "EEG + audio -> audio".
  eeg_only : all audio inputs nulled, one pass. Same information as the mTRF baseline,
             so this is the fair comparison.
  tf       : teacher forcing (true audio up to t). NOT a decoding number (it sees the audio);
             used for the world-model diagnostics only.
  Every mode is also run with shuffled EEG (another window's EEG): *_eeg_gain =
  r_env(true EEG) - r_env(shuffled EEG) is what the EEG actually adds.
  dnll_shuf / dnll_null = next-patch L1 error with shuffled / null EEG minus true EEG
  (teacher forced, > 0 means the model uses the EEG); wm_mm_acc = P(true EEG beats shuffled).

Usage
  python world_model_v3.py train --run runs/world_model_v3 --steps 20000
  python world_model_v3.py eval  --run runs/world_model_v3
Outputs in --run: best.pt, last.pt, args.json, val_history.json, metrics.json, per_subject.csv
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
import torch.nn as nn
import torch.nn.functional as F

from mtrf_baseline import evaluate as window_metrics
from world_model_alice import HELDOUT, ROOT, ActionEncoder, AliceSeq, EEGPatchEncoder, env_bands, \
    pearson_loss, transformer

MODES = ("rollout", "eeg_only", "tf")


# ============================================================================== model
class WorldModelV3(nn.Module):
    def __init__(self, a, positions, n_slots, n_mel, N):
        super().__init__()
        d, self.P, self.n_mel, self.look = a.dim, a.patch, n_mel, a.lookahead
        self.eeg = EEGPatchEncoder(positions, n_slots, a.patch, d, p=a.dropout)
        self.aud = ActionEncoder(n_mel, a.patch, d)
        self.null_s = nn.Parameter(torch.zeros(d))   # missing audio state
        self.null_a = nn.Parameter(torch.zeros(d))   # missing EEG action
        self.inp = nn.Linear((2 + a.lookahead) * d, d)
        self.pos = nn.Parameter(torch.randn(1, N - 1, d) * 0.02)
        self.tf = transformer(d, a.layers, a.heads, a.dropout)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, a.patch * n_mel))
        self.register_buffer("mask", torch.triu(torch.ones(N - 1, N - 1, dtype=torch.bool), 1), persistent=False)

    def forward(self, s, e):
        """s: audio states [B,N,d], e: EEG actions [B,N,d] -> Mel of frames P..N*P  [B,(N-1)*P,n_mel]."""
        B, N, d = s.shape
        acts = [e[:, 1:]]
        for k in range(1, self.look + 1):  # lookahead EEG patches, null past the end of the sequence
            acts.append(torch.cat([e[:, 1 + k:], self.null_a.to(e.dtype).expand(B, min(k, N - 1), d)], 1))
        h = self.inp(torch.cat([s[:, :-1]] + acts, -1)) + self.pos
        return self.out(self.tf(h, mask=self.mask)).reshape(B, (N - 1) * self.P, self.n_mel)


def build(a, d: AliceSeq, dev):
    return WorldModelV3(a, np.array(d.meta["positions"]), d.unknown + 1, d.mel.shape[1], d.N).to(dev)


# ============================================================================= losses
def losses(model, b, a):
    x, y_all = b["x"], b["mel"]
    xa = x
    if a.chan_drop > 0:
        xa = xa * (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) >= a.chan_drop)
    if a.noise > 0:
        xa = xa + a.noise * torch.randn_like(xa)
    sid = b["sid"].clone()
    sid[torch.rand(len(sid), device=sid.device) < a.subject_drop] = model.eeg.subject.w.shape[0] - 1

    e = model.eeg(xa, sid)                       # [B,N,d] EEG actions
    s = model.aud(y_all)                         # [B,N,d] audio states
    B, N, d = s.shape
    dev, C0, P = s.device, a.ctx_patches, a.patch
    tgt_span = torch.arange(N, device=dev) >= C0
    y = y_all[:, P:]                             # frames predicted by tokens 0..N-2

    # scheduled sampling: target-span audio inputs <- the model's own one-step predictions
    if a.self_feed > 0:
        with torch.no_grad():
            m0 = model(s, e).detach()
        s_hat = model.aud(torch.cat([y_all[:, :P], m0.to(y_all.dtype)], 1))
        use = (torch.rand(B, 1, device=dev) < a.self_feed) & tgt_span
        s = torch.where(use[..., None], s_hat, s)

    # audio-state dropout
    drop = torch.rand(B, N, device=dev) < a.state_drop
    drop |= (torch.rand(B, 1, device=dev) < a.state_drop_seq) & tgt_span
    drop |= torch.rand(B, 1, device=dev) < a.state_drop_all
    s_in = torch.where(drop[..., None], model.null_s.expand_as(s), s)
    # EEG-action dropout
    keep = (torch.rand(B, N, device=dev) >= a.act_drop) & (torch.rand(B, 1, device=dev) >= a.act_drop_seq)
    e_in = torch.where(keep[..., None], e, model.null_a.expand_as(e))

    m = model(s_in, e_in)
    l_pred = F.l1_loss(m, y) + a.l_corr * pearson_loss(m, y)
    loss, out = a.l_pred * l_pred, dict(pred=l_pred)

    if a.l_anc > 0:
        err = lambda ee: (model(s_in, ee) - y).abs().mean((1, 2))
        ar = torch.arange(B, device=dev)
        errs = [err(e)] + [err(e[(ar + off) % B]) for off in (torch.randperm(B - 1)[:a.anc_neg] + 1).tolist()]
        logits = -torch.stack(errs, 1) / a.anc_tau
        l_anc = F.cross_entropy(logits, torch.zeros(B, dtype=torch.long, device=dev))
        loss = loss + a.l_anc * l_anc
        out.update(anc=l_anc, anc_acc=(logits.argmax(1) == 0).float().mean())
    return loss, out


# ============================================================================ decoding
@torch.no_grad()
def decode(model, s, e, C0, mode):
    """-> predicted Mel of the target window (frames C0*P .. N*P)  [B, T, n_mel]."""
    B, N, d = s.shape
    P = model.P
    ns = model.null_s.to(s.dtype).expand(B, N, d)
    if mode == "tf":
        return model(s, e)[:, (C0 - 1) * P:]
    if mode == "eeg_only":
        return model(ns, e)[:, (C0 - 1) * P:]
    cur, out = torch.cat([s[:, :C0], ns[:, C0:]], 1), []
    for t in range(C0, N):                     # open loop: feed each decoded patch back as a state
        m = model(cur, e)[:, (t - 1) * P:t * P]
        out.append(m)
        if t < N - 1:
            cur = cur.clone()
            cur[:, t] = model.aud(m)[:, 0].to(cur.dtype)
    return torch.cat(out, 1)


@torch.no_grad()
def run_windows(model, d: AliceSeq, w: pd.DataFrame, a, dev):
    """Decode every window in every mode with true and shuffled EEG; keeps envelope + bands only."""
    model.eval()
    C0, ctx = a.ctx_patches, d.ctx
    R = {k: [] for k in ["y"] + [m + sfx for m in MODES for sfx in ("", "_shuf")]}
    subj, starts, errs = [], [], []
    for i in range(0, len(w), a.eval_batch):
        r = w.iloc[i:i + a.eval_batch]
        b = d.assemble(r.subject.tolist(), (r.audio_start - ctx).tolist(), (r.eeg_start - ctx).tolist(), dev)
        y = b["mel"][:, ctx:]
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            e = model.eeg(b["x"], b["sid"]).float()
            s = model.aud(b["mel"]).float()
            e_shuf = e.roll(max(1, len(r) // 2), 0)
            e_null = model.null_a.float().expand_as(e)
            for m in MODES:
                for sfx, ee in (("", e), ("_shuf", e_shuf)):
                    p = decode(model, s, ee, C0, m).float()
                    R[m + sfx].append(env_bands(p.cpu().numpy()))
                    if m == "tf":
                        errs.append((sfx or "_true", (p - y).abs().mean((1, 2)).cpu().numpy()))
            p_null = decode(model, s, e_null, C0, "tf").float()
            errs.append(("_null", (p_null - y).abs().mean((1, 2)).cpu().numpy()))
        R["y"].append(env_bands(y.cpu().numpy()))
        subj += r.subject.tolist()
        starts.append(r.audio_start.to_numpy())
    model.train()
    R = {k: np.concatenate(v) for k, v in R.items()}
    for k in ("_true", "_shuf", "_null"):
        R["err" + k] = np.concatenate([v for n, v in errs if n == k])
    return R, np.array(subj), np.concatenate(starts)


def summarize(R, subj, starts, T):
    rows = []
    for s in sorted(set(subj)):
        k = subj == s
        row = dict(subject=s, n_test_windows=int(k.sum()))
        for m in MODES:
            r = window_metrics(R[m][k], R["y"][k], starts[k], T)
            rs = window_metrics(R[m + "_shuf"][k], R["y"][k], starts[k], T)
            row.update({f"{m}_{q}": v for q, v in r.items() if q != "n_test_windows"})
            row[f"{m}_r_above_null"] = r["r_env"] - r["r_env_null"]
            row[f"{m}_r_env_shufeeg"] = rs["r_env"]
            row[f"{m}_mm_acc_shufeeg"] = rs["mm_acc"]
            row[f"{m}_eeg_gain"] = r["r_env"] - rs["r_env"]
        et, es, en = R["err_true"][k], R["err_shuf"][k], R["err_null"][k]
        row.update(dnll_shuf=float((es - et).mean()), dnll_null=float((en - et).mean()),
                   wm_mm_acc=float((et < es).mean()))
        rows.append(row)
    return pd.DataFrame(rows)


def mean_metrics(m: pd.DataFrame) -> dict:
    return {k: float(m[k].mean()) for k in m.columns if k not in ("subject", "group", "n_test_windows")}


# ============================================================================== train
def train(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = AliceSeq(a)
    va = d.eval_windows("val", d.seen, a.val_per_subject)
    n_tr = sum(len(v[0]) for v in d.train_starts.values())
    print(f"seen {len(d.seen)} subjects, held out {d.heldout}; sequence {d.L} frames = {d.N} patches "
          f"({a.ctx_patches} context + {d.Nt} target) of {a.patch} frames; state = Mel ({d.mel.shape[1]}), "
          f"action = EEG (lookahead {a.lookahead}); {n_tr} valid training start frames; val windows {len(va)}",
          flush=True)
    model = build(a, d, dev)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(a), indent=1))
    best, bad, hist, acc, t0 = -1e9, 0, [], {}, time.time()
    for step in range(1, a.steps + 1):
        b = d.train_batch(a.batch, dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            loss, parts = losses(model, b, a)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        for k, v in parts.items():
            acc[k] = acc.get(k, 0.0) + float(v)
        if step % 100 == 0:
            print(f"step {step} | " + " ".join(f"{k} {v / 100:.4f}" for k, v in acc.items())
                  + f" | lr {sched.get_last_lr()[0]:.2e} | {time.time() - t0:.0f}s", flush=True)
            acc = {}
        if step % a.eval_every == 0 or step == a.steps:
            v = dict(step=step, **mean_metrics(summarize(*run_windows(model, d, va, a, dev), d.T)))
            hist.append(v)
            tag = ""
            if v[a.select] > best:
                best, bad, tag = v[a.select], 0, " (best)"
                torch.save(dict(model=model.state_dict(), step=step, args=vars(a), val=v), run / "best.pt")
            else:
                bad += 1
            print(f"== val step {step}: " + " | ".join(
                f"{m} r {v[m + '_r_env']:.4f} mm {v[m + '_mm_acc']:.3f} gain {v[m + '_eeg_gain']:.4f}"
                for m in MODES) + f" | dNLL_shuf {v['dnll_shuf']:.4f} dNLL_null {v['dnll_null']:.4f} "
                f"wm_mm_acc {v['wm_mm_acc']:.3f}{tag}", flush=True)
            (run / "val_history.json").write_text(json.dumps(hist, indent=1))
            if a.save_every and step % a.save_every == 0:  # per-step checkpoints for eval_world_model_curves.py
                (run / "ckpt").mkdir(exist_ok=True)
                torch.save(dict(model=model.state_dict(), step=step, args=vars(a)), run / "ckpt" / f"step_{step:06d}.pt")
            if a.patience and bad >= a.patience:
                print(f"early stop: no {a.select} gain in {bad} evaluations (best {best:.4f})", flush=True)
                break
    torch.save(dict(model=model.state_dict(), step=step, args=vars(a)), run / "last.pt")
    evaluate(a, d)


def evaluate(a, d: AliceSeq | None = None):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = d or AliceSeq(a)
    model = build(a, d, dev)
    ck = torch.load(Path(a.run) / "best.pt", map_location=dev)
    model.load_state_dict(ck["model"])
    base = pd.read_csv(a.baseline) if Path(a.baseline).exists() else None
    out, rows = dict(checkpoint_step=int(ck["step"])), []
    for name, subs, bmodel in (("seen", d.seen, "subject"), ("unseen", d.heldout, "pooled")):
        te = d.eval_windows("test", subs)
        if not len(te):
            continue
        m = summarize(*run_windows(model, d, te, a, dev), d.T)
        m.insert(0, "group", name)
        rows.append(m)
        res = mean_metrics(m)
        res.update(n_subjects=int(len(m)), n_test_windows=int(len(te)))
        if base is not None:
            bb = base[(base.model == bmodel) & base.subject.isin(m.subject)]
            res["baseline"] = {k: float(bb[k].mean()) for k in ("r_env", "r_env_null", "r_bands", "mm_acc")}
            res["baseline"]["r_above_null"] = float((bb.r_env - bb.r_env_null).mean())
            res["baseline_model"] = f"mtrf {bmodel}"
        out[name] = res
        print(f"{name}: " + json.dumps(res), flush=True)
    pd.concat(rows).to_csv(Path(a.run) / "per_subject.csv", index=False)
    (Path(a.run) / "metrics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


# =============================================================================== main
if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("cmd", choices=["train", "eval"])
    p.add_argument("--data", default=str(ROOT / "D_T_processed_v2"))
    p.add_argument("--dataset", default=str(ROOT / "Dataset"))
    p.add_argument("--run", default=str(ROOT / "runs" / "world_model_v3"))
    p.add_argument("--baseline", default=str(ROOT / "runs" / "mtrf_baseline_v2" / "results.csv"))
    p.add_argument("--heldout", nargs="*", default=HELDOUT)
    p.add_argument("--exclude", nargs="*", default=[])
    # sequence / model
    p.add_argument("--patch", type=int, default=30, help="frames per patch (120 Hz): 30 = 250 ms")
    p.add_argument("--ctx-patches", type=int, default=4, help="context patches before the 3 s target window")
    p.add_argument("--lookahead", type=int, default=1, help="extra future EEG patches seen per prediction")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    # optimisation
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.05)
    # losses
    p.add_argument("--l-pred", type=float, default=1.0)
    p.add_argument("--l-corr", type=float, default=1.0)
    p.add_argument("--l-anc", type=float, default=0.5, help="EEG-contrastive next-audio loss (0 = off)")
    p.add_argument("--anc-neg", type=int, default=3, help="shuffled EEG sequences per sample")
    p.add_argument("--anc-tau", type=float, default=0.05)
    # input noise / regularisation
    p.add_argument("--state-drop", type=float, default=0.15, help="prob. of nulling one audio input patch")
    p.add_argument("--state-drop-seq", type=float, default=0.3, help="prob. of nulling the whole target-span audio")
    p.add_argument("--state-drop-all", type=float, default=0.2, help="prob. of nulling all audio (EEG-only)")
    p.add_argument("--self-feed", type=float, default=0.25, help="prob. of feeding back own predictions")
    p.add_argument("--act-drop", type=float, default=0.1, help="prob. of nulling one EEG patch")
    p.add_argument("--act-drop-seq", type=float, default=0.1, help="prob. of nulling all EEG of a sample")
    p.add_argument("--subject-drop", type=float, default=0.1, help="prob. of the 'unknown subject' slot")
    p.add_argument("--chan-drop", type=float, default=0.1)
    p.add_argument("--noise", type=float, default=0.1)
    # evaluation
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batch", type=int, default=128)
    p.add_argument("--val-per-subject", type=int, default=150)
    p.add_argument("--select", default="rollout_mm_acc", help="val metric for best.pt, e.g. eeg_only_mm_acc")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--save-every", type=int, default=0,
                   help="also save run/ckpt/step_N.pt every N steps (multiple of --eval-every; 0 = off)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    a.action = "mel"  # AliceSeq: load Mel only (the state); EEG is the action here
    {"train": train, "eval": evaluate}[a.cmd](a)

"""Action-conditioned EEG world model for the Alice data (D_T_processed_v2).

Framing
  environment = the listener's brain, observation o_t = EEG patch, action a_t = audio patch.
  The audio "acts" on the brain; the EEG is what we observe of the resulting state.
  Patches are P frames at 120 Hz (default 30 = 250 ms). EEG is already 150 ms later than
  the audio (prepare_alice_dataset.py), so patch t of EEG and patch t of audio are aligned
  with the brain delay built in.

  sequence = ctx context patches (default 4 = 1 s) + the 3 s target window (12 patches)

Model
  E    EEG patch encoder: spatial attention over sensors + per-subject layer
       (train_clip_decoder.py) -> per-patch conv -> z_t  (a patch never sees other patches)
  E~   EMA copy of E: JEPA-style prediction targets (no gradient, clean EEG input)
  A    action encoder: per-frame wav2vec (or Mel) -> per-patch embedding u_t
  f    dynamics, causal transformer:  z_hat_{t+1} = f(z_<=t, u_<=t+1)
  g    inverse dynamics, bidirectional transformer over z_1..z_N:
       -> per-patch action embedding e_t (contrastive vs u_t) and the 120-band log-Mel
          of every frame (the decoded audio, scored like mtrf_baseline / sense_alice)

Losses
  L_dyn    ||f(z, u_dropped) - E~(x)_{t+1}||^2           (actions randomly nulled)
  L_anc    action-contrastive dynamics: the TRUE action sequence must predict the future
           EEG latents better than K shuffled ones (cross-entropy over -MSE / tau).
           Stops the dynamics from ignoring the audio (EEG history alone predicts a lot).
  L_clip   patch-level CLIP between g's e_t and u_t (near-duplicate audio masked)
  L_mel    L1 + (1 - Pearson) between g's Mel and the true Mel
  L_var    VICReg variance term on z (no collapse);  L_rec  raw EEG patch reconstruction

Decoding audio from 3 s of EEG
  amortized : g(E(EEG)) -> Mel [360, 120] -> r_env / mm_acc (same code as the baselines)
  planning  : retrieval among all unique test windows by g's embeddings (top-1/10), then
              re-rank the top-K with the world model: roll f with each candidate's audio
              (context actions nulled, so audio continuity cannot leak the position) and
              keep the candidate that best predicts the observed EEG latents.

World-model diagnostics (test): dynamics error with true vs shuffled vs null audio;
  dNLL_shuf = err(shuffled) - err(true) > 0 means the model uses the audio;
  wm_mm_acc = fraction of windows where true audio beats shuffled audio (chance 0.5).

Training windows start at random frames (not word onsets) inside one clip and inside
training-sentence audio, so the model cannot learn the shared word-onset window shape
(the sense_alice job 552929 problem). Test/val use the word-onset windows of windows.csv
so the numbers compare with runs/mtrf_baseline_v2 and runs/sense_alice*.

Usage
  python world_model_alice.py train --run runs/world_model --steps 20000
  python world_model_alice.py eval  --run runs/world_model
Outputs in --run: best.pt, last.pt, val_history.json, metrics.json, per_subject.csv
"""

from __future__ import annotations

import argparse
import copy
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
from train_clip_decoder import SpatialAttention, SubjectLayer

ROOT = Path(__file__).resolve().parent
HELDOUT = ["S03", "S13", "S22", "S32", "S41", "S48"]  # same unseen subjects as sense_alice.py
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}
N_BANDS = 8


# =============================================================================== data
def frame_labels(dataset: Path, meta: dict):
    """Per global audio frame: sentence split code and clip number (as in sense_alice.py)."""
    from prepare_alice_dataset import assign_splits
    words = pd.read_csv(dataset / "AliceChapterOne-EEG.csv")
    split = words.Sentence.map(assign_splits(words)).map(SPLIT_CODE).to_numpy()
    fs, offs, lens = meta["fs"], meta["clip_offsets"], meta["clip_lengths"]
    n = offs[-1] + lens[-1]
    lab, clip = np.full(n, -1, np.int8), np.zeros(n, np.int16)
    for c, (o, L) in enumerate(zip(offs, lens), start=1):
        clip[o:o + L] = c
        w = words[words.Segment == c].sort_values("onset")
        on = np.clip(np.round(w.onset.to_numpy() * fs).astype(int), 0, L)
        for k, (s0, code) in enumerate(zip(on, split[w.index.to_numpy()])):
            s1 = on[k + 1] if k + 1 < len(on) else L
            lab[o + (0 if k == 0 else s0):o + s1] = code
    return lab, clip


class AliceSeq:
    """Continuous EEG / audio arrays in RAM, random training sequences, fixed eval windows."""

    def __init__(self, a):
        data = Path(a.data)
        self.meta = meta = json.loads((data / "meta.json").read_text())
        self.T, self.P = meta["window_samples"], a.patch
        assert self.T % self.P == 0, "window must be a whole number of patches"
        self.ctx = a.ctx_patches * self.P
        self.L = self.ctx + self.T
        self.N, self.Nt = self.L // self.P, self.T // self.P
        self.eegs = {s: np.load(data / "eeg" / f"{s}.npy") for s in meta["subjects"]}  # f16 [61, n]
        self.mel = np.load(data / "speech_mel.npy")  # f16 [frames, 120]
        self.act = self.mel if a.action == "mel" else np.load(data / "speech_wav2vec.npy")
        self.windows = pd.read_csv(data / "windows.csv")

        self.heldout = [s for s in a.heldout if s in meta["subjects"]]
        self.seen = [s for s in meta["subjects"] if s not in self.heldout and s not in a.exclude]
        self.sid = {s: i for i, s in enumerate(self.seen)}
        self.unknown = len(self.seen)  # subject slot for unseen subjects (trained by subject dropout)

        # EEG index = audio index + per-(subject, clip) offset (constant up to +-1 rounding)
        w = self.windows.assign(d=self.windows.eeg_start - self.windows.audio_start)
        off = w.groupby(["subject", "audio_clip"]).d.median().round().astype(int)

        lab, _ = frame_labels(Path(a.dataset), meta)
        ok = np.concatenate([[0], np.cumsum(lab == SPLIT_CODE["train"])])
        self.train_starts = {}
        for s in self.seen:
            n, A, D = self.eegs[s].shape[1], [], []
            for (s2, c), d in off.items():
                if s2 != s:
                    continue
                o, Lc = meta["clip_offsets"][c - 1], meta["clip_lengths"][c - 1]
                st = np.arange(o, o + Lc - self.L + 1)
                good = (ok[st + self.L] - ok[st] == self.L) & (st + d >= 0) & (st + d + self.L <= n)
                A.append(st[good])
                D.append(np.full(good.sum(), d))
            self.train_starts[s] = (np.concatenate(A), np.concatenate(D))
        self.rng = np.random.default_rng(a.seed)

    # ------------------------------------------------------------------ batches
    def assemble(self, subs, a0, e0, dev):
        x = np.stack([self.eegs[s][:, e:e + self.L] for s, e in zip(subs, e0)])
        act = np.stack([self.act[a:a + self.L] for a in a0])
        mel = np.stack([self.mel[a:a + self.L] for a in a0])
        sid = [self.sid.get(s, self.unknown) for s in subs]
        t = lambda v: torch.from_numpy(v).to(dev, non_blocking=True).float()
        return dict(x=t(x), act=t(act), mel=t(mel), sid=torch.tensor(sid, device=dev),
                    aid=torch.as_tensor(np.asarray(a0), device=dev))

    def train_batch(self, B, dev):
        subs = [self.seen[i] for i in self.rng.integers(len(self.seen), size=B)]
        a0, e0 = [], []
        for s in subs:
            A, D = self.train_starts[s]
            k = self.rng.integers(len(A))
            a0.append(int(A[k]))
            e0.append(int(A[k] + D[k]))
        return self.assemble(subs, a0, e0, dev)

    def eval_windows(self, split, subjects, per_subject=None, seed=0):
        """Word-onset windows of windows.csv with ctx patches of context that stay in the clip."""
        w = self.windows[(self.windows.split == split) & self.windows.subject.isin(subjects)].copy()
        clip_start = np.array(self.meta["clip_offsets"])[w.audio_clip.to_numpy() - 1]
        n = w.subject.map(lambda s: self.eegs[s].shape[1]).to_numpy()
        keep = ((w.audio_start - self.ctx >= clip_start) & (w.eeg_start - self.ctx >= 0)
                & (w.eeg_start + self.T <= n))
        w = w[keep.to_numpy()]
        if per_subject:
            w = w.sample(frac=1.0, random_state=seed).groupby("subject").head(per_subject)
        return w.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # mixed for shuffles


# ============================================================================== model
class EEGPatchEncoder(nn.Module):
    """[B, C, L] -> [B, N, d]; each patch is encoded on its own (no leakage across time)."""

    def __init__(self, positions, n_slots, P, d, d_sa=128, width=96, p=0.1):
        super().__init__()
        self.P = P
        self.sa = SpatialAttention(positions, d_sa)
        self.merge = nn.Conv1d(d_sa, d_sa, 1)
        self.subject = SubjectLayer(n_slots, d_sa)
        self.local = nn.Sequential(nn.Conv1d(d_sa, width, 5, padding=2), nn.GELU(), nn.Dropout(p),
                                   nn.Conv1d(width, width, 5, padding=2), nn.GELU())
        self.proj = nn.Linear(width * P, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, sid):
        h = self.subject(self.merge(self.sa(x)), sid)  # [B, D, L]
        B, D, L = h.shape
        N = L // self.P
        h = h.reshape(B, D, N, self.P).permute(0, 2, 1, 3).reshape(B * N, D, self.P)
        return self.norm(self.proj(self.local(h).flatten(1)).reshape(B, N, -1))


class ActionEncoder(nn.Module):
    """[B, L, Fa] audio features -> [B, N, d] per-patch action embeddings."""

    def __init__(self, Fa, P, d, d_frame=64):
        super().__init__()
        self.P = P
        self.frame = nn.Sequential(nn.Linear(Fa, d_frame), nn.GELU())
        self.patch = nn.Sequential(nn.Linear(P * d_frame, d), nn.LayerNorm(d))

    def forward(self, act):
        B, L, _ = act.shape
        return self.patch(self.frame(act).reshape(B, L // self.P, -1))


def transformer(d, layers, heads, p):
    layer = nn.TransformerEncoderLayer(d, heads, 4 * d, p, activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


class Dynamics(nn.Module):
    """Causal: token t = [z_t, u_{t+1}] -> prediction of z_{t+1}. Output [B, N-1, d]."""

    def __init__(self, d, N, layers, heads, p):
        super().__init__()
        self.inp = nn.Linear(2 * d, d)
        self.pos = nn.Parameter(torch.randn(1, N - 1, d) * 0.02)
        self.tf = transformer(d, layers, heads, p)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.register_buffer("mask", torch.triu(torch.ones(N - 1, N - 1, dtype=torch.bool), 1), persistent=False)

    def forward(self, z, u):
        h = self.inp(torch.cat([z[:, :-1], u[:, 1:]], -1)) + self.pos
        return self.out(self.tf(h, mask=self.mask))


class InverseDynamics(nn.Module):
    """Bidirectional over z_1..z_N -> action embeddings e [B,N,d] and Mel [B, N*P, n_mel]."""

    def __init__(self, d, N, P, n_mel, layers, heads, p):
        super().__init__()
        self.P, self.n_mel = P, n_mel
        self.pos = nn.Parameter(torch.randn(1, N, d) * 0.02)
        self.tf = transformer(d, layers, heads, p)
        self.act = nn.Linear(d, d)
        self.mel = nn.Linear(d, P * n_mel)
        self.smooth = nn.Conv1d(n_mel, n_mel, 5, padding=2)  # across patch borders

    def forward(self, z):
        h = self.tf(z + self.pos)
        B, N, _ = h.shape
        m = self.mel(h).reshape(B, N * self.P, self.n_mel)
        return self.act(h), m + self.smooth(m.transpose(1, 2)).transpose(1, 2)


class WorldModel(nn.Module):
    def __init__(self, a, positions, n_slots, C, Fa, n_mel, N):
        super().__init__()
        d = a.dim
        self.enc = EEGPatchEncoder(positions, n_slots, a.patch, d, p=a.dropout)
        self.act_enc = ActionEncoder(Fa, a.patch, d)
        self.null_u = nn.Parameter(torch.zeros(d))
        self.dyn = Dynamics(d, N, a.layers, a.heads, a.dropout)
        self.idm = InverseDynamics(d, N, a.patch, n_mel, a.layers, a.heads, a.dropout)
        self.recon = nn.Linear(d, C * a.patch)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))


def build(a, d: AliceSeq, dev):
    model = WorldModel(a, np.array(d.meta["positions"]), d.unknown + 1, len(d.meta["channels"]),
                       d.act.shape[1], d.mel.shape[1], d.N).to(dev)
    tgt = copy.deepcopy(model.enc).eval()
    for q in tgt.parameters():
        q.requires_grad_(False)
    return model, tgt


@torch.no_grad()
def ema_update(tgt, enc, m):
    for pt, p in zip(tgt.parameters(), enc.parameters()):
        pt.mul_(m).add_(p.detach(), alpha=1 - m)
    for bt, b in zip(tgt.buffers(), enc.buffers()):
        bt.copy_(b)


# ============================================================================= losses
def pearson_loss(p, y):  # [B,T,K] -> 1 - mean Pearson r over time
    p = p - p.mean(1, keepdim=True)
    y = y - y.mean(1, keepdim=True)
    return 1 - ((p * y).sum(1) / (p.norm(dim=1) * y.norm(dim=1) + 1e-6)).mean()


def dyn_err(model, z, u, zt, first=0):
    """Per-sample mean squared error of the dynamics prediction, transitions into patches >= first+1."""
    return ((model.dyn(z, u) - zt[:, 1:]) ** 2)[:, first:].mean((1, 2))


def losses(model, tgt, b, a, step):
    x = b["x"]
    xa = x
    if a.chan_drop > 0:
        xa = xa * (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) >= a.chan_drop)
    if a.noise > 0:
        xa = xa + a.noise * torch.randn_like(xa)
    sid = b["sid"].clone()
    sid[torch.rand(len(sid), device=sid.device) < a.subject_drop] = model.enc.subject.w.shape[0] - 1

    z = model.enc(xa, sid)                       # [B,N,d]
    with torch.no_grad():
        zt = tgt(x, sid)                          # clean EEG targets
    u = model.act_enc(b["act"])                  # [B,N,d]
    B, N, d = z.shape
    pre = step < a.pretrain_steps

    # dynamics with random action dropout (whole sequence and single patches)
    keep = torch.rand(B, N, device=z.device) >= a.act_drop
    keep &= (torch.rand(B, 1, device=z.device) >= a.act_drop_seq)
    if pre:
        keep[:] = False
    u_in = torch.where(keep[..., None], u, model.null_u.expand_as(u))
    l_dyn = ((model.dyn(z, u_in) - zt[:, 1:]) ** 2).mean()

    out = dict(dyn=l_dyn)
    loss = a.l_dyn * l_dyn
    if not pre:
        # action-contrastive dynamics: true audio vs K shuffled audio sequences
        errs = [dyn_err(model, z, u, zt)]
        ar = torch.arange(B, device=z.device)
        for off in (torch.randperm(B - 1)[:a.anc_neg] + 1).tolist():  # distinct non-zero shifts: never own audio
            errs.append(dyn_err(model, z, u[(ar + off) % B], zt))
        logits = -torch.stack(errs, 1) / a.anc_tau
        l_anc = F.cross_entropy(logits, torch.zeros(B, dtype=torch.long, device=z.device))

        # inverse dynamics: patch-level CLIP + Mel regression
        e, m = model.idm(z)
        en, un = F.normalize(e.flatten(0, 1), dim=-1), F.normalize(u.flatten(0, 1), dim=-1)
        lg = model.logit_scale.exp().clamp(max=100) * en @ un.T
        ids = (b["aid"][:, None] + torch.arange(N, device=z.device) * a.patch).flatten()
        dup = ((ids[:, None] - ids[None, :]).abs() < a.patch // 2) & ~torch.eye(len(ids), dtype=torch.bool,
                                                                                device=z.device)
        lg = lg.masked_fill(dup, float("-inf"))
        tgt_idx = torch.arange(len(ids), device=z.device)
        l_clip = 0.5 * (F.cross_entropy(lg, tgt_idx) + F.cross_entropy(lg.T, tgt_idx))
        l_mel = F.l1_loss(m, b["mel"]) + a.l_corr * pearson_loss(m, b["mel"])
        loss = loss + a.l_anc * l_anc + a.l_clip * l_clip + a.l_mel * l_mel
        out.update(anc=l_anc, clip=l_clip, mel=l_mel,
                   anc_acc=(logits.argmax(1) == 0).float().mean())

    std = z.flatten(0, 1).float().std(0)
    l_var = F.relu(1 - std).mean()
    xr = model.recon(z).reshape(B, N, x.shape[1], a.patch).permute(0, 2, 1, 3).reshape_as(x)
    l_rec = F.mse_loss(xr, x)
    loss = loss + a.l_var * l_var + a.l_rec * l_rec
    out.update(var=l_var, rec=l_rec)
    return loss, out


# ============================================================================ evaluation
def env_bands(m):  # [n,T,120] -> [n,T,1+N_BANDS], same definitions as mtrf_baseline.py
    return np.concatenate([m.mean(-1, keepdims=True), m.reshape(*m.shape[:2], N_BANDS, -1).mean(-1)], -1)


@torch.no_grad()
def run_windows(model, tgt, d: AliceSeq, w: pd.DataFrame, a, dev, retrieval=False):
    """Decode every window; world-model diagnostics; optional retrieval + planning re-rank."""
    model.eval()
    ctx, Nt = d.ctx, d.Nt
    first = d.N - Nt - 1  # dynamics outputs predicting the target patches
    cand = None
    if retrieval:
        cand = w.drop_duplicates("audio_start").audio_start.to_numpy()
        pos = {v: i for i, v in enumerate(cand)}
        Uc = []
        for i in range(0, len(cand), 256):
            act = np.stack([d.act[s:s + d.T] for s in cand[i:i + 256]])
            Uc.append(model.act_enc(torch.from_numpy(act).to(dev).float()))
        Uc = torch.cat(Uc)                            # [K, Nt, d]
        Ucn = F.normalize(Uc, dim=-1)
    P, Y, subj, starts, rows = [], [], [], [], []
    for i in range(0, len(w), a.eval_batch):
        r = w.iloc[i:i + a.eval_batch]
        b = d.assemble(r.subject.tolist(), (r.audio_start - ctx).tolist(), (r.eeg_start - ctx).tolist(), dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            z = model.enc(b["x"], b["sid"]).float()
            zt = tgt(b["x"], b["sid"]).float()
            u = model.act_enc(b["act"]).float()
            e, m = model.idm(z)
            null = model.null_u.expand_as(u)
            e_true = dyn_err(model, z, u, zt, first).float()
            e_shuf = dyn_err(model, z, u.roll(len(r) // 2, 0), zt, first).float()
            e_null = dyn_err(model, z, null, zt, first).float()
        P.append(m[:, ctx:].float().cpu().numpy())
        Y.append(b["mel"][:, ctx:].cpu().numpy())
        subj += r.subject.tolist()
        starts.append(r.audio_start.to_numpy())
        row = dict(subject=r.subject.to_numpy(), err_true=e_true.cpu().numpy(),
                   err_shuf=e_shuf.cpu().numpy(), err_null=e_null.cpu().numpy())
        if retrieval:
            en = F.normalize(e[:, -Nt:].float(), dim=-1)
            S = torch.einsum("btd,ktd->bk", en, Ucn) / Nt           # [B, K]
            true = torch.tensor([pos[v] for v in r.audio_start], device=dev)
            rank = (S > S.gather(1, true[:, None])).sum(1)
            K = min(a.rerank_k, S.shape[1])
            top = S.topk(K, dim=1).indices                         # [B, K]
            B = len(r)
            u_c = null[:1, :1].expand(B * K, d.N, -1).clone()     # context actions nulled
            u_c[:, -Nt:] = Uc[top.flatten()]
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                err = dyn_err(model, z.repeat_interleave(K, 0), u_c, zt.repeat_interleave(K, 0), first)
            err = err.float().reshape(B, K)
            wm_pick = top.gather(1, err.argmin(1, keepdim=True))[:, 0]
            row.update(idm_rank=rank.cpu().numpy(), in_topk=(top == true[:, None]).any(1).cpu().numpy(),
                       wm_top1=(wm_pick == true).cpu().numpy())
        rows.append(row)
    model.train()
    df = pd.concat([pd.DataFrame(x) for x in rows], ignore_index=True)
    return np.concatenate(P), np.concatenate(Y), np.array(subj), np.concatenate(starts), df, \
        (len(cand) if cand is not None else None)


def summarize(P, Y, subj, starts, df, T, n_cand=None):
    Pe, Ye = env_bands(P), env_bands(Y)
    out = []
    for s in sorted(set(subj)):
        k = subj == s
        r = window_metrics(Pe[k], Ye[k], starts[k], T)
        g = df[df.subject == s]
        row = dict(subject=s, r_above_null=r["r_env"] - r["r_env_null"], **r,
                   dnll_shuf=float((g.err_shuf - g.err_true).mean()),
                   dnll_null=float((g.err_null - g.err_true).mean()),
                   wm_mm_acc=float((g.err_true < g.err_shuf).mean()))
        if "idm_rank" in g:
            row.update(top1=float((g.idm_rank < 1).mean()), top10=float((g.idm_rank < 10).mean()),
                       recall_at_k=float(g.in_topk.mean()), wm_rerank_top1=float(g.wm_top1.mean()),
                       n_candidates=n_cand)
        out.append(row)
    return pd.DataFrame(out)


METRICS = ["r_env", "r_env_null", "r_above_null", "r_bands", "mm_acc", "dnll_shuf", "dnll_null", "wm_mm_acc",
           "top1", "top10", "recall_at_k", "wm_rerank_top1"]


# ============================================================================== train
def train(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = AliceSeq(a)
    va = d.eval_windows("val", d.seen, a.val_per_subject)
    n_tr = sum(len(v[0]) for v in d.train_starts.values())
    print(f"seen {len(d.seen)} subjects, held out {d.heldout}; sequence {d.L} frames = {d.N} patches "
          f"({a.ctx_patches} context + {d.Nt} target) of {a.patch} frames; action={a.action} "
          f"(F={d.act.shape[1]}); {n_tr} valid training start frames; val windows {len(va)}", flush=True)
    model, tgt = build(a, d, dev)
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
            loss, parts = losses(model, tgt, b, a, step)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        ema_update(tgt, model.enc, a.ema)
        for k, v in parts.items():
            acc[k] = acc.get(k, 0.0) + float(v)
        if step % 100 == 0:
            print(f"step {step} | " + " ".join(f"{k} {v / 100:.4f}" for k, v in acc.items())
                  + f" | lr {sched.get_last_lr()[0]:.2e} | {time.time() - t0:.0f}s", flush=True)
            acc = {}
        if step % a.eval_every == 0 or step == a.steps:
            m = summarize(*run_windows(model, tgt, d, va, a, dev)[:5], d.T)
            v = dict(step=step, **{k: float(m[k].mean()) for k in METRICS if k in m})
            hist.append(v)
            score, tag = v[a.select], ""
            if step > a.pretrain_steps and score > best:
                best, bad, tag = score, 0, " (best)"
                torch.save(dict(model=model.state_dict(), tgt=tgt.state_dict(), step=step, args=vars(a), val=v),
                           run / "best.pt")
            elif step > a.pretrain_steps:
                bad += 1
            print(f"== val step {step}: r_env {v['r_env']:.4f} above_null {v['r_above_null']:.4f} "
                  f"mm_acc {v['mm_acc']:.3f} | dNLL_shuf {v['dnll_shuf']:.4f} dNLL_null {v['dnll_null']:.4f} "
                  f"wm_mm_acc {v['wm_mm_acc']:.3f}{tag}", flush=True)
            (run / "val_history.json").write_text(json.dumps(hist, indent=1))
            if a.save_every and step % a.save_every == 0:  # per-step checkpoints for eval_world_model_curves.py
                (run / "ckpt").mkdir(exist_ok=True)
                torch.save(dict(model=model.state_dict(), tgt=tgt.state_dict(), step=step, args=vars(a)),
                           run / "ckpt" / f"step_{step:06d}.pt")
            if a.patience and bad >= a.patience:
                print(f"early stop: no {a.select} gain in {bad} evaluations (best {best:.4f})", flush=True)
                break
    torch.save(dict(model=model.state_dict(), tgt=tgt.state_dict(), step=step, args=vars(a)), run / "last.pt")
    evaluate(a, d)


def evaluate(a, d: AliceSeq | None = None):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = d or AliceSeq(a)
    model, tgt = build(a, d, dev)
    ck = torch.load(Path(a.run) / "best.pt", map_location=dev)
    model.load_state_dict(ck["model"])
    tgt.load_state_dict(ck["tgt"])
    base = pd.read_csv(a.baseline) if Path(a.baseline).exists() else None
    out, rows = dict(checkpoint_step=int(ck["step"])), []
    for name, subs, bmodel in (("seen", d.seen, "subject"), ("unseen", d.heldout, "pooled")):
        te = d.eval_windows("test", subs)
        if not len(te):
            continue
        res_all = run_windows(model, tgt, d, te, a, dev, retrieval=True)
        m = summarize(*res_all[:5], d.T, n_cand=res_all[5])
        m.insert(0, "group", name)
        rows.append(m)
        res = {k: float(m[k].mean()) for k in METRICS if k in m}
        res.update(n_subjects=int(len(m)), n_test_windows=int(len(te)), n_candidates=int(res_all[5]),
                   chance_top1=1 / res_all[5], chance_top10=min(1.0, 10 / res_all[5]))
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
    p.add_argument("--run", default=str(ROOT / "runs" / "world_model"))
    p.add_argument("--baseline", default=str(ROOT / "runs" / "mtrf_baseline_v2" / "results.csv"))
    p.add_argument("--heldout", nargs="*", default=HELDOUT)
    p.add_argument("--exclude", nargs="*", default=[])
    # sequence / model
    p.add_argument("--action", choices=["wav2vec", "mel"], default="wav2vec", help="audio features used as actions")
    p.add_argument("--patch", type=int, default=30, help="frames per patch (120 Hz): 30 = 250 ms")
    p.add_argument("--ctx-patches", type=int, default=4, help="context patches before the 3 s target window")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    # optimisation
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--pretrain-steps", type=int, default=0, help="first steps: EEG dynamics only, no actions")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--ema", type=float, default=0.996)
    # losses
    p.add_argument("--l-dyn", type=float, default=1.0)
    p.add_argument("--l-anc", type=float, default=0.5, help="action-contrastive dynamics loss")
    p.add_argument("--anc-neg", type=int, default=3, help="shuffled action sequences per sample")
    p.add_argument("--anc-tau", type=float, default=0.05)
    p.add_argument("--l-clip", type=float, default=1.0)
    p.add_argument("--l-mel", type=float, default=1.0)
    p.add_argument("--l-corr", type=float, default=1.0)
    p.add_argument("--l-var", type=float, default=1.0)
    p.add_argument("--l-rec", type=float, default=0.1)
    # regularisation
    p.add_argument("--act-drop", type=float, default=0.15, help="prob. of nulling one action patch")
    p.add_argument("--act-drop-seq", type=float, default=0.15, help="prob. of nulling all actions of a sample")
    p.add_argument("--subject-drop", type=float, default=0.1, help="prob. of the 'unknown subject' slot")
    p.add_argument("--chan-drop", type=float, default=0.1)
    p.add_argument("--noise", type=float, default=0.1)
    # evaluation
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batch", type=int, default=128)
    p.add_argument("--val-per-subject", type=int, default=150)
    p.add_argument("--select", choices=["mm_acc", "r_above_null", "wm_mm_acc"], default="mm_acc")
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--rerank-k", type=int, default=10, help="IDM candidates re-ranked by the world model")
    p.add_argument("--save-every", type=int, default=0,
                   help="also save run/ckpt/step_N.pt every N steps (multiple of --eval-every; 0 = off)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    {"train": train, "eval": evaluate}[a.cmd](a)

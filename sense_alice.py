"""SENSE-style EEG -> speech model for the Alice data (D_T_processed_v2).

Adapts SENSE (paper2.pdf; full N400 implementation in sense.py) to naturalistic
listening, and reports the SAME metrics as mtrf_baseline.py so the two compare
directly.

Model (EEG encoder reused from sense.py)
  EEG window [61, 3 s + context] -> channel gate (auditory warm start)
    -> {graph conv over Easycap-M10 geometry + sigma(g) * CNN skip}
    -> Conv1d stride 3 -> 4 bidirectional S4 blocks -> M [192, T/3]
  mel head    : M upsampled x3, FiLM-conditioned on g_eeg (detached, as the paper's
                conditioning bridge), residual convs -> 120-band log-mel at 120 Hz
  auxiliaries : EEG self-reconstruction (cosine, sense.py), CTC phoneme head on M
                (ARPAbet from the window's words via g2p_en), ESC: g_eeg = W_eeg(mean M)
                cosine-aligned to the frozen CLIP text embedding of the window's words
  Loss = L1(mel) + corr_w * (1 - Pearson per band) + 1.0 recon + 0.3 CTC + 0.5 ESC

Adaptations to this dataset (vs the paper)
  * no congruent/incongruent trials: ESC only on "predictable" windows, i.e. mean
    RNN surprisal of the window's words below the training median (--esc-mode)
  * no VITS: the decoder regresses the D_T mel (so metrics match the baseline);
    `synth` turns predictions into audio with Griffin-Lim (rough, for listening)
  * CTC on a linear head over M (sense.py uses a Conformer + LSTM decoder)
  * CLIP text embeddings are mean-centred over the corpus before the cosine loss
    (raw CLIP text vectors share one dominant direction)
  * EEG input = the baseline's lag range: 0.1 s before to 0.25 s after the 3 s
    window (eeg_start already includes the 150 ms brain delay)

Against overfitting / window-shape leakage (job 552929: val peaked at 2k steps; every
window starts 0.5 s before a word onset, so a deep model can output the average window
envelope from position alone -> chance-level r_env ~0.16 instead of ~0.005)
  * training windows are randomly shifted by up to +-0.5 s (EEG and audio together),
    only to positions that stay inside one clip and inside training-sentence audio
  * best.pt is chosen by match-mismatch accuracy (robust to the shared shape);
    r_above_null = r_env - r_env_null is reported next to r_env
  * early stopping (--patience), channel dropout, EEG noise, dropout 0.2, weight decay 0.05
  The CTC targets / CLIP text still describe the unshifted window (small mismatch).

Splits
  sentences : the sentence-block train/val/test split of prepare_alice_dataset.py
  subjects  : --heldout subjects never seen in training ("unseen"); the rest "seen"
  checkpoint selection on val windows of seen subjects (mean envelope r)

Metrics (test sentences, per subject then averaged), as in mtrf_baseline.py
  r_env, r_bands, r_env_null, mm_acc; compared with runs/mtrf_baseline_v2/results.csv
  (subject model for seen subjects, pooled model for unseen subjects)

Usage
  python sense_alice.py train --data D_T_processed_v2 --run runs/sense_alice
  python sense_alice.py eval  --data D_T_processed_v2 --run runs/sense_alice
  python sense_alice.py synth --data D_T_processed_v2 --run runs/sense_alice
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mtrf_baseline import evaluate as window_metrics
from sense import D, VOCAB, EEGModule, recon_loss

ROOT = Path(__file__).resolve().parent
PRE, POST = 12, 30  # EEG samples of context before / after the window (120 Hz)
HELDOUT = ["S03", "S13", "S22", "S32", "S41", "S48"]  # spread over ID blocks and quality
AUDITORY = ["T7", "T8", "FT7", "FT8", "TP7", "TP8"]
N_BANDS = 8


# =============================================================================== data
def geometry(channels):
    """Easycap-M10 3-D positions -> Gaussian-kernel adjacency, auditory electrode indices."""
    import mne
    mont = mne.channels.make_standard_montage("easycap-M10").get_positions()["ch_pos"]
    pos = torch.tensor(np.stack([mont[c] for c in channels]), dtype=torch.float32)
    dist = torch.cdist(pos, pos)
    nn_d = dist.masked_fill(torch.eye(len(pos), dtype=torch.bool), 1e9).min(1).values.median()
    A = torch.exp(-dist ** 2 / (2 * nn_d ** 2)) * (dist < 2 * nn_d)  # sigma = neighbour spacing
    A.fill_diagonal_(0)
    A = A / A.sum(1, keepdim=True).clamp_min(1e-8)
    std = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    u = F.normalize(pos, dim=1)
    aud = sorted({int((u @ F.normalize(torch.tensor(std[n], dtype=torch.float32), dim=0)).argmax())
                  for n in AUDITORY})
    return A, aud


def text_features(dataset: Path, windows: pd.DataFrame, meta: dict, cache: Path) -> dict:
    """{segment_id: dict(phon, clip, surprisal)} for every window anchor, cached."""
    if cache.exists():
        return torch.load(cache)
    from g2p_en import G2p
    from transformers import CLIPModel, CLIPTokenizer

    words = pd.read_csv(dataset / "AliceChapterOne-EEG.csv")
    fs, T, offs = meta["fs"], meta["window_samples"], meta["clip_offsets"]
    seg = windows.drop_duplicates("segment_id")[["segment_id", "text", "audio_clip", "audio_start"]]
    g2p = G2p()
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    out, texts = {}, []
    for r in seg.itertuples():
        t0 = (r.audio_start - offs[r.audio_clip - 1]) / fs
        inside = words[(words.Segment == r.audio_clip) & (words.offset > t0) & (words.onset < t0 + T / fs)]
        ph = [VOCAB.index(p.rstrip("012")) for p in g2p(str(r.text)) if p.rstrip("012") in VOCAB]
        out[int(r.segment_id)] = dict(phon=torch.tensor(ph, dtype=torch.long),
                                      surprisal=float(inside.RNN.mean()) if len(inside) else float("nan"))
        texts.append(str(r.text))
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), 256):
            b = tok(texts[i:i + 256], return_tensors="pt", padding=True, truncation=True)
            o = clip.get_text_features(**b)
            if not torch.is_tensor(o):  # newer transformers return a model-output object
                o = o.text_embeds if getattr(o, "text_embeds", None) is not None else o.pooler_output
                if o.shape[-1] != clip.config.projection_dim:  # un-projected pooled output
                    o = clip.text_projection(o)
            embs.append(F.normalize(o, dim=-1))
    E = torch.cat(embs)
    E = F.normalize(E - E.mean(0), dim=-1)  # remove the shared CLIP direction
    for k, e in zip(out, E):
        out[k]["clip"] = e
    torch.save(out, cache)
    print(f"text features: {len(out)} windows, mean {np.mean([len(v['phon']) for v in out.values()]):.1f} "
          f"phonemes per window", flush=True)
    return out


SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


def frame_labels(dataset: Path, meta: dict):
    """Per global audio frame (120 Hz, clips concatenated): sentence split code and clip number.
    A frame belongs to the word whose [onset, next onset) covers it (pauses go to the previous word)."""
    from prepare_alice_dataset import assign_splits
    words = pd.read_csv(dataset / "AliceChapterOne-EEG.csv")
    split = words.Sentence.map(assign_splits(words)).map(SPLIT_CODE).to_numpy()
    fs, offs, lens = meta["fs"], meta["clip_offsets"], meta["clip_lengths"]
    n = offs[-1] + lens[-1]
    lab, clip = np.full(n, -1, np.int8), np.zeros(n, np.int16)
    for c, (o, L) in enumerate(zip(offs, lens), start=1):
        clip[o:o + L] = c
        w = words[words.Segment == c].sort_values("onset")  # words has a RangeIndex
        on = np.clip(np.round(w.onset.to_numpy() * fs).astype(int), 0, L)
        for k, (s0, code) in enumerate(zip(on, split[w.index.to_numpy()])):
            s1 = on[k + 1] if k + 1 < len(on) else L
            lab[o + (0 if k == 0 else s0):o + s1] = code
    return lab, clip


class Windows(Dataset):
    def __init__(self, df, eegs, mel, text, T, esc_thr, esc_mode, aug=None):
        self.subj = df.subject.to_numpy()
        self.eeg_start = df.eeg_start.to_numpy(dtype=np.int64)
        self.audio_start = df.audio_start.to_numpy(dtype=np.int64)
        self.seg = df.segment_id.to_numpy(dtype=np.int64)
        self.eegs, self.mel, self.text, self.T = eegs, mel, text, T
        self.esc_thr, self.esc_mode = esc_thr, esc_mode
        self.aug = aug  # None (eval) or dict(jitter, lab, clip, chan_drop, noise)

    def __len__(self):
        return len(self.subj)

    def jitter(self, s, a, n_eeg):
        """Random shift (same for EEG and audio) that keeps the window inside one clip,
        inside training-sentence audio and inside the recording; 0 if none found."""
        J, lab, clip = self.aug["jitter"], self.aug["lab"], self.aug["clip"]
        for _ in range(6):
            j = random.randint(-J, J)
            a0, a1 = a + j, a + j + self.T
            if (a0 >= 0 and a1 <= len(lab) and s + j - PRE >= 0 and s + j + self.T + POST <= n_eeg
                    and clip[a0] == clip[a1 - 1] and (lab[a0:a1] == SPLIT_CODE["train"]).all()):
                return j
        return 0

    def __getitem__(self, i):
        s, a = int(self.eeg_start[i]), int(self.audio_start[i])
        eeg = self.eegs[self.subj[i]]
        if self.aug and self.aug["jitter"] > 0:
            j = self.jitter(s, a, eeg.shape[1])
            s, a = s + j, a + j
        x = torch.from_numpy(eeg[:, s - PRE:s + self.T + POST].astype(np.float32))
        if self.aug:
            if self.aug["chan_drop"] > 0:
                x = x * (torch.rand(x.size(0), 1) >= self.aug["chan_drop"])
            if self.aug["noise"] > 0:
                x = x + self.aug["noise"] * torch.randn_like(x)
        y = torch.from_numpy(self.mel[a:a + self.T].astype(np.float32))
        t = self.text[int(self.seg[i])]
        w = {"all": 1.0, "none": 0.0}.get(self.esc_mode, float(t["surprisal"] < self.esc_thr))
        return dict(eeg=x, mel=y, phon=t["phon"], clip=t["clip"], esc_w=w,
                    subject=self.subj[i], audio_start=a)


def collate(b):
    pl = torch.tensor([len(x["phon"]) for x in b])
    phon = torch.stack([F.pad(x["phon"], (0, int(pl.max()) - len(x["phon"]))) for x in b])
    return dict(eeg=torch.stack([x["eeg"] for x in b]), mel=torch.stack([x["mel"] for x in b]),
                phon=phon, phon_len=pl, clip=torch.stack([x["clip"] for x in b]),
                esc_w=torch.tensor([x["esc_w"] for x in b]),
                subject=[x["subject"] for x in b], audio_start=np.array([x["audio_start"] for x in b]))


def load_data(a):
    data = Path(a.data)
    meta = json.loads((data / "meta.json").read_text())
    T = meta["window_samples"]
    windows = pd.read_csv(data / "windows.csv")
    eegs = {s: np.load(data / "eeg" / f"{s}.npy") for s in meta["subjects"]}  # float16, in RAM
    n = windows.subject.map(lambda s: eegs[s].shape[1])
    windows = windows[(windows.eeg_start - PRE >= 0) & (windows.eeg_start + T + POST <= n)]
    mel = np.load(data / "speech_mel.npy")
    Path(a.run).mkdir(parents=True, exist_ok=True)
    text = text_features(Path(a.dataset), windows, meta, Path(a.run) / "text_features.pt")
    heldout = [s for s in a.heldout if s in meta["subjects"]]
    seen = [s for s in meta["subjects"] if s not in heldout and s not in a.exclude]
    tr = windows[(windows.split == "train") & windows.subject.isin(seen)]
    esc_thr = float(np.nanmedian([text[int(k)]["surprisal"] for k in tr.segment_id.unique()]))
    lab, clip = frame_labels(Path(a.dataset), meta)
    aug = dict(jitter=a.jitter, lab=lab, clip=clip, chan_drop=a.chan_drop, noise=a.noise)
    mk = lambda df, train=False: Windows(df, eegs, mel, text, T, esc_thr, a.esc_mode,
                                         aug if train else None)
    return dict(meta=meta, T=T, windows=windows, mk=mk, seen=seen, heldout=heldout, eegs=eegs,
                mel=mel, esc_thr=esc_thr)


# ============================================================================== model
class SenseAlice(nn.Module):
    def __init__(self, A, aud, C, n_mel, clip_dim=512, p=0.1):
        super().__init__()
        self.eeg = EEGModule(A, aud, C)
        self.ctc = nn.Linear(D, len(VOCAB))
        self.W_eeg = nn.Linear(D, clip_dim)
        self.film = nn.Linear(clip_dim, 2 * D)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.up = nn.ConvTranspose1d(D, D, 3, 3)
        self.blocks = nn.ModuleList(nn.Sequential(nn.Conv1d(D, D, 5, padding=2), nn.GELU(), nn.Dropout(p),
                                                  nn.Conv1d(D, D, 5, padding=2)) for _ in range(3))
        self.norms = nn.ModuleList(nn.GroupNorm(1, D) for _ in range(3))
        self.out = nn.Conv1d(D, n_mel, 1)

    def forward(self, x, T):
        B, _, L = x.shape
        lens = torch.full((B,), L, device=x.device, dtype=torch.long)
        M, lens3, xh = self.eeg(x, lens)
        g = self.W_eeg(M.mean(-1))
        gam, bet = self.film(g.detach()).chunk(2, -1)  # conditioning bridge: no grad into M / W_eeg
        h = self.up(M)[..., PRE:PRE + T]
        h = h * (1 + gam[..., None]) + bet[..., None]
        for blk, nrm in zip(self.blocks, self.norms):
            h = h + blk(nrm(h))
        return dict(mel=self.out(h).transpose(1, 2), ctc=self.ctc(M.transpose(1, 2)), g=g, xh=xh,
                    lens=lens, lens3=lens3)


def pearson_loss(p, y):  # [B,T,K], 1 - mean Pearson r over time
    p = p - p.mean(1, keepdim=True)
    y = y - y.mean(1, keepdim=True)
    r = (p * y).sum(1) / (p.norm(dim=1) * y.norm(dim=1) + 1e-6)
    return 1 - r.mean()


def losses(model, b, T, a):
    o = model(b["eeg"], T)
    l_l1 = F.l1_loss(o["mel"], b["mel"])
    l_corr = pearson_loss(o["mel"], b["mel"])
    l_recon = recon_loss(b["eeg"], o["xh"], o["lens"])
    logp = o["ctc"].log_softmax(-1).transpose(0, 1)  # [T3,B,V]
    l_ctc = F.ctc_loss(logp, b["phon"], o["lens3"], b["phon_len"], blank=0, zero_infinity=True)
    w = b["esc_w"]
    cos = F.cosine_similarity(o["g"], b["clip"], dim=-1)
    l_esc = ((1 - cos) * w).sum() / w.sum().clamp_min(1.0)
    loss = (l_l1 + a.l_corr * l_corr + a.l_recon * l_recon + a.l_ctc * l_ctc + a.l_esc * l_esc)
    return loss, dict(l1=l_l1, corr=l_corr, recon=l_recon, ctc=l_ctc, esc=l_esc)


def build(d, dev, p=0.1):
    A, aud = geometry(d["meta"]["channels"])
    return SenseAlice(A, aud, len(d["meta"]["channels"]), d["mel"].shape[1], p=p).to(dev)


# ============================================================================ metrics
def env_bands(m):  # [n,T,120] -> [n,T,1+N_BANDS], same definitions as mtrf_baseline.py
    return np.concatenate([m.mean(-1, keepdims=True), m.reshape(*m.shape[:2], N_BANDS, -1).mean(-1)], -1)


@torch.no_grad()
def predict(model, ds, T, dev, batch=128):
    model.eval()
    P, Y, subj, starts = [], [], [], []
    for b in DataLoader(ds, batch, collate_fn=collate, num_workers=4):
        P.append(model(b["eeg"].to(dev), T)["mel"].float().cpu().numpy())
        Y.append(b["mel"].numpy())
        subj += b["subject"]
        starts.append(b["audio_start"])
    model.train()
    return np.concatenate(P), np.concatenate(Y), np.array(subj), np.concatenate(starts)


def per_subject_metrics(P, Y, subj, starts, T):
    Pe, Ye = env_bands(P), env_bands(Y)
    rows = []
    for s in sorted(set(subj)):
        k = subj == s
        r = window_metrics(Pe[k], Ye[k], starts[k], T)
        rows.append(dict(subject=s, r_above_null=r["r_env"] - r["r_env_null"], **r))
    return pd.DataFrame(rows)


# ============================================================================== train
def train(a):
    torch.manual_seed(a.seed), random.seed(a.seed), np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = load_data(a)
    T, W = d["T"], d["windows"]
    tr = W[(W.split == "train") & W.subject.isin(d["seen"])]
    va = W[(W.split == "val") & W.subject.isin(d["seen"])]
    va = va.sample(frac=1.0, random_state=0).groupby("subject").head(a.val_per_subject)
    print(f"seen {len(d['seen'])} subjects, held out {d['heldout']}, excluded {a.exclude}; "
          f"train {len(tr)} / val {len(va)} windows; ESC on windows with surprisal < {d['esc_thr']:.2f} "
          f"(mode {a.esc_mode}); aug: jitter +-{a.jitter} samples, chan_drop {a.chan_drop}, "
          f"noise {a.noise}; select by {a.select}, patience {a.patience}", flush=True)
    dl = DataLoader(d["mk"](tr, train=True), a.batch, shuffle=True, collate_fn=collate,
                    num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)
    model = build(d, dev, a.dropout)
    print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    run = Path(a.run)
    step, best, hist, t0, bad = 0, -1.0, [], time.time(), 0
    acc, stop = {}, False
    while step < a.steps and not stop:
        for b in dl:
            b = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()}
            loss, parts = losses(model, b, T, a)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if a.save_every and step % a.save_every == 0:  # opt-in, for eval_sense_curves.py
                (run / "ckpt").mkdir(exist_ok=True)
                torch.save(dict(model=model.state_dict(), step=step, args=vars(a)), run / "ckpt" / f"step_{step:06d}.pt")
            for k, v in parts.items():
                acc[k] = acc.get(k, 0.0) + float(v)
            if step % 100 == 0:
                print(f"step {step} | " + " ".join(f"{k} {v / 100:.4f}" for k, v in acc.items())
                      + f" | lr {sched.get_last_lr()[0]:.2e} | {time.time() - t0:.0f}s", flush=True)
                acc = {}
            if step % a.eval_every == 0 or step >= a.steps:
                m = per_subject_metrics(*predict(model, d["mk"](va), T, dev), T)
                v = dict(step=step, r_env=float(m.r_env.mean()), r_env_null=float(m.r_env_null.mean()),
                         r_above_null=float(m.r_above_null.mean()), mm_acc=float(m.mm_acc.mean()))
                hist.append(v)
                score, tag = v[a.select], ""
                if score > best:
                    best, bad, tag = score, 0, " (best)"
                    torch.save(dict(model=model.state_dict(), step=step, args=vars(a), val=v), run / "best.pt")
                else:
                    bad += 1
                print(f"== val step {step}: r_env {v['r_env']:.4f} null {v['r_env_null']:.4f} "
                      f"above_null {v['r_above_null']:.4f} mm_acc {v['mm_acc']:.3f}{tag}", flush=True)
                (run / "val_history.json").write_text(json.dumps(hist, indent=1))
                if a.patience and bad >= a.patience:
                    print(f"early stop: no {a.select} gain in {bad} evaluations (best {best:.4f})", flush=True)
                    stop = True
            if step >= a.steps or stop:
                break
    torch.save(dict(model=model.state_dict(), step=step, args=vars(a)), run / "last.pt")
    evaluate(a)


# =============================================================================== eval
def evaluate(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = load_data(a)
    T, W = d["T"], d["windows"]
    model = build(d, dev)
    ck = torch.load(Path(a.run) / "best.pt", map_location=dev)
    model.load_state_dict(ck["model"])
    base = pd.read_csv(a.baseline) if Path(a.baseline).exists() else None
    out, all_rows = {}, []
    for name, subs, bmodel in (("seen", d["seen"], "subject"), ("unseen", d["heldout"], "pooled")):
        te = W[(W.split == "test") & W.subject.isin(subs)]
        if not len(te):
            continue
        m = per_subject_metrics(*predict(model, d["mk"](te), T, dev), T)
        m.insert(0, "group", name)
        all_rows.append(m)
        res = {k: float(m[k].mean()) for k in ("r_env", "r_env_null", "r_above_null", "r_bands", "mm_acc")}
        if base is not None:
            bb = base[(base.model == bmodel) & base.subject.isin(m.subject)]
            res["baseline"] = {k: float(bb[k].mean()) for k in ("r_env", "r_env_null", "r_bands", "mm_acc")}
            res["baseline"]["r_above_null"] = float((bb.r_env - bb.r_env_null).mean())
            res["baseline_model"] = f"mtrf {bmodel}"
        res["n_subjects"] = int(len(m))
        out[name] = res
        print(f"{name} ({len(m)} subjects, test sentences): " + json.dumps(res), flush=True)
    out["checkpoint_step"] = int(ck["step"])
    pd.concat(all_rows).to_csv(Path(a.run) / "per_subject.csv", index=False)
    (Path(a.run) / "metrics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


# ============================================================================== synth
def synth(a):
    """Griffin-Lim audio for a few test windows: prediction, true-mel upper bound, original."""
    import soundfile as sf
    import torchaudio
    from scipy.signal import resample_poly

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = load_data(a)
    T, W, meta = d["T"], d["windows"], d["meta"]
    model = build(d, dev)
    model.load_state_dict(torch.load(Path(a.run) / "best.pt", map_location=dev)["model"])
    audio = Path(a.dataset) / "audio"

    # per-band stats of the un-standardized log-mel (same transform as prepare_alice_dataset.py)
    mel_fn = torchaudio.transforms.MelSpectrogram(16000, n_fft=512, hop_length=128, n_mels=120, normalized=True)
    wavs, logs = {}, []
    for c in range(1, len(meta["clip_offsets"]) + 1):
        w, sr = sf.read(audio / f"DownTheRabbitHoleFinal_SoundFile{c}.wav", dtype="float32")
        w = resample_poly(w.mean(1) if w.ndim > 1 else w, 16000, sr).astype(np.float32)
        wavs[c] = w
        logs.append(torch.log(1e-5 + mel_fn(torch.from_numpy(w))).T)
    L = torch.cat(logs)
    mu, sd = L.mean(0), L.std(0)
    inv = torchaudio.transforms.InverseMelScale(n_stft=257, n_mels=120, sample_rate=16000)
    gl = torchaudio.transforms.GriffinLim(n_fft=512, hop_length=128, power=2.0, normalized=True, n_iter=64)

    def to_wav(m):  # standardized [T,120] at 120 Hz -> waveform
        lm = torch.from_numpy(m).float() * sd + mu
        lm = F.interpolate(lm.T[None], size=int(round(T / meta["fs"] * 125)), mode="linear")[0]
        return gl(inv(lm.exp().clamp_min(0))).numpy()

    out = Path(a.run) / "samples"
    out.mkdir(exist_ok=True)
    picks = []
    for s in (d["heldout"][:2] + d["seen"][:2]):
        te = W[(W.split == "test") & (W.subject == s)].drop_duplicates("audio_start")
        picks.append(te.sort_values("audio_start").iloc[::max(1, len(te) // a.n_synth)][:a.n_synth])
    sel = pd.concat(picks)
    P, Y, subj, starts = predict(model, d["mk"](sel), T, dev)
    for i, r in enumerate(sel.itertuples()):
        t0 = (r.audio_start - meta["clip_offsets"][r.audio_clip - 1]) / meta["fs"]
        orig = wavs[r.audio_clip][int(t0 * 16000): int((t0 + T / meta["fs"]) * 16000)]
        tag = f"{r.subject}_clip{r.audio_clip}_{t0:.1f}s"
        sf.write(out / f"{tag}_pred.wav", to_wav(P[i]), 16000)
        sf.write(out / f"{tag}_truemel.wav", to_wav(Y[i]), 16000)
        sf.write(out / f"{tag}_orig.wav", orig, 16000)
        (out / f"{tag}.txt").write_text(str(r.text))
    print(f"wrote {len(sel)} x 3 wavs to {out}", flush=True)


# =============================================================================== main
if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("cmd", choices=["train", "eval", "synth"])
    p.add_argument("--data", default=str(ROOT / "D_T_processed_v2"))
    p.add_argument("--dataset", default=str(ROOT / "Dataset"))
    p.add_argument("--run", default=str(ROOT / "runs" / "sense_alice"))
    p.add_argument("--baseline", default=str(ROOT / "runs" / "mtrf_baseline_v2" / "results.csv"))
    p.add_argument("--heldout", nargs="*", default=HELDOUT)
    p.add_argument("--exclude", nargs="*", default=[], help="subjects left out entirely, e.g. S11")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--l-corr", type=float, default=1.0)
    p.add_argument("--l-recon", type=float, default=1.0)
    p.add_argument("--l-ctc", type=float, default=0.3)
    p.add_argument("--l-esc", type=float, default=0.5)
    p.add_argument("--esc-mode", choices=["low_surprisal", "all", "none"], default="low_surprisal")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=0,
                   help="also save run/ckpt/step_N.pt every N steps (for eval_sense_curves.py); 0 = off")
    p.add_argument("--select", choices=["mm_acc", "r_above_null", "r_env"], default="mm_acc",
                   help="validation metric for best.pt (r_env is inflated by the shared window shape)")
    p.add_argument("--patience", type=int, default=6, help="evaluations without gain before stopping; 0 = off")
    p.add_argument("--jitter", type=int, default=60, help="max random shift of training windows (samples)")
    p.add_argument("--chan-drop", type=float, default=0.1, help="prob. of zeroing each EEG channel (train)")
    p.add_argument("--noise", type=float, default=0.1, help="Gaussian noise std added to EEG (train)")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--val-per-subject", type=int, default=150)
    p.add_argument("--n-synth", type=int, default=3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    {"train": train, "eval": evaluate, "synth": synth}[a.cmd](a)

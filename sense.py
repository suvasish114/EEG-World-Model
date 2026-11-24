"""SENSE: Semantic Neural Speech Synthesis from Brain Dynamics via Spatial Graph Encoding
(Park et al., NeurIPS 2026, arXiv:2609.37601) -- minimal single-file implementation.

Data: N400 corpus of Toffolo, Freedman & Foxe (2022), Dryad
  EEG     doi:10.5061/dryad.6wwpzgmx4  (BIDS-style, 128-ch BioSemi .bdf, 512 Hz, 24 subjects)
  stimuli doi:10.5061/dryad.9ghx3ffkg  (442 wavs NPC_*/NPI_*, N400Stimset_stimuli_parameters.tsv)

Pipeline (paper Sec. 3 + App. A):
  EEG X[128,T] -> channel gate -> {graph conv over BioSemi geometry  +  sigma(g) * CNN skip}
     -> Conv1d stride 3 -> 4 bidirectional S4 blocks -> M[192,T/3]
  aux (training only): EEG decoder (cosine recon), Conformer + LSTM-attention phoneme predictor (CTC)
  ESC: g_eeg = W_eeg(masked-mean(M)), cosine-aligned to sg[W_clip(CLIP-text)] on congruent trials only
  speech: connector(sg[M]) -> VITS prior; posterior enc + flow + HiFi-GAN, MPD/MSD, conditioned on g
  conditioning bridge: p=0.33 g=sg[g_eeg], else per-sample {W_clip(c_text), null} with prob 0.5

Usage:
  python sense.py prepare --bids <EEG root with sub-XX/> --stimuli <dir with wavs + parameters tsv> --out data/n400_sense
  python sense.py train   --data data/n400_sense --run runs/sense
  python sense.py eval    --data data/n400_sense --run runs/sense

Deviations (marked "NOTE" below): ARPAbet phonemes via g2p_en instead of the 51-symbol IPA set
(espeak needs sudo); connector uses absolute-position-free MHA instead of VITS relative attention;
EEG prior is linearly stretched to the spectrogram frame count (EEG 85.3 Hz vs mel 86.1 Hz);
evaluation reports Mel-Corr (+STOI if pystoi installed) only, not MCD/WER/BERTScore/CLIP-Sim.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:  # older torch
    from torch.nn.utils import weight_norm

# ----------------------------------------------------------------------------- config
EEG_FS = 256                       # Hz after resampling (A.1)
SR, NFFT, HOP, NMEL = 22050, 1024, 256, 80
D = 192                            # latent dim
SEG_FRAMES = 32                    # VITS segment_size 8192 samples
AUDITORY = ["T7", "T8", "FT7", "FT8", "TP7", "TP8"]
EXCLUDED_SUBJ = {5, 10, 15, 18}
TEST_SUBJ = {23, 24}
N_TEST_SENT = 40                   # 400 train / 40 held-out sentences per subject
L = dict(mel=45.0, kl=1.0, recon=1.0, ctc=0.3, esc=0.5)
P_EEG, LR, BETAS, WD, GAMMA = 0.33, 2e-4, (0.8, 0.99), 0.01, 0.999875
MAX_PH = 50

ARPA = ("AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH "
        "UH UW V W Y Z ZH").split()
VOCAB = ["<blank>", "<sos>", "<eos>"] + ARPA + [" ", ",", ".", "?", "!"]
SOS = 1


def stem(name) -> str:
    s = Path(str(name)).name
    return s[:-4] if s.lower().endswith(".wav") else s


def seq_mask(lens, T):
    return torch.arange(T, device=lens.device)[None] < lens[:, None]


# ----------------------------------------------------------------------------- prepare
def electrode_geometry():
    """BioSemi-128 positions (m), Gaussian-kernel adjacency (A.1 eq.), auditory electrode indices."""
    import mne
    bio = mne.channels.make_standard_montage("biosemi128").get_positions()["ch_pos"]
    names = list(bio)
    pos = torch.tensor(np.stack([bio[n] for n in names]), dtype=torch.float32)
    dist = torch.cdist(pos, pos)
    A = torch.exp(-dist ** 2 / (2 * 0.02 ** 2)) * (dist < 0.04)
    A.fill_diagonal_(0)
    A = A / A.sum(1, keepdim=True).clamp_min(1e-8)
    # BioSemi uses A1..D32 labels: map the 10-20 auditory sites to the nearest electrode by direction
    std = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    u = F.normalize(pos, dim=1)
    aud = sorted({int((u @ F.normalize(torch.tensor(std[n], dtype=torch.float32), dim=0)).argmax())
                  for n in AUDITORY})
    return names, pos, A, aud


def preprocess_subject(bdf: Path, names):
    """A.1: 60 Hz notch, 0.5-50 Hz Butterworth, ICA (EOG-correlated comps removed), resample 256 Hz."""
    import mne
    raw = mne.io.read_raw_bdf(bdf, preload=True, verbose="error")
    eog = [c for c in raw.ch_names if c.upper().startswith("EXG")]
    raw.set_channel_types({c: "eog" for c in eog}, verbose="error")
    raw.set_montage(mne.channels.make_standard_montage("biosemi128"), on_missing="ignore", verbose="error")
    ch_tsv = Path(str(bdf).replace("_eeg.bdf", "_channels.tsv"))
    if ch_tsv.exists():
        import pandas as pd
        t = pd.read_csv(ch_tsv, sep="\t")
        if "status" in t:
            raw.info["bads"] = [n for n, s in zip(t["name"], t["status"]) if str(s).lower() == "bad" and n in names]
    raw.notch_filter(60, picks=["eeg", "eog"], verbose="error")
    raw.filter(0.5, 50, picks=["eeg", "eog"], method="iir", iir_params=dict(order=4, ftype="butter"), verbose="error")
    if raw.info["bads"]:
        raw.interpolate_bads(reset_bads=True, verbose="error")
    ica = mne.preprocessing.ICA(n_components=20, random_state=0, max_iter="auto", verbose="error")
    ica.fit(raw, picks="eeg", decim=3, verbose="error")
    if eog:
        try:
            ica.exclude, _ = ica.find_bads_eog(raw, ch_name=eog, verbose="error")
        except Exception as e:  # flat / unconnected EXG channels
            print(f"  ICA EOG detection skipped: {e}")
        ica.apply(raw, verbose="error")
    raw.resample(EEG_FS, verbose="error")
    missing = [n for n in names if n not in raw.ch_names]
    assert not missing, f"{bdf.name}: channels not found {missing[:5]}"
    x = raw.get_data(picks=names)
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)  # per-channel z-score
    return x.astype(np.float32)


def prepare(a):
    import pandas as pd
    import soundfile as sf
    from g2p_en import G2p
    from transformers import CLIPModel, CLIPTokenizer

    out = Path(a.out)
    (out / "eeg").mkdir(parents=True, exist_ok=True)
    names, pos, A, aud = electrode_geometry()
    torch.save(dict(names=names, pos=pos, A=A, aud=aud), out / "geometry.pt")

    # ---- stimuli: text, phonemes, CLIP text embedding, 22.05 kHz audio
    sdir = Path(a.stimuli)
    tsv = next(sdir.rglob("*stimuli_parameters.tsv"))
    tab = pd.read_csv(tsv, sep="\t")
    words = [c for c in tab.columns if str(c).strip().isdigit()]
    text = {stem(r["stim_file"]): " ".join(str(r[c]) for c in words if pd.notna(r[c]) and str(r[c]).strip())
            for _, r in tab.iterrows()}
    wavs = {stem(p): p for p in sdir.rglob("*.wav")}
    g2p = G2p()
    tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    stim = {}
    for k, sent in text.items():
        if k not in wavs:
            print(f"  no wav for {k}, skipped")
            continue
        y, sr = sf.read(wavs[k], dtype="float32", always_2d=True)
        y = torchaudio.functional.resample(torch.from_numpy(y.mean(1)), sr, SR)
        y = y[: len(y) // HOP * HOP]
        ph = [VOCAB.index(p.rstrip("012")) for p in g2p(sent) if p.rstrip("012") in VOCAB]  # NOTE: ARPAbet
        with torch.no_grad():
            c = clip.get_text_features(**tok([sent], return_tensors="pt", padding=True))[0]
        stim[k] = dict(text=sent, wav=y.half(), phon=torch.tensor(ph[:MAX_PH]), clip=c,
                       congruent="NPC_" in k.upper())
    torch.save(stim, out / "stimuli.pt")
    print(f"stimuli: {len(stim)} sentences")

    # ---- EEG: one window per sentence, from sentence onset for stim_dur seconds
    root = Path(a.bids)
    for bdf in sorted(root.glob("sub-*/eeg/*_eeg.bdf")):
        sub = int(bdf.name.split("_")[0].split("-")[1])
        ev = pd.read_csv(next(bdf.parent.glob("*_events.tsv")), sep="\t")
        x = preprocess_subject(bdf, names)
        trials = []
        for _, r in ev.iterrows():
            k = stem(r.get("stim_file", ""))
            if k not in stim or not any(t in k.upper() for t in ("NPC_", "NPI_")):
                continue
            on = r["stim_onset"] if "stim_onset" in ev and pd.notna(r["stim_onset"]) else r["onset"]
            dur = r["stim_dur"] if "stim_dur" in ev and pd.notna(r["stim_dur"]) else len(stim[k]["wav"]) / SR
            s, n = int(round(float(on) * EEG_FS)), int(round(float(dur) * EEG_FS))
            if s + n <= x.shape[1]:
                trials.append((k, torch.from_numpy(x[:, s:s + n]).half()))
        torch.save(dict(sub=sub, trials=trials), out / "eeg" / f"sub-{sub:02d}.pt")
        print(f"sub-{sub:02d}: {len(trials)} trials")


# ----------------------------------------------------------------------------- data
class N400(Dataset):
    def __init__(self, items, stim):
        self.items, self.stim = items, stim

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        sub, k, eeg = self.items[i]
        s = self.stim[k]
        return dict(eeg=eeg.float(), wav=s["wav"].float(), phon=s["phon"], clip=s["clip"],
                    cong=s["congruent"], key=f"sub-{sub:02d}_{k}")


def collate(b):
    el = torch.tensor([x["eeg"].shape[1] for x in b])
    T = (int(el.max()) + 2) // 3 * 3
    eeg = torch.stack([F.pad(x["eeg"], (0, T - x["eeg"].shape[1])) for x in b])
    wl = torch.tensor([len(x["wav"]) for x in b])
    wav = torch.stack([F.pad(x["wav"], (0, int(wl.max()) - len(x["wav"]))) for x in b])
    pl = torch.tensor([len(x["phon"]) for x in b])
    phon = torch.stack([F.pad(x["phon"], (0, int(pl.max()) - len(x["phon"]))) for x in b])
    return dict(eeg=eeg, eeg_len=el, wav=wav, wav_len=wl, phon=phon, phon_len=pl,
                clip=torch.stack([x["clip"] for x in b]), cong=torch.tensor([x["cong"] for x in b]),
                key=[x["key"] for x in b])


def splits(data: Path):
    stim = torch.load(data / "stimuli.pt")
    keys = sorted(stim)
    test_sent = set(random.Random(0).sample(keys, N_TEST_SENT))
    out = {s: [] for s in ("train", "audio", "subject", "both")}
    for f in sorted((data / "eeg").glob("sub-*.pt")):
        d = torch.load(f)
        if d["sub"] in EXCLUDED_SUBJ:
            continue
        for k, x in d["trials"]:
            held_s, held_k = d["sub"] in TEST_SUBJ, k in test_sent
            name = {(0, 0): "train", (0, 1): "audio", (1, 0): "subject", (1, 1): "both"}[(held_s, held_k)]
            out[name].append((d["sub"], k, x))
    return {k: N400(v, stim) for k, v in out.items()}


# ----------------------------------------------------------------------------- audio features
_FB = {}


def spec_fn(y):  # VITS spectrogram_torch, y [B,N] -> [B,513,N/HOP]
    y = F.pad(y[:, None], ((NFFT - HOP) // 2,) * 2, mode="reflect")[:, 0]
    s = torch.stft(y, NFFT, HOP, NFFT, window=torch.hann_window(NFFT, device=y.device), center=False,
                   return_complex=True)
    return (s.abs() ** 2 + 1e-6).sqrt()


def mel_fn(spec):
    if spec.device not in _FB:
        _FB[spec.device] = torchaudio.functional.melscale_fbanks(
            NFFT // 2 + 1, 0.0, SR / 2, NMEL, SR, norm="slaney", mel_scale="slaney").T.to(spec.device)
    return torch.log(torch.clamp(_FB[spec.device] @ spec, min=1e-5))


# ----------------------------------------------------------------------------- EEG module (3.1, A.2)
class SpatialGNN(nn.Module):
    def __init__(self, A, dn=64):
        super().__init__()
        self.register_buffer("A", A)  # fixed, never trained
        self.inp = nn.Sequential(nn.Linear(1, dn), nn.GELU())
        self.f = nn.ModuleList(nn.Sequential(nn.Linear(dn, dn), nn.LayerNorm(dn), nn.GELU(), nn.Dropout(0.1))
                               for _ in range(2))
        self.out = nn.Linear(dn, D)

    def forward(self, x):  # [B,C,T] -> [B,D,T]
        h = self.inp(x.transpose(1, 2).unsqueeze(-1))                 # [B,T,C,dn]
        for f in self.f:
            h = h + f(torch.einsum("ij,btjd->btid", self.A, h))      # residual graph conv
        return self.out(h.mean(2)).transpose(1, 2)


class S4Block(nn.Module):
    """Bidirectional diagonal S4 (S4D-Lin kernel), FFT convolution, GELU + GLU output, pre-norm residual."""

    def __init__(self, d=D, n=64, p=0.1):
        super().__init__()
        H, N = 2 * d, n // 2
        self.log_dt = nn.Parameter(torch.rand(H) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        self.log_A_re = nn.Parameter(torch.log(0.5 * torch.ones(H, N)))
        self.A_im = nn.Parameter(math.pi * torch.arange(N).float().repeat(H, 1))
        self.C = nn.Parameter(torch.randn(H, N, 2) * 0.5 ** 0.5)
        self.Dskip = nn.Parameter(torch.randn(d))
        self.norm, self.drop = nn.LayerNorm(d), nn.Dropout(p)
        self.out = nn.Sequential(nn.Conv1d(d, 2 * d, 1), nn.GLU(dim=1))

    def kernel(self, L):
        dt = self.log_dt.exp()[:, None]
        A = -self.log_A_re.exp() + 1j * self.A_im
        C = torch.view_as_complex(self.C) * ((dt * A).exp() - 1) / A
        V = torch.exp((dt * A)[..., None] * torch.arange(L, device=A.device))
        return 2 * torch.einsum("hn,hnl->hl", C, V).real

    def forward(self, x):  # [B,d,L]
        Lx = x.size(-1)
        z = self.norm(x.transpose(1, 2)).transpose(1, 2)
        kf, kb = self.kernel(Lx).chunk(2, 0)
        k = F.pad(kf, (0, Lx)) + F.pad(kb.flip(-1), (Lx, 0))
        y = torch.fft.irfft(torch.fft.rfft(z, n=2 * Lx) * torch.fft.rfft(k, n=2 * Lx), n=2 * Lx)[..., :Lx]
        y = self.out(self.drop(F.gelu(y + z * self.Dskip[:, None])))
        return x + self.drop(y)


class EEGModule(nn.Module):
    def __init__(self, A, aud, C=128):
        super().__init__()
        w = torch.zeros(C)
        w[aud] = 2.0                                                  # auditory warm start
        self.w = nn.Parameter(w)
        self.gnn = SpatialGNN(A)
        self.cnn = nn.Sequential(nn.Conv1d(C, D, 4, 1, 3), nn.GELU(), nn.Conv1d(D, D, 4, 1, 3), nn.GELU())
        self.g = nn.Parameter(torch.tensor(-2.0))
        self.down = nn.Conv1d(D, D, 3, 3, 1)
        self.s4 = nn.ModuleList(S4Block() for _ in range(4))
        self.dec = nn.Sequential(nn.ConvTranspose1d(D, D, 3, 3, 0), nn.GELU(),          # training only
                                 nn.ConvTranspose1d(D, D, 4, 1, 3), nn.GELU(), nn.ConvTranspose1d(D, C, 4, 1, 3))

    def forward(self, x, lens):  # x [B,C,T], T % 3 == 0
        T = x.size(-1)
        mask = seq_mask(lens, T)[:, None].float()
        xa = x * torch.sigmoid(self.w)[None, :, None]
        hg, hc = self.gnn(xa), self.cnn(xa)[..., :T]
        hc = hc * hg.std((1, 2), keepdim=True) / (hc.std((1, 2), keepdim=True) + 1e-5)
        h = (hg + torch.sigmoid(self.g) * hc) * mask
        lens3 = (lens + 2) // 3
        m3 = seq_mask(lens3, T // 3)[:, None].float()
        M = self.down(h) * m3
        for b in self.s4:
            M = b(M) * m3
        xh = self.dec(M)
        return M, lens3, F.pad(xh, (0, T - xh.size(-1)))


# ----------------------------------------------------------------------------- phoneme predictor (3.2, A.3)
class AttnDecoder(nn.Module):
    """LSTM decoder with location-based attention; fully autoregressive (teacher forcing 0)."""

    def __init__(self, V, d=D):
        super().__init__()
        self.emb, self.lstm = nn.Embedding(V, d), nn.LSTMCell(2 * d, d)
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, 1)
        self.loc, self.loc_p = nn.Conv1d(1, 32, 31, padding=15), nn.Linear(32, d)
        self.out = nn.Linear(2 * d, V)

    def forward(self, enc, mask, steps):
        B, T, d = enc.shape
        h = c = ctx = enc.new_zeros(B, d)
        a = enc.new_zeros(B, T)
        y = torch.full((B,), SOS, device=enc.device)
        keys, outs = self.k(enc), []
        for _ in range(steps):
            h, c = self.lstm(torch.cat([self.emb(y), ctx], -1), (h, c))
            e = self.v(torch.tanh(keys + self.q(h)[:, None] + self.loc_p(self.loc(a[:, None]).transpose(1, 2))))
            a = e.squeeze(-1).masked_fill(~mask, -1e4).softmax(-1)
            ctx = torch.bmm(a[:, None], enc)[:, 0]
            logit = self.out(torch.cat([h, ctx], -1))
            outs.append(logit)
            y = logit.argmax(-1)
        return torch.stack(outs)  # [steps,B,V]


class PhonemePredictor(nn.Module):
    def __init__(self, V=len(VOCAB)):
        super().__init__()
        self.conf = torchaudio.models.Conformer(input_dim=D, num_heads=8, ffn_dim=4 * D, num_layers=1,
                                                depthwise_conv_kernel_size=31, dropout=0.1)
        self.dec = AttnDecoder(V)

    def forward(self, M, lens, steps):
        h, _ = self.conf(M.transpose(1, 2), lens)
        return self.dec(h, seq_mask(lens, h.size(1)), steps)


# ----------------------------------------------------------------------------- connector (A.2)
class Connector(nn.Module):
    """6-layer self-attention encoder (VITS text-encoder shape) + 1x1 proj to prior mean / log-std.
    NOTE: plain MHA instead of VITS windowed relative-position attention."""

    def __init__(self, n=6, heads=2, ffn=768, k=3, p=0.1):
        super().__init__()
        self.att = nn.ModuleList(nn.MultiheadAttention(D, heads, dropout=p, batch_first=True) for _ in range(n))
        self.f1 = nn.ModuleList(nn.Conv1d(D, ffn, k, padding=k // 2) for _ in range(n))
        self.f2 = nn.ModuleList(nn.Conv1d(ffn, D, k, padding=k // 2) for _ in range(n))
        self.n1 = nn.ModuleList(nn.LayerNorm(D) for _ in range(n))
        self.n2 = nn.ModuleList(nn.LayerNorm(D) for _ in range(n))
        self.drop, self.proj = nn.Dropout(p), nn.Conv1d(D, 2 * D, 1)

    def forward(self, x, mask):  # x [B,D,T], mask [B,T] bool
        m = mask[:, None].float()
        h = (x * m).transpose(1, 2)
        for att, f1, f2, n1, n2 in zip(self.att, self.f1, self.f2, self.n1, self.n2):
            h = n1(h + self.drop(att(h, h, h, key_padding_mask=~mask)[0]))
            f = f2(self.drop(F.relu(f1(h.transpose(1, 2) * m))) * m)
            h = n2(h + self.drop(f.transpose(1, 2)))
        return (self.proj(h.transpose(1, 2)) * m).chunk(2, 1)


# ----------------------------------------------------------------------------- VITS decoder (compact, VITS defaults)
def pad_for(k, d=1):
    return (k * d - d) // 2


class WN(nn.Module):
    def __init__(self, h, k, dil, n, gin=D):
        super().__init__()
        self.h, self.n = h, n
        self.inl = nn.ModuleList(weight_norm(nn.Conv1d(h, 2 * h, k, dilation=dil ** i, padding=pad_for(k, dil ** i)))
                                 for i in range(n))
        self.rs = nn.ModuleList(weight_norm(nn.Conv1d(h, 2 * h if i < n - 1 else h, 1)) for i in range(n))
        self.cond = weight_norm(nn.Conv1d(gin, 2 * h * n, 1))

    def forward(self, x, mask, g):
        out, gc = 0, self.cond(g)
        for i in range(self.n):
            xi = self.inl[i](x) + gc[:, i * 2 * self.h:(i + 1) * 2 * self.h]
            r = self.rs[i](torch.tanh(xi[:, :self.h]) * torch.sigmoid(xi[:, self.h:]))
            if i < self.n - 1:
                x, out = (x + r[:, :self.h]) * mask, out + r[:, self.h:]
            else:
                out = out + r
        return out * mask


class PosteriorEncoder(nn.Module):
    def __init__(self, cin=NFFT // 2 + 1):
        super().__init__()
        self.pre, self.enc, self.proj = nn.Conv1d(cin, D, 1), WN(D, 5, 1, 16), nn.Conv1d(D, 2 * D, 1)

    def forward(self, spec, mask, g):
        m, logs = (self.proj(self.enc(self.pre(spec) * mask, mask, g)) * mask).chunk(2, 1)
        return (m + torch.randn_like(m) * logs.exp()) * mask, m, logs


class Coupling(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre, self.enc, self.post = nn.Conv1d(D // 2, D, 1), WN(D, 5, 1, 4), nn.Conv1d(D, D // 2, 1)
        nn.init.zeros_(self.post.weight)
        nn.init.zeros_(self.post.bias)

    def forward(self, x, mask, g, reverse=False):
        x0, x1 = x.chunk(2, 1)
        m = self.post(self.enc(self.pre(x0) * mask, mask, g)) * mask  # mean-only coupling
        x1 = (x1 - m) * mask if reverse else m + x1 * mask
        return torch.cat([x0, x1], 1)


class Flow(nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.layers = nn.ModuleList(Coupling() for _ in range(n))

    def forward(self, x, mask, g, reverse=False):
        for l in (reversed(self.layers) if reverse else self.layers):
            if reverse:
                x = l(x.flip(1), mask, g, reverse=True)
            else:
                x = l(x, mask, g).flip(1)
        return x


class ResBlock1(nn.Module):
    def __init__(self, ch, k, dil=(1, 3, 5)):
        super().__init__()
        self.c1 = nn.ModuleList(weight_norm(nn.Conv1d(ch, ch, k, 1, padding=pad_for(k, d), dilation=d)) for d in dil)
        self.c2 = nn.ModuleList(weight_norm(nn.Conv1d(ch, ch, k, 1, padding=pad_for(k))) for _ in dil)

    def forward(self, x):
        for a, b in zip(self.c1, self.c2):
            x = x + b(F.leaky_relu(a(F.leaky_relu(x, 0.1)), 0.1))
        return x


class Generator(nn.Module):  # HiFi-GAN v1 as configured in VITS (x256 upsampling)
    def __init__(self, ch=512, rates=(8, 8, 2, 2), ks=(16, 16, 4, 4), rk=(3, 7, 11)):
        super().__init__()
        self.pre, self.cond = nn.Conv1d(D, ch, 7, 1, 3), nn.Conv1d(D, ch, 1)
        self.ups = nn.ModuleList(weight_norm(nn.ConvTranspose1d(ch // 2 ** i, ch // 2 ** (i + 1), k, u, (k - u) // 2))
                                 for i, (u, k) in enumerate(zip(rates, ks)))
        self.res = nn.ModuleList(ResBlock1(ch // 2 ** (i + 1), k) for i in range(len(rates)) for k in rk)
        self.post, self.nk = nn.Conv1d(ch // 2 ** len(rates), 1, 7, 1, 3, bias=False), len(rk)

    def forward(self, x, g):
        x = self.pre(x) + self.cond(g)
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, 0.1))
            x = sum(self.res[i * self.nk + j](x) for j in range(self.nk)) / self.nk
        return torch.tanh(self.post(F.leaky_relu(x)))


class DiscP(nn.Module):
    def __init__(self, p):
        super().__init__()
        ch = [1, 32, 128, 512, 1024, 1024]
        self.p = p
        self.convs = nn.ModuleList(weight_norm(nn.Conv2d(ch[i], ch[i + 1], (5, 1), (3, 1) if i < 4 else 1, (2, 0)))
                                   for i in range(5))
        self.post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, (1, 0)))

    def forward(self, x):
        b, c, t = x.shape
        if t % self.p:
            x = F.pad(x, (0, self.p - t % self.p), "reflect")
        x, fm = x.view(b, c, -1, self.p), []
        for l in self.convs:
            x = F.leaky_relu(l(x), 0.1)
            fm.append(x)
        x = self.post(x)
        return x.flatten(1), fm + [x]


class DiscS(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = [(1, 16, 15, 1, 1, 7), (16, 64, 41, 4, 4, 20), (64, 256, 41, 4, 16, 20), (256, 1024, 41, 4, 64, 20),
               (1024, 1024, 41, 4, 256, 20), (1024, 1024, 5, 1, 1, 2)]
        self.convs = nn.ModuleList(weight_norm(nn.Conv1d(i, o, k, s, groups=g, padding=p)) for i, o, k, s, g, p in cfg)
        self.post = weight_norm(nn.Conv1d(1024, 1, 3, 1, 1))

    def forward(self, x):
        fm = []
        for l in self.convs:
            x = F.leaky_relu(l(x), 0.1)
            fm.append(x)
        x = self.post(x)
        return x.flatten(1), fm + [x]


class MPD(nn.Module):
    def __init__(self):
        super().__init__()
        self.ds = nn.ModuleList([DiscS()] + [DiscP(p) for p in (2, 3, 5, 7, 11)])

    def forward(self, y, yh):
        r, g = [d(y) for d in self.ds], [d(yh) for d in self.ds]
        return [o for o, _ in r], [o for o, _ in g], [f for _, f in r], [f for _, f in g]


# ----------------------------------------------------------------------------- full model
class SENSE(nn.Module):
    def __init__(self, A, aud):
        super().__init__()
        self.eeg, self.phon = EEGModule(A, aud), PhonemePredictor()                     # optimizer 1
        self.conn, self.W_eeg, self.W_clip = Connector(), nn.Linear(D, D), nn.Linear(512, D)  # optimizer 2
        self.null = nn.Parameter(torch.zeros(D))
        self.enc_q, self.flow, self.gen = PosteriorEncoder(), Flow(), Generator()
        self.disc = MPD()                                                                # discriminator optimizer

    def speech_params(self):
        return [p for m in (self.conn, self.W_eeg, self.W_clip, self.enc_q, self.flow, self.gen)
                for p in m.parameters()] + [self.null]

    @staticmethod
    def masked_mean(M, lens):
        m = seq_mask(lens, M.size(-1))[:, None].float()
        return (M * m).sum(-1) / m.sum(-1)

    def prior(self, M, lens3, frames):
        """Connector on sg[M], then stretch each sample's prior to its spectrogram length (NOTE)."""
        m_p, logs_p = self.conn(M.detach(), seq_mask(lens3, M.size(-1)))
        Ty = int(frames.max())
        out = M.new_zeros(2, M.size(0), D, Ty)
        for i in range(M.size(0)):
            st = torch.stack([m_p[i, :, :lens3[i]], logs_p[i, :, :lens3[i]]])
            out[:, i, :, :frames[i]] = F.interpolate(st, size=int(frames[i]), mode="linear", align_corners=False)
        return out[0], out[1]

    @torch.no_grad()
    def infer(self, eeg, eeg_len, noise=0.667):
        M, lens3, _ = self.eeg(eeg, eeg_len)
        g = self.W_eeg(self.masked_mean(M, lens3))[..., None]                    # g_eeg, no text, CFG scale 0
        frames = (lens3 * 3 * SR / EEG_FS / HOP).round().long()
        m_p, logs_p = self.prior(M, lens3, frames)
        ym = seq_mask(frames, m_p.size(-1))[:, None].float()
        z = self.flow((m_p + torch.randn_like(m_p) * logs_p.exp() * noise) * ym, ym, g, reverse=True)
        return self.gen(z * ym, g)[:, 0], frames * HOP


def rand_slice(x, lens, n):
    ids = (torch.rand(x.size(0), device=x.device) * (lens - n + 1).clamp_min(1)).long()
    return torch.stack([x[i, ..., j:j + n] for i, j in enumerate(ids.tolist())]), ids


def kl_loss(z_p, logs_q, m_p, logs_p, mask):
    kl = logs_p - logs_q - 0.5 + 0.5 * (z_p - m_p) ** 2 * torch.exp(-2 * logs_p)
    return (kl * mask).sum() / mask.sum()


def recon_loss(x, xh, lens):  # length-masked cosine reconstruction loss
    m = seq_mask(lens, x.size(-1))[:, None].float()
    num = (x * xh * m).sum((1, 2))
    den = (x * x * m).sum((1, 2)).sqrt() * (xh * xh * m).sum((1, 2)).sqrt() + 1e-8
    return (1 - num / den).mean()


# ----------------------------------------------------------------------------- train
def train(a):
    torch.manual_seed(a.seed), random.seed(a.seed), np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, run = Path(a.data), Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    geo = torch.load(data / "geometry.pt")
    ds = splits(data)
    print({k: len(v) for k, v in ds.items()})
    dl = DataLoader(ds["train"], a.batch, shuffle=True, collate_fn=collate, num_workers=4, drop_last=True)

    model = SENSE(geo["A"], geo["aud"]).to(dev)
    kw = dict(lr=LR, betas=BETAS, weight_decay=WD)
    opt_e = torch.optim.AdamW(list(model.eeg.parameters()) + list(model.phon.parameters()), **kw)
    opt_g = torch.optim.AdamW(model.speech_params(), **kw)
    opt_d = torch.optim.AdamW(model.disc.parameters(), **kw)
    scheds = [torch.optim.lr_scheduler.ExponentialLR(o, GAMMA) for o in (opt_e, opt_g, opt_d)]
    step, ck = 0, run / "ckpt.pt"
    if ck.exists():
        s = torch.load(ck, map_location=dev)
        model.load_state_dict(s["model"])
        for o, so in zip((opt_e, opt_g, opt_d, *scheds), s["opt"]):
            o.load_state_dict(so)
        step = s["step"]
        print(f"resumed at step {step}")

    model.train()
    while step < a.steps:
        for b in dl:
            b = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in b.items()}
            # ---- EEG module + auxiliary objectives
            M, lens3, xh = model.eeg(b["eeg"], b["eeg_len"])
            l_recon = recon_loss(b["eeg"], xh, b["eeg_len"])
            steps = int(b["phon_len"].max())
            logp = model.phon(M, lens3, steps).log_softmax(-1)
            l_ctc = F.ctc_loss(logp, b["phon"], b["phon_len"], b["phon_len"], blank=0, zero_infinity=True)
            mbar = model.masked_mean(M, lens3)
            g_clip = model.W_clip(b["clip"])
            c = b["cong"]
            l_esc = (1 - F.cosine_similarity(model.W_eeg(mbar)[c], g_clip.detach()[c], -1)).mean() if c.any() \
                else M.new_zeros(())
            # ---- conditioning bridge
            if random.random() < P_EEG:
                g = model.W_eeg(mbar.detach()).detach()          # gradients blocked at M and W_eeg
            else:
                keep = torch.rand(len(c), 1, device=dev) < 0.5   # CFG dropout: text vs null
                g = torch.where(keep, g_clip, model.null.expand_as(g_clip))
            g = g[..., None]
            # ---- VITS
            frames = b["wav_len"] // HOP
            spec = spec_fn(b["wav"])[..., :int(frames.max())]
            ym = seq_mask(frames, spec.size(-1))[:, None].float()
            z, m_q, logs_q = model.enc_q(spec, ym, g)
            z_p = model.flow(z, ym, g)
            m_p, logs_p = model.prior(M, lens3, frames)
            zs, ids = rand_slice(z, frames, SEG_FRAMES)
            y_hat = model.gen(zs, g)
            y = torch.stack([b["wav"][i, j * HOP:(j + SEG_FRAMES) * HOP] for i, j in enumerate(ids.tolist())])[:, None]
            # discriminator step
            r, f, _, _ = model.disc(y, y_hat.detach())
            l_d = sum(((1 - dr) ** 2).mean() + (dg ** 2).mean() for dr, dg in zip(r, f))
            opt_d.zero_grad()
            l_d.backward()
            nn.utils.clip_grad_norm_(model.disc.parameters(), 1e3)
            opt_d.step()
            # generator / encoder step
            _, f, fm_r, fm_g = model.disc(y, y_hat)
            l_adv = sum(((1 - dg) ** 2).mean() for dg in f)
            l_fm = 2 * sum(F.l1_loss(gl, rl.detach()) for fr, fg in zip(fm_r, fm_g) for rl, gl in zip(fr, fg))
            mel_full = mel_fn(spec)
            mel_t = torch.stack([mel_full[i, :, j:j + SEG_FRAMES] for i, j in enumerate(ids.tolist())])
            l_mel = F.l1_loss(mel_fn(spec_fn(y_hat[:, 0])), mel_t)
            l_kl = kl_loss(z_p, logs_q, m_p, logs_p, ym)
            loss = (L["mel"] * l_mel + l_adv + l_fm + L["kl"] * l_kl
                    + L["recon"] * l_recon + L["ctc"] * l_ctc + L["esc"] * l_esc)
            opt_e.zero_grad()
            opt_g.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1e3)
            opt_e.step()
            opt_g.step()
            step += 1
            if step % 100 == 0:
                print(f"step {step} | mel {l_mel:.3f} kl {l_kl:.3f} adv {l_adv:.3f} fm {l_fm:.3f} d {l_d:.3f} "
                      f"| recon {l_recon:.3f} ctc {l_ctc:.3f} esc {l_esc:.3f}", flush=True)
            if step % 5000 == 0 or step >= a.steps:
                torch.save(dict(model=model.state_dict(), step=step,
                                opt=[o.state_dict() for o in (opt_e, opt_g, opt_d, *scheds)]), ck)
            if step >= a.steps:
                break
        for s in scheds:
            s.step()  # exponential decay per epoch


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(a):
    import soundfile as sf
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, run = Path(a.data), Path(a.run)
    geo = torch.load(data / "geometry.pt")
    model = SENSE(geo["A"], geo["aud"]).to(dev).eval()
    model.load_state_dict(torch.load(run / "ckpt.pt", map_location=dev)["model"])
    try:
        from pystoi import stoi
    except ImportError:
        stoi = None
    res, ds = {}, splits(data)
    for name in ("audio", "subject", "both"):
        out = run / "samples" / name
        out.mkdir(parents=True, exist_ok=True)
        mc, st = [], []
        for b in DataLoader(ds[name], a.batch, collate_fn=collate):
            wav, n = model.infer(b["eeg"].to(dev), b["eeg_len"].to(dev))
            for i in range(len(wav)):
                yh = wav[i, :int(n[i])].cpu()
                y = b["wav"][i, :int(b["wav_len"][i])]
                k = min(len(y), len(yh)) // HOP * HOP
                mh, mt = mel_fn(spec_fn(yh[None, :k]))[0], mel_fn(spec_fn(y[None, :k]))[0]
                mc.append(float(torch.corrcoef(torch.stack([mh.flatten(), mt.flatten()]))[0, 1]))
                if stoi:
                    r = lambda s: torchaudio.functional.resample(s, SR, 16000).numpy()
                    st.append(stoi(r(y[:k]), r(yh[:k]), 16000))
                sf.write(out / f"{b['key'][i]}.wav", yh.numpy(), SR)
        res[name] = dict(n=len(mc), mel_corr=100 * float(np.mean(mc)), stoi=float(np.mean(st)) if st else None)
        print(name, res[name], flush=True)
    (run / "metrics.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("prepare")
    q.add_argument("--bids", required=True)
    q.add_argument("--stimuli", required=True)
    q.add_argument("--out", default="data/n400_sense")
    for n in ("train", "eval"):
        q = sp.add_parser(n)
        q.add_argument("--data", default="data/n400_sense")
        q.add_argument("--run", default="runs/sense")
        q.add_argument("--batch", type=int, default=16)
        q.add_argument("--steps", type=int, default=200_000)
        q.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    {"prepare": prepare, "train": train, "eval": evaluate}[a.cmd](a)

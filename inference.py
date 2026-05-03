#!/usr/bin/env python3
"""
inference.py — Generate effect predictions for Guitar-TECHS clean excerpts.

Loads the ConditionalUNet (log-spec) and convVAE_deterministic models,
processes multiple Guitar-TECHS direct-input WAV files, and saves
predicted audio for 10 effects per model.

Usage:
    python3 inference.py
"""

import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import soundfile as sf

# ═══════════════════════════════════════════════════════════════
# Constants (must match training)
# ═══════════════════════════════════════════════════════════════

SR = 16000
N_FFT = 512
HOP = 128
EPS = 1e-9
LOG_ALPHA = 200.0
DUR = 4.9  # seconds per chunk

FREQ_BINS = N_FFT // 2 + 1          # 257
TARGET_FRAMES = int((SR * DUR) / HOP) + 1  # 613
N_EFFECTS = 13

# Effect index mapping (alphabetically sorted EGFxSet folder names)
EFFECT_NAMES = [
    "Clean",           # 0
    "bluesDriver",     # 1
    "chorus",          # 2
    "digitalDelay",    # 3
    "flanger",         # 4
    "hallReverb",      # 5
    "phaser",          # 6
    "plateReverb",     # 7
    "rat",             # 8
    "spring-Reverb",   # 9
    "sweepEcho",       # 10
    "tapeEcho",        # 11
    "tubeScreamer",    # 12
]

# Display names for the website
DISPLAY_NAMES = {
    "digitalDelay": "Digital Delay",
    "flanger": "Flanger",
    "hallReverb": "Hall Reverb",
    "phaser": "Phaser",
    "plateReverb": "Plate Reverb",
    "rat": "RAT Distortion",
    "spring-Reverb": "Spring Reverb",
    "sweepEcho": "Sweep Echo",
    "tapeEcho": "Tape Echo",
    "tubeScreamer": "Tube Screamer",
}

# Effects to include (excluding Clean, bluesDriver, chorus)
INCLUDED_EFFECTS = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

# Guitar-TECHS excerpts to use (indices into the 12 files, 1-based)
# Pick 4 diverse excerpts
EXCERPT_INDICES = [1, 3, 7, 10]
EXCERPT_LABELS = {
    1: "Excerpt 1 — Full Band Piece",
    3: "Excerpt 3 — Melodic Passage",
    7: "Excerpt 7 — Rhythmic Section",
    10: "Excerpt 10 — Expressive Solo",
}

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
GUITAR_TECHS_DIR = PROJECT_DIR / "guitartechs" / "P3_music" / "audio" / "directinput"
UNET_CKPT = PROJECT_DIR / "model_checkpoints" / "latest_unet_clean2fx_logspec.pt"
CONVVAE_CKPT = PROJECT_DIR / "model_checkpoints" / "convVAE_clean2fx_final.pt"
AUDIO_OUT = SCRIPT_DIR / "audio"


# ═══════════════════════════════════════════════════════════════
# Model Definitions (copied from notebooks)
# ═══════════════════════════════════════════════════════════════

# --- U-Net ---

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        *,
        n_effects: int,
        cond_dim: int = 64,
        cond_ch: int = 64,
        cond_drop: float = 0.15,
        c1: int = 32,
        c2: int = 64,
        c3: int = 128,
        c4: int = 256,
        c5: int = 512,
    ):
        super().__init__()
        self.cond_drop = cond_drop

        # Encoder
        self.enc1 = DoubleConv(1, c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(c4, c5)
        self.effect_emb = nn.Embedding(n_effects, cond_dim)
        self.cond_proj = nn.Linear(cond_dim, cond_ch)
        self.bottleneck_fuse = DoubleConv(c5 + cond_ch, c5)

        # FiLM layers
        self.film4 = nn.Linear(cond_dim, c4 * 2)
        self.film3 = nn.Linear(cond_dim, c3 * 2)
        self.film2 = nn.Linear(cond_dim, c2 * 2)
        self.film1 = nn.Linear(cond_dim, c1 * 2)
        self.film_skip4 = nn.Linear(cond_dim, c4 * 2)
        self.film_skip3 = nn.Linear(cond_dim, c3 * 2)
        self.film_skip2 = nn.Linear(cond_dim, c2 * 2)
        self.film_skip1 = nn.Linear(cond_dim, c1 * 2)

        # Decoder
        self.dec4 = DoubleConv(c5 + c4, c4)
        self.dec3 = DoubleConv(c4 + c3, c3)
        self.dec2 = DoubleConv(c3 + c2, c2)
        self.dec1 = DoubleConv(c2 + c1, c1)
        self.out_conv = nn.Conv2d(c1, 1, kernel_size=1)

    def _film(self, x, film_layer, cond_vec):
        gamma_beta = film_layer(cond_vec)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta

    def forward(self, x):
        specA, effB = x

        if specA.dim() != 3:
            raise ValueError(f"Expected specA shape (B, F, T), got {tuple(specA.shape)}")

        x0 = specA.unsqueeze(1)

        # Encoder
        e1 = self.enc1(x0)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        cond_vec = self.effect_emb(effB.long())

        if self.training and self.cond_drop > 0:
            mask = (torch.rand(cond_vec.shape[0], 1, device=cond_vec.device) > self.cond_drop).float()
            cond_vec = cond_vec * mask

        cond_spatial = self.cond_proj(cond_vec)
        cond_spatial = cond_spatial.unsqueeze(-1).unsqueeze(-1)
        cond_spatial = cond_spatial.expand(-1, -1, b.shape[-2], b.shape[-1])
        b = torch.cat([b, cond_spatial], dim=1)
        b = self.bottleneck_fuse(b)

        # Decoder with FiLM + gated skips
        e4 = self._film(e4, self.film_skip4, cond_vec)
        d4 = F.interpolate(b, size=e4.shape[-2:], mode="nearest")
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        d4 = self._film(d4, self.film4, cond_vec)

        e3 = self._film(e3, self.film_skip3, cond_vec)
        d3 = F.interpolate(d4, size=e3.shape[-2:], mode="nearest")
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d3 = self._film(d3, self.film3, cond_vec)

        e2 = self._film(e2, self.film_skip2, cond_vec)
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="nearest")
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d2 = self._film(d2, self.film2, cond_vec)

        e1 = self._film(e1, self.film_skip1, cond_vec)
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="nearest")
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        d1 = self._film(d1, self.film1, cond_vec)

        delta = self.out_conv(d1).squeeze(1)
        out = F.softplus(specA + delta, beta=1)
        return out


# --- ConvVAE ---

def enc_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

def dec_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class convVAE_deterministic(nn.Module):
    def __init__(
        self,
        *,
        freq_bins: int,
        target_frames: int,
        z_dim: int,
        n_effects: int,
        cond_dim: int = 4,
        c1: int = 32,
        c2: int = 64,
        c3: int = 128,
        c4: int = 256,
        c5: int = 512,
    ):
        super().__init__()

        self.freq_bins = int(freq_bins)
        self.target_frames = int(target_frames)
        self.z_dim = int(z_dim)
        self.n_effects = int(n_effects)
        self.cond_dim = int(cond_dim)

        self.effect_emb = nn.Embedding(self.n_effects, self.cond_dim)

        self.enc_blocks = nn.ModuleList([
            enc_block(1, c1),
            enc_block(c1, c2),
            enc_block(c2, c3),
            enc_block(c3, c4),
            enc_block(c4, c5),
        ])

        with torch.no_grad():
            h = torch.zeros(1, 1, self.freq_bins, self.target_frames)
            enc_hw = []
            for blk in self.enc_blocks:
                h = blk(h)
                enc_hw.append((int(h.shape[-2]), int(h.shape[-1])))

            self._enc_C = int(h.shape[1])
            self._enc_H = int(h.shape[2])
            self._enc_W = int(h.shape[3])
            self._dec_targets = enc_hw[-2::-1] + [(self.freq_bins, self.target_frames)]
            enc_flat_dim = self._enc_C * self._enc_H * self._enc_W

        self.fc_mu = nn.Linear(enc_flat_dim, self.z_dim)
        self.fc_logvar = nn.Linear(enc_flat_dim, self.z_dim)
        self.fc_dec = nn.Linear(self.z_dim + self.cond_dim, enc_flat_dim)

        self.dec_blocks = nn.ModuleList([
            dec_block(c5, c4),
            dec_block(c4, c3),
            dec_block(c3, c2),
            dec_block(c2, c1),
            dec_block(c1, c1),
        ])
        self.out_conv = nn.Conv2d(c1, 1, kernel_size=3, padding=1)

    def encoder(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected x as (B, F, T), got shape {tuple(x.shape)}")
        h = x.unsqueeze(1)
        for blk in self.enc_blocks:
            h = blk(h)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        log_var = log_var.clamp(min=-10.0, max=10.0)
        return mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder(self, z, effB):
        cond = self.effect_emb(effB.long())
        h = torch.cat([z, cond], dim=1)
        h = self.fc_dec(h)
        h = h.view(-1, self._enc_C, self._enc_H, self._enc_W)
        for target_hw, blk in zip(self._dec_targets, self.dec_blocks):
            h = F.interpolate(h, size=target_hw, mode="nearest")
            h = blk(h)
        h = self.out_conv(h)
        h = F.relu(h)
        return h.squeeze(1)

    def forward(self, x, deterministic_latent=False):
        specA, effB = x
        mu, log_var = self.encoder(specA)
        if deterministic_latent:
            z = mu
        else:
            z = self.sampling(mu, log_var)
        recon = self.decoder(z, effB)
        return recon, mu, log_var


# ═══════════════════════════════════════════════════════════════
# Audio Processing Utilities
# ═══════════════════════════════════════════════════════════════

def load_and_resample(wav_path: Path) -> np.ndarray:
    """Load a WAV file, convert to mono, resample to SR."""
    y, orig_sr = librosa.load(str(wav_path), sr=SR, mono=True)
    return y.astype(np.float32)


def audio_to_spec_linear(y: np.ndarray) -> torch.Tensor:
    """Audio → normalized linear magnitude spectrogram (F, T)."""
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    norm_max = np.max(S) + EPS
    S = S / norm_max

    if S.shape[1] < TARGET_FRAMES:
        pad = np.zeros((S.shape[0], TARGET_FRAMES - S.shape[1]))
        S = np.concatenate([S, pad], axis=1)
    else:
        S = S[:, :TARGET_FRAMES]

    return torch.from_numpy(S.astype(np.float32)), norm_max


def spec_to_audio(spec: np.ndarray, n_iter: int = 64) -> np.ndarray:
    """Magnitude spectrogram → audio via Griffin-Lim."""
    y = librosa.griffinlim(spec, n_fft=N_FFT, hop_length=HOP, n_iter=n_iter)
    # Normalize
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / (peak + 1e-9)
    return y.astype(np.float32)


def linear_to_log(spec: torch.Tensor) -> torch.Tensor:
    """Linear spec → log-compressed spec (as used by U-Net training)."""
    return torch.log1p(LOG_ALPHA * spec)


def log_to_linear(spec: torch.Tensor) -> torch.Tensor:
    """Log-compressed spec → linear spec."""
    return torch.clamp((torch.exp(spec) - 1.0) / LOG_ALPHA, min=0.0)


def save_wav(y: np.ndarray, path: Path):
    """Save audio as 16kHz mono WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, SR)


# ═══════════════════════════════════════════════════════════════
# Model Loading
# ═══════════════════════════════════════════════════════════════

def load_unet(ckpt_path: Path, device: torch.device) -> ConditionalUNet:
    model = ConditionalUNet(
        n_effects=N_EFFECTS,
        cond_dim=64,
        cond_ch=64,
        cond_drop=0.15,
        c1=32, c2=64, c3=128, c4=256, c5=512,
    ).to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(f"  ✅ U-Net loaded (epoch {ckpt.get('epoch', '?')})")
    return model


def load_convvae(ckpt_path: Path, device: torch.device) -> convVAE_deterministic:
    model = convVAE_deterministic(
        freq_bins=FREQ_BINS,
        target_frames=TARGET_FRAMES,
        z_dim=512,
        n_effects=N_EFFECTS,
    ).to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(f"  ✅ ConvVAE loaded (epoch {ckpt.get('epoch', '?')})")
    return model


# ═══════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_unet(model, spec_lin: torch.Tensor, eff_idx: int, device: torch.device):
    """Run U-Net inference. Returns (audio, pred_spec_linear)."""
    spec_log = linear_to_log(spec_lin).unsqueeze(0).to(device)
    effB = torch.tensor([eff_idx], device=device)

    pred_log = model((spec_log, effB))
    pred_lin = log_to_linear(pred_log.squeeze(0).cpu())
    pred_np = pred_lin.numpy()
    return spec_to_audio(pred_np), pred_np


@torch.no_grad()
def predict_convvae(model, spec_lin: torch.Tensor, eff_idx: int, device: torch.device):
    """Run ConvVAE inference. Returns (audio, pred_spec_linear)."""
    spec_in = spec_lin.unsqueeze(0).to(device)
    effB = torch.tensor([eff_idx], device=device)

    pred, _, _ = model((spec_in, effB), deterministic_latent=True)
    pred_np = torch.clamp(pred.squeeze(0).cpu(), min=0.0).numpy()
    return spec_to_audio(pred_np), pred_np


def save_spectrogram(spec_np: np.ndarray, path: Path, title: str = ""):
    """Save a magnitude spectrogram as a dark-themed PNG image."""
    path.parent.mkdir(parents=True, exist_ok=True)

    spec_db = 20 * np.log10(np.maximum(spec_np, 1e-8))

    fig, ax = plt.subplots(1, 1, figsize=(6, 2.4), dpi=150)
    fig.patch.set_facecolor("#12121a")
    ax.set_facecolor("#12121a")

    img = ax.imshow(
        spec_db,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest",
    )

    # Axis labels with proper units
    n_freq, n_time = spec_np.shape
    freq_ticks = np.linspace(0, n_freq - 1, 5)
    freq_labels = [f"{int(t * SR / N_FFT)}" for t in freq_ticks]
    ax.set_yticks(freq_ticks)
    ax.set_yticklabels(freq_labels, fontsize=7, color="#9898b0")
    ax.set_ylabel("Hz", fontsize=7, color="#9898b0", labelpad=2)

    time_ticks = np.linspace(0, n_time - 1, 5)
    time_labels = [f"{t * HOP / SR:.1f}" for t in time_ticks]
    ax.set_xticks(time_ticks)
    ax.set_xticklabels(time_labels, fontsize=7, color="#9898b0")
    ax.set_xlabel("Time (s)", fontsize=7, color="#9898b0", labelpad=2)

    if title:
        ax.set_title(title, fontsize=8, color="#e8e8f0", pad=4)

    ax.tick_params(axis="both", colors="#5c5c78", length=2, width=0.5)
    for spine in ax.spines.values():
        spine.set_color("#2a2a40")
        spine.set_linewidth(0.5)

    fig.tight_layout(pad=0.5)
    fig.savefig(str(path), facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def pick_chunk(y: np.ndarray) -> np.ndarray:
    """Pick a representative chunk from a longer audio signal.
    Chooses the chunk with the highest RMS energy (most musical content).
    """
    chunk_len = int(SR * DUR)
    if len(y) <= chunk_len:
        # Pad if shorter
        return np.pad(y, (0, max(0, chunk_len - len(y))))

    # Slide through and pick highest-energy chunk
    best_start = 0
    best_rms = 0.0
    step = chunk_len // 4  # 25% overlap

    for start in range(0, len(y) - chunk_len, step):
        chunk = y[start:start + chunk_len]
        rms = np.sqrt(np.mean(chunk ** 2))
        if rms > best_rms:
            best_rms = rms
            best_start = start

    return y[best_start:best_start + chunk_len]


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # Verify paths
    if not GUITAR_TECHS_DIR.exists():
        print(f"❌ Guitar-TECHS directory not found: {GUITAR_TECHS_DIR}")
        sys.exit(1)
    if not UNET_CKPT.exists():
        print(f"❌ U-Net checkpoint not found: {UNET_CKPT}")
        sys.exit(1)
    if not CONVVAE_CKPT.exists():
        print(f"❌ ConvVAE checkpoint not found: {CONVVAE_CKPT}")
        sys.exit(1)

    # Load models
    print("Loading models...")
    unet = load_unet(UNET_CKPT, device)
    convvae = load_convvae(CONVVAE_CKPT, device)
    print()

    # Process excerpts
    manifest = {"excerpts": [], "effects": [], "models": ["unet", "convvae"]}

    # Build effects list for manifest
    for eff_idx in sorted(INCLUDED_EFFECTS):
        name = EFFECT_NAMES[eff_idx]
        manifest["effects"].append({
            "index": eff_idx,
            "name": name,
            "displayName": DISPLAY_NAMES[name],
        })

    for excerpt_num in EXCERPT_INDICES:
        wav_name = f"directinput_{excerpt_num:02d}.wav"
        wav_path = GUITAR_TECHS_DIR / wav_name

        if not wav_path.exists():
            print(f"⚠️  Skipping {wav_name} — not found")
            continue

        excerpt_id = f"excerpt_{excerpt_num:02d}"
        label = EXCERPT_LABELS.get(excerpt_num, f"Excerpt {excerpt_num}")

        print(f"Processing {wav_name} ({label})...")

        # Load and pick best chunk
        y_full = load_and_resample(wav_path)
        y_chunk = pick_chunk(y_full)

        # Save clean input
        clean_path = AUDIO_OUT / excerpt_id / "clean.wav"
        save_wav(y_chunk, clean_path)
        print(f"  💾 Saved clean input ({DUR}s chunk)")

        # Compute linear spectrogram
        spec_lin, norm_max = audio_to_spec_linear(y_chunk)

        # Save clean input spectrogram
        clean_spec_path = AUDIO_OUT / excerpt_id / "specs" / "clean.png"
        save_spectrogram(spec_lin.numpy(), clean_spec_path, title="Clean Input")
        print(f"  📊 Saved clean spectrogram")

        excerpt_manifest = {
            "id": excerpt_id,
            "label": label,
            "cleanAudio": f"audio/{excerpt_id}/clean.wav",
            "cleanSpec": f"audio/{excerpt_id}/specs/clean.png",
            "predictions": {},
        }

        # Run predictions for each included effect
        for eff_idx in sorted(INCLUDED_EFFECTS):
            eff_name = EFFECT_NAMES[eff_idx]
            display_name = DISPLAY_NAMES[eff_name]

            # U-Net prediction
            print(f"  🔮 U-Net → {display_name}...", end="", flush=True)
            y_unet, spec_unet = predict_unet(unet, spec_lin, eff_idx, device)
            unet_path = AUDIO_OUT / excerpt_id / "unet" / f"{eff_name}.wav"
            save_wav(y_unet, unet_path)
            unet_spec_path = AUDIO_OUT / excerpt_id / "specs" / f"unet_{eff_name}.png"
            save_spectrogram(spec_unet, unet_spec_path, title=f"U-Net → {display_name}")
            print(" ✓")

            # ConvVAE prediction
            print(f"  🔮 ConvVAE → {display_name}...", end="", flush=True)
            y_convvae, spec_convvae = predict_convvae(convvae, spec_lin, eff_idx, device)
            convvae_path = AUDIO_OUT / excerpt_id / "convvae" / f"{eff_name}.wav"
            save_wav(y_convvae, convvae_path)
            convvae_spec_path = AUDIO_OUT / excerpt_id / "specs" / f"convvae_{eff_name}.png"
            save_spectrogram(spec_convvae, convvae_spec_path, title=f"ConvVAE → {display_name}")
            print(" ✓")

            excerpt_manifest["predictions"][eff_name] = {
                "unet": f"audio/{excerpt_id}/unet/{eff_name}.wav",
                "convvae": f"audio/{excerpt_id}/convvae/{eff_name}.wav",
                "unetSpec": f"audio/{excerpt_id}/specs/unet_{eff_name}.png",
                "convvaeSpec": f"audio/{excerpt_id}/specs/convvae_{eff_name}.png",
            }

        manifest["excerpts"].append(excerpt_manifest)
        print()

    # Save manifest
    manifest_path = SCRIPT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"📋 Saved manifest: {manifest_path}")
    print("✅ Done!")


if __name__ == "__main__":
    main()

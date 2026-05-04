#!/usr/bin/env python3
"""
regenerate_all.py — Regenerate ALL website audio + spectrograms.

For EGFxSet samples: uses AugmentedPairDataV3 (from cache) to render
properly normalised clean/effect pairs. The model receives the exact same
input distribution it was trained on.

For Guitar-TECHS samples: uses the SHARED_MAX_CORRECTION factors to
approximate the shared-max normalisation the models expect.
"""

import json, sys, random, pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import librosa
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CACHE_PATH = PROJECT_DIR / "egfx_cache" / "aug_pair_v3_train_sr16k_dur4p9.pkl"
UNET_CKPT  = PROJECT_DIR / "model_checkpoints" / "latest_unet_clean2fx_logspec.pt"
CONVVAE_CKPT = PROJECT_DIR / "model_checkpoints" / "convVAE_clean2fx_final.pt"
GUITAR_TECHS_DIR = PROJECT_DIR / "guitartechs" / "P3_music" / "audio" / "directinput"
AUDIO_OUT  = SCRIPT_DIR / "audio"
MANIFEST   = SCRIPT_DIR / "manifest.json"

# ── constants (must match training) ────────────────────────────
SR = 16000
N_FFT = 512
HOP = 128
EPS = 1e-9
LOG_ALPHA = 200.0
DUR = 4.9
FREQ_BINS = N_FFT // 2 + 1          # 257
TARGET_FRAMES = int((SR * DUR) / HOP) + 1  # 613
N_EFFECTS = 13
MIN_SOUND = 0.15

EXCLUDED_EFFECTS = {"BluesDriver", "Chorus", "Clean"}

# Shared-max correction for Guitar-TECHS (clean-only) input
SHARED_MAX_CORRECTION = {
    3:  1.028,   # Digital Delay
    4:  1.499,   # Flanger
    5:  1.045,   # Hall Reverb
    6:  1.082,   # Phaser
    7:  1.002,   # Plate Reverb
    8:  1.270,   # RAT
    9:  1.048,   # Spring Reverb
    10: 1.015,   # Sweep Echo
    11: 1.167,   # Tape Echo
    12: 1.162,   # Tube Screamer
}

# Effect display mapping
EFFECT_DISPLAY = {
    "Digital Delay": ("digitalDelay",  "Digital Delay"),
    "Flanger":       ("flanger",       "Flanger"),
    "Hall Reverb":   ("hallReverb",    "Hall Reverb"),
    "Phaser":        ("phaser",        "Phaser"),
    "Plate Reverb":  ("plateReverb",   "Plate Reverb"),
    "RAT":           ("rat",           "RAT Distortion"),
    "Spring Reverb": ("spring-Reverb", "Spring Reverb"),
    "Sweep Echo":    ("sweepEcho",     "Sweep Echo"),
    "TapeEcho":      ("tapeEcho",      "Tape Echo"),
    "TubeScreamer":  ("tubeScreamer",  "Tube Screamer"),
}

# ── import model classes from inference.py ─────────────────────
sys.path.insert(0, str(SCRIPT_DIR))
from inference import (
    ConditionalUNet, convVAE_deterministic,
    load_unet, load_convvae,
    linear_to_log, log_to_linear,
)


# ══════════════════════════════════════════════════════════════
#  AugmentedPairDataV3 passage rendering (from notebook)
# ══════════════════════════════════════════════════════════════

def silence(duration):
    return np.zeros(max(1, int(SR * duration)), dtype=np.float32)

def safe_slice(y, dur):
    want = max(1, int(SR * dur))
    return y[:want] if len(y) >= want else np.pad(y, (0, want - len(y)))


class PassageRenderer:
    """Replicates AugmentedPairDataV3's passage rendering from the cache."""

    def __init__(self, cache_path):
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        self.audio_dict    = payload["audio_dict"]
        self.keys_by_pair  = payload["keys_by_pair"]
        self.eff_pairs     = payload["eff_pairs"]
        self.effects       = payload["effects"]
        self.effect_to_idx = payload["effect_to_idx"]
        self.idx_to_effect = {v: k for k, v in self.effect_to_idx.items()}

        # Filter to Clean -> X pairs only
        self.clean_pairs = [p for p in self.eff_pairs
                            if p[0] == "Clean" and p[1] not in EXCLUDED_EFFECTS]
        print(f"  Loaded cache: {len(self.audio_dict)} notes, "
              f"{len(self.clean_pairs)} Clean->Effect pairs")

    # ── plan / resolve / render: exact copies from the notebook ──

    def _sample_event_count(self, rng):
        p = rng.random()
        if p < 0.10: return 1
        if p < 0.30: return 2
        if p < 0.60: return 3
        if p < 0.80: return 4
        if p < 0.95: return 5
        return 6

    def make_plan(self, rng, keys):
        types  = ["tone", "chord", "melody", "silence"]
        probs  = [0.35, 0.35, 0.25, 0.05]
        n_events = self._sample_event_count(rng)
        plan = []
        for _ in range(n_events):
            et = rng.choices(types, weights=probs, k=1)[0]
            if et == "silence":
                plan.append(("silence", [], rng.uniform(0.10, 0.90)))
            elif et == "tone":
                plan.append(("tone", [rng.choice(keys)], None))
            elif et == "chord":
                n = rng.randint(2, 5)
                plan.append(("chord", [rng.choice(keys) for _ in range(n)], None))
            elif et == "melody":
                n = rng.randint(2, 6)
                ks = [rng.choice(keys) for _ in range(n)]
                raw = np.array([rng.uniform(0.20, 0.80) for _ in range(n)])
                block = rng.uniform(0.4, 2.0)
                plan.append(("melody", ks, ((raw / raw.sum()) * block).tolist()))
        return plan

    def resolve(self, rng, plan):
        T = DUR; out = []; t = 0.0
        for i, (etype, keys, extra) in enumerate(plan):
            remaining = T - t
            if i == len(plan) - 1:
                if etype == "melody":
                    raw = np.array(extra)
                    out.append(("melody", keys, ((raw / raw.sum()) * remaining).tolist()))
                elif etype == "silence":
                    out.append(("silence", [], remaining))
                else:
                    out.append((etype, keys, max(MIN_SOUND, remaining)))
                break
            if remaining < MIN_SOUND:
                out.append(("silence", [], remaining)); break
            if etype == "silence":
                dur = min(extra, max(MIN_SOUND, remaining - MIN_SOUND))
                out.append(("silence", [], dur)); t += dur
            elif etype == "tone":
                dur = rng.uniform(MIN_SOUND, min(1.0, remaining - MIN_SOUND))
                out.append(("tone", keys, dur)); t += dur
            elif etype == "chord":
                dur = rng.uniform(MIN_SOUND, min(1.2, remaining - MIN_SOUND))
                out.append(("chord", keys, dur)); t += dur
            elif etype == "melody":
                raw = np.array(extra)
                block = min(2.0, remaining - MIN_SOUND)
                if block < 0.2: block = remaining
                out.append(("melody", keys, ((raw / raw.sum()) * block).tolist()))
                t += block
        return out

    def render(self, effect, timeline):
        parts = []
        for etype, keys, extra in timeline:
            if etype == "silence":
                parts.append(silence(extra)); continue
            if etype == "tone":
                y = self.audio_dict[keys[0]][effect]
                parts.append(safe_slice(y, extra)); continue
            if etype == "chord":
                wavs = [self.audio_dict[k][effect] for k in keys]
                L = max(1, int(SR * extra))
                stacked = []
                for w in wavs:
                    if len(w) >= L: stacked.append(w[:L])
                    else: stacked.append(np.pad(w, (0, L - len(w))))
                mix = np.sum(np.stack(stacked), axis=0)
                mix /= np.max(np.abs(mix)) + EPS
                parts.append(mix.astype(np.float32)); continue
            if etype == "melody":
                segs = []
                for k, d in zip(keys, extra):
                    y = self.audio_dict[k][effect]
                    segs.append(safe_slice(y, d))
                mel = np.concatenate(segs)
                mel /= np.max(np.abs(mel)) + EPS
                parts.append(mel.astype(np.float32)); continue
        full = np.concatenate(parts)
        want = int(SR * DUR)
        if len(full) > want: full = full[:want]
        else: full = np.pad(full, (0, want - len(full)))
        return full.astype(np.float32)

    def generate_sample(self, seed, target_effect):
        """Generate a Clean/Effect pair exactly like __getitem__."""
        rng = random.Random(seed)
        keys = self.keys_by_pair[("Clean", target_effect)]
        raw_plan = self.make_plan(rng, keys)
        timeline = self.resolve(rng, raw_plan)

        yA = self.render("Clean", timeline)
        yB = self.render(target_effect, timeline)

        # shared-max normalisation (EXACTLY like __getitem__)
        SA_raw = np.abs(librosa.stft(yA, n_fft=N_FFT, hop_length=HOP))
        SB_raw = np.abs(librosa.stft(yB, n_fft=N_FFT, hop_length=HOP))
        shared_max = max(SA_raw.max(), SB_raw.max()) + EPS

        A_lin = self._to_spec(SA_raw, shared_max)
        return yA, A_lin, shared_max

    def _to_spec(self, S_raw, norm_max):
        S = S_raw / norm_max
        if S.shape[1] < TARGET_FRAMES:
            pad = np.zeros((S.shape[0], TARGET_FRAMES - S.shape[1]))
            S = np.concatenate([S, pad], axis=1)
        else:
            S = S[:, :TARGET_FRAMES]
        return torch.from_numpy(S.astype(np.float32))


# ══════════════════════════════════════════════════════════════
#  Helper functions
# ══════════════════════════════════════════════════════════════

def spec_to_audio(spec_np, n_iter=64):
    y = librosa.griffinlim(spec_np, n_fft=N_FFT, hop_length=HOP, n_iter=n_iter)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y.astype(np.float32)


def audio_to_spec(y):
    """Convert audio to normalised linear spectrogram tensor."""
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    norm_max = np.max(S) + EPS
    S = S / norm_max
    if S.shape[1] < TARGET_FRAMES:
        pad = np.zeros((S.shape[0], TARGET_FRAMES - S.shape[1]))
        S = np.concatenate([S, pad], axis=1)
    else:
        S = S[:, :TARGET_FRAMES]
    return torch.from_numpy(S.astype(np.float32)), norm_max


def save_wav(y, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, SR)


def save_spectrogram(spec_np, path, title=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    spec_db = 20 * np.log10(np.maximum(spec_np, 1e-8))
    fig, ax = plt.subplots(1, 1, figsize=(6, 2.4), dpi=150)
    fig.patch.set_facecolor("#12121a")
    ax.set_facecolor("#12121a")
    ax.imshow(spec_db, aspect="auto", origin="lower", cmap="magma",
              interpolation="nearest", vmin=-80, vmax=0)

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
        spine.set_color("#2a2a40"); spine.set_linewidth(0.5)
    fig.tight_layout(pad=0.5)
    fig.savefig(str(path), facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def pick_chunk(y, sr=SR, dur=DUR):
    chunk_len = int(sr * dur)
    if len(y) <= chunk_len:
        return np.pad(y, (0, max(0, chunk_len - len(y))))
    best_start, best_rms = 0, 0.0
    step = chunk_len // 4
    for start in range(0, len(y) - chunk_len, step):
        rms = np.sqrt(np.mean(y[start:start + chunk_len] ** 2))
        if rms > best_rms:
            best_rms = rms; best_start = start
    return y[best_start:best_start + chunk_len]


# ══════════════════════════════════════════════════════════════
#  Predict functions
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def predict_unet(model, spec_lin, eff_idx, device, apply_correction=False):
    if apply_correction:
        correction = SHARED_MAX_CORRECTION.get(eff_idx, 1.0)
        spec_lin = spec_lin / correction

    spec_log = linear_to_log(spec_lin).unsqueeze(0).to(device)
    effB = torch.tensor([eff_idx], device=device)
    pred_log = model((spec_log, effB))
    pred_lin = log_to_linear(pred_log.squeeze(0).cpu())
    pred_np = pred_lin.numpy()
    return spec_to_audio(pred_np), pred_np


@torch.no_grad()
def predict_convvae(model, spec_lin, eff_idx, device, apply_correction=False):
    if apply_correction:
        correction = SHARED_MAX_CORRECTION.get(eff_idx, 1.0)
        spec_lin = spec_lin / correction

    spec_in = spec_lin.unsqueeze(0).to(device)
    effB = torch.tensor([eff_idx], device=device)
    pred, _, _ = model((spec_in, effB), deterministic_latent=True)
    pred_np = torch.clamp(pred.squeeze(0).cpu(), min=0.0).numpy()
    return spec_to_audio(pred_np), pred_np


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def main():
    device = torch.device("cpu")
    print("Loading models...")
    unet = load_unet(UNET_CKPT, device)
    convvae = load_convvae(CONVVAE_CKPT, device)

    print("Loading AugmentedPairDataV3 cache...")
    renderer = PassageRenderer(CACHE_PATH)

    target_effects = sorted(
        [(n, i) for n, i in renderer.effect_to_idx.items() if n not in EXCLUDED_EFFECTS],
        key=lambda x: x[1],
    )

    manifest = {
        "excerpts": [],
        "effects": [
            {"index": 3,  "name": "digitalDelay",  "displayName": "Digital Delay"},
            {"index": 4,  "name": "flanger",        "displayName": "Flanger"},
            {"index": 5,  "name": "hallReverb",     "displayName": "Hall Reverb"},
            {"index": 6,  "name": "phaser",         "displayName": "Phaser"},
            {"index": 7,  "name": "plateReverb",    "displayName": "Plate Reverb"},
            {"index": 8,  "name": "rat",            "displayName": "RAT Distortion"},
            {"index": 9,  "name": "spring-Reverb",  "displayName": "Spring Reverb"},
            {"index": 10, "name": "sweepEcho",      "displayName": "Sweep Echo"},
            {"index": 11, "name": "tapeEcho",       "displayName": "Tape Echo"},
            {"index": 12, "name": "tubeScreamer",   "displayName": "Tube Screamer"},
        ],
        "models": [
            {"id": "unet",    "name": "U-Net (Log)",  "color": "#7c6ef0"},
            {"id": "convvae", "name": "ConvVAE",      "color": "#4ecdc4"},
        ],
    }

    # ──────────────────────────────────────────────────────────
    # Part 1: Guitar-TECHS excerpts (with correction factors)
    # ──────────────────────────────────────────────────────────
    print("\n═══ Guitar-TECHS Excerpts ═══")

    gt_files = sorted(GUITAR_TECHS_DIR.glob("*.wav"))
    gt_indices = [1, 3, 7, 10]
    gt_labels = {
        1: "Excerpt 1", 3: "Excerpt 3",
        7: "Excerpt 7", 10: "Excerpt 10",
    }

    for idx in gt_indices:
        wav_path = gt_files[idx - 1]
        excerpt_id = f"excerpt_{idx:02d}"
        label = gt_labels[idx]
        print(f"\n  {label} ({wav_path.name})")

        y_raw, _ = librosa.load(str(wav_path), sr=SR, mono=True)
        y_chunk = pick_chunk(y_raw)

        # Save original clean audio
        save_wav(y_chunk / (np.max(np.abs(y_chunk)) + EPS), AUDIO_OUT / excerpt_id / "clean.wav")

        # Spectrogram for model input (normalised by clean_max)
        spec_lin, _ = audio_to_spec(y_chunk)

        # Save clean spectrogram
        save_spectrogram(spec_lin.numpy(), AUDIO_OUT / excerpt_id / "specs" / "clean.png",
                         title="Clean Input")

        excerpt_entry = {
            "id": excerpt_id, "label": label,
            "cleanAudio": f"audio/{excerpt_id}/clean.wav",
            "cleanSpec": f"audio/{excerpt_id}/specs/clean.png",
            "predictions": {},
        }

        for eff_name, eff_idx in target_effects:
            eff_key, display = EFFECT_DISPLAY[eff_name]
            print(f"    {display}...", end="", flush=True)

            y_u, s_u = predict_unet(unet, spec_lin, eff_idx, device, apply_correction=True)
            save_wav(y_u, AUDIO_OUT / excerpt_id / "unet" / f"{eff_key}.wav")
            save_spectrogram(s_u, AUDIO_OUT / excerpt_id / "specs" / f"unet_{eff_key}.png",
                             title=f"U-Net -> {display}")

            y_c, s_c = predict_convvae(convvae, spec_lin, eff_idx, device, apply_correction=True)
            save_wav(y_c, AUDIO_OUT / excerpt_id / "convvae" / f"{eff_key}.wav")
            save_spectrogram(s_c, AUDIO_OUT / excerpt_id / "specs" / f"convvae_{eff_key}.png",
                             title=f"ConvVAE -> {display}")

            excerpt_entry["predictions"][eff_key] = {
                "unet": f"audio/{excerpt_id}/unet/{eff_key}.wav",
                "convvae": f"audio/{excerpt_id}/convvae/{eff_key}.wav",
                "unetSpec": f"audio/{excerpt_id}/specs/unet_{eff_key}.png",
                "convvaeSpec": f"audio/{excerpt_id}/specs/convvae_{eff_key}.png",
            }
            print(" done")

        manifest["excerpts"].append(excerpt_entry)

    # ──────────────────────────────────────────────────────────
    # Part 2: EGFxSet samples via AugmentedPairDataV3
    #         (proper shared-max normalisation, no correction)
    # ──────────────────────────────────────────────────────────
    print("\n═══ EGFxSet Excerpts (AugmentedPairDataV3) ═══")

    # Use 4 seeds that produce good-sounding passages
    egfxset_seeds = [42, 137, 314, 500]
    egfxset_labels = ["EGFxSet A", "EGFxSet B", "EGFxSet C", "EGFxSet D"]

    for p_idx, (seed, label) in enumerate(zip(egfxset_seeds, egfxset_labels)):
        excerpt_id = f"egfxset_{p_idx + 1:02d}"
        print(f"\n  {label} (seed={seed})")

        # For each effect, we generate a SEPARATE passage with proper shared-max.
        # We use the first effect's clean audio as the "clean input" for the UI.
        clean_audio_saved = False

        excerpt_entry = {
            "id": excerpt_id, "label": label,
            "cleanAudio": f"audio/{excerpt_id}/clean.wav",
            "cleanSpec": f"audio/{excerpt_id}/specs/clean.png",
            "predictions": {},
        }

        for eff_name, eff_idx in target_effects:
            eff_key, display = EFFECT_DISPLAY[eff_name]
            print(f"    {display}...", end="", flush=True)

            # Render passage with proper shared-max normalisation
            y_clean, A_lin, shared_max = renderer.generate_sample(seed, eff_name)

            # Save clean audio from the FIRST effect only (it's the same
            # clean passage since the seed + timeline are the same)
            if not clean_audio_saved:
                y_clean_norm = y_clean / (np.max(np.abs(y_clean)) + EPS)
                save_wav(y_clean_norm, AUDIO_OUT / excerpt_id / "clean.wav")
                save_spectrogram(A_lin.numpy(),
                                 AUDIO_OUT / excerpt_id / "specs" / "clean.png",
                                 title="Clean Input")
                clean_audio_saved = True

            # NO correction needed — A_lin already has shared-max normalisation
            y_u, s_u = predict_unet(unet, A_lin, eff_idx, device, apply_correction=False)
            save_wav(y_u, AUDIO_OUT / excerpt_id / "unet" / f"{eff_key}.wav")
            save_spectrogram(s_u, AUDIO_OUT / excerpt_id / "specs" / f"unet_{eff_key}.png",
                             title=f"U-Net -> {display}")

            y_c, s_c = predict_convvae(convvae, A_lin, eff_idx, device, apply_correction=False)
            save_wav(y_c, AUDIO_OUT / excerpt_id / "convvae" / f"{eff_key}.wav")
            save_spectrogram(s_c, AUDIO_OUT / excerpt_id / "specs" / f"convvae_{eff_key}.png",
                             title=f"ConvVAE -> {display}")

            excerpt_entry["predictions"][eff_key] = {
                "unet": f"audio/{excerpt_id}/unet/{eff_key}.wav",
                "convvae": f"audio/{excerpt_id}/convvae/{eff_key}.wav",
                "unetSpec": f"audio/{excerpt_id}/specs/unet_{eff_key}.png",
                "convvaeSpec": f"audio/{excerpt_id}/specs/convvae_{eff_key}.png",
            }
            print(" done")

        manifest["excerpts"].append(excerpt_entry)

    # ── Save manifest ──────────────────────────────────────────
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Manifest saved: {len(manifest['excerpts'])} excerpts")
    print("Done!")


if __name__ == "__main__":
    main()

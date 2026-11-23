"""Training-time audio augmentation for SER.

The benchmark so far trains on clean, fixed-length features.  With only 4-24
speakers per corpus the models overfit speaker identity, so augmenting the
*waveform* before feature extraction is the highest-leverage change: it makes
the same utterance look like a different recording of a different voice.

All functions are deterministic given ``seed`` so runs stay reproducible.
Augmentation is applied to the TRAIN split only; val/test stay clean.
"""

from __future__ import annotations

import numpy as np

# Per-augmentation application probability inside ``augment_waveform``.
DEFAULT_PROBS = {
    "noise": 0.5,
    "gain": 0.5,
    "time_stretch": 0.4,
    "pitch_shift": 0.4,
    "time_shift": 0.3,
}


def add_noise(wav: np.ndarray, rng: np.random.Generator, snr_db: tuple = (10.0, 25.0)) -> np.ndarray:
    """Add white Gaussian noise at a random SNR in ``snr_db``."""
    snr = rng.uniform(*snr_db)
    sig_power = float(np.mean(wav ** 2))
    if sig_power <= 0:
        return wav
    noise_power = sig_power / (10.0 ** (snr / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=wav.shape)
    return (wav + noise).astype(np.float32)


def random_gain(wav: np.ndarray, rng: np.random.Generator, db: tuple = (-6.0, 6.0)) -> np.ndarray:
    """Scale amplitude by a random gain in dB."""
    g = 10.0 ** (rng.uniform(*db) / 20.0)
    return (wav * g).astype(np.float32)


def time_shift(wav: np.ndarray, rng: np.random.Generator, max_frac: float = 0.15) -> np.ndarray:
    """Roll the waveform by up to ``max_frac`` of its length (circular)."""
    if wav.size == 0:
        return wav
    shift = int(rng.uniform(-max_frac, max_frac) * wav.size)
    return np.roll(wav, shift).astype(np.float32)


def time_stretch(wav: np.ndarray, rng: np.random.Generator, sr: int = 16000,
                 rate: tuple = (0.9, 1.1)) -> np.ndarray:
    """Speed perturbation via librosa; falls back to the input on failure."""
    try:
        import librosa

        r = float(rng.uniform(*rate))
        return librosa.effects.time_stretch(wav, rate=r).astype(np.float32)
    except Exception:
        return wav


def pitch_shift(wav: np.ndarray, rng: np.random.Generator, sr: int = 16000,
                steps: tuple = (-2.0, 2.0)) -> np.ndarray:
    """Pitch shift by a random number of semitones; falls back on failure."""
    try:
        import librosa

        n = float(rng.uniform(*steps))
        return librosa.effects.pitch_shift(wav, sr=sr, n_steps=n).astype(np.float32)
    except Exception:
        return wav


def augment_waveform(
    wav: np.ndarray,
    seed: int,
    sr: int = 16000,
    probs: dict | None = None,
) -> np.ndarray:
    """Apply a random subset of augmentations to one waveform.

    Each augmentation fires independently with its probability, so the same
    utterance can come out clean, lightly perturbed, or heavily perturbed
    across epochs.
    """
    p = dict(DEFAULT_PROBS)
    if probs:
        p.update(probs)
    rng = np.random.default_rng(seed)
    out = np.asarray(wav, dtype=np.float32)

    if rng.random() < p["gain"]:
        out = random_gain(out, rng)
    if rng.random() < p["noise"]:
        out = add_noise(out, rng)
    if rng.random() < p["time_shift"]:
        out = time_shift(out, rng)
    if rng.random() < p["time_stretch"]:
        out = time_stretch(out, rng, sr=sr)
    if rng.random() < p["pitch_shift"]:
        out = pitch_shift(out, rng, sr=sr)
    return out


def augment_batch(
    paths: list[str],
    seed: int,
    sr: int = 16000,
    probs: dict | None = None,
) -> list[np.ndarray]:
    """Load and augment a batch of wav files (mono, ``sr``).

    Used by the feature extractors when augmentation is enabled.  Loading is
    delegated to soundfile/librosa so behaviour matches the clean pipeline.
    """
    import soundfile as sf

    out: list[np.ndarray] = []
    for i, p in enumerate(paths):
        audio, file_sr = sf.read(p, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != sr:
            import librosa

            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
        out.append(augment_waveform(audio, seed=seed + i, sr=sr, probs=probs))
    return out

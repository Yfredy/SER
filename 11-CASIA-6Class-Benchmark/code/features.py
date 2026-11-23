"""Feature extraction frontends plus a content-addressed feature cache.

The cache keys on both the number of paths and a content hash of those paths.
This fixes the benchmark defect where the cache was keyed only on sample count,
so two corpora with equal counts silently reused each other's features.

Frontends:
- ``timnet_mfcc``: TIM-Net MFCC, mirrors ``TIM-Net_SER/Code/extract_feature.py``.
- ``whisper_512``: ERT-WFCCL Whisper features, delegates to the existing
  ``ert_wfccl.features.extract_whisper_features`` (not reimplemented).
"""

import hashlib
import json
import os
import sys

import numpy as np

_ERT_WFCCL_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "ERT-WFCCL", "projects", "ERT-WFCCL")
)

# ---------------------------------------------------------------------------
# Cache keys
# ---------------------------------------------------------------------------


def _file_digest(path: str) -> str:
    """SHA-256 of a file's bytes, hex-encoded (full length).

    A content digest is preferred over ``(basename, size)`` because the four
    corpora reuse filenames and two utterances can coincidentally share a byte
    size: a ``(basename, size)`` key would collide, silently cross-hitting the
    cache.  Absolute paths were rejected too: they make the cache non-portable
    and fail to detect an in-place edit that preserves size.  These are short
    wavs, so hashing their bytes is cheap.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def paths_hash(paths: list[str]) -> str:
    """Content hash of a path sequence: sha256 of ``(digest(path),)`` list.

    The digest covers each file's bytes, so the key changes when a file is
    edited in place (even without a size change) and is identical for
    byte-identical files regardless of their directory.  Returns the first 16
    hex characters.
    """
    entries = [_file_digest(p) for p in paths]
    return hashlib.sha256(json.dumps(entries).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def cache_array(cache_dir: str, name: str, paths: list[str], fn) -> np.ndarray:
    """Return ``fn(paths)``, cached to ``<cache_dir>/<name>.npz``.

    A cache HIT requires *both* ``int(z["n"]) == len(paths)`` and
    ``str(z["hash"]) == paths_hash(paths)``.  On miss, ``fn`` is called with
    the whole list of paths and must return the stacked array; the result is
    written atomically (temp name + ``os.replace``) so a half-written cache can
    never be read as valid, and a corrupt cache file triggers recomputation
    rather than silently serving wrong features.

    ``fn`` takes the list (not per-path) because all three real frontends are
    batched; Whisper especially benefits from GPU batching.
    """
    os.makedirs(cache_dir, exist_ok=True)
    out_path = os.path.join(cache_dir, f"{name}.npz")
    target_hash = paths_hash(paths)

    try:
        with np.load(out_path) as z:
            if int(z["n"]) == len(paths) and str(z["hash"]) == target_hash:
                return z["feat"]
    except (OSError, ValueError, KeyError):
        # Missing or corrupt cache: fall through and recompute.
        pass

    feat = np.asarray(fn(paths), dtype=np.float32)

    # ``np.savez_compressed`` appends ``.npz`` if the name lacks it, so give the
    # temp file a name that already ends in ``.npz`` and place it in the same
    # directory so ``os.replace`` is atomic.
    tmp_path = os.path.join(cache_dir, f".{name}.{os.getpid()}.npz")
    try:
        np.savez_compressed(tmp_path, feat=feat, n=len(paths), hash=target_hash)
        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return feat


# ---------------------------------------------------------------------------
# TIM-Net MFCC frontend
# ---------------------------------------------------------------------------

_MEAN_SIGNAL_LENGTH = 96000


def timnet_mfcc(path: str, mean_signal_length: int = _MEAN_SIGNAL_LENGTH,
                n_mfcc: int = 39) -> np.ndarray:
    """TIM-Net MFCC features for one utterance: ``(188, 39)`` float32.

    Mirrors ``TIM-Net_SER/Code/extract_feature.py::get_feature``:
    - ``librosa.load(path)`` with the DEFAULT sampling rate (22050); no ``sr=``
      override.
    - shorter than ``mean_signal_length``: center zero-pad; longer: center-crop.
    - ``librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=n_mfcc)`` then transpose.
    """
    import librosa

    signal, fs = librosa.load(path)
    s_len = len(signal)
    if s_len < mean_signal_length:
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem), "constant",
                        constant_values=0)
    else:
        pad_len = (s_len - mean_signal_length) // 2
        signal = signal[pad_len:pad_len + mean_signal_length]
    mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=n_mfcc)
    return np.ascontiguousarray(np.transpose(mfcc), dtype=np.float32)


# ---------------------------------------------------------------------------
# ERT-WFCCL Whisper frontend
# ---------------------------------------------------------------------------


def whisper_512(paths: list[str], model_path: str, device=None, batch_size=8) -> np.ndarray:
    """Whisper global features ``(N, 512)`` for a list of audio paths.

    Delegates directly to the existing ``ert_wfccl.features.extract_whisper_features``
    (``(paths, model_path, device=None, batch_size=8, loader=None)``); not
    reimplemented here.
    """
    if _ERT_WFCCL_ROOT not in sys.path:
        sys.path.insert(0, _ERT_WFCCL_ROOT)
    from ert_wfccl.features import extract_whisper_features

    return extract_whisper_features(paths, model_path, device=device, batch_size=batch_size)

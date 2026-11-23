"""Tests for the feature cache keyed on content hashes.

The cache defect this guards against: keying the cache only on the number of
paths, so two corpora with equal sample counts silently reuse each other's
features.  ``cache_array`` therefore keys on both ``n`` and ``paths_hash``,
and ``paths_hash`` digests the *content* of the files (not just name + size),
so a same-basename/same-size file with different bytes can never cross-hit.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmark_ser import features


# ---------------------------------------------------------------------------
# Brief's tests, rewritten to the list-taking ``fn`` contract.
# ---------------------------------------------------------------------------


def test_same_paths_hit_cache(tmp_path):
    calls = {"n": 0}

    def fn(paths):
        calls["n"] += 1
        return np.zeros((len(paths), 4), dtype=np.float32)

    p1 = str(tmp_path / "a.wav")
    open(p1, "wb").write(b"x" * 10)
    features.cache_array(str(tmp_path), "t", [p1], fn)
    features.cache_array(str(tmp_path), "t", [p1], fn)
    assert calls["n"] == 1


def test_same_count_different_content_misses_cache(tmp_path):
    calls = {"n": 0}

    def fn(paths):
        calls["n"] += 1
        return np.array([os.path.getsize(p) for p in paths], dtype=np.float32)

    p1 = str(tmp_path / "a.wav")
    p2 = str(tmp_path / "b.wav")
    open(p1, "wb").write(b"x" * 10)
    features.cache_array(str(tmp_path), "t", [p1], fn)
    open(p2, "wb").write(b"y" * 20)
    out = features.cache_array(str(tmp_path), "t", [p2], fn)
    assert calls["n"] == 2
    assert out[0] == 20.0


# ---------------------------------------------------------------------------
# Additional tests the brief lacks.
# ---------------------------------------------------------------------------


def test_same_basename_size_different_content_no_cross_hit(tmp_path):
    """Same basename AND same size but different bytes must not cross-hit.

    This is the real four-corpus risk: corpora reuse filenames, and two
    utterances can coincidentally share a byte size.  A (basename, size)-only
    hash collides here; the content digest must not.
    """
    da = tmp_path / "a"
    db = tmp_path / "b"
    da.mkdir()
    db.mkdir()
    pa = str(da / "x.wav")
    pb = str(db / "x.wav")

    content_a = b"q" * 100  # same length -> same size
    content_b = b"r" * 100  # same length, different bytes
    open(pa, "wb").write(content_a)
    open(pb, "wb").write(content_b)

    assert os.path.getsize(pa) == os.path.getsize(pb)
    assert os.path.basename(pa) == os.path.basename(pb)
    assert features.paths_hash([pa]) != features.paths_hash([pb])

    calls = {"n": 0}

    def fn(paths):
        calls["n"] += 1
        # Content-dependent features: the two files must produce different arrays.
        with open(paths[0], "rb") as f:
            byte = f.read(1)[0]
        return np.full((1, 1), float(byte), dtype=np.float32)

    out_a = features.cache_array(str(tmp_path), "same", [pa], fn)
    out_b = features.cache_array(str(tmp_path), "same", [pb], fn)
    assert calls["n"] == 2
    assert out_a[0, 0] == float(content_a[0])
    assert out_b[0, 0] == float(content_b[0])


def test_identical_bytes_different_dir_is_same_content(tmp_path):
    """Byte-identical files are the *same content* regardless of directory.

    The content digest intentionally hashes them equal: their features are
    bit-identical, so a cache hit is correct, not a cross-contamination bug.
    """
    da = tmp_path / "a"
    db = tmp_path / "b"
    da.mkdir()
    db.mkdir()
    pa = str(da / "x.wav")
    pb = str(db / "x.wav")
    data = b"z" * 100
    open(pa, "wb").write(data)
    open(pb, "wb").write(data)

    assert features.paths_hash([pa]) == features.paths_hash([pb])

    calls = {"n": 0}

    def fn(paths):
        calls["n"] += 1
        return np.full((len(paths), 1), 7.0, dtype=np.float32)

    out_a = features.cache_array(str(tmp_path), "ident", [pa], fn)
    out_b = features.cache_array(str(tmp_path), "ident", [pb], fn)
    # Same content -> one computation, and the cached array is valid for both.
    assert calls["n"] == 1
    assert np.array_equal(out_a, out_b)


def test_hash_detects_in_place_edit_preserving_size(tmp_path):
    """Editing a file in place without changing its size must change the hash."""
    p = tmp_path / "a.wav"
    p.write_bytes(b"a" * 50)
    h1 = features.paths_hash([str(p)])
    p.write_bytes(b"b" * 50)  # same size, different bytes
    h2 = features.paths_hash([str(p)])
    assert os.path.getsize(str(p)) == 50
    assert h1 != h2


def test_corrupt_cache_is_recomputed(tmp_path):
    """A half-written / corrupt npz must never be served as valid features."""
    p = tmp_path / "a.wav"
    p.write_bytes(b"x" * 10)
    cache_dir = str(tmp_path)
    npz_path = os.path.join(cache_dir, "t.npz")

    # Simulate a crashed write: garbage bytes at the cache path.
    with open(npz_path, "wb") as f:
        f.write(b"not a real npz file")

    calls = {"n": 0}

    def fn(paths):
        calls["n"] += 1
        return np.ones((len(paths), 2), dtype=np.float32)

    out = features.cache_array(cache_dir, "t", [str(p)], fn)
    assert calls["n"] == 1
    assert out.shape == (1, 2)
    assert out[0, 0] == 1.0

    # And the repaired cache now loads cleanly.
    calls["n"] = 0
    features.cache_array(cache_dir, "t", [str(p)], fn)
    assert calls["n"] == 0


# ---------------------------------------------------------------------------
# TIM-Net MFCC frontend shape checks.
# ---------------------------------------------------------------------------


def _write_wav(path, sr, n_samples):
    import soundfile as sf

    rng = np.random.default_rng(0)
    sig = rng.standard_normal(n_samples).astype(np.float32)
    sig /= np.abs(sig).max() + 1e-8
    sf.write(path, sig, sr)


def test_timnet_mfcc_shape_exact_96000(tmp_path):
    p = str(tmp_path / "x.wav")
    _write_wav(p, 22050, 96000)
    feat = features.timnet_mfcc(p)
    assert feat.shape == (188, 39), feat.shape
    assert feat.dtype == np.float32


def test_timnet_mfcc_shape_short_signal_padded(tmp_path):
    p = str(tmp_path / "short.wav")
    _write_wav(p, 22050, 16000)
    feat = features.timnet_mfcc(p)
    assert feat.shape == (188, 39), feat.shape


def test_timnet_mfcc_shape_long_signal_cropped(tmp_path):
    p = str(tmp_path / "long.wav")
    _write_wav(p, 22050, 200000)
    feat = features.timnet_mfcc(p)
    assert feat.shape == (188, 39), feat.shape

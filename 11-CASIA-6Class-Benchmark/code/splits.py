"""Deterministic train/val/test splits across four corpora for SER benchmarking.

Generates 24 auditable split entries (4 corpora x 2 protocols x 3 seeds), each
with a content hash so different runs can detect drift.
"""

import hashlib
import json
import os
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Split functions
# ---------------------------------------------------------------------------


def random_split(
    samples,
    seed: int,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
) -> dict[str, list[int]]:
    """Stratified random split by label.

    Returns ``{"train", "val", "test"}`` lists of int indices (into *samples*),
    each sorted ascending.  Every non-empty class is guaranteed at least one
    sample in test; train receives all remaining samples after test and val
    are carved out, so a singleton class (n=1) puts its only sample in test
    and none in train.
    """
    rng = np.random.default_rng(seed)

    # Group indices by label
    by_label: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        by_label[s.label].append(i)

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []

    for _label, indices in sorted(by_label.items()):
        n = len(indices)
        arr = np.array(indices)
        rng.shuffle(arr)
        shuffled: list[int] = arr.tolist()

        test_n = max(1, round(n * test_ratio)) if n > 0 else 0
        if test_n > n:
            test_n = n

        remaining_n = n - test_n

        if remaining_n > 0:
            val_n = round(remaining_n * val_ratio)
            if val_n > remaining_n:
                val_n = remaining_n
        else:
            val_n = 0

        test.extend(shuffled[:test_n])
        val.extend(shuffled[test_n : test_n + val_n])
        train.extend(shuffled[test_n + val_n :])

    return {
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
    }


def speaker_split(
    samples,
    seed: int,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, list[int]]:
    """Speaker-disjoint split.

    Shuffles speakers (deterministically via *seed*), allocates by *ratios*,
    then maps speaker sets back to sample indices.  Guarantees at least one
    speaker in validation and test; returns ``{"train", "val", "test"}``.
    """
    by_speaker: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        by_speaker[s.speaker].append(i)

    speakers = sorted(by_speaker.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(speakers)

    n = len(speakers)

    val_count = max(1, round(n * ratios[1]))
    test_count = max(1, round(n * ratios[2]))
    train_count = n - val_count - test_count

    # If train_count becomes negative (extreme edge), push back into train.
    if train_count < 0:
        test_count += train_count  # train_count is negative, so this reduces test
        if test_count < 1:
            test_count = 1
        train_count = n - val_count - test_count

    train_speakers: set[str] = set(speakers[:train_count])
    val_speakers: set[str] = set(speakers[train_count : train_count + val_count])
    test_speakers: set[str] = set(speakers[train_count + val_count :])

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []

    for sp, idxs in by_speaker.items():
        if sp in train_speakers:
            train.extend(idxs)
        elif sp in val_speakers:
            val.extend(idxs)
        else:
            test.extend(idxs)

    return {
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
    }


# ---------------------------------------------------------------------------
# Content hash
# ---------------------------------------------------------------------------


def file_hash(split: dict) -> str:
    """SHA-256 hex digest of ``{"train", "val", "test"}``, first 16 chars."""
    payload = json.dumps(
        {"train": split["train"], "val": split["val"], "test": split["test"]},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Build a single split (with fallback logic)
# ---------------------------------------------------------------------------


def build(samples, protocol: str, seed: int) -> dict:
    """Build one split dict for a single corpus.

    Parameters
    ----------
    samples : list[Sample]
        Must all belong to the same ``dataset``.
    protocol : {"random", "speaker"}
    seed : int

    Returns
    -------
    dict with keys: dataset, protocol, seed, hash, train, val, test,
    speaker_leak, fallback_reason.
    """
    if not samples:
        raise ValueError("samples must not be empty")

    dataset: str = samples[0].dataset

    if protocol == "random":
        result = random_split(samples, seed)
    elif protocol == "speaker":
        result = speaker_split(samples, seed)
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    speaker_leak: bool = False
    fallback_reasons: list[str] = []

    # ------------------------------------------------------------------
    # Speaker-protocol fallback: ensure every class present in the
    # corpus also appears in the test set.  When a class is missing we
    # move the *whole owning speaker* (so speakers stay disjoint) from
    # train into test, deterministically.
    # ------------------------------------------------------------------
    if protocol == "speaker":
        present_classes: set[str] = {s.label for s in samples}
        test_labels: set[str] = {samples[i].label for i in result["test"]}
        missing = present_classes - test_labels

        if missing:
            fallback_rng = np.random.default_rng(seed + 9999)
            train_set: set[int] = set(result["train"])
            test_set: set[int] = set(result["test"])

            for cls in sorted(missing):
                # Which speakers in train carry this class?
                candidates: list[str] = sorted(
                    {
                        samples[i].speaker
                        for i in train_set
                        if samples[i].label == cls
                    }
                )
                if not candidates:
                    # Class exists in corpus but somehow not in any train
                    # speaker -- should not happen with our corpora.
                    continue

                # Deterministic pick
                idx = int(fallback_rng.integers(0, len(candidates)))
                chosen = candidates[idx]

                # Move the whole speaker
                moved: list[int] = []
                for i in list(train_set):
                    if samples[i].speaker == chosen:
                        train_set.discard(i)
                        test_set.add(i)
                        moved.append(i)

                fallback_reasons.append(
                    f"moved speaker '{chosen}' ({len(moved)} samples) "
                    f"to cover class '{cls}'"
                )

            if fallback_reasons:
                speaker_leak = True
                result["train"] = sorted(train_set)
                result["test"] = sorted(test_set)

    # ------------------------------------------------------------------
    # Guard: validation set must never be empty.
    # ------------------------------------------------------------------
    if len(result["val"]) == 0:
        # Move one sample (or speaker for speaker protocol) from train to val.
        if protocol == "speaker":
            # Move one speaker from train into val
            train_speakers = sorted({samples[i].speaker for i in result["train"]})
            if train_speakers:
                rng = np.random.default_rng(seed + 7777)
                pick_idx = int(rng.integers(0, len(train_speakers)))
                chosen = train_speakers[pick_idx]

                train_set = set(result["train"])
                val_set: set[int] = set(result["val"])
                moved = []
                for i in list(train_set):
                    if samples[i].speaker == chosen:
                        train_set.discard(i)
                        val_set.add(i)
                        moved.append(i)
                result["train"] = sorted(train_set)
                result["val"] = sorted(val_set)
                fallback_reasons.append(
                    f"val was empty; moved speaker '{chosen}' "
                    f"({len(moved)} samples) from train to val"
                )
        else:
            # random protocol: move one sample
            if result["train"]:
                train = list(result["train"])
                rng = np.random.default_rng(seed + 7777)
                pick_idx = int(rng.integers(0, len(train)))
                moved = train.pop(pick_idx)
                result["train"] = sorted(train)
                result["val"] = sorted([moved])
                fallback_reasons.append(
                    f"val was empty; moved sample index {moved} from train to val"
                )

    # ------------------------------------------------------------------
    # Assemble final dict
    # ------------------------------------------------------------------
    result["dataset"] = dataset
    result["protocol"] = protocol
    result["seed"] = seed
    result["hash"] = file_hash(result)
    result["speaker_leak"] = speaker_leak
    result["fallback_reason"] = "; ".join(fallback_reasons) if fallback_reasons else None

    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_all(by_dataset: dict[str, list], path: str) -> str:
    """Build all 24 splits and write them as JSON.

    Keys are ``{dataset}_{protocol}_seed{seed}`` (e.g.  ``emodb_random_seed0``).
    Returns *path*.
    """
    all_splits: dict[str, dict] = {}
    for ds in sorted(by_dataset):
        samples = by_dataset[ds]
        for protocol in ("random", "speaker"):
            for seed in (0, 1, 2):
                key = f"{ds}_{protocol}_seed{seed}"
                all_splits[key] = build(samples, protocol, seed)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(all_splits, f, indent=2, sort_keys=True)

    return path


def load_all(path: str) -> dict:
    """Load the splits JSON file produced by :func:`save_all`."""
    with open(path) as f:
        return json.load(f)
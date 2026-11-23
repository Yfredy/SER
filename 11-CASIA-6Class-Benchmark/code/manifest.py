"""Sample manifest: scan four corpora, normalize labels, and expose auditable stats."""

import glob
import os
from collections import defaultdict
from dataclasses import dataclass, field

from benchmark_ser.labels import CLASSES, normalize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATASETS: dict[str, str] = {
    "emodb": os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "_local", "datasets", "emodb_by_class")
    ),
    "casia": os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "dataset", "CASIA", "CASIA")
    ),
    "savee": os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "dataset", "SAVEE", "wav")
    ),
    "ravdess": os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "dataset", "RAVDESS", "wav")
    ),
}

# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """One labelled utterance from a corpus."""

    path: str
    dataset: str
    label: str
    speaker: str
    uid: str

    # Automatically sort by (dataset, uid)
    sort_key: tuple[str, str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sort_key", (self.dataset, self.uid))


# ---------------------------------------------------------------------------
# Speaker extraction
# ---------------------------------------------------------------------------


def sample_speaker(dataset: str, path: str) -> str:
    """Extract speaker identity from a wav file path, per corpus convention."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if dataset in ("emodb", "savee"):
        return stem[:2]
    elif dataset == "casia":
        return stem.split("-")[2]
    elif dataset == "ravdess":
        return stem.split("-")[0]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Internal scan (shared between load_manifest and audit)
# ---------------------------------------------------------------------------


def _scan_corpora():
    """Yield (path, dataset, raw_label, stem) for every wav on disk across all corpora.

    Uses ``glob(root / * / *.wav)`` exactly once to avoid double-counting or
    picking up junk directories (``os.walk`` would grab ``__MACOSX`` etc.).
    """
    for ds, root in DATASETS.items():
        for path in sorted(glob.glob(os.path.join(root, "*", "*.wav"))):
            stem = os.path.splitext(os.path.basename(path))[0]
            raw_label = os.path.basename(os.path.dirname(path))
            yield path, ds, raw_label, stem


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_manifest() -> list[Sample]:
    """Return all retained samples across four corpora, sorted by (dataset, uid)."""
    samples: list[Sample] = []
    for path, ds, raw_label, stem in _scan_corpora():
        label = normalize(raw_label)
        if label is None:
            continue
        uid = f"{ds}:{stem}"
        try:
            speaker = sample_speaker(ds, path)
        except Exception:
            print(f"WARNING: cannot parse speaker for {path}")
            continue
        samples.append(Sample(path=path, dataset=ds, label=label, speaker=speaker, uid=uid))
    samples.sort(key=lambda s: s.sort_key)
    return samples


def by_dataset(samples: list[Sample] | None = None) -> dict[str, list[Sample]]:
    """Group samples by ``dataset`` field into ``{emodb: [...], ...}``.

    When *samples* is None the full manifest is loaded.
    """
    if samples is None:
        samples = load_manifest()
    groups: dict[str, list[Sample]] = {ds: [] for ds in DATASETS}
    for s in samples:
        groups[s.dataset].append(s)
    return groups


def audit() -> dict:
    """Return per-dataset accounting of raw wavs, kept/dropped/unparsed samples,
    class distribution, and speaker count.

    ``class_counts`` is pre-populated with all six ``CLASSES`` keys set to 0
    so the report can express "EMO-DB has no surprise".
    """
    result: dict[str, dict] = {}
    for ds in DATASETS:
        result[ds] = {
            "raw_wav": 0,
            "kept": 0,
            "unparsed": 0,
            "dropped_labels": defaultdict(int),
            "class_counts": {c: 0 for c in CLASSES},
            "speakers": set(),
        }

    for path, ds, raw_label, stem in _scan_corpora():
        a = result[ds]
        a["raw_wav"] += 1
        label = normalize(raw_label)
        if label is None:
            a["dropped_labels"][raw_label] += 1
            continue
        try:
            speaker = sample_speaker(ds, path)
        except Exception:
            a["unparsed"] += 1
            print(f"WARNING: cannot parse speaker for {path}")
            continue
        a["kept"] += 1
        a["class_counts"][label] += 1
        a["speakers"].add(speaker)

    # Finalise mutable accumulators
    for ds in result:
        result[ds]["speakers"] = len(result[ds]["speakers"])
        result[ds]["dropped_labels"] = dict(result[ds]["dropped_labels"])

    return result
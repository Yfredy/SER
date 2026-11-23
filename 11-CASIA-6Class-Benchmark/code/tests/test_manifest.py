import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmark_ser import manifest
from benchmark_ser.labels import CLASSES


def test_manifest_counts_match_disk():
    samples = manifest.load_manifest()
    audit = manifest.audit()
    assert len(samples) == sum(a["kept"] for a in audit.values())
    for ds, root in manifest.DATASETS.items():
        on_disk = len(glob.glob(os.path.join(root, "*", "*.wav")))
        a = audit[ds]
        assert a["raw_wav"] == on_disk, (ds, a["raw_wav"], on_disk)
        assert a["kept"] + a["unparsed"] + sum(a["dropped_labels"].values()) == on_disk


def test_emodb_has_no_surprise():
    audit = manifest.audit()
    assert audit["emodb"]["class_counts"]["surprise"] == 0
    assert audit["casia"]["class_counts"]["surprise"] > 0


def test_all_labels_in_unified_space():
    for s in manifest.load_manifest():
        assert s.label in CLASSES


def test_speakers_parsed():
    audit = manifest.audit()
    assert audit["emodb"]["speakers"] == 10
    assert audit["casia"]["speakers"] == 4
    assert audit["savee"]["speakers"] == 4
    assert audit["ravdess"]["speakers"] == 24


def test_by_dataset():
    """by_dataset() produces per-corpus buckets with correct keys, sizes, and no cross-talk."""
    groups = manifest.by_dataset()

    # 1. Keys are exactly the four corpora
    assert set(groups.keys()) == {"emodb", "casia", "savee", "ravdess"}

    # 2. Concatenation of all values equals load_manifest()
    all_samples = manifest.load_manifest()
    grouped_samples = []
    for ds in sorted(groups.keys()):
        grouped_samples.extend(groups[ds])
    assert grouped_samples == all_samples

    # 3. Every sample in bucket k has .dataset == k
    for ds, samples in groups.items():
        for s in samples:
            assert s.dataset == ds, f"{s.uid} in bucket {ds} but has dataset {s.dataset}"

    # 4. Total count is 3084 under the CASIA-style 6-class space
    assert len(all_samples) == 3084
    assert sum(len(v) for v in groups.values()) == 3084

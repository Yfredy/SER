import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmark_ser import manifest, splits


def _by_dataset():
    out = {}
    for s in manifest.load_manifest():
        out.setdefault(s.dataset, []).append(s)
    return out


def test_hash_is_deterministic():
    s = _by_dataset()["emodb"]
    a = splits.build(s, "random", 0)
    b = splits.build(s, "random", 0)
    assert a["hash"] == b["hash"]
    assert a["train"] == b["train"]


def test_partitions_are_disjoint_and_complete():
    for ds, s in _by_dataset().items():
        for protocol in ("random", "speaker"):
            for seed in (0, 1, 2):
                sp = splits.build(s, protocol, seed)
                tr, va, te = set(sp["train"]), set(sp["val"]), set(sp["test"])
                assert not (tr & va) and not (tr & te) and not (va & te)
                assert tr | va | te == set(range(len(s))), (ds, protocol, seed)
                assert sp["dataset"] == ds


def test_every_present_class_in_test():
    for ds, s in _by_dataset().items():
        present = {x.label for x in s}
        for protocol in ("random", "speaker"):
            for seed in (0, 1, 2):
                sp = splits.build(s, protocol, seed)
                got = {s[i].label for i in sp["test"]}
                missing = present - got
                assert not missing, (ds, protocol, seed, missing)


def test_speaker_split_has_no_overlap():
    for ds, s in _by_dataset().items():
        sp = splits.build(s, "speaker", 0)
        tr = {s[i].speaker for i in sp["train"]}
        va = {s[i].speaker for i in sp["val"]}
        te = {s[i].speaker for i in sp["test"]}
        assert not (tr & va) and not (tr & te) and not (va & te), ds


def test_speaker_protocol_seeds_produce_distinct_partitions():
    """Regression: speaker-protocol shuffle was a no-op (GH#fix-shuffle-bug).

    Before the fix, ``rng.shuffle(np.array(speakers))`` shuffled a
    throwaway array and discarded the result, so every seed produced the
    same sorted-speaker order and identical hashes.  This test catches
    that by asserting that all three seeds do *not* yield identical hashes
    for any corpus (at least 2 distinct).  For corpora with more than 2
    speakers we additionally assert all 3 hashes are distinct.
    """
    for ds, s in _by_dataset().items():
        hashes = [splits.build(s, "speaker", seed)["hash"] for seed in (0, 1, 2)]
        n_distinct = len(set(hashes))
        # Every corpus has at least 2 speakers, so at least 2 distinct hashes.
        assert n_distinct >= 2, (
            f"{ds} speaker protocol: expected at least 2 distinct hashes "
            f"(shuffle was inert), got {n_distinct}: {hashes}"
        )
        # Corpora with >2 speakers: 3 seeds should produce 3 distinct splits.
        speakers_in_corpus = len({x.speaker for x in s})
        if speakers_in_corpus > 2:
            assert n_distinct == 3, (
                f"{ds} speaker protocol ({speakers_in_corpus} speakers): "
                f"expected 3 distinct hashes, got {n_distinct}: {hashes}"
            )


def test_save_and_load_roundtrip(tmp_path):
    by_ds = _by_dataset()
    p = str(tmp_path / "splits.json")
    splits.save_all(by_ds, p)
    back = splits.load_all(p)
    assert len(back) == 24
    assert back["emodb_random_seed0"]["hash"] == splits.build(by_ds["emodb"], "random", 0)["hash"]
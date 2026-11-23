import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmark_ser.labels import CLASSES, normalize


def test_classes_fixed_order():
    assert CLASSES == ("angry", "fear", "happy", "neutral", "sad", "surprise")


def test_normalize_synonyms():
    assert normalize("anger") == "angry"
    assert normalize("Angry") == "angry"
    assert normalize("happiness") == "happy"
    assert normalize("sadness") == "sad"
    assert normalize("fearful") == "fear"
    assert normalize("neutral") == "neutral"


def test_normalize_rejects_unmapped():
    assert normalize("boredom") is None
    assert normalize("calm") is None
    assert normalize("disgust") is None

"""CASIA-style 6-class emotion label space for four-corpus SER benchmarking.

The label set mirrors the corpora's own convention (angry, fear, happy,
neutral, sad, surprise) rather than a disgust-based one, so CASIA contributes
all six classes.  EMO-DB has no ``surprise`` in the corpus, so it only fills
five of the six -- that asymmetry is inherent to the data and is disclosed in
the report rather than papered over.
"""

CLASSES: tuple[str, ...] = ("angry", "fear", "happy", "neutral", "sad", "surprise")

_SYNONYMS: dict[str, str | None] = {
    # Canonical self-mappings
    "angry": "angry",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprise": "surprise",
    # Synonym mappings
    "anger": "angry",
    "happiness": "happy",
    "sadness": "sad",
    "fearful": "fear",
    "surprised": "surprise",
    # Explicitly rejected (mapped to None so normalize returns None)
    "boredom": None,
    "calm": None,
    "disgust": None,
}


def normalize(raw: str) -> str | None:
    """Map a raw corpus label to a unified CLASSES label, or None if unmapped."""
    return _SYNONYMS.get(raw.strip().lower())

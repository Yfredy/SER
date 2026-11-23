"""Aggregate per-run result JSONs into merged rows and summary statistics.

Unit contract: every metric stays in the 0-1 float range produced by the
runners.  ``render.build_markdown`` is the only place that multiplies by 100
for human-readable percentages.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

# Row schema shared by all three runners.  ``n_concepts`` is ERT-WFCCL only and
# is carried through verbatim when present; ``n_classes_present`` is consumed by
# ``render`` for the metric-denominator caveat.
_METRICS = ("accuracy", "macro_precision", "macro_recall", "macro_f1")

_RESULTS_DIR = Path(__file__).resolve().parent / "results"


def _load_json(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON array of row objects")
    return data


def merge(paths: list[str]) -> list[dict]:
    """Load *paths* and return a single list of unified row dicts.

    Each runner row already matches the shared schema, so this is a plain
    concatenation; no unit conversion happens here.
    """
    rows: list[dict] = []
    for p in paths:
        rows.extend(_load_json(p))
    return rows


def summary(rows: list[dict]) -> dict:
    """Aggregate metrics per ``(dataset, protocol, model)``.

    Returns ``{(dataset, protocol, model): {metric: (mean, std, n)}}`` where
    ``mean``/``std`` are 0-1 floats (``statistics.mean`` / ``statistics.pstdev``)
    and ``n`` is the seed count.
    """
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for r in rows:
        key = (r["dataset"], r["protocol"], r["model"])
        groups.setdefault(key, []).append(r)

    out: dict[tuple[str, str, str], dict] = {}
    for key, grp in groups.items():
        entry: dict[str, tuple[float, float, int]] = {}
        for metric in _METRICS:
            values = [float(r[metric]) for r in grp]
            mean = statistics.mean(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            entry[metric] = (mean, std, len(values))
        out[key] = entry
    return out


def main(result_paths: list[str], out_path: str | Path) -> list[dict]:
    """Merge *result_paths* and write the merged rows to *out_path*."""
    rows = merge(result_paths)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return rows
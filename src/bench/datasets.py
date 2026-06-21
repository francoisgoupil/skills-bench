"""Registry of benchmark use cases.

Each entry returns a Dataset: a (X, y) pair plus the metadata skore/scoring needs.
Swap the loaders for your own CSVs/warehouse pulls — the rest of the pipeline is agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd
from sklearn.datasets import fetch_openml


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: pd.Series
    task: str  # "binary-classification" | "regression"
    pos_label: object | None = None
    description: str = ""


def _match_pos_label(y: pd.Series, pos_label: object | None) -> object | None:
    """Align pos_label with y's actual label values (e.g. int 1 -> str '1')."""
    if pos_label is None:
        return None
    labels = set(y.unique())
    if pos_label in labels:
        return pos_label
    as_str = str(pos_label)
    if as_str in labels:
        return as_str
    if isinstance(pos_label, str) and pos_label.isdigit():
        as_int = int(pos_label)
        if as_int in labels:
            return as_int
    return pos_label


def _openml(name: str, data_id: int, target: str, pos_label=None) -> Callable[[], Dataset]:
    def load() -> Dataset:
        bunch = fetch_openml(data_id=data_id, as_frame=True)
        df = bunch.frame
        y = df[target]
        X = df.drop(columns=[target])
        return Dataset(
            name=name, X=X, y=y, task="binary-classification",
            pos_label=_match_pos_label(y, pos_label),
        )
    return load


# NOTE: these data_ids are placeholders — point them at the datasets that match your
# actual benchmark (the four charts: predictive maintenance, fraud, churn, credit scoring).
REGISTRY: dict[str, Callable[[], Dataset]] = {
    "predictive-maintenance": _openml("predictive-maintenance", data_id=42890, target="Machine failure", pos_label=1),
    "fraud-detection":        _openml("fraud-detection", data_id=1597, target="Class", pos_label=1),
    "churn-prediction":       _openml("churn-prediction", data_id=40701, target="class", pos_label=1),
    "credit-scoring":         _openml("credit-scoring", data_id=31, target="class", pos_label="bad"),
}


def load(name: str) -> Dataset:
    if name not in REGISTRY:
        raise KeyError(f"Unknown dataset {name!r}. Known: {list(REGISTRY)}")
    return REGISTRY[name]()

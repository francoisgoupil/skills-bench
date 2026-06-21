"""Turn an AgentRun into (a) an ML performance report logged to Skore Hub, and
(b) a row in a local cost table recording the token cost of producing it.

Performance and cost are joined later (in the dashboard) on the run `key`.
"""
from __future__ import annotations

import os
import sqlite3
import time

from sklearn.base import clone

from .agent import AgentRun
from .datasets import Dataset

COST_DB = os.environ.get("BENCH_COST_DB", "runs.db")


def hub_project_name(dataset_name: str, *, prefix: str | None = None) -> str:
    """One Skore Hub project per dataset inside the workspace."""
    if prefix:
        return f"{prefix}-{dataset_name}"
    return dataset_name


def _project(project_name: str, *, hub: bool = True):
    from skore import Project
    if hub and os.environ.get("SKORE_WORKSPACE"):
        workspace = os.environ["SKORE_WORKSPACE"]
        return Project(f"{workspace}/{project_name}", mode="hub")
    return Project(project_name, mode="local")


def _attach_agent_metrics(report, run: AgentRun) -> None:
    """Register agent cost and total tokens burned as skore custom metrics (synced to Hub)."""
    usage = run.usage

    def _constant(value: float, metric_name: str):
        def _scorer(estimator, X, y_true) -> float:  # noqa: ARG001
            return float(value)
        _scorer.__name__ = metric_name
        return _scorer

    for name, value, verbose_name, greater_is_better in (
        ("cost_usd", usage.cost_usd, "Token cost (USD)", False),
        ("tokens_burned", usage.tokens_burned, "Tokens burned", False),
    ):
        report.metrics.add(
            _constant(value, name),
            name=name,
            verbose_name=verbose_name,
            greater_is_better=greater_is_better,
            position="last",
        )


def score_and_log(
    run: AgentRun,
    dataset: Dataset,
    *,
    project_prefix: str | None = None,
    hub: bool = True,
) -> dict:
    """Evaluate the estimator with skore, push to Hub, persist token cost."""
    from skore import evaluate

    hub_project = hub_project_name(dataset.name, prefix=project_prefix)
    key = f"{dataset.name}::{run.config.key}::{int(time.time())}"

    from .agent import _validate_pipeline
    if err := _validate_pipeline(run.estimator):
        raise ValueError(f"Pipeline rejected before scoring: {err}")

    # 1. Performance — skore owns the held-out split for a fair, comparable measure.
    #    clone() keeps hyperparameters but drops fitted state so we refit on train only.
    report = evaluate(
        clone(run.estimator), dataset.X, dataset.y,
        splitter=0.2, pos_label=dataset.pos_label,
    )
    _attach_agent_metrics(report, run)

    # 2. Persist to Skore Hub when configured (best-effort — local mode can fail on
    #    SQLite 3.49+ due to a diskcache compatibility issue).
    skore_error = None
    try:
        _project(hub_project, hub=hub).put(key, report)
    except Exception as exc:
        skore_error = str(exc)
        if hub:
            raise RuntimeError(f"Skore Hub upload failed for {key}: {exc}") from exc

    metrics = _extract_metrics(report)

    # 3. Local cost sidecar (offline fallback + dashboard); Hub gets the same via custom metrics.
    _record_cost(
        key=key, dataset=dataset.name, hub_project=hub_project, config=run.config.key,
        model=run.config.model,
        use_skills=int(run.config.use_skills),
        tokens_burned=run.usage.tokens_burned,
        cost_usd=run.usage.cost_usd,
        **metrics,
    )
    result = {"key": key, "hub_project": hub_project, "cost_usd": run.usage.cost_usd,
              "tokens_burned": run.usage.tokens_burned, **metrics}
    if skore_error:
        result["skore_warning"] = skore_error
    return result


_SKORE_SCALAR_METRICS = (
    "roc_auc", "accuracy", "precision", "recall", "brier_score", "log_loss",
    "r2", "rmse", "mae", "mape", "fit_time", "predict_time", "score",
)


def _extract_metrics(report) -> dict:
    """Pull scalar metrics from a skore report for the local cost table."""
    metrics: dict[str, float] = {}
    accessor = report.metrics
    for name in _SKORE_SCALAR_METRICS:
        fn = getattr(accessor, name, None)
        if fn is None or not callable(fn):
            continue
        try:
            val = fn()
        except Exception:
            continue
        if isinstance(val, dict):
            continue
        if isinstance(val, (int, float)) and val == val:
            metrics[name] = float(val)
    return metrics


def _record_cost(**row) -> None:
    con = sqlite3.connect(COST_DB)
    con.execute(
        """CREATE TABLE IF NOT EXISTS runs (
               key TEXT PRIMARY KEY, dataset TEXT, hub_project TEXT, config TEXT, model TEXT,
               use_skills INT, tokens_burned INT, cost_usd REAL,
               ts REAL DEFAULT (strftime('%s','now')))"""
    )
    _TEXT = {"hub_project", "dataset", "config", "model", "key"}
    _INT = {"use_skills", "tokens_burned"}
    existing = {r[1] for r in con.execute("PRAGMA table_info(runs)")}
    for col in row:
        if col not in existing:
            if col in _TEXT:
                col_type = "TEXT"
            elif col in _INT:
                col_type = "INT"
            else:
                col_type = "REAL"
            con.execute(f"ALTER TABLE runs ADD COLUMN {col} {col_type}")
            existing.add(col)
    cols = ",".join(row)
    con.execute(f"INSERT OR REPLACE INTO runs ({cols}) VALUES ({','.join('?' * len(row))})",
                list(row.values()))
    con.commit()
    con.close()

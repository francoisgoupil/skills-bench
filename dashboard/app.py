"""Monitoring dashboard: ML performance vs agent cost, per use case.

Performance comes from skore (one Hub project per dataset); token cost from runs.db.
They join on the run `key`.

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def bootstrap_env() -> None:
    """Load settings from Streamlit secrets (cloud) or .env (local). Never log secrets."""
    load_dotenv(ROOT / ".env")
    try:
        for key, value in st.secrets.items():
            if isinstance(value, str):
                os.environ[key] = value
    except Exception:
        pass
    # skore SDK reads SKORE_HUB_API_KEY; bench hub_client.py uses SKORE_HUB_TOKEN.
    if os.environ.get("SKORE_HUB_TOKEN") and not os.environ.get("SKORE_HUB_API_KEY"):
        os.environ["SKORE_HUB_API_KEY"] = os.environ["SKORE_HUB_TOKEN"]
    if os.environ.get("SKORE_HUB_API") and not os.environ.get("SKORE_HUB_URI"):
        os.environ["SKORE_HUB_URI"] = os.environ["SKORE_HUB_API"]


def default_cost_db() -> str:
    override = os.environ.get("BENCH_COST_DB")
    if override:
        return override
    local = ROOT / "runs.db"
    if local.exists():
        return str(local)
    return str(ROOT / "data" / "runs.public.db")


bootstrap_env()

from bench.scoring import hub_project_name

st.set_page_config(page_title="skills-bench", layout="wide")
st.title("Model × Probabl Skills — performance vs cost")

PREFIX = st.text_input("Hub project prefix (optional)", value="")
COST_DB = default_cost_db()

# skore scalar metrics (exclude agent-side custom metrics used as x-axis).
SKORE_METRICS = frozenset({
    "roc_auc", "accuracy", "precision", "recall", "brier_score", "log_loss",
    "r2", "rmse", "mae", "mape", "score", "fit_time", "predict_time",
})

# Legacy runs.db columns from an older _extract_metrics bug (estimator class names).
LEGACY_ROC_AUC_COLUMNS = (
    "HistGradientBoostingClassifier",
    "RandomForestClassifier",
    "LogisticRegression",
    "VotingClassifier",
)

# Default y-axis per dataset (all current use cases are binary classification).
DATASET_DEFAULT_METRIC = {
    "churn-prediction": "roc_auc",
    "fraud-detection": "roc_auc",
    "credit-scoring": "roc_auc",
    "predictive-maintenance": "roc_auc",
}

METRIC_LABELS = {
    "roc_auc": "ROC AUC",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "brier_score": "Brier score",
    "log_loss": "Log loss",
}

DATASET_NOTES = {
    "churn-prediction": (
        "Binary churn classification — ROC AUC is the recommended metric "
        "(threshold-independent, robust to class imbalance)."
    ),
    "fraud-detection": (
        "Binary fraud classification — ROC AUC is the recommended metric "
        "(threshold-independent ranking; robust to rare fraud class). "
        "Use recall as a follow-up when choosing an operational cutoff."
    ),
    "predictive-maintenance": (
        "Binary failure prediction — ROC AUC is the recommended metric "
        "(threshold-independent; failures are typically rare). "
        "Use recall as a follow-up to limit missed failures / downtime."
    ),
    "credit-scoring": (
        "Binary default risk (bad vs good) — ROC AUC is the recommended metric "
        "(standard for credit scoring; robust to class imbalance). "
        "Use precision/recall as follow-ups once you pick an approval cutoff."
    ),
}


def hub_login() -> str | None:
    """Authenticate to Skore Hub once per session."""
    if not os.environ.get("SKORE_WORKSPACE"):
        return "SKORE_WORKSPACE is not set (use .env locally or Streamlit secrets on Cloud)."
    if not os.environ.get("SKORE_HUB_API_KEY"):
        return (
            "SKORE_HUB_API_KEY is not set — add your Skore Hub API key in "
            "`.streamlit/secrets.toml` or Streamlit Cloud app secrets."
        )
    try:
        import skore
        skore.login()
    except Exception as exc:
        return str(exc)
    return None


@st.cache_data(ttl=60)
def load_cost() -> pd.DataFrame:
    if not os.path.exists(COST_DB):
        return pd.DataFrame()
    con = sqlite3.connect(COST_DB)
    df = pd.read_sql("SELECT * FROM runs", con)
    con.close()
    df["hub_project"] = df["dataset"].map(
        lambda d: hub_project_name(d, prefix=PREFIX or None)
    )
    if "tokens_burned" not in df.columns and {"input_tokens", "output_tokens"} <= set(df.columns):
        df["tokens_burned"] = df["input_tokens"].fillna(0) + df["output_tokens"].fillna(0)
    return df


@st.cache_data(ttl=60)
def load_performance(hub_project: str) -> pd.DataFrame:
    """skore Hub project summary -> one row per logged report, with metric columns."""
    from skore import Project
    workspace = os.environ["SKORE_WORKSPACE"]
    frame = Project(f"{workspace}/{hub_project}", mode="hub").summarize().frame()
    return frame.reset_index(drop=True)


def backfill_roc_auc(df: pd.DataFrame) -> pd.DataFrame:
    """Recover roc_auc from legacy estimator-named columns in runs.db."""
    out = df.copy()
    if "roc_auc" not in out.columns:
        out["roc_auc"] = float("nan")
    out["roc_auc"] = pd.to_numeric(out["roc_auc"], errors="coerce")
    missing = out["roc_auc"].isna()
    for col in LEGACY_ROC_AUC_COLUMNS:
        if col not in out.columns:
            continue
        legacy = pd.to_numeric(out[col], errors="coerce")
        out.loc[missing, "roc_auc"] = legacy.loc[missing]
        missing = out["roc_auc"].isna()
    return out


def latest_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the most recent row per config (re-runs append new keys)."""
    if "ts" not in df.columns:
        return df.drop_duplicates(subset=["config"], keep="last")
    return (
        df.sort_values("ts")
        .drop_duplicates(subset=["config"], keep="last")
        .reset_index(drop=True)
    )


def performance_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c in SKORE_METRICS and pd.api.types.is_numeric_dtype(df[c])
    ]


def pick_default_metric(dataset: str, options: list[str]) -> str | None:
    preferred = DATASET_DEFAULT_METRIC.get(dataset, "roc_auc")
    if preferred in options:
        return preferred
    for fallback in ("roc_auc", "recall", "precision", "accuracy"):
        if fallback in options:
            return fallback
    return options[0] if options else None


def skill_pairs(configs: pd.Series) -> list[tuple[str, str]]:
    """Return (base, base+skills) config name pairs present in the data."""
    names = set(configs.dropna())
    pairs: list[tuple[str, str]] = []
    for base in sorted(c for c in names if not str(c).endswith("+skills")):
        skilled = f"{base}+skills"
        if skilled in names:
            pairs.append((base, skilled))
    return pairs


def _pct_change(before: float, after: float) -> float | None:
    if before == 0 or pd.isna(before) or pd.isna(after):
        return None
    return (after - before) / before * 100.0


def _performance_verdict(delta: float, *, tie_threshold: float = 0.005) -> str:
    if abs(delta) < tie_threshold:
        return "unchanged"
    return "improved" if delta > 0 else "lower"


def _cost_verdict(delta: float, pct: float | None, *, rel_tie_pct: float = 5.0) -> str:
    """delta = skilled − base on cost axis (lower is better)."""
    if pct is not None and abs(pct) < rel_tie_pct:
        return "similar cost"
    if pct is None and abs(delta) < 1e-9:
        return "similar cost"
    return "cheaper" if delta < 0 else "costlier"


def skills_outcome(d_perf: float, d_cost: float, pct_cost: float | None) -> str:
    """Classify a base → +skills pair as win, lose, or tie for chart coloring."""
    perf_v = _performance_verdict(d_perf)
    cost_v = _cost_verdict(d_cost, pct_cost)
    if perf_v == "lower":
        return "lose"
    if perf_v == "improved":
        return "win"
    if cost_v == "cheaper":
        return "win"
    if cost_v == "costlier":
        return "lose"
    return "tie"


SKILL_LINE_COLORS = {
    "win": "#22c55e",
    "lose": "#ef4444",
    "tie": "#94a3b8",
}


def build_scatter_with_skill_lines(
    df: pd.DataFrame,
    metric: str,
    x_axis: str,
    *,
    x_label: str,
    y_label: str,
) -> go.Figure:
    """Scatter of all configs with green/red/gray lines linking each model pair."""
    fig = go.Figure()
    line_legend_shown: set[str] = set()

    for base, skilled in skill_pairs(df["config"]):
        if base not in df["config"].values or skilled not in df["config"].values:
            continue
        row_b = df.loc[df["config"] == base].iloc[0]
        row_s = df.loc[df["config"] == skilled].iloc[0]
        d_perf = float(row_s[metric]) - float(row_b[metric])
        d_cost = float(row_s[x_axis]) - float(row_b[x_axis])
        outcome = skills_outcome(d_perf, d_cost, _pct_change(float(row_b[x_axis]), float(row_s[x_axis])))
        color = SKILL_LINE_COLORS[outcome]
        legend_label = {"win": "Skills win", "lose": "Skills lose", "tie": "Neutral"}[outcome]

        fig.add_trace(go.Scatter(
            x=[row_b[x_axis], row_s[x_axis]],
            y=[row_b[metric], row_s[metric]],
            mode="lines",
            line=dict(color=color, width=2.5),
            name=legend_label,
            legendgroup=outcome,
            showlegend=legend_label not in line_legend_shown,
            hovertemplate=(
                f"{base} → {skilled}<br>"
                f"{y_label}: {row_b[metric]:.3f} → {row_s[metric]:.3f}<br>"
                f"{x_label}: {row_b[x_axis]:,.0f} → {row_s[x_axis]:,.0f}"
                "<extra></extra>"
            ),
        ))
        line_legend_shown.add(legend_label)

    scatter = px.scatter(
        df,
        x=x_axis,
        y=metric,
        color="config",
        text="config",
        labels={x_axis: x_label, metric: y_label},
    )
    scatter.update_traces(textposition="top center", mode="markers+text")
    for trace in scatter.data:
        fig.add_trace(trace)

    fig.update_layout(
        title=f"{y_label} vs {x_label}",
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend=dict(title="Config"),
    )
    return fig


def analyze_skills(
    df: pd.DataFrame,
    metric: str,
    x_axis: str,
    *,
    dataset: str,
) -> str:
    """Build a short narrative comparing each model vs its +skills variant."""
    pairs = skill_pairs(df["config"])
    y_label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())
    x_name = "tokens burned" if x_axis == "tokens_burned" else "token cost (USD)"

    if not pairs:
        return (
            "**Skills analysis** — No paired runs found. "
            "Log both `model` and `model+skills` configs for this use case."
        )

    lines = ["**Skills analysis**", ""]
    perf_wins = perf_losses = perf_ties = 0
    cost_wins = cost_losses = 0
    best_efficiency: tuple[str, float] | None = None  # (config, metric / x_axis)

    for base, skilled in pairs:
        if base not in df["config"].values or skilled not in df["config"].values:
            continue
        row_b = df.loc[df["config"] == base].iloc[0]
        row_s = df.loc[df["config"] == skilled].iloc[0]
        perf_b, perf_s = float(row_b[metric]), float(row_s[metric])
        cost_b, cost_s = float(row_b[x_axis]), float(row_s[x_axis])
        d_perf = perf_s - perf_b
        d_cost = cost_s - cost_b
        pct_cost = _pct_change(cost_b, cost_s)

        perf_v = _performance_verdict(d_perf)
        cost_v = _cost_verdict(d_cost, pct_cost)

        if perf_v == "improved":
            perf_wins += 1
        elif perf_v == "lower":
            perf_losses += 1
        else:
            perf_ties += 1
        if cost_v == "cheaper":
            cost_wins += 1
        elif cost_v == "costlier":
            cost_losses += 1

        for cfg, perf_val, cost_val in ((base, perf_b, cost_b), (skilled, perf_s, cost_s)):
            if cost_val > 0:
                eff = perf_val / cost_val
                if best_efficiency is None or eff > best_efficiency[1]:
                    best_efficiency = (cfg, eff)

        perf_part = (
            f"{y_label} **{perf_v}** ({perf_b:.3f} → {perf_s:.3f}, "
            f"{'+' if d_perf >= 0 else ''}{d_perf:.3f})"
        )
        if pct_cost is not None:
            cost_part = f"agent cost **{cost_v}** ({cost_b:,.0f} → {cost_s:,.0f} {x_name}, {pct_cost:+.0f}%)"
        else:
            cost_part = f"agent cost **{cost_v}** ({cost_b:,.0f} → {cost_s:,.0f} {x_name})"

        if perf_v == "improved" and cost_v == "cheaper":
            takeaway = "Skills helped on both quality and cost."
        elif perf_v == "improved" and cost_v == "costlier":
            takeaway = "Skills traded higher agent cost for better model quality."
        elif perf_v == "lower" and cost_v == "cheaper":
            takeaway = "Skills reduced cost but hurt model quality."
        elif perf_v == "lower" and cost_v == "costlier":
            takeaway = "Skills hurt on both quality and cost for this model."
        elif perf_v == "unchanged" and cost_v == "cheaper":
            takeaway = "Similar model quality at lower agent cost."
        elif perf_v == "unchanged" and cost_v == "costlier":
            takeaway = "Similar model quality at higher agent cost."
        else:
            takeaway = "Mixed or neutral impact — compare the scatter plot."

        lines.append(f"- **{base}** → **{skilled}**: {perf_part}; {cost_part}. {takeaway}")

    n = len(pairs)
    lines.extend(["", "**Conclusion**"])
    if perf_wins > perf_losses:
        perf_summary = (
            f"Probabl skills **raised {y_label}** in {perf_wins}/{n} model families"
            + (f" ({perf_ties} tie{'s' if perf_ties != 1 else ''})" if perf_ties else "")
            + "."
        )
    elif perf_losses > perf_wins:
        perf_summary = (
            f"Skills **lowered {y_label}** in {perf_losses}/{n} model families"
            + (f" ({perf_ties} tie{'s' if perf_ties != 1 else ''})" if perf_ties else "")
            + "."
        )
    else:
        perf_summary = f"{y_label} was **mixed or flat** across model families ({perf_ties} ties)."

    if cost_wins > cost_losses:
        cost_summary = f"Skills **reduced {x_name}** in {cost_wins}/{n} pairs."
    elif cost_losses > cost_wins:
        cost_summary = f"Skills **increased {x_name}** in {cost_losses}/{n} pairs."
    else:
        cost_summary = f"Agent cost was **similar** with and without skills."

    lines.append(f"On **{dataset}**, {perf_summary} {cost_summary}")

    if best_efficiency is not None:
        cfg, _ = best_efficiency
        row = df.loc[df["config"] == cfg].iloc[0]
        skill_note = " (with skills)" if str(cfg).endswith("+skills") else ""
        lines.append(
            f"Best efficiency ({y_label} per unit {x_name}): **{cfg}**{skill_note} "
            f"({row[metric]:.3f} {y_label.lower()} at {row[x_axis]:,.0f} {x_name})."
        )

    return "\n".join(lines)


hub_error = hub_login()
if hub_error:
    st.sidebar.warning(f"Skore Hub: {hub_error} — using local runs.db only.")

cost = load_cost()
if cost.empty:
    st.warning("No runs yet. Run `python -m bench.run ...` first.")
    st.stop()

with st.sidebar:
    st.header("Plot axes")
    x_axis = st.radio(
        "Agent cost (x-axis)",
        options=["tokens_burned", "cost_usd"],
        index=0,
        format_func=lambda v: "Tokens burned" if v == "tokens_burned" else "Token cost (USD)",
    )
    x_label = "Tokens burned" if x_axis == "tokens_burned" else "Token cost (USD)"

for ds_name, sub in cost.groupby("dataset"):
    hub_project = hub_project_name(ds_name, prefix=PREFIX or None)
    st.subheader(f"{ds_name}  ·  hub project `{hub_project}`")
    if ds_name in DATASET_NOTES:
        st.caption(DATASET_NOTES[ds_name])

    df = latest_runs(sub)
    hub_note = None
    try:
        perf = load_performance(hub_project)
        if "key" in perf.columns:
            metric_cols = [c for c in perf.columns if c in SKORE_METRICS]
            merge_cols = ["key", *metric_cols]
            df = df.drop(columns=[c for c in metric_cols if c in df.columns], errors="ignore")
            df = df.merge(perf[merge_cols], on="key", how="left")
        else:
            hub_note = "Hub summary has no `key` column — cannot merge performance metrics."
    except Exception as exc:
        hub_note = f"Hub unavailable for `{hub_project}`: {exc}"

    if hub_note and hub_error is None:
        st.info(hub_note)
    elif hub_note:
        st.caption(hub_note)

    df = backfill_roc_auc(df)
    metric_options = performance_columns(df)
    default_metric = pick_default_metric(ds_name, metric_options)
    if not metric_options:
        st.warning("No skore performance metrics found for this dataset.")
        st.dataframe(
            df[["config", "model", "use_skills", "tokens_burned", "cost_usd"]],
            use_container_width=True,
        )
        continue

    metric = st.selectbox(
        "Performance metric (y-axis)",
        metric_options,
        index=metric_options.index(default_metric),
        format_func=lambda m: METRIC_LABELS.get(m, m.replace("_", " ").title()),
        key=f"metric-{ds_name}",
    )
    y_label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())

    plot_df = df.dropna(subset=[metric, x_axis], how="any")
    if plot_df.empty:
        st.warning(f"No rows with both `{metric}` and `{x_axis}`.")
        st.dataframe(df, use_container_width=True)
        continue

    fig = build_scatter_with_skill_lines(
        plot_df, metric, x_axis, x_label=x_label, y_label=y_label,
    )
    st.caption(
        "Lines link each model to its +skills variant: "
        "**green** = skills win (better or equal quality, lower or equal cost), "
        "**red** = skills lose, **gray** = neutral."
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        plot_df[["config", "model", "use_skills", "tokens_burned", "cost_usd", metric]].sort_values(x_axis),
        use_container_width=True,
    )
    st.markdown(analyze_skills(plot_df, metric, x_axis, dataset=ds_name))

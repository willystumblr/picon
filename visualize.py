import os
import json
from typing import List, Dict, Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ==============================
# Data loading
# ==============================

def read_json_safe(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return {}


def load_metrics_for_baseline(base_dir: str, baseline: str) -> pd.DataFrame:
    """
    Load required metrics from JSON files under {base_dir}/{baseline}.
    Returns a DataFrame with one row per file.
    Columns:
      - baseline
      - file
      - external_total_evaluations
      - external_consistency_rate
      - internal_consistency_rate
      - inter_session_score
    """
    dir_path = os.path.join(base_dir, baseline)
    rows = []

    if not os.path.isdir(dir_path):
        return pd.DataFrame(
            columns=[
                "baseline",
                "file",
                "external_total_evaluations",
                "external_consistency_rate",
                "internal_consistency_rate",
                "inter_session_score",
            ]
        )

    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(dir_path, fname)
        data = read_json_safe(fpath)

        ext = data.get("external_consistency", {}) or {}
        intl = data.get("internal_consistency", {}) or {}
        inter = data.get("inter_session_score", {}) or {}

        rows.append(
            {
                "baseline": baseline,
                "file": fname,
                "external_total_evaluations": ext.get("total_evaluations"),
                "external_consistency_rate": ext.get("consistency_rate"),
                "internal_consistency_rate": intl.get("consistency_rate"),
                "inter_session_score": inter.get("inter_session_score"),
            }
        )

    return pd.DataFrame(rows)


def load_all_baselines(base_dir: str, baselines: List[str]) -> pd.DataFrame:
    parts = [load_metrics_for_baseline(base_dir, b) for b in baselines]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(
            columns=[
                "baseline",
                "file",
                "external_total_evaluations",
                "external_consistency_rate",
                "internal_consistency_rate",
                "inter_session_score",
            ]
        )
    return pd.concat(parts, ignore_index=True)


# ==============================
# Plot helpers
# ==============================

def box_with_mean(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    plot_df = df[[x_col, y_col]].dropna()
    if plot_df.empty:
        st.info(f"No valid data for {y_col}.")
        return

    fig = px.box(plot_df, x=x_col, y=y_col, points="all", title=title)
    mean_df = plot_df.groupby(x_col, as_index=False)[y_col].mean()
    fig.add_trace(
        go.Scatter(
            x=mean_df[x_col],
            y=mean_df[y_col],
            mode="markers+text",
            marker_symbol="diamond",
            marker_size=10,
            marker_color="black",
            text=mean_df[y_col].round(4).astype(str),
            textposition="top center",
            name="mean",
        )
    )
    st.plotly_chart(fig, use_container_width=True)


def bar_sum_by_group(df: pd.DataFrame, group_col: str, value_col: str, title: str, y_title: str):
    plot_df = df[[group_col, value_col]].dropna()
    if plot_df.empty:
        st.info(f"No valid data for {value_col}.")
        return

    agg = plot_df.groupby(group_col, as_index=False)[value_col].sum()
    fig = px.bar(agg, x=group_col, y=value_col, text=value_col, title=title)
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_title=y_title)
    st.plotly_chart(fig, use_container_width=True)


# ==============================
# Streamlit App
# ==============================

st.set_page_config(page_title="Consistency Metrics (2025-11-16)", layout="wide")
st.title("Consistency Metrics Dashboard — 2025-11-16")

with st.form("controls"):
    base_dir = st.text_input("Base directory", value="data/2025_11_16")
    default_baselines = ["characterai", "human_simulacra", "opencharacter"]
    baselines = st.multiselect(
        "Baselines (child directories)",
        options=default_baselines,
        default=default_baselines,
    )
    submitted = st.form_submit_button("Load")

if not submitted:
    st.stop()

df = load_all_baselines(base_dir, baselines)

if df.empty:
    st.warning("No metrics found. Verify base directory and baselines.")
    st.stop()

st.subheader("Per-file Metrics")
st.dataframe(
    df.sort_values(["baseline", "file"]).reset_index(drop=True),
    use_container_width=True,
)

# Aggregated summary by baseline
st.subheader("Summary by Baseline")
summary = (
    df.groupby("baseline", as_index=False)
    .agg(
        external_total_evaluations_sum=("external_total_evaluations", "sum"),
        external_consistency_rate_mean=("external_consistency_rate", "mean"),
        internal_consistency_rate_mean=("internal_consistency_rate", "mean"),
        inter_session_score_mean=("inter_session_score", "mean"),
        files=("file", "count"),
    )
)
st.dataframe(summary, use_container_width=True)

st.markdown("---")
st.header("Visualizations")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### External — total_evaluations (sum by baseline)")
    bar_sum_by_group(
        df,
        group_col="baseline",
        value_col="external_total_evaluations",
        title="External total_evaluations — sum by baseline",
        y_title="total_evaluations (sum)",
    )

with col2:
    st.markdown("### External — consistency_rate (per-file distribution)")
    box_with_mean(
        df,
        x_col="baseline",
        y_col="external_consistency_rate",
        title="External consistency_rate by baseline",
    )

col3, col4 = st.columns(2)
with col3:
    st.markdown("### Internal — consistency_rate (per-file distribution)")
    box_with_mean(
        df,
        x_col="baseline",
        y_col="internal_consistency_rate",
        title="Internal consistency_rate by baseline",
    )

with col4:
    st.markdown("### Inter-session — inter_session_score (per-file distribution)")
    box_with_mean(
        df,
        x_col="baseline",
        y_col="inter_session_score",
        title="Inter-session score by baseline",
    )

st.markdown(
    """
    - external_consistency.total_evaluations → summed per baseline (bar)
    - external_consistency.consistency_rate → per-file distribution (box + mean)
    - internal_consistency.consistency_rate → per-file distribution (box + mean)
    - inter_session_score.inter_session_score → per-file distribution (box + mean)
    """
)


# ==============================
# Abstain results (data/2025_11_16/abstain_results)
# ==============================

st.markdown("---")
st.header("Abstain Results — 2025-11-16")

def _is_abstain_true(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        lv = v.strip().lower()
        return lv in {"true", "partially true", "partially_true", "partial", "yes"}
    return False


def load_abstain_for_baseline(abstain_base_dir: str, baseline: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (per_file_df, per_file_type_counts_df)
    per_file_df columns: baseline, file, total_count, abstain_true_count, abstain_rate
    per_file_type_counts_df columns: baseline, file, abstain_type, count
    """
    dir_path = os.path.join(abstain_base_dir, baseline)
    per_file_rows = []
    type_rows = []

    if not os.path.isdir(dir_path):
        return (
            pd.DataFrame(columns=["baseline", "file", "total_count", "abstain_true_count", "abstain_rate"]),
            pd.DataFrame(columns=["baseline", "file", "abstain_type", "count"]),
        )

    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(dir_path, fname)
        data = read_json_safe(fpath)

        # Prefer explicit abstain_rate if present
        explicit_rate = data.get("abstain_rate")
        results = data.get("results", [])
        total = len(results)

        # Count abstain-true items (including partial true)
        abstain_true_records = []
        type_counter: Dict[str, int] = {}
        for r in results:
            a = r.get("abstain")
            if _is_abstain_true(a):
                abstain_true_records.append(r)
                t = r.get("abstain_type")
                if t is None:
                    continue
                t_str = str(t).strip()
                if not t_str or t_str.lower() == "none":
                    continue
                type_counter[t_str] = type_counter.get(t_str, 0) + 1

        true_count = len(abstain_true_records)
        computed_rate = (true_count / total) if total else 0.0
        rate = explicit_rate if isinstance(explicit_rate, (int, float)) else computed_rate

        per_file_rows.append(
            {
                "baseline": baseline,
                "file": fname,
                "total_count": total,
                "abstain_true_count": true_count,
                "abstain_rate": rate,
            }
        )

        for t, c in type_counter.items():
            type_rows.append({"baseline": baseline, "file": fname, "abstain_type": t, "count": c})

    return pd.DataFrame(per_file_rows), pd.DataFrame(type_rows)


def load_abstain_all(abstain_base_dir: str, baselines: List[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts_files = []
    parts_types = []
    for b in baselines:
        df_f, df_t = load_abstain_for_baseline(abstain_base_dir, b)
        if not df_f.empty:
            parts_files.append(df_f)
        if not df_t.empty:
            parts_types.append(df_t)
    files_df = pd.concat(parts_files, ignore_index=True) if parts_files else pd.DataFrame(
        columns=["baseline", "file", "total_count", "abstain_true_count", "abstain_rate"]
    )
    types_df = pd.concat(parts_types, ignore_index=True) if parts_types else pd.DataFrame(
        columns=["baseline", "file", "abstain_type", "count"]
    )
    return files_df, types_df


# Load and visualize abstain
abstain_base_dir = os.path.join("data", "2025_11_16", "abstain_results")
abs_files_df, abs_types_df = load_abstain_all(abstain_base_dir, baselines)

if abs_files_df.empty and abs_types_df.empty:
    st.info("No abstain data found under data/2025_11_16/abstain_results for selected baselines.")
else:
    st.subheader("Abstain — Per-file Summary")
    st.dataframe(
        abs_files_df.sort_values(["baseline", "abstain_rate", "file"], ascending=[True, False, True]).reset_index(drop=True),
        use_container_width=True,
    )

    st.markdown("### Abstain rate (per-file distribution)")
    box_with_mean(
        abs_files_df,
        x_col="baseline",
        y_col="abstain_rate",
        title="Abstain rate by baseline",
    )

    st.markdown("### Average number of each abstain_type per file")
    if abs_types_df.empty:
        st.info("No abstain types present (after filtering).")
    else:
        avg_types = (
            abs_types_df
            .groupby(["baseline", "abstain_type"], as_index=False)["count"]
            .mean()
            .rename(columns={"count": "avg_count_per_file"})
        )
        fig_types = px.bar(
            avg_types,
            x="abstain_type",
            y="avg_count_per_file",
            color="baseline",
            barmode="group",
            text=avg_types["avg_count_per_file"].round(2),
            title="Average count per file by abstain_type and baseline",
        )
        fig_types.update_traces(textposition="outside")
        fig_types.update_layout(yaxis_title="avg count per file")
        st.plotly_chart(fig_types, use_container_width=True)

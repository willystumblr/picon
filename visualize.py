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

        ext = data.get("external_consistency", {})
        intl = data.get("internal_consistency", {})
        rpt = data.get("repeat_score", {})

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
            textfont=dict(
                size=14,         # ← 글씨 크게
                color="black",   # ← 글씨 색 진하게
                family="Arial"   # (선택) 좀 읽기 쉬운 폰트
            ),
            name="mean"
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
    - external_consistency_rate = external_consistency.consistency_rate  
    - internal_consistency_rate = internal_consistency.consistency_rate  
    - repeat_score = repeat.repeat_score  
    - 검정색 다이아몬드 = 각 그룹의 평균  
    """
)

import os
import json
from typing import List, Dict, Any, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import mannwhitneyu, ttest_ind

import logging

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ==============================
# Data loading (수정 없음)
# ==============================

def read_json_safe(path: str) -> Dict[str, Any]:
    """JSON 파일을 안전하게 읽습니다."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return {}


def do_eval(baseline, final_result, model, path) -> Dict[str, Any]:
    """EvaluatorTestEnv를 사용하여 평가를 수행하고 결과를 반환합니다."""
    from src.env.evaluator_test_env import EvaluatorTestEnv
    from src.utils import write_json
    evaluator_env = EvaluatorTestEnv(
        model=model,
        interview_path=path,
    )
    try:
        evaluator_env.reset()
        evaluator_env.step()
        final_result["first_conflict_turn"] = evaluator_env.first_conflict_turn
        if "external_consistency" not in final_result:
            final_result["external_consistency"] = {
                "mean_evaluations": evaluator_env.external_count,
                "conflict_count": evaluator_env.external_conflict,
                "plausible_count": evaluator_env.external_plausible,
                "consistency_rate": (evaluator_env.external_plausible / evaluator_env.external_count) if evaluator_env.external_count > 0 else None,
                "conflict_verdicts": evaluator_env.external_conflict_verdicts
            }
        if "internal_consistency" not in final_result:
            final_result["internal_consistency"] = {
                "mean_evaluations": evaluator_env.internal_count,
                "conflict_count": evaluator_env.internal_conflict,
                "plausible_count": evaluator_env.internal_plausible,
                "consistency_rate": (evaluator_env.internal_plausible / evaluator_env.internal_count) if evaluator_env.internal_count > 0 else None,
                "conflict_verdicts": evaluator_env.internal_conflict_verdicts
            }
        if "inter_session_score" not in final_result:
            final_result["inter_session_score"] = {
                "inter_session_score": evaluator_env.inter_session_score,
                "inter_session_results": evaluator_env.inter_session_results
            }
        if "abstention_analysis" not in final_result:
            final_result["abstention_analysis"] = {
                "abstention_rate": evaluator_env.abstention_rate,
                "abstention_results": evaluator_env.abstention_results
            }
        write_json(final_result, f"/home/edlab/sjim/interrogation_eval/data/{baseline}/{os.path.basename(path)}")
        return final_result
    except Exception as e:
        # error details logging
        logger.error(f"Error during evaluation for {path}: {e}", exc_info=True)
        return {}

def load_metrics_for_baseline(baseline_name: str, full_path: str) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    주어진 전체 경로(full_path)에서 JSON 파일을 한 번만 로드하고,
    메트릭 DataFrame과 abstention 데이터를 함께 반환합니다.
    Returns: (metrics_df, abstention_data_list)
    """
    dir_path = full_path
    rows = []
    abstention_data = []

    columns = [
        "baseline",
        "file",
        "external_mean_evaluations",
        "external_consistency_rate",
        "internal_consistency_rate",
        "repeat_score",
        "inter_session_score",
        "abstention_rate",
    ]

    if not os.path.isdir(dir_path):
        return pd.DataFrame(columns=columns), []

    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(dir_path, fname)
        data = read_json_safe(fpath)

        if "external_consistency" not in data or "internal_consistency" not in data or "abstention_analysis" not in data:
            print(f"[INFO] Missing keys in {fpath}, re-evaluating... (current keys: {list(data.keys())})")
            data = do_eval(baseline_name, data, model="gpt-5", path=fpath)

        # 1. 외부/내부 일관성 로딩
        ext = data.get("external_consistency", {})
        intl = data.get("internal_consistency", {})
        
        # 2. 반복 점수 로딩
        repeat_score = data.get("repeat_score", {}).get("repeat_score")
        if repeat_score is None:
            repeat_score = data.get("repeat", {}).get("repeat_score")
        if repeat_score is None:
            repeat_score = data.get("session_1", {}).get("repeat", {}).get("repeat_score")
        
        # 3. inter_session_score 로딩
        inter_session_score = data.get("inter_session_score", {}).get("inter_session_score")
        
        # 4. abstention_rate 로딩
        abstention_rate = data.get("abstention_analysis", {}).get("abstention_rate")
        
        # 5. abstention_results 로딩 (한 번에 처리)
        abstention_results = data.get("abstention_analysis", {}).get("abstention_results", [])
        for result in abstention_results:
            abstain_type = result.get("abstain_type", "unknown")
            abstention_data.append({
                "baseline": baseline_name,
                "file": fname,
                "abstain_type": abstain_type
            })
            
        rows.append({
            "baseline": baseline_name,
            "file": fname,
            "external_mean_evaluations": ext.get("total_evaluations"),
            "external_consistency_rate": ext.get("consistency_rate"),
            "internal_consistency_rate": intl.get("consistency_rate"),
            "repeat_score": repeat_score,
            "inter_session_score": inter_session_score,
            "abstention_rate": abstention_rate,
        })

    return pd.DataFrame(rows), abstention_data


# ==============================
# Plot helpers (수정 없음)
# ==============================

def box_with_mean(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    """그룹별 Box Plot과 평균 Scatter를 함께 표시합니다."""
    plot_df = df[[x_col, y_col]].dropna()
    if plot_df.empty:
        st.info(f"No valid data for {y_col}.")
        return

    color_map = {
        "human_interview": "#FFD700",
        "characterai": "#1f77b4",
        "human_simulacra": "#ff7f0e",
        "opencharacter": "#2ca02c",
        "consistent_llm": "#9467bd"
    }
    
    category_order = ["characterai", "human_simulacra", "opencharacter", "consistent_llm", "human_interview"]
    
    fig = px.box(plot_df, x=x_col, y=y_col, points="all", title=title,
                 color=x_col, color_discrete_map=color_map,
                 category_orders={x_col: category_order})
    
    fig.update_layout(
        font=dict(size=16),
        xaxis=dict(tickfont=dict(size=14)),
        yaxis=dict(tickfont=dict(size=14)),
        title_font=dict(size=16),
        margin=dict(l=80, r=80, t=100, b=80),
        height=600
    )
    
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
            textfont=dict(size=16, color="black", family="Arial"),
            name="Mean"
        )
    )
    st.plotly_chart(fig, use_container_width=True)


def bar_mean_by_group(df: pd.DataFrame, group_col: str, value_col: str, title: str, y_title: str):
    """그룹별 평균을 표시하는 Bar Chart를 생성합니다."""
    plot_df = df[[group_col, value_col]].dropna()
    if plot_df.empty:
        st.info(f"No valid data for {value_col}.")
        return

    color_map = {
        "human_interview": "#FFD700",
        "characterai": "#1f77b4",
        "human_simulacra": "#ff7f0e",
        "opencharacter": "#2ca02c",
        "consistent_llm": "#9467bd"
    }

    category_order = ["characterai", "human_simulacra", "opencharacter", "consistent_llm", "human_interview"]

    agg = plot_df.groupby(group_col, as_index=False)[value_col].mean()
    agg[value_col] = agg[value_col].round(2)
    fig = px.bar(agg, x=group_col, y=value_col, text=value_col, title=title,
                 color=group_col, color_discrete_map=color_map,
                 category_orders={group_col: category_order})
    fig.update_traces(textposition="outside", textfont_size=16)
    fig.update_layout(
        yaxis_title=y_title,
        font=dict(size=16),
        xaxis=dict(tickfont=dict(size=14)),
        yaxis=dict(tickfont=dict(size=14)),
        title_font=dict(size=16),
        margin=dict(l=80, r=80, t=120, b=120),
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)


# ==============================
# Streamlit App
# ==============================
print("Starting Streamlit app...")
st.set_page_config(page_title="Consistency Metrics (2025-11-16)", layout="wide")
st.title("Consistency Metrics Dashboard — 2025-11-16")
print("Page configured.")
# ------------------------------
# 1. Baseline 선택 및 개별 경로 입력
# ------------------------------
with st.form("controls"):
    
    st.subheader("1. 분석할 베이스라인 선택")
    default_baselines = ["characterai", "human_simulacra", "opencharacter", "human_interview", "consistent_llm"]
    selected_baselines = st.multiselect(
        "베이스라인 선택",
        options=default_baselines,
        default=default_baselines,
    )

    baseline_paths = {}
    if selected_baselines:
        st.subheader("2. 각 베이스라인의 결과 경로 입력")
        for i, b in enumerate(selected_baselines):
            path_input = st.text_input(
                f"'{b}'의 경로 (예: data/2025_11_16/{b})",
                value=f"/home/edlab/sjim/interrogation_eval/data/{b}",
                key=f"path_input_{b}"
            )
            baseline_paths[b] = path_input

    submitted = st.form_submit_button("3. 데이터 로드 및 시각화")

if not submitted:
    st.stop()

# ------------------------------
# 데이터 로드 (한 번만 파일 읽기)
# ------------------------------
all_parts = []
all_abstention_data = []
loading_errors = []



for name, path in baseline_paths.items():
    if path:
        df_part, abstention_part = load_metrics_for_baseline(name, path)
        if df_part.empty:
            loading_errors.append(f"경로 '{path}'에서 '{name}' 베이스라인의 데이터를 찾을 수 없습니다.")
        else:
            all_parts.append(df_part)
            all_abstention_data.extend(abstention_part)
    else:
        loading_errors.append(f"'{name}' 베이스라인의 경로가 비어있습니다. 건너뜁니다.")

if loading_errors:
    for err in loading_errors:
        st.warning(err)

df = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame(
    columns=[
        "baseline", "file", "external_mean_evaluations", "external_consistency_rate",
        "internal_consistency_rate", "repeat_score", "inter_session_score", "abstention_rate",
    ]
)

if df.empty:
    st.error("분석할 메트릭을 찾지 못했습니다. 경로를 확인해주세요.")
    st.stop()

st.success(f"총 {len(df)}개 파일의 메트릭을 성공적으로 로드했습니다.")


# ------------------------------
# 경로 재설정 UI
# ------------------------------
st.markdown("---")
st.subheader("경로 재설정 (선택사항)")

with st.expander("베이스라인별 경로 재설정하기"):
    with st.form("reconfig_form"):
        st.write("현재 로드된 베이스라인의 경로를 변경하고 다시 로드할 수 있습니다.")
        
        new_baseline_paths = {}
        for baseline in selected_baselines:
            current_path = baseline_paths.get(baseline, f"data/2025_11_16/{baseline}")
            new_path = st.text_input(
                f"'{baseline}'의 새 경로",
                value=current_path,
                key=f"reconfig_{baseline}"
            )
            new_baseline_paths[baseline] = new_path
        
        resubmit = st.form_submit_button("경로 업데이트 및 재로드")
        
        if resubmit:
            all_parts = []
            all_abstention_data = []
            loading_errors = []
            
            for name, path in new_baseline_paths.items():
                if path:
                    df_part, abstention_part = load_metrics_for_baseline(name, path)
                    if df_part.empty:
                        loading_errors.append(f"경로 '{path}'에서 '{name}' 베이스라인의 데이터를 찾을 수 없습니다.")
                    else:
                        all_parts.append(df_part)
                        all_abstention_data.extend(abstention_part)
                else:
                    loading_errors.append(f"'{name}' 베이스라인의 경로가 비어있습니다. 건너킵니다.")
            
            if loading_errors:
                for err in loading_errors:
                    st.warning(err)
            
            df = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame(
                columns=[
                    "baseline", "file", "external_mean_evaluations", "external_consistency_rate",
                    "internal_consistency_rate", "repeat_score", "inter_session_score", "abstention_rate",
                ]
            )
            
            if df.empty:
                st.error("분석할 메트릭을 찾지 못했습니다. 경로를 확인해주세요.")
            else:
                st.success(f"총 {len(df)}개 파일의 메트릭을 성공적으로 재로드했습니다.")
                baseline_paths = new_baseline_paths

# ------------------------------
# 데이터 표시 및 시각화 (수정 없음)
# ------------------------------

st.markdown("---")
st.subheader("Per-file Metrics")
st.dataframe(
    df.sort_values(["baseline", "file"]).reset_index(drop=True),
    use_container_width=True,
)

st.subheader("Summary by Baseline")
summary = (
    df.groupby("baseline", as_index=False)
    .agg(
        external_mean_evaluations_mean=("external_mean_evaluations", "mean"),
        external_consistency_rate_mean=("external_consistency_rate", "mean"),
        internal_consistency_rate_mean=("internal_consistency_rate", "mean"),
        repeat_score_mean=("repeat_score", "mean"),
        inter_session_score_mean=("inter_session_score", "mean"),
        abstention_rate_mean=("abstention_rate", "mean"),
        files=("file", "count"),
    )
)
st.dataframe(summary, use_container_width=True)

st.markdown("---")
st.header("Visualizations")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### External — mean_evaluations (baseline별 평균)")
    bar_mean_by_group(
        df,
        group_col="baseline",
        value_col="external_mean_evaluations",
        title="External mean_evaluations — baseline별 평균",
        y_title="mean_evaluations (평균)",
    )

with col2:
    st.markdown("### External — consistency_rate (파일별 분포)")
    box_with_mean(
        df,
        x_col="baseline",
        y_col="external_consistency_rate",
        title="External consistency_rate (baseline별)",
    )

col3, col4 = st.columns(2)
with col3:
    st.markdown("### Internal — consistency_rate (파일별 분포)")
    box_with_mean(
        df,
        x_col="baseline",
        y_col="internal_consistency_rate",
        title="Internal consistency_rate (baseline별)",
    )

with col4:
    st.markdown("### Repeat — repeat_score (파일별 분포)")
    box_with_mean(
        df,
        x_col="baseline",
        y_col="repeat_score",
        title="Repeat score (baseline별)",
    )

col5, col6 = st.columns(2)
with col5:
    st.markdown("### Inter-Session Score (파일별 분포)")
    box_with_mean(
        df,
        x_col="baseline",
        y_col="inter_session_score",
        title="Inter-Session Score (baseline별)",
    )

with col6:
    st.markdown("### Abstention Rate (파일별 분포)")
    box_with_mean(
        df,
        x_col="baseline",
        y_col="abstention_rate",
        title="Abstention Rate (baseline별)",
    )


# ------------------------------
# 통계적 유의성 검정 (수정 없음)
# ------------------------------
st.markdown("---")
st.header("Statistical Significance Tests")
st.markdown("### Comparison: t-test vs Mann-Whitney U Test (vs human_interview)")

metrics_to_test = [
    "external_consistency_rate",
    "internal_consistency_rate", 
    "repeat_score",
    "inter_session_score",
    "abstention_rate"
]

human_data = df[df["baseline"] == "human_interview"]
other_baselines = [b for b in selected_baselines if b != "human_interview"]

stat_results = []

for metric in metrics_to_test:
    human_values = human_data[metric].dropna()
    
    if len(human_values) < 2:
        continue
        
    for baseline in other_baselines:
        baseline_data = df[df["baseline"] == baseline]
        baseline_values = baseline_data[metric].dropna()
        
        if len(baseline_values) < 2:
            continue
            
        t_statistic, t_pvalue = ttest_ind(baseline_values, human_values)
        u_statistic, u_pvalue = mannwhitneyu(baseline_values, human_values, alternative='two-sided')
        
        mean_diff = baseline_values.mean() - human_values.mean()
        
        t_sig = "***" if t_pvalue < 0.001 else "**" if t_pvalue < 0.01 else "*" if t_pvalue < 0.05 else "ns"
        u_sig = "***" if u_pvalue < 0.001 else "**" if u_pvalue < 0.01 else "*" if u_pvalue < 0.05 else "ns"
            
        stat_results.append({
            "Metric": metric,
            "Baseline": baseline,
            "Baseline Mean": f"{baseline_values.mean():.4f}",
            "Human Mean": f"{human_values.mean():.4f}",
            "Mean Diff": f"{mean_diff:.4f}",
            "n (Baseline)": len(baseline_values),
            "n (Human)": len(human_values),
            "t-test p-value": f"{t_pvalue:.4f}",
            "t-test Sig": t_sig,
            "Mann-Whitney p-value": f"{u_pvalue:.4f}",
            "Mann-Whitney Sig": u_sig,
        })

stat_df = pd.DataFrame(stat_results)

if not stat_df.empty:
    for metric in metrics_to_test:
        metric_data = stat_df[stat_df["Metric"] == metric]
        if not metric_data.empty:
            st.markdown(f"#### {metric}")
            display_df = metric_data.drop(columns=["Metric"])
            st.dataframe(display_df, use_container_width=True)
            st.markdown("---")
    
    st.markdown("""
    **유의수준:**
    - `***` p < 0.001 (매우 유의함)
    - `**` p < 0.01 (유의함)
    - `*` p < 0.05 (약간 유의함)
    - `ns` p ≥ 0.05 (유의하지 않음)
    
    **Mean Diff**: (Baseline 평균) - (Human 평균). 양수면 Baseline이 더 높음.
    
    **t-test**: 정규분포 가정, 평균 차이 검정 (모수 검정)
    
    **Mann-Whitney U**: 정규분포 가정 불필요, 순위 기반 검정 (비모수 검정)
    
    **해석**: 두 검정 결과가 비슷하면 데이터가 정규분포에 가깝고, 다르면 분포가 편향되었거나 이상치가 있을 가능성이 높습니다.
    """)
else:
    st.info("통계 검정을 수행할 충분한 데이터가 없습니다.")

# ------------------------------
# Abstention Type 분석 (수정된 부분)
# ------------------------------
st.markdown("---")
st.header("Abstention Type Analysis")

abstention_df = pd.DataFrame(all_abstention_data)
abstention_df = abstention_df[abstention_df["abstain_type"].notna() & (abstention_df["abstain_type"] != "none")]

if not abstention_df.empty:
    abstain_count = abstention_df.groupby(["baseline", "abstain_type"], as_index=False).size()
    abstain_count.columns = ["baseline", "abstain_type", "count"]
    
    color_map = {
        "human_interview": "#FFD700",
        "characterai": "#1f77b4",
        "human_simulacra": "#ff7f0e",
        "opencharacter": "#2ca02c",
        "consistent_llm": "#9467bd"
    }
    
    category_order = ["characterai", "human_simulacra", "opencharacter", "consistent_llm", "human_interview"]

    # 그래프를 넣을 두 개의 컬럼 생성
    col_abstain_1, col_abstain_2 = st.columns(2)

    # 1. Grouped Bar Chart
    with col_abstain_1:
        st.subheader("Abstention Type Count by Baseline")
        fig = px.bar(
            abstain_count,
            x="baseline",
            y="count",
            color="abstain_type",
            title="Grouped Bar Chart",
            barmode="group",
            category_orders={"baseline": category_order}
        )
        
        fig.update_layout(
            font=dict(size=14),
            xaxis=dict(tickfont=dict(size=12), tickangle=-45, title="Baseline"),
            yaxis=dict(tickfont=dict(size=12), title="Count"),
            title_font=dict(size=14),
            margin=dict(l=50, r=50, t=60, b=80),
            height=400, # 그래프 높이도 줄임
            legend=dict(title="Abstain Type", font=dict(size=12), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # use_container_width=True를 사용하여 컬럼의 너비에 맞춤 (가로 길이 줄이는 핵심)
        st.plotly_chart(fig, use_container_width=True)
    
    # 2. Stacked Bar Chart
    with col_abstain_2:
        st.subheader("Abstention Type Distribution by Baseline")
        fig2 = px.bar(
            abstain_count,
            x="baseline",
            y="count",
            color="abstain_type",
            title="Stacked Bar Chart",
            barmode="stack",
            category_orders={"baseline": category_order}
        )
        
        fig2.update_layout(
            font=dict(size=14),
            xaxis=dict(tickfont=dict(size=12), tickangle=-45, title="Baseline"),
            yaxis=dict(tickfont=dict(size=12), title="Count"),
            title_font=dict(size=14),
            margin=dict(l=50, r=50, t=60, b=80),
            height=400, # 그래프 높이도 줄임
            legend=dict(title="Abstain Type", font=dict(size=12), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig2, use_container_width=True)

    # 테이블은 컬럼 없이 전체 너비로 표시
    st.subheader("Abstention Type Count Table")
    pivot_table = abstain_count.pivot(index="baseline", columns="abstain_type", values="count").fillna(0).astype(int)
    pivot_table = pivot_table.reindex(category_order)
    st.dataframe(pivot_table, use_container_width=True)
    
else:
    st.info("Abstention type 데이터를 찾을 수 없습니다.")

st.markdown("---")
st.markdown("""
- external_consistency_rate = external\_consistency.consistency\_rate  
- internal_consistency\_rate = internal\_consistency.consistency\_rate  
- **repeat\_score** = `repeat.repeat_score` 또는 `repeat_score.repeat_score` (두 구조 중 하나에서 로드됨)
- **inter\_session\_score** = `inter_session_score.inter_session_score`
- **abstention\_rate** = `abstention_analysis.abstention_rate`
- 검정색 다이아몬드 = 각 그룹의 평균""")
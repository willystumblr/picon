import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

#################################
# 1. 데이터 로더 (폴더 하나)
#################################

def load_metrics_from_dir(dir_path: str, source_label: str) -> pd.DataFrame:
    """
    dir_path 안의 모든 .json 파일에서 metric들을 수집해서
    DataFrame으로 반환. 이 때 'source' 컬럼에 source_label을 붙인다.
    """
    rows = []

    if not os.path.isdir(dir_path):
        # 폴더 없으면 빈 df
        return pd.DataFrame(columns=[
            "file",
            "external_consistency_rate",
            "internal_consistency_rate",
            "repeat_score",
            "source"
        ])

    for fname in os.listdir(dir_path):
        if not fname.endswith(".json"):
            continue

        fpath = os.path.join(dir_path, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] {fname} load error: {e}")
            continue

        ext = data.get("external_consistency", {})
        intl = data.get("internal_consistency", {})
        rpt = data.get("repeat_score", {})

        rows.append({
            "file": fname,
            "external_consistency_rate": ext.get("consistency_rate", None),
            "internal_consistency_rate": intl.get("consistency_rate", None),
            "repeat_score": rpt.get("repeat_score", None),
            "source": source_label.split('/')[-1],  # 폴더명만 쓰기
        })

    return pd.DataFrame(rows)


#################################
# 2. 여러 폴더 한 번에 로드
#################################

def load_from_multiple_dirs(dir_list_raw: str) -> pd.DataFrame:
    """
    "cai, cai_v2, experiment/run3" 처럼 콤마로 구분된 입력 문자열을 받아서
    각 폴더에서 df를 로드하고 concat.
    """
    # 공백 제거 + 빈 문자열 제거
    dirs = [d.strip() for d in dir_list_raw.split(",") if d.strip()]

    dfs = []
    for d in dirs:
        df_d = load_metrics_from_dir(d, source_label=d)
        dfs.append(df_d)

    if not dfs:
        return pd.DataFrame(columns=[
            "file",
            "external_consistency_rate",
            "internal_consistency_rate",
            "repeat_score",
            "source"
        ])

    return pd.concat(dfs, ignore_index=True)


#################################
# 3. Plot helper들
#################################

def boxplot_metric_multi(df: pd.DataFrame, value_col: str, title: str):
    """
    여러 폴더(source)에서 같은 metric(value_col)을 비교하는 box plot.
    x축: source (폴더명), y축: metric 값
    그리고 각 source별 mean 값을 box 위에 다이아몬드 마커+텍스트로 overlay.
    """
    plot_df = df[["source", value_col]].dropna()

    if plot_df.empty:
        st.info(f"{value_col} 에 유효한 값이 없습니다.")
        return

    # 1) 기본 box plot
    fig = px.box(
        plot_df,
        x="source",
        y=value_col,
        points="all",
        title=title
    )

    # 2) source별 평균 계산
    mean_df = plot_df.groupby("source", as_index=False)[value_col].mean()

    # 3) 평균 overlay
    fig.add_trace(
        go.Scatter(
            x=mean_df["source"],
            y=mean_df[value_col],
            mode="markers+text",
            marker_symbol="diamond",
            marker_size=9,
            marker_color="black",
            text=mean_df[value_col].round(4).astype(str),
            textposition="top center",
            textfont=dict(
                size=12,         # ← 글씨 크게
                color="black",   # ← 글씨 색 진하게
                family="Arial"   # (선택) 좀 읽기 쉬운 폰트
            ),
            name="mean"
        )
    )

    st.plotly_chart(fig, use_container_width=True)



#################################
# 4. Streamlit UI
#################################

st.set_page_config(
    page_title="Consistency Metrics Dashboard (Multi-Experiment)",
    layout="wide"
)

st.title("Consistency Metrics Dashboard (Multi-Experiment)")
st.markdown(
    """
    여러 실험 결과 폴더를 한 번에 비교할 수 있는 대시보드입니다.  
    예시 입력:  
    `cai, cai_v2, /home/james/results/cai_ablation`  
    콤마(,)로 나누고 엔터(↩) 치면 로드됩니다.
    """
)

# 🔴 폴더 경로 입력 + 엔터로 submit 되도록 form 사용
with st.form(key="dir_form"):
    dir_list_raw = st.text_input(
        "데이터 폴더 경로(콤마로 여러 개 입력)",
        value="cai"
    )
    submitted = st.form_submit_button("로드")  # Enter 쳐도 submit

if not submitted:
    # 아직 submit 안 했을 때는 이후 내용 그리지 않음
    st.stop()

# 사용자가 엔터 or 버튼으로 제출하면 그 순간부터 아래 실행:

# 데이터 로드
df_all = load_from_multiple_dirs(dir_list_raw)

if df_all.empty:
    st.error("유효한 데이터가 없습니다. 폴더 경로를 확인해주세요.")
    st.stop()

st.subheader("Raw extracted metrics (통합)")
st.dataframe(df_all)

#################################
# 5. 폴더별 box plot (metric별)
#################################

st.markdown("## Metric별 Box Plot (폴더 비교 + mean 표시)")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### External Consistency Rate")
    boxplot_metric_multi(
        df_all,
        value_col="external_consistency_rate",
        title="external_consistency.consistency_rate by folder"
    )

with col2:
    st.markdown("### Internal Consistency Rate")
    boxplot_metric_multi(
        df_all,
        value_col="internal_consistency_rate",
        title="internal_consistency.consistency_rate by folder"
    )

with col3:
    st.markdown("### Repeat Score")
    boxplot_metric_multi(
        df_all,
        value_col="repeat_score",
        title="repeat.repeat_score by folder"
    )

#################################
# 6. 추가: 모든 metric vs 모든 폴더 한 번에 보기
#################################


st.markdown(
    """
    - external_consistency_rate = external_consistency.consistency_rate  
    - internal_consistency_rate = internal_consistency.consistency_rate  
    - repeat_score = repeat.repeat_score  
    - 검정색 다이아몬드 = 각 그룹의 평균  
    """
)
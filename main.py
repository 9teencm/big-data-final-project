import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# 假設這一行路徑是對的，請確保 about_model 資料夾與此 script 在同層或正確路徑
from about_model.randomforest_model import Model


@st.cache_resource
def load_models():
    """載入低/中/高時長模型，避免重複 IO。"""
    return [
        ("低時長模型", Model("低時長模型.joblib"), np.arange(0, 101, 1) / 10.0),      # 0.0 ~ 10.0 分
        ("中時長模型", Model("中時長模型.joblib"), np.arange(101, 301, 1) / 10.0),   # 10.1 ~ 30.0 分
        ("高時長模型", Model("高時長模型.joblib"), np.arange(301, 1201, 1) / 10.0), # 30.1 ~ 120.1 分
    ]


def predict_best_for_score(score: int):
    """給定前測分數，回傳三個模型的最佳影片/AI 時長。"""
    ai_range = np.arange(1, 15, 1) / 10.0  # 0.1 ~ 1.4
    rows = []
    
    # 載入模型
    models_data = load_models()
    
    for name, model, video_range in models_data:
        grid = np.array(list(itertools.product(video_range, ai_range)))
        grid_video = grid[:, 0]
        grid_ai = grid[:, 1]

        X_batch = pd.DataFrame(
            {
                "pretest_score": np.full(len(grid), score),
                "video_duration": grid_video,
                "ai_partner": grid_ai,
            }
        )

        preds = model.model.predict(X_batch.values)
        idx = np.argmax(preds)
        rows.append(
            {
                "模型": name,
                "前測分數": score,
                "推薦影片時長(分)": round(float(grid_video[idx]), 1),
                "推薦AI時長(分)": round(float(grid_ai[idx] * 60), 1),
            }
        )
    return rows


def page_calculation():
    """頁面功能 1：計算最佳化推薦"""
    st.title("前測分數推薦 - 低/中/高時長模型")
    st.write("請輸入前測分數，模型將計算出最佳的影片時長與 AI 使用時長建議。")

    # 將輸入框移至主畫面
    score = st.number_input("請輸入前測分數 (1-100)", min_value=1, max_value=100, value=60, step=1)

    if st.button("計算推薦", type="primary"):
        with st.spinner("模型計算中..."):
            rows = predict_best_for_score(score)
            df = pd.DataFrame(rows)
        
        st.success("計算完成！推薦結果如下：")
        st.dataframe(df, use_container_width=True)

        # 下載區域
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "下載結果 (CSV)",
                data=df.to_csv(index=False, encoding="utf-8-sig"),
                file_name=f"recommendations_score_{score}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col2:
            st.download_button(
                "下載結果 (JSON)",
                data=df.to_json(orient="records", force_ascii=False),
                file_name=f"recommendations_score_{score}.json",
                mime="application/json",
                use_container_width=True
            )


def page_charts():
    """頁面功能 2：圖表展示"""
    st.title("已生成圖表展示")
    st.write("瀏覽並下載分析圖表（低/中/高時長三個資料夾）。")

    base_dir = Path(__file__).resolve().parent
    chart_dirs = [
        ("低時長預測", base_dir / "低時長預測"),
        ("中時長預測", base_dir / "中時長預測"),
        ("高時長預測", base_dir / "高時長預測"),
    ]

    entries = []
    for label, cdir in chart_dirs:
        if cdir.exists():
            for p in sorted(list(cdir.glob("*.png")) + list(cdir.glob("*.jpg"))):
                entries.append((f"{label} / {p.name}", p))

    if not entries:
        st.warning("未在「低/中/高時長預測」資料夾找到 .png 或 .jpg 圖檔。")
        return

    display_names = [e[0] for e in entries]
    choice = st.selectbox("請選擇要檢視的圖表", display_names)
    selected_path = dict(entries)[choice]

    st.image(str(selected_path), caption=choice, use_column_width=True)
    st.download_button(
        "下載此圖檔",
        data=selected_path.read_bytes(),
        file_name=selected_path.name,
        mime="image/png" if selected_path.suffix.lower() == ".png" else "image/jpeg",
    )


def main():
    # 側邊欄導覽
    st.sidebar.title("功能選單")
    app_mode = st.sidebar.radio("請選擇功能", ["計算最佳化推薦", "圖表展示"])

    st.sidebar.markdown("---")
    st.sidebar.info("切換上方選項以使用不同功能。")

    # 根據選擇渲染不同頁面
    if app_mode == "計算最佳化推薦":
        page_calculation()
    elif app_mode == "圖表展示":
        page_charts()


if __name__ == "__main__":
    main()
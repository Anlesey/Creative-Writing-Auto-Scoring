import streamlit as st

# 展示比赛信息卡片
def get_model_options_selectbox(key=None):
    # 模型
    return st.selectbox(
        label="Model",
        options=("gpt-4o-mini", "gpt-4o", "deepseek-r1-250120"
        # "Anlesey/ernie-3.0-mini-zh-finetuned-aut", 
        ),
        key=key
    )

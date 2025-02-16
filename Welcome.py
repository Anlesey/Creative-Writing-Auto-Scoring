
import os
import sys
import streamlit as st
import requests
from Utils import *
import time  # for measuring time duration of API calls
from openai import OpenAI
from Utils.Utils import request_for_model_score
from Utils.components import get_model_options_selectbox
import os

# _______________________________________________________________________
st.write("### 单例测试")
# 选择物品、模型、答案

# 模型
model_name = get_model_options_selectbox(key='singleton')

# 物品
text = st.text_input("答案", value='')

# -------------------------
if st.button("计算创造力得分"):
    result_json = {}
    st.write("输入:", text)
    st.write("评分模型:", model_name)
    st.write("输出:")

    # ------------------------- 打接口 -------------------------

    score, err = request_for_model_score(model_name=model_name, text=text)

    if err is not None:
        st.error(err)
    else:
        st.write(f'新颖性评分：{score[0]}\n有效性评分：{score[1]}')
    
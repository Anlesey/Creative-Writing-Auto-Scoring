import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import io
import numpy as np
import time
from Utils.components import get_model_options_selectbox
from Utils.creativity_scoring import get_default_system_prompt, process_dataframe_for_scoring
from openai import OpenAI
from scipy.stats import f, pearsonr, spearmanr

# 修正会话状态初始化
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# 确保cancel_scoring在使用前已初始化
if "cancel_scoring" not in st.session_state:
    st.session_state.cancel_scoring = False

def update_key():
    st.session_state.uploader_key += 1

def set_cancel_scoring():
    st.session_state.cancel_scoring = True

def check_cancel():
    return st.session_state.cancel_scoring

# 修改样本文件验证函数
def validate_samples_file(df):
    required_columns = ['role', 'content']
    if not all(column in df.columns for column in required_columns):
        return "样本文件必须包含以下列：role, content"
    return None

def validate_test_file(df):
    if 'text' not in df.columns:
        return "测试文件必须包含text列"
    return None

def process_file(df, model_name, sys_prompt, samples_df):
    # 重置取消状态
    st.session_state.cancel_scoring = False
    
    # 添加取消按钮
    cancel_col = st.empty()
    with cancel_col.container():
        st.button("取消评分", on_click=set_cancel_scoring, key="cancel_button")

    # 创建进度条
    progress_bar = st.progress(0)
    
    # 定义进度回调函数
    def update_progress(progress):
        progress_bar.progress(progress)
    
    # 使用抽象的处理函数，不再需要手动处理API密钥
    processed_df, correlation_results, has_original_scores = process_dataframe_for_scoring(
        df=df,
        model_name=model_name,
        sys_prompt=sys_prompt,
        samples_df=samples_df,
        progress_callback=update_progress,
        cancel_check=check_cancel,
        st_secrets=st.secrets  # 传递secrets对象
    )
    
    # 移除取消按钮
    cancel_col.empty()
    
    # 如果取消了评分，显示警告
    if st.session_state.cancel_scoring:
        st.warning("评分已取消！")
    
    return processed_df, correlation_results, has_original_scores

def main():
    st.write("### 自定义System Prompt和Few-shot样本测试")
    
    # 选择模型
    model_name = get_model_options_selectbox(key='custom_batch')
    
    # System Prompt输入 - 使用抽象的默认提示词
    sys_prompt = st.text_area(
        "System Prompt",
        value=get_default_system_prompt(),
        height=200
    )
    
    # 创建两列布局，同时显示两个文件上传框
    col1, col2 = st.columns(2)
    
    with col1:
        # 上传Few-shot样本文件（可选）
        st.write("#### Few-shot样本文件(可选)")
        samples_file = st.file_uploader(
            "上传包含role、content列的样本文件",
            type=["xlsx", "csv"],
            key=f'samples_uploader_{st.session_state.uploader_key}'
        )
    
    with col2:
        # 上传待评分文件
        st.write("#### 待评分文件")
        test_file = st.file_uploader(
            "上传包含text列的测试文件",
            type=["xlsx", "csv"],
            key=f'test_uploader_{st.session_state.uploader_key}'
        )
    
    # 只要上传了测试文件就可以进行处理
    if test_file:
        # 读取样本文件（如果有）
        samples_df = None
        if samples_file:
            samples_df = pd.read_excel(samples_file) if samples_file.name.endswith('.xlsx') else pd.read_csv(samples_file)
            validation_error = validate_samples_file(samples_df)
            if validation_error:
                st.error(validation_error)
                return
        
        # 读取测试文件
        df = pd.read_csv(test_file) if test_file.name.endswith('.csv') else pd.read_excel(test_file)
        validation_error = validate_test_file(df)
        if validation_error:
            st.error(validation_error)
            return
        
        # 添加开始评分按钮
        if st.button("开始评分", use_container_width=True, disabled=not sys_prompt or not test_file, key='custom_batch_process'):
            st.info("自动评分中...")
            processed_df, correlation_results, has_original_scores = process_file(df, model_name, sys_prompt, samples_df)
            st.success("评分完成！")
            # 显示相关性结果，但不使用可视化
            if has_original_scores:
                st.write("### 相关性分析结果")
                
                # 创建相关性结果表格 - 只显示创造力评分与originality的相关性
                correlation_table = pd.DataFrame({
                    '维度': ['创造力 (Pearson)', '创造力 (Spearman)'],
                    '相关系数': [
                        correlation_results['pearson']['creativity'][0],
                        correlation_results['spearman']['creativity'][0]
                    ],
                    'p值': [
                        correlation_results['pearson']['creativity'][1],
                        correlation_results['spearman']['creativity'][1]
                    ]
                })
                
                st.dataframe(correlation_table.style.format({
                    '相关系数': '{:.4f}',
                    'p值': '{:.4f}'
                }))
            
            # 下载结果
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                processed_df.to_excel(writer, index=False, sheet_name='Sheet1')
            
            output.seek(0)
            st.download_button(
                label="下载评分结果",
                data=output,
                file_name=f"{model_name}_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                on_click=update_key
            )

    st.divider()
    st.markdown(
    '''
    #### 说明
    1. Few-shot样本文件格式要求：
    - 文件格式：.xlsx或.csv
    - 必须包含以下列：
        - **role**：消息角色（system/user/assistant）
        - **content**：消息内容

    2. 测试文件格式要求：
    - 文件格式：.xlsx或.csv
    - 必须包含以下列：
        - **text**：待评分的文本
    - 如果包含以下列，将计算相关性：
        - **originality**：原始创造力评分
    ''')

if __name__ == "__main__":
    main()
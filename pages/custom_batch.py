import streamlit as st
import pandas as pd
import io
import numpy as np
import time
from Utils.Utils import get_finturned_model_response_openai
from Utils.components import get_model_options_selectbox
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

# 新增函数：从新格式的样本文件构建消息
def build_messages_from_samples(samples_df, sys_prompt):
    messages = [{"role": "system", "content": sys_prompt}]
    
    if samples_df is not None:
        for i, row in samples_df.iterrows():
            messages.append({"role": row['role'], "content": row['content']})
    
    return messages

def process_file(df, model_name, sys_prompt, samples_df):
    # 重置取消状态
    st.session_state.cancel_scoring = False
    
    # 检查是否已有评分列
    has_original_scores = 'originality' in df.columns
    
    df['创造力评分'] = np.nan
    df['评分理由'] = ""
    df['Error'] = np.nan
    progress_bar = st.progress(0)

    # 添加取消按钮
    cancel_col = st.empty()
    with cancel_col.container():
        st.button("取消评分", on_click=set_cancel_scoring, key="cancel_button")

    OPENAI_API_KEY=st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=OPENAI_API_KEY)

    # 构建基础消息
    base_messages = build_messages_from_samples(samples_df, sys_prompt)

    for i, row in df.iterrows():
        # 检查是否取消
        if st.session_state.cancel_scoring:
            st.warning("评分已取消！")
            cancel_col.empty()  # 移除取消按钮
            return df, {}, has_original_scores
            
        text = f"{row['text']}"
        
        # 复制基础消息并添加当前用户查询
        messages = base_messages.copy()
        messages.append({"role": "user", "content": text})
        
        # 调用API
        retries = 0
        max_retries = 5
        while retries < max_retries:
            # 再次检查是否取消
            if st.session_state.cancel_scoring:
                st.warning("评分已取消！")
                cancel_col.empty()  # 移除取消按钮
                return df, {}, has_original_scores
                
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0,
                    max_tokens=200  # 增加token数以容纳评分理由
                )
                
                reply_content = response.choices[0].message.content
                
                # 解析回复内容，提取评分和理由
                if "【创造力评分】" in reply_content and "【评分理由】" in reply_content:
                    score_text = reply_content.split("【创造力评分】：")[1].split("分")[0].strip()
                    score = float(score_text)
                    
                    reason_parts = reply_content.split("【评分理由】：")
                    if len(reason_parts) > 1:
                        reason = reason_parts[1].strip()
                    else:
                        reason = "未提供评分理由"
                    
                    df.at[i, '创造力评分'] = score
                    df.at[i, '评分理由'] = reason
                    break
                else:
                    retries += 1
                    time.sleep(0.05)
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    df.at[i, 'Error'] = str(e)
                time.sleep(0.05)
        
        progress_bar.progress((i + 1) / len(df))
    
    # 移除取消按钮
    cancel_col.empty()
    
    if df['Error'].isna().sum() == df.shape[0]:
        del df['Error']
    
    # 修复相关性计算中的列名错误
    correlation_results = {}
    if has_original_scores:
        # 只计算与originality的相关性
        pearson_creativity, p_value_pearson = pearsonr(df['originality'], df['创造'])
        spearman_creativity, p_value_spearman = spearmanr(df['originality'], df['创造'])
        
        correlation_results = {
            'pearson': {
                'creativity': (pearson_creativity, p_value_pearson)
            },
            'spearman': {
                'creativity': (spearman_creativity, p_value_spearman)
            }
        }
    
    return df, correlation_results, has_original_scores

def main():
    st.write("### 自定义System Prompt和Few-shot样本测试")
    
    # 选择模型
    model_name = get_model_options_selectbox(key='custom_batch')
    
    # System Prompt输入
    sys_prompt = st.text_area(
        "System Prompt",
        value='''你是一位创造力研究领域的专家，擅长分析文本中体现的创造性思维，并根据被试在开放性问题中的回答进行创造力评分。你具备深入分析问题解决策略的能力，能够结合评分维度对文本进行合理打分，并输出清晰、结构化的评分理由。

研究背景：
- 研究目的：本任务旨在评估学生在“智慧博物馆”主题情境下完成的创造力写作题中的文本创造力水平。该题目要求学生结合智慧博物馆的背景与相关技术（如信息技术、人工智能、3D打印、数字交互、互联网等），为“老年人在博物馆中参观存在困难”的现实问题提供原创性的解决方案或设计构想。
- 测验题目：
你和小组成员们到当地博物馆实地考察时发现：喜爱传统文化的博物馆游览者中有一类老年人群体，他们极其热爱历史和传统文化，但是，随着年龄的增大，他们在游览时遇到越来越多的问题。比如，这类游览者的体力已经不足以支撑他们在博物馆中随意走动，游览藏品，而且他们的视力也逐渐减弱，对于一些摆放位置较远的展品已经无法看清。请你针对这类游览者遇到的问题，发挥创造力和想象力，设计一个你认为能够最好解决该问题的、最新颖的观展方案。
- 数据：
被试生成的解决方案或设计文本。

评分目标：
请你阅读被试生成的文本对其创造力进行1-10分评分，1分表示几乎无创新；10分表示文本极具创新性，并说明评分依据。

评分步骤：
1、充分阅读文本
2、分析文本中体现的思维策略
3、给出评分与理由

输出要求：
1.	每条文本的创造力评分。
2.	简要叙述评分原因。

输入：被试的作答文本

输出格式：
【创造力评分】：X分  
【评分理由】：（简要叙述打分依据） ''',
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
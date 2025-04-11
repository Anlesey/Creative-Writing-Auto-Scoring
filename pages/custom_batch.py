import streamlit as st
import pandas as pd
import io
import numpy as np
from Utils.Utils import request_for_model_score, get_finturned_model_response_openai
from Utils.components import get_model_options_selectbox
from openai import OpenAI
from scipy.stats import pearsonr, spearmanr

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
def update_key():
    st.session_state.uploader_key += 1

def validate_file(df):
    required_columns = ['text', 'originality', 'usefulness']
    if not all(column in df.columns for column in required_columns):
        return "样本文件必须包含以下列：text, originality, usefulness"
    return None

def validate_test_file(df):
    if 'text' not in df.columns:
        return "测试文件必须包含text列"
    return None

def process_file(df, model_name, sys_prompt, samples_df):
    # 检查是否已有评分列
    has_original_scores = 'originality' in df.columns and 'usefulness' in df.columns
    
    df['新颖性'] = np.nan
    df['有效性'] = np.nan
    df['Error'] = np.nan
    progress_bar = st.progress(0)

    OPENAI_API_KEY=st.secrets["OPENAI_API_KEY"]
    client = OpenAI(api_key=OPENAI_API_KEY)

    for i, row in df.iterrows():
        text = f"{row['text']}"
        scores, err = get_finturned_model_response_openai(
            client=client, 
            text=text, 
            model_name=model_name,
            sys_prompt=sys_prompt,
            d_fewshot=samples_df
        )
        if err is None:
            df.at[i, '新颖性'] = scores[0]
            df.at[i, '有效性'] = scores[1]
        else:
            df.at[i, 'Error'] = err
        progress_bar.progress((i + 1) / len(df))
    
    if df['Error'].isna().sum() == df.shape[0]:
        del df['Error']
    
    # 计算相关性
    correlation_results = {}
    if has_original_scores:
        # 计算Pearson相关系数
        pearson_originality, p_value_pearson_orig = pearsonr(df['originality'], df['新颖性'])
        pearson_usefulness, p_value_pearson_use = pearsonr(df['usefulness'], df['有效性'])
        
        # 计算Spearman相关系数
        spearman_originality, p_value_spearman_orig = spearmanr(df['originality'], df['新颖性'])
        spearman_usefulness, p_value_spearman_use = spearmanr(df['usefulness'], df['有效性'])
        
        correlation_results = {
            'pearson': {
                'originality': (pearson_originality, p_value_pearson_orig),
                'usefulness': (pearson_usefulness, p_value_pearson_use)
            },
            'spearman': {
                'originality': (spearman_originality, p_value_spearman_orig),
                'usefulness': (spearman_usefulness, p_value_spearman_use)
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
        value='''请你作为创造力研究领域的专业研究者，为创意写作任务中被试的作答评分。
任务背景：被试被要求为博物馆中的老年游览者发现一项亟待解决的体验问题，并现有技术为老年游览者设计一个能够最好解决该问题的、最新颖的观展方案。例如，可以设计博物馆的藏品展示、游览方式，或使用互联网技术通过智能终端解决问题。
要求：对于被试的回答，你需要评价的分数有两项：原创性、有效性。分值为0~10：1分代表该作答不具备原创性/有效性，10分代表该作答极具原创性/有效性。
输出规范：直接给出原创性、有效性两个评分结果，以英文逗号分隔。评分需保留一位小数。请直接给出分数结果，不需要任何其他额外说明。''',
        height=200
    )
    
    # 上传Few-shot样本文件
    st.write("#### 上传Few-shot样本文件")
    samples_file = st.file_uploader(
        "上传包含text、originality、usefulness列的样本文件",
        type=["xlsx"],
        key=f'samples_uploader_{st.session_state.uploader_key}'
    )

    if samples_file:
        samples_df = pd.read_excel(samples_file)
        validation_error = validate_samples_file(samples_df)
        if validation_error:
            st.error(validation_error)
            return
    
    # 上传待评分文件 - 这部分现在是独立的
    st.write("#### 上传待评分文件")
    test_file = st.file_uploader(
        "上传包含text列的测试文件 (如果包含originality和usefulness列，将计算相关性)",
        type=["xlsx", "csv"],
        key=f'test_uploader_{st.session_state.uploader_key}'
    )

    if test_file:
        df = pd.read_csv(test_file) if test_file.name.endswith('.csv') else pd.read_excel(test_file)
        
        validation_error = validate_test_file(df)
        if validation_error:
            st.error(validation_error)
            return

        st.info("自动评分中...")
        processed_df, correlation_results, has_original_scores = process_file(df, model_name, sys_prompt, samples_df)
        st.success("评分完成！")
        
        # 显示相关性结果
        if has_original_scores:
            st.write("### 相关性分析结果")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Pearson相关系数")
                pearson_df = pd.DataFrame({
                    '维度': ['原创性', '有效性'],
                    '相关系数': [
                        correlation_results['pearson']['originality'][0],
                        correlation_results['pearson']['usefulness'][0]
                    ],
                    'p值': [
                        correlation_results['pearson']['originality'][1],
                        correlation_results['pearson']['usefulness'][1]
                    ]
                })
                st.dataframe(pearson_df.style.format({
                    '相关系数': '{:.4f}',
                    'p值': '{:.4f}'
                }))
            
            with col2:
                st.write("#### Spearman相关系数")
                spearman_df = pd.DataFrame({
                    '维度': ['原创性', '有效性'],
                    '相关系数': [
                        correlation_results['spearman']['originality'][0],
                        correlation_results['spearman']['usefulness'][0]
                    ],
                    'p值': [
                        correlation_results['spearman']['originality'][1],
                        correlation_results['spearman']['usefulness'][1]
                    ]
                })
                st.dataframe(spearman_df.style.format({
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
            file_name="custom_processed_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            on_click=update_key
        )

    st.divider()
    st.markdown(
    '''
    #### 说明
    1. Few-shot样本文件格式要求：
    - 文件格式：.xlsx
    - 必须包含以下列：
        - **text**：示例文本
        - **originality**：原创性得分
        - **usefulness**：有效性得分

    2. 测试文件格式要求：
    - 文件格式：.xlsx或.csv
    - 必须包含以下列：
        - **text**：待评分的文本
    - 如果包含以下列，将计算相关性：
        - **originality**：原始原创性得分
        - **usefulness**：原始有效性得分
    ''')

if __name__ == "__main__":
    main()
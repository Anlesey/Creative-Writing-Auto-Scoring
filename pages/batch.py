import streamlit as st
import pandas as pd
import re
import io
import numpy as np

from Utils.Utils import request_for_model_score
from Utils.components import get_model_options_selectbox

# 用于在用户下载后更新当前文件
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
def update_key():
    st.session_state.uploader_key += 1


def validate_file(df):
    required_columns = ['ID', 'text']
    if not all(column in df.columns for column in required_columns):
        return "文件必须包含以下列：ID, text"
    if df['ID'].duplicated().any():
        return "ID列不能包含重复值"
    if not df['ID'].apply(lambda x: re.match(r'^\w+$', str(x))).all():
        return "ID列只能包含字母、数字和下划线"
    return None

def process_file(df, model_name):
    df['新颖性'] = np.nan
    df['有效性'] = np.nan
    df['Error'] = np.nan
    progress_bar = st.progress(0)

    for i, row in df.iterrows():
        text = f"{row['text']}"
        scores, err = request_for_model_score(model_name, text)
        if err is None:
            df.at[i, '新颖性'] = scores[0]
            df.at[i, '有效性'] = scores[1]
        else:
            df.at[i, 'Error'] = err
        progress_bar.progress((i + 1) / len(df))
    if df['Error'].isna().sum() == df.shape[0]:
        del df['Error']
    return df

def main():
    st.write("### 文件批处理")
    model_name = get_model_options_selectbox(key='batch')
    
    uploaded_file = st.file_uploader(
        label="请上传包含ID、text两列的excel或csv数据文件", 
        type=["csv", "xlsx"],
        key=f'uploader_{st.session_state.uploader_key}'
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

        validation_error = validate_file(df)
        if validation_error:
            st.error(validation_error)
            return

        st.info("自动评分中...")
        processed_df = process_file(df, model_name)

        st.success("评分完成！")
        
        # 将 DataFrame 保存为 Excel 格式
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            processed_df.to_excel(writer, index=False, sheet_name='Sheet1')

        # 将指针移回开始位置
        output.seek(0)
        uploaded_file = None
        # 创建下载按钮
        download_link = st.download_button(
            label="下载评分结果",
            data=output,
            file_name="processed_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            on_click = update_key
        )
    
    st.divider()
    st.markdown(
    '''
    #### 说明
    数据格式要求：
    - 文件格式：.xlsx或.csv。必须包含ID、text两列。
    - **ID**：唯一标识符，不能重复。必须仅由字母、数字、下划线组成。
    - **text**：待评分的文本。

    输入示例：
    | ID  | text  |
    |-----|-------|
    | 1   | 从前有座山，山里有座庙 |
    | 2   | 庙里有个老和尚和一个小和尚 |

    输出示例：
    | ID  | text  |新颖性  | 有效性  |
    |-----|-------| -------  | -------  |
    | 1   | 从前有座山，山里有座庙 | 1.0  | 1.0  |
    | 2   | 庙里有个老和尚和一个小和尚 | 1.0  | 1.0  |
    ''')

if __name__ == "__main__":
    main()

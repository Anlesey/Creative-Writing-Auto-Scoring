import numpy as np
import pandas as pd
import time
import streamlit as st
from openai import OpenAI

starting_system_prompt = '''请你作为创造力研究领域的专业研究者，为创意写作任务中被试的作答评分。
任务背景：被试被要求为博物馆中的老年游览者发现一项亟待解决的体验问题，并现有技术为老年游览者设计一个能够最好解决该问题的、最新颖的观展方案。例如，可以设计博物馆的藏品展示、游览方式，或使用互联网技术通过智能终端解决问题。
要求：对于被试的回答，你需要评价的分数有两项：原创性、有效性。分值为0~10：1分代表该作答不具备原创性/有效性，10分代表该作答极具原创性/有效性。
输出规范：直接给出原创性、有效性两个评分结果，以英文逗号分隔。评分需保留一位小数。请直接给出分数结果，不需要任何其他额外说明。'''

d_fewshot = pd.read_excel('data/samples.xlsx')

def get_fewshot_sample_messages(d_fewshot=d_fewshot):
    messages = []
    for i,r in d_fewshot.iterrows():
        messages = messages + [
            {"role": "user", "content": r['text']},
            {"role": "assistant", "content": f"{r['originality']},{r['usefulness']}"},
        ]
    return messages

# 输出:分数;错误信息
def get_finturned_model_response_openai(client, text, model_name, max_retries=5):
    retries = 0
    while retries < max_retries:
        # send a ChatCompletion request to count to 100
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": starting_system_prompt}] 
                + get_fewshot_sample_messages() 
                + [{"role": "user", "content": text}],
            temperature=0,
            max_tokens=10
        )
        # print the time delay and text received
        reply_content = response.choices[0].message.content
        try:
            result = [float(x.strip()) for x in reply_content.split(',')]
            scores = [result[0], result[1]] # 检查是否为两位数组
            return scores, None
        except ValueError:
            retries += 1
            time.sleep(0.05)  # Optional: wait for a short period before retrying
    
    return None, response

def request_for_model_score(model_name, text):
    if model_name=="gpt-4o-mini":
        OPENAI_API_KEY=st.secrets["OPENAI_API_KEY"]
        client = OpenAI(api_key=OPENAI_API_KEY)
        scores, err = get_finturned_model_response_openai(client, text, model_name)
    else:
        st.error('Model is not available!')
        return None, None
    return scores, err

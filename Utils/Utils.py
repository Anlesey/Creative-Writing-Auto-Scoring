import numpy as np
import pandas as pd
import time
import streamlit as st
from openai import OpenAI

# starting_system_prompt = '''请你作为创造力研究领域的专业研究者，为创意写作任务中被试的作答评分。
# 任务背景：被试被要求为博物馆中的老年游览者发现一项亟待解决的体验问题，并现有技术为老年游览者设计一个能够最好解决该问题的、最新颖的观展方案。例如，可以设计博物馆的藏品展示、游览方式，或使用互联网技术通过智能终端解决问题。
# 要求：对于被试的回答，你需要评价的分数有两项：原创性、有效性。分值为0~10：1分代表该作答不具备原创性/有效性，10分代表该作答极具原创性/有效性。
# 输出规范：直接给出原创性、有效性两个评分结果，以英文逗号分隔。评分需保留一位小数。请直接给出分数结果，不需要任何其他额外说明。'''

starting_system_prompt = '''你是一位创造力研究领域的专家，擅长分析文本中体现的创造性思维，并根据被试在开放性问题中的回答进行创造力评分。你具备深入分析问题解决策略的能力，能够结合评分维度对文本进行合理打分，并输出清晰、结构化的评分理由。

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
【评分理由】：（简要叙述打分依据） '''


d_fewshot = pd.read_excel('data/samples.xlsx')

def get_fewshot_sample_messages(samples_df=d_fewshot):
    messages = []
    if samples_df is None or len(samples_df) == 0:
        return messages
    for i,r in samples_df.iterrows():
        messages = messages + [
            {"role": "user", "content": r['text']},
            {"role": "assistant", "content": f"{r['originality']},{r['usefulness']}"},
        ]
    return messages

# 输出:分数;错误信息
def get_finturned_model_response_openai(client, text, model_name, sys_prompt=starting_system_prompt, samples_df=d_fewshot, max_retries=5):
    retries = 0
    while retries < max_retries:
        # send a ChatCompletion request to count to 100
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": sys_prompt}] 
                + get_fewshot_sample_messages(samples_df=d_fewshot) 
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
    if model_name in ["gpt-4o", "gpt-4o-mini"]:
        OPENAI_API_KEY=st.secrets["OPENAI_API_KEY"]
        client = OpenAI(api_key=OPENAI_API_KEY)
        scores, err = get_finturned_model_response_openai(client, text, model_name)
    else:
        st.error('Model is not available!')
        return None, None
    return scores, err

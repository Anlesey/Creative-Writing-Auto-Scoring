import time
import numpy as np
import os
from openai import OpenAI

def get_api_key(model_name, st_secrets=None):
    """
    根据模型名称获取适当的API密钥
    
    参数:
    - model_name: 模型名称
    - st_secrets: streamlit的secrets对象
    
    返回:
    - api_key: 适用于该模型的API密钥
    """
    # 默认的API密钥 - 使用环境变量或其他安全方式存储
    
    if st_secrets is None:
        return None
    
    try:
        # 根据模型类型选择合适的API密钥
        if "deepseek" in model_name.lower():
            # 尝试获取ARK_API_KEY，如果不存在则使用默认值
            return st_secrets.get("ARK_API_KEY", None)
        else:
            # 尝试获取OPENAI_API_KEY，如果不存在则使用默认值
            return st_secrets.get("OPENAI_API_KEY", None)
    except Exception as e:
        # 如果获取过程中出现任何错误，返回默认密钥
        print(f"获取API密钥时出错: {str(e)}")
        return None


def get_default_system_prompt():
    """返回默认的系统提示词"""
    return '''你是一位创造力研究领域的专家，擅长分析文本中体现的创造性思维，并根据被试在开放性问题中的回答进行创造力评分。你具备深入分析问题解决策略的能力，能够结合评分维度对文本进行合理打分，并输出清晰、结构化的评分理由。

研究背景：
- 研究目的：本任务旨在评估学生在"智慧博物馆"主题情境下完成的创造力写作题中的文本创造力水平。该题目要求学生结合智慧博物馆的背景与相关技术（如信息技术、人工智能、3D打印、数字交互、互联网等），为"老年人在博物馆中参观存在困难"的现实问题提供原创性的解决方案或设计构想。
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

def build_messages_from_samples(samples_df, sys_prompt):
    """从样本文件构建消息列表"""
    messages = [{"role": "system", "content": sys_prompt}]
    
    if samples_df is not None:
        for i, row in samples_df.iterrows():
            messages.append({"role": row['role'], "content": row['content']})
    
    return messages

def score_creativity(client, text, model_name, messages, max_retries=5, check_cancel_func=None):
    """
    使用OpenAI API对文本进行创造力评分
    
    参数:
    - client: OpenAI客户端
    - text: 待评分文本
    - model_name: 使用的模型名称
    - messages: 基础消息列表
    - max_retries: 最大重试次数
    - check_cancel_func: 检查是否取消的函数
    
    返回:
    - scores: 包含新颖性、有效性、创造性评分的字典
    - error: 错误信息
    """
    # 复制基础消息并添加当前用户查询
    current_messages = messages.copy()
    current_messages.append({"role": "user", "content": text})
    
    retries = 0
    while retries < max_retries:
        # 检查是否取消
        if check_cancel_func and check_cancel_func():
            return None, "评分已取消"
            
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=current_messages,
                temperature=0,
                max_tokens=200  # 增加token数以容纳评分理由
            )
            
            reply_content = response.choices[0].message.content
            
            # 解析回复内容，提取三个维度的评分
            scores = {}
            
            # 提取新颖性评分
            if "新颖性：" in reply_content:
                novelty_parts = reply_content.split("新颖性：")[1].split("\n")[0].strip()
                try:
                    scores['新颖性'] = float(novelty_parts)
                except:
                    pass
            
            # 提取有效性评分
            if "有效性：" in reply_content:
                usefulness_parts = reply_content.split("有效性：")[1].split("\n")[0].strip()
                try:
                    scores['有效性'] = float(usefulness_parts)
                except:
                    pass
            
            # 提取创造性评分
            if "创造性：" in reply_content:
                creativity_parts = reply_content.split("创造性：")[1].split("\n")[0].strip()
                try:
                    scores['创造性'] = float(creativity_parts)
                except:
                    pass
            
            # 检查是否成功提取了所有三个维度的评分
            if len(scores) == 3:
                return scores, None
            else:
                retries += 1
                time.sleep(0.05)
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                return None, str(e)
            time.sleep(0.05)
    
    return None, "无法解析模型响应"

def process_dataframe_for_scoring(df, model_name, sys_prompt, samples_df, api_key=None, progress_callback=None, cancel_check=None, st_secrets=None):
    """
    处理DataFrame进行批量评分
    
    参数:
    - df: 包含待评分文本的DataFrame
    - model_name: 使用的模型名称
    - sys_prompt: 系统提示词
    - samples_df: 样本DataFrame
    - api_key: OpenAI API密钥（可选，如果不提供则自动获取）
    - progress_callback: 进度回调函数
    - cancel_check: 检查是否取消的函数
    - st_secrets: streamlit的secrets对象
    
    返回:
    - processed_df: 处理后的DataFrame
    - correlation_results: 相关性结果
    - has_original_scores: 是否有原始评分
    """
    # 检查是否已有评分列
    has_original_scores = 'originality' in df.columns and 'usefulness' in df.columns
    
    # 添加评分列
    df['新颖性'] = np.nan
    df['有效性'] = np.nan
    df['创造性'] = np.nan
    df['Error'] = np.nan

    # 如果没有提供API密钥，则自动获取
    if api_key is None:
        api_key = get_api_key(model_name, st_secrets)

    # 获取适当的客户端和模型名称
    client, actual_model = get_client(model_name, api_key)

    # 构建基础消息
    base_messages = build_messages_from_samples(samples_df, sys_prompt)
    
    for i, row in df.iterrows():
        # 检查是否取消
        if cancel_check and cancel_check():
            return df, {}, has_original_scores
            
        text = f"{row['text']}"
        
        # 调用评分函数
        scores, error = score_creativity(
            client=client,
            text=text,
            model_name=model_name,
            messages=base_messages,
            check_cancel_func=cancel_check
        )
        
        if error is None and scores:
            df.at[i, '新颖性'] = scores.get('新颖性', np.nan)
            df.at[i, '有效性'] = scores.get('有效性', np.nan)
            df.at[i, '创造性'] = scores.get('创造性', np.nan)
        else:
            df.at[i, 'Error'] = error or "未能获取评分"
        
        # 更新进度
        if progress_callback:
            progress_callback((i + 1) / len(df))
    
    if df['Error'].isna().sum() == df.shape[0]:
        del df['Error']
    
    # 计算相关性
    correlation_results = {}
    if has_original_scores:
        from scipy.stats import pearsonr, spearmanr
        
        # 计算新颖性相关性
        pearson_novelty, p_value_pearson_nov = pearsonr(df['originality'], df['新颖性'])
        spearman_novelty, p_value_spearman_nov = spearmanr(df['originality'], df['新颖性'])
        
        # 计算有效性相关性
        pearson_usefulness, p_value_pearson_use = pearsonr(df['usefulness'], df['有效性'])
        spearman_usefulness, p_value_spearman_use = spearmanr(df['usefulness'], df['有效性'])
        
        # 如果有创造性原始评分，也计算创造性相关性
        if 'creativity' in df.columns:
            pearson_creativity, p_value_pearson_cre = pearsonr(df['creativity'], df['创造性'])
            spearman_creativity, p_value_spearman_cre = spearmanr(df['creativity'], df['创造性'])
            
            correlation_results = {
                'pearson': {
                    'novelty': (pearson_novelty, p_value_pearson_nov),
                    'usefulness': (pearson_usefulness, p_value_pearson_use),
                    'creativity': (pearson_creativity, p_value_pearson_cre)
                },
                'spearman': {
                    'novelty': (spearman_novelty, p_value_spearman_nov),
                    'usefulness': (spearman_usefulness, p_value_spearman_use),
                    'creativity': (spearman_creativity, p_value_spearman_cre)
                }
            }
        else:
            correlation_results = {
                'pearson': {
                    'novelty': (pearson_novelty, p_value_pearson_nov),
                    'usefulness': (pearson_usefulness, p_value_pearson_use)
                },
                'spearman': {
                    'novelty': (spearman_novelty, p_value_spearman_nov),
                    'usefulness': (spearman_usefulness, p_value_spearman_use)
                }
            }
    
    return df, correlation_results, has_original_scores


def get_client(model_name, api_key):
    """
    根据模型名称获取适当的客户端
    
    参数:
    - model_name: 模型名称
    - api_key: API密钥
    
    返回:
    - client: OpenAI客户端
    - actual_model: 实际使用的模型名称
    """
    # 检查是否是deepseek模型
    if "deepseek" in model_name.lower():
        # 使用deepseek的API配置
        client = OpenAI(
            api_key=api_key,
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )
        # 使用deepseek的模型名称
        actual_model = "deepseek-r1-250120"
    else:
        # 使用标准OpenAI配置
        client = OpenAI(api_key=api_key)
        actual_model = model_name
    
    return client, actual_model
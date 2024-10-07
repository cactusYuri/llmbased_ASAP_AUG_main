import random
from http import HTTPStatus
import dashscope
import json
import os
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import re

# 设置 API 密钥列表
api_keys = [
    'sk-4------------------------------3', 
    'sk-e------------------------------0',
    'sk-5------------------------------f',
    'sk-d------------------------------a',
]

def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def save_to_file(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def append_to_file(file_path, content):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(content)

def call_dashscope(conversation_history, api_key):
    dashscope.api_key = api_key
    response = dashscope.Generation.call(
        model="qwen-max-0403",
        messages=conversation_history,
        seed=random.randint(1, 10000),
        result_format='message',
        temperature=1.0
    )
    return response

def generate_response(model_type, conversation_history, api_key):
    dashscope.api_key = api_key
    response = dashscope.Generation.call(
        model=model_type,
        messages=conversation_history,
        seed=random.randint(1, 10000),
        result_format='message',
        temperature=1.0
    )

    if response.status_code == HTTPStatus.OK:
        response_content = response.output['choices'][0]['message']['content']
        
        # 将assistant的回复添加到messages列表中
        conversation_history.append({'role': 'assistant', 'content': response_content})
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
        # 如果响应失败，将最后一条user message从messages列表里删除，确保user/assistant消息交替出现
        conversation_history.pop()
    
    return conversation_history

def multi_round_essay_test(task_id, prompt, standard, api_key):
    # 初始化对话历史
    conversation_history_g = [
        {'role': 'system', 'content': 'Your job is to produce an article of same quality as the example junior student text, imitating the text quality even it is bad. Your response only contains the essay, DO NOT CONTAIN ANY OTHER THINGS LIKE SELF-EVALUATION.'}
    ]
    
    # 第一轮对话
    conversation_history_g.append({'role': 'user', 'content': standard})
    conversation_history_g.append({'role': 'assistant', 'content': 'OK, I will generate the essay as the standard of evaluation. My response only contains the essay, DO NOT CONTAIN ANY OTHER THINGS LIKE SELF-EVALUATION OR SCORE.'})                            
    conversation_history_g.append({'role': 'user', 'content': prompt})
    conversation_history_g = generate_response('qwen-max-0403', conversation_history_g, api_key)
    
    generated_essay = conversation_history_g[-1]['content']

    return task_id, generated_essay

def main():
    prompt = load_file('prompt.txt')
    standard = load_file('standard.txt')
    result_file = 'result.txt'
    # 清空结果文件
    save_to_file(result_file, "")
    # 并行执行任务
    results = []
    for i in range(10):
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(multi_round_essay_test, task_id, prompt, standard, api_keys[task_id % len(api_keys)]) for task_id in range(4)]
            for future in futures:
                results.append(future.result())
    
    # 按任务ID顺序写入结果文件
    results.sort(key=lambda x: x[0])
    for task_id, generated_essay in results:
        append_to_file(result_file, f"Task {task_id}:\n")
        append_to_file(result_file, "--------------------\nGenerated Essay:\n--------------------\n")
        append_to_file(result_file, generated_essay)
        append_to_file(result_file, "\n\n\n")


if __name__ == '__main__':
    main()

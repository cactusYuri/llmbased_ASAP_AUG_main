import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import language_tool_python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import reader
from reader import read_essays_from_file
from reader import read_essays_by_score
import json
import warnings
from transformers import logging
from tqdm import tqdm
import re



# 忽略transformers库中的警告消息
warnings.filterwarnings("ignore", category=UserWarning, module='transformers')
logging.set_verbosity_error()


def analyze_structure(text):
    paragraphs = text.split('\n\n')
    paragraph_lengths = [len(p.split()) for p in paragraphs]
    avg_paragraph_length = sum(paragraph_lengths) / len(paragraph_lengths)

    sentences = sent_tokenize(text)
    sentence_lengths = [len(word_tokenize(s)) for s in sentences]
    avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)

    return avg_paragraph_length, avg_sentence_length


def check_grammar_and_spelling(text):
    tool = language_tool_python.LanguageToolPublicAPI('en-US')
    matches = tool.check(text)
    num_errors = len(matches)
    return num_errors


def analyze_texts_by_score(file_path, score_range, max_per_score):
    score_data = {score: {'avg_paragraph_length': [], 'avg_sentence_length': [], 'num_errors': []} for score in
                  score_range}
    essay_data = []

    for score in score_range:
        essays = read_essays_by_score(file_path, score)
        for i, essay in enumerate(
                tqdm(essays, desc=f"Processing score {score}", total=min(len(essays), max_per_score))):
            if i >= max_per_score:
                break
            avg_paragraph_length, avg_sentence_length = analyze_structure(essay)
            num_errors = check_grammar_and_spelling(essay)
            score_data[score]['avg_paragraph_length'].append(avg_paragraph_length)
            score_data[score]['avg_sentence_length'].append(avg_sentence_length)
            score_data[score]['num_errors'].append(num_errors)
            essay_data.append({
                'score': score,
                'avg_paragraph_length': avg_paragraph_length,
                'avg_sentence_length': avg_sentence_length,
                'num_errors': num_errors,
                'text': essay
            })

    return score_data, essay_data


def calculate_averages(score_data):
    average_data = {}
    for score, data in score_data.items():
        if len(data['avg_paragraph_length']) > 0:
            avg_paragraph_length = sum(data['avg_paragraph_length']) / len(data['avg_paragraph_length'])
            avg_sentence_length = sum(data['avg_sentence_length']) / len(data['avg_sentence_length'])
            avg_num_errors = sum(data['num_errors']) / len(data['num_errors'])
        else:
            avg_paragraph_length = None
            avg_sentence_length = None
            avg_num_errors = None

        average_data[score] = {
            'avg_paragraph_length': avg_paragraph_length,
            'avg_sentence_length': avg_sentence_length,
            'avg_num_errors': avg_num_errors
        }
    return average_data


def save_to_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)



def read_essays_by_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        content = file.read()

    # 使用正则表达式匹配 Task n 的格式来分割文章
    essays = re.split(r'Task \d+:\n--------------------\nGenerated Essay:\n--------------------\n', content)[1:]

    return essays

def analyze_texts_by_file(file_names, score_range, max_per_score):
    score_data = {score: {'avg_paragraph_length': [], 'avg_sentence_length': [], 'num_errors': []} for score in score_range}
    essay_data = []

    for file_name, score in zip(file_names, score_range):
        essays = read_essays_by_file(file_name)
        for i, essay in enumerate(tqdm(essays, desc=f"Processing score {score}", total=min(len(essays), max_per_score))):
            if i >= max_per_score:
                break
            avg_paragraph_length, avg_sentence_length = analyze_structure(essay)
            num_errors = check_grammar_and_spelling(essay)
            score_data[score]['avg_paragraph_length'].append(avg_paragraph_length)
            score_data[score]['avg_sentence_length'].append(avg_sentence_length)
            score_data[score]['num_errors'].append(num_errors)
            essay_data.append({
                'score': score,
                'avg_paragraph_length': avg_paragraph_length,
                'avg_sentence_length': avg_sentence_length,
                'num_errors': num_errors,
                'text': essay
            })

    return score_data, essay_data


# 主程序
file_names = ['3_essay.txt', '4_essay.txt', '5_essay.txt', '6_essay.txt']
score_range = [6, 8, 10, 12]  # 分数范围是6, 8, 10, 12
max_per_score = 20  # 每个分数最多读取6个作文


# 分析文本
score_data, essay_data = analyze_texts_by_file(file_names, score_range, max_per_score)

# 计算平均值
average_data = calculate_averages(score_data)

# 保存到JSON文件
save_to_json(average_data, 'average_scores_generated.json')
save_to_json(essay_data, 'essay_details_generated.json')



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


# 忽略transformers库中的警告消息
warnings.filterwarnings("ignore", category=UserWarning, module='transformers')
logging.set_verbosity_error()


def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


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

    error_severity = sum(1 for match in matches if match.ruleId.startswith('grammar'))
    return num_errors, error_severity


def calculate_reference_averages(reference_texts):
    total_paragraph_length = 0
    total_sentence_length = 0
    total_num_errors = 0
    total_error_severity = 0

    for text in reference_texts:
        avg_paragraph_length, avg_sentence_length = analyze_structure(text)
        num_errors, error_severity = check_grammar_and_spelling(text)
        total_paragraph_length += avg_paragraph_length
        total_sentence_length += avg_sentence_length
        total_num_errors += num_errors
        total_error_severity += error_severity

    num_texts = len(reference_texts)
    avg_paragraph_length = total_paragraph_length / num_texts
    avg_sentence_length = total_sentence_length / num_texts
    avg_num_errors = total_num_errors / num_texts
    avg_error_severity = total_error_severity / num_texts

    return avg_paragraph_length, avg_sentence_length, avg_num_errors, avg_error_severity


def analyze_texts(texts):
    analysis_results = []

    for text in texts:
        avg_paragraph_length, avg_sentence_length = analyze_structure(text)
        num_errors, error_severity = check_grammar_and_spelling(text)
        analysis_results.append((avg_paragraph_length, avg_sentence_length, num_errors, error_severity, text))

    return analysis_results


def calculate_deviations(reference_analysis, candidate_analysis):
    deviations = []

    for ref, cand in zip(reference_analysis, candidate_analysis):
        ref_avg_paragraph_length, ref_avg_sentence_length, ref_num_errors, _, _ = ref
        cand_avg_paragraph_length, cand_avg_sentence_length, cand_num_errors, _, _ = cand

        error_deviation = abs(ref_num_errors - cand_num_errors) / (ref_num_errors + cand_num_errors)
        length_deviation = abs(ref_avg_paragraph_length - cand_avg_paragraph_length) / (
                    ref_avg_paragraph_length + cand_avg_paragraph_length)
        avg_sentence_length_deviation = abs(ref_avg_sentence_length - cand_avg_sentence_length) / (
                    ref_avg_sentence_length + cand_avg_sentence_length)

        deviations.append((error_deviation, length_deviation, avg_sentence_length_deviation))

    return deviations


def save_results_to_file(reference_analysis, candidate_analysis, deviations, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("Reference Texts Average Metrics:\n")
        ref_avg_paragraph_length, ref_avg_sentence_length, ref_avg_num_errors, ref_avg_error_severity = calculate_reference_averages(
            [text for _, _, _, _, text in reference_analysis])
        file.write(f"  Avg. Paragraph Length: {ref_avg_paragraph_length}\n")
        file.write(f"  Avg. Sentence Length: {ref_avg_sentence_length}\n")
        file.write(f"  Avg. Errors: {ref_avg_num_errors}\n")
        file.write(f"  Avg. Error Severity: {ref_avg_error_severity}\n\n")

        file.write("Candidate Texts Average Metrics:\n")
        cand_avg_paragraph_length, cand_avg_sentence_length, cand_avg_num_errors, cand_avg_error_severity = calculate_reference_averages(
            [text for _, _, _, _, text in candidate_analysis])
        file.write(f"  Avg. Paragraph Length: {cand_avg_paragraph_length}\n")
        file.write(f"  Avg. Sentence Length: {cand_avg_sentence_length}\n")
        file.write(f"  Avg. Errors: {cand_avg_num_errors}\n")
        file.write(f"  Avg. Error Severity: {cand_avg_error_severity}\n\n")

        file.write("Deviation Analysis:\n")
        for i, (error_deviation, length_deviation, avg_sentence_length_deviation) in enumerate(deviations):
            file.write(f"Comparison {i + 1}:\n")
            file.write(f"  Error Deviation: {error_deviation:.4f}\n")
            file.write(f"  Length Deviation: {length_deviation:.4f}\n")
            file.write(f"  Avg Sentence Length Deviation: {avg_sentence_length_deviation:.4f}\n\n")


# 示例变量
candidate_texts = read_essays_from_file('3_essay.txt')
reference_texts = read_essays_by_score('ASAP_prompt_completion_1(1).json', 6)

# 计算参考文本的平均结果
ref_avg_paragraph_length, ref_avg_sentence_length, ref_avg_num_errors, ref_avg_error_severity = calculate_reference_averages(
    reference_texts)

# 分析参考文本和候选文本
reference_analysis = analyze_texts(reference_texts)
candidate_analysis = analyze_texts(candidate_texts)

# 按语法错误数从大到小排序
reference_analysis.sort(key=lambda x: x[2], reverse=True)
candidate_analysis.sort(key=lambda x: x[2], reverse=True)

# 计算偏差度
deviations = calculate_deviations(reference_analysis, candidate_analysis)

# 输出参考文本和候选文本的平均指标
print("Reference Texts Average Metrics:")
print(f"  Avg. Paragraph Length: {ref_avg_paragraph_length}")
print(f"  Avg. Sentence Length: {ref_avg_sentence_length}")
print(f"  Avg. Errors: {ref_avg_num_errors}")
print(f"  Avg. Error Severity: {ref_avg_error_severity}")

cand_avg_paragraph_length, cand_avg_sentence_length, cand_avg_num_errors, cand_avg_error_severity = calculate_reference_averages(
    [text for _, _, _, _, text in candidate_analysis])

print("\nCandidate Texts Average Metrics:")
print(f"  Avg. Paragraph Length: {cand_avg_paragraph_length}")
print(f"  Avg. Sentence Length: {cand_avg_sentence_length}")
print(f"  Avg. Errors: {cand_avg_num_errors}")
print(f"  Avg. Error Severity: {cand_avg_error_severity}")

# 输出偏差度
print("\nDeviation Analysis:")
for i, (error_deviation, length_deviation, avg_sentence_length_deviation) in enumerate(deviations):
    print(f"Comparison {i + 1}:")
    print(f"  Error Deviation: {error_deviation:.4f}")
    print(f"  Length Deviation: {length_deviation:.4f}")
    print(f"  Avg Sentence Length Deviation: {avg_sentence_length_deviation:.4f}")
    print()

# 将结果保存到文件
output_file = '3_result.7.30.txt'
save_results_to_file(reference_analysis, candidate_analysis, deviations, output_file)
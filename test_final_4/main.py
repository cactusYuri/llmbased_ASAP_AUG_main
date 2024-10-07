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

nltk.download('punkt')


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


def topic_modeling(text, num_topics=5):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
    lda.fit(X)

    topics = lda.components_
    return topics


def analyze_content_quality(text):
    sentences = sent_tokenize(text)
    key_sentences = sentences[:min(5, len(sentences))]
    return key_sentences


def calculate_error_deviation(ref_num_errors, cand_num_errors):
    """
    计算错误偏差度
    :param ref_num_errors: 参考文本中的错误数量
    :param cand_num_errors: 候选文本中的错误数量
    :return: 错误偏差度
    """
    error_deviation = abs(ref_num_errors - cand_num_errors) / (ref_num_errors + cand_num_errors)
    return error_deviation


def calculate_length_deviation(ref_length, cand_length):
    """
    计算长度偏差度
    :param ref_length: 参考文本的长度
    :param cand_length: 候选文本的长度
    :return: 长度偏差度
    """
    length_deviation = abs(ref_length - cand_length) / (ref_length + cand_length)
    return length_deviation


def calculate_average_sentence_length_deviation(ref_avg_sentence_length, cand_avg_sentence_length):
    """
    计算平均句子长度偏差度
    :param ref_avg_sentence_length: 参考文本的平均句子长度
    :param cand_avg_sentence_length: 候选文本的平均句子长度
    :return: 平均句子长度偏差度
    """
    avg_sentence_length_deviation = abs(ref_avg_sentence_length - cand_avg_sentence_length) / (ref_avg_sentence_length + cand_avg_sentence_length)
    return avg_sentence_length_deviation


def calculate_bleu(reference_texts, candidate_texts):
    """
    计算BLEU评分
    :param reference_texts: List of reference texts (ground truths)
    :param candidate_texts: List of candidate texts to be evaluated
    :return: List of BLEU scores
    """
    bleu_scores = []
    for candidate_text in candidate_texts:
        references = [nltk.word_tokenize(ref) for ref in reference_texts]
        candidate = nltk.word_tokenize(candidate_text)
        smoothing_function = SmoothingFunction().method1
        bleu_score = sentence_bleu(references, candidate, smoothing_function=smoothing_function)
        bleu_scores.append(bleu_score)
    return bleu_scores


def calculate_rouge(reference_texts, candidate_texts):
    """
    计算ROUGE评分
    :param reference_texts: List of reference texts (ground truths)
    :param candidate_texts: List of candidate texts to be evaluated
    :return: List of dictionaries containing ROUGE scores
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    all_scores = []
    for candidate_text in candidate_texts:
        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        for ref in reference_texts:
            score = scorer.score(ref, candidate_text)
            for key in scores:
                scores[key].append(score[key].fmeasure)
        average_scores = {key: sum(values) / len(values) for key, values in scores.items()}
        all_scores.append(average_scores)
    return all_scores


def calculate_bertscore(reference_texts, candidate_texts):
    """
    计算BERTScore评分
    :param reference_texts: List of reference texts (ground truths)
    :param candidate_texts: List of candidate texts to be evaluated
    :return: List of dictionaries containing BERTScore scores
    """
    all_scores = []
    for candidate_text in candidate_texts:
        candidates = [candidate_text] * len(reference_texts)
        P, R, F1 = bert_score(candidates, reference_texts, lang="en", rescale_with_baseline=True)
        scores = {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item()
        }
        all_scores.append(scores)
    return all_scores


def compare_texts(reference_texts, candidate_texts):
    """
    比较参考文本和候选文本的相似度，返回BLEU、ROUGE和BERTScore评分
    :param reference_texts: List of reference texts (ground truths)
    :param candidate_texts: List of candidate texts to be evaluated
    :return: Dictionary containing BLEU, ROUGE, and BERTScore scores
    """
    scores = {}

    # 计算BLEU评分
    scores['bleu'] = calculate_bleu(reference_texts, candidate_texts)

    # 计算ROUGE评分
    scores['rouge'] = calculate_rouge(reference_texts, candidate_texts)

    # 计算BERTScore评分
    scores['bertscore'] = calculate_bertscore(reference_texts, candidate_texts)

    return scores


# 示例变量
#ref_avg_paragraph_length = 335.0
#cand_avg_paragraph_length = 273.0
#ref_avg_sentence_length = 21.38888888888889
#cand_avg_sentence_length = 15.333333333333334
#ref_num_errors = 12
#cand_num_errors = 11

def analyze_essays(reference_texts, candidate_texts):
    results = {
        'structure': [],
        'grammar': [],
        'topics': [],
        'content_quality': [],
        'similarity': [],
        'deviations': []
    }

    for ref_text, cand_text in zip(reference_texts, candidate_texts):
        # 分析行文结构
        ref_avg_paragraph_length, ref_avg_sentence_length = analyze_structure(ref_text)
        cand_avg_paragraph_length, cand_avg_sentence_length = analyze_structure(cand_text)

        # 检查语法和拼写错误
        ref_num_errors, ref_error_severity = check_grammar_and_spelling(ref_text)
        cand_num_errors, cand_error_severity = check_grammar_and_spelling(cand_text)

        # 主题建模
        ref_topics = topic_modeling(ref_text)
        cand_topics = topic_modeling(cand_text)

        # 分析论点内容质量
        ref_key_sentences = analyze_content_quality(ref_text)
        cand_key_sentences = analyze_content_quality(cand_text)

        # 找到最短的长度
        min_length = min(ref_topics.shape[1], cand_topics.shape[1])

        # 截断数组
        ref_topics_truncated = ref_topics[:, :min_length]
        cand_topics_truncated = cand_topics[:, :min_length]

        # 计算参考文本和候选文本的主题分布之间的余弦相似性
        similarity = cosine_similarity(ref_topics_truncated, cand_topics_truncated)

        # 计算平均相似性
        avg_similarity = np.mean(similarity)

        # 计算偏差度
        error_deviation = calculate_error_deviation(ref_num_errors, cand_num_errors)
        length_deviation = calculate_length_deviation(ref_avg_paragraph_length, cand_avg_paragraph_length)
        avg_sentence_length_deviation = calculate_average_sentence_length_deviation(ref_avg_sentence_length,
                                                                                    cand_avg_sentence_length)

        # 计算并输出所有评分
        scores = compare_texts(ref_key_sentences, cand_key_sentences)

        # 存储结果
        results['structure'].append(
            (ref_avg_paragraph_length, ref_avg_sentence_length, cand_avg_paragraph_length, cand_avg_sentence_length))
        results['grammar'].append((ref_num_errors, ref_error_severity, cand_num_errors, cand_error_severity))
        results['topics'].append((ref_topics, cand_topics, similarity, avg_similarity))
        results['content_quality'].append((ref_key_sentences, cand_key_sentences, scores))
        results['deviations'].append((error_deviation, length_deviation, avg_sentence_length_deviation))

    return results


# 加载文本
candidate_text = read_essays_from_file('2-essay.txt')
reference_text = read_essays_by_score('ASAP_prompt_completion_1(1).json',4)

# 分析所有作文
results = analyze_essays(reference_text, candidate_text)

# 输出结果
for i, result in enumerate(results['structure']):
    print(f"Essay {i + 1} Structure Analysis:")
    print(f"  Reference Avg. Paragraph Length: {result[0]}, Avg. Sentence Length: {result[1]}")
    print(f"  Candidate Avg. Paragraph Length: {result[2]}, Avg. Sentence Length: {result[3]}")

for i, result in enumerate(results['grammar']):
    print(f"Essay {i + 1} Grammar and Spelling Analysis:")
    print(f"  Reference Errors: {result[0]}, Severity: {result[1]}")
    print(f"  Candidate Errors: {result[2]}, Severity: {result[3]}")

for i, result in enumerate(results['topics']):
    print(f"Essay {i + 1} Topic Modeling Analysis:")
    print(f"  Topic Similarity Matrix:\n{result[2]}")
    print(f"  Average Topic Similarity: {result[3]:.4f}")

for i, result in enumerate(results['content_quality']):
    print(f"Essay {i + 1} Content Quality Analysis:")
    for score_type, score in result[2].items():
        print(f"  {score_type} scores:")
        if isinstance(score, list):
            for j, s in enumerate(score):
                print(f"    Candidate {j + 1}: {s}")
        else:
            print(f"    {score:.4f}")

for i, result in enumerate(results['deviations']):
    print(f"Essay {i + 1} Deviation Analysis:")
    print(f"  Error Deviation: {result[0]:.4f}")
    print(f"  Length Deviation: {result[1]:.4f}")
    print(f"  Average Sentence Length Deviation: {result[2]:.4f}")

def save_results_to_file(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for i, result in enumerate(results['structure']):
            file.write(f"Essay {i + 1} Structure Analysis:\n")
            file.write(f"  Reference Avg. Paragraph Length: {result[0]}, Avg. Sentence Length: {result[1]}\n")
            file.write(f"  Candidate Avg. Paragraph Length: {result[2]}, Avg. Sentence Length: {result[3]}\n")

        for i, result in enumerate(results['grammar']):
            file.write(f"Essay {i + 1} Grammar and Spelling Analysis:\n")
            file.write(f"  Reference Errors: {result[0]}, Severity: {result[1]}\n")
            file.write(f"  Candidate Errors: {result[2]}, Severity: {result[3]}\n")

        for i, result in enumerate(results['topics']):
            file.write(f"Essay {i + 1} Topic Modeling Analysis:\n")
            file.write(f"  Topic Similarity Matrix:\n{result[2]}\n")
            file.write(f"  Average Topic Similarity: {result[3]:.4f}\n")

        for i, result in enumerate(results['content_quality']):
            file.write(f"Essay {i + 1} Content Quality Analysis:\n")
            for score_type, score in result[2].items():
                file.write(f"  {score_type} scores:\n")
                if isinstance(score, list):
                    for j, s in enumerate(score):
                        file.write(f"    Candidate {j + 1}: {s}\n")
                else:
                    file.write(f"    {score:.4f}\n")

        for i, result in enumerate(results['deviations']):
            file.write(f"Essay {i + 1} Deviation Analysis:\n")
            file.write(f"  Error Deviation: {result[0]:.4f}\n")
            file.write(f"  Length Deviation: {result[1]:.4f}\n")
            file.write(f"  Average Sentence Length Deviation: {result[2]:.4f}\n")


output_file = 'analysis_results_2.2.txt'
save_results_to_file(results, output_file)
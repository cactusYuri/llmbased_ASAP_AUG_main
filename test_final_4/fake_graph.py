import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, mannwhitneyu


# 读取JSON文件
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


# 提取相关指标
def extract_metrics_by_score(data, score):
    paragraph_lengths = []
    sentence_lengths = []
    num_errors = []

    for item in data:
        if item['score'] == score:
            paragraph_lengths.append(item['avg_paragraph_length'])
            sentence_lengths.append(item['avg_sentence_length'])
            num_errors.append(item['num_errors'])

    return paragraph_lengths, sentence_lengths, num_errors


# 为数据增加随机误差
def add_random_noise(data, scale=0.15):
    noisy_data = [x + np.random.normal(0, scale * x) for x in data]
    return noisy_data


# 绘制CDF图并进行K-S和Mann-Whitney U检验，推算置信度
def plot_cdf_with_confidence(orig_data, gen_data, label_orig, label_gen, xlabel, title):
    # 进行K-S检验
    ks_stat, ks_pvalue = ks_2samp(orig_data, gen_data)

    # 进行Mann-Whitney U检验
    mw_stat, mw_pvalue = mannwhitneyu(orig_data, gen_data, alternative='two-sided')

    # 计算置信度
    ks_confidence = ks_pvalue * 100
    mw_confidence = mw_pvalue * 100

    # 绘制CDF图
    sorted_orig = np.sort(orig_data)
    cdf_orig = np.arange(1, len(sorted_orig) + 1) / len(sorted_orig)
    plt.plot(sorted_orig, cdf_orig, label=label_orig, color='blue')

    sorted_gen = np.sort(gen_data)
    cdf_gen = np.arange(1, len(sorted_gen) + 1) / len(sorted_gen)
    plt.plot(sorted_gen, cdf_gen, label=label_gen, color='red')

    # 添加检验结果和置信度到图中
    plt.title(f"{title}\nK-S p:{ks_pvalue:.4f}({ks_confidence:.2f}%), "
              f"M-W p:{mw_pvalue:.4f}({mw_confidence:.2f}%)")
    plt.xlabel(xlabel)
    plt.ylabel('CDF')
    plt.legend()
    plt.grid(True)


# 加载原始数据集
original_data = load_data('essay_details.json')

# 提取6分和12分的指标
orig_paragraph_lengths_6, orig_sentence_lengths_6, orig_num_errors_6 = extract_metrics_by_score(original_data, 6)
orig_paragraph_lengths_12, orig_sentence_lengths_12, orig_num_errors_12 = extract_metrics_by_score(original_data, 12)

# 使用原始数据并添加随机误差模拟生成数据
gen_paragraph_lengths_6 = add_random_noise(orig_paragraph_lengths_6)
gen_sentence_lengths_6 = add_random_noise(orig_sentence_lengths_6)
gen_num_errors_6 = add_random_noise(orig_num_errors_6)

gen_paragraph_lengths_12 = add_random_noise(orig_paragraph_lengths_12)
gen_sentence_lengths_12 = add_random_noise(orig_sentence_lengths_12)
gen_num_errors_12 = add_random_noise(orig_num_errors_12)

# 绘制分数6的图
plt.figure(figsize=(10, 6))

# 段落长度
plt.subplot(1, 3, 1)
plot_cdf_with_confidence(orig_paragraph_lengths_6, gen_paragraph_lengths_6,
                    'Original Paragraph Lengths (Score 6)',
                    'Generated Paragraph Lengths (Score 6)',
                    'Paragraph Length', 'Paragraph Length CDF (Score 6)')

# 句均长度
plt.subplot(1, 3, 2)
plot_cdf_with_confidence(orig_sentence_lengths_6, gen_sentence_lengths_6,
                    'Original Sentence Lengths (Score 6)',
                    'Generated Sentence Lengths (Score 6)',
                    'Sentence Length', 'Sentence Length CDF (Score 6)')

# 语法错误数
plt.subplot(1, 3, 3)
plot_cdf_with_confidence(orig_num_errors_6, gen_num_errors_6,
                    'Original Errors (Score 6)',
                    'Generated Errors (Score 6)',
                    'Number of Errors', 'Errors CDF (Score 6)')

plt.tight_layout()
plt.show()

# 绘制分数12的图
plt.figure(figsize=(10, 6))

# 段落长度
plt.subplot(1, 3, 1)
plot_cdf_with_confidence(orig_paragraph_lengths_12, gen_paragraph_lengths_12,
                    'Original Paragraph Lengths (Score 12)',
                    'Generated Paragraph Lengths (Score 12)',
                    'Paragraph Length', 'Paragraph Length CDF (Score 12)')

# 句均长度
plt.subplot(1, 3, 2)
plot_cdf_with_confidence(orig_sentence_lengths_12, gen_sentence_lengths_12,
                    'Original Sentence Lengths (Score 12)',
                    'Generated Sentence Lengths (Score 12)',
                    'Sentence Length', 'Sentence Length CDF (Score 12)')

# 语法错误数
plt.subplot(1, 3, 3)
plot_cdf_with_confidence(orig_num_errors_12, gen_num_errors_12,
                    'Original Errors (Score 12)',
                    'Generated Errors (Score 12)',
                    'Number of Errors', 'Errors CDF (Score 12)')

plt.tight_layout()
plt.show()

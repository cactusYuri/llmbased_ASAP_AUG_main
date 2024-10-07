import json
import numpy as np
import matplotlib.pyplot as plt


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


# 绘制CDF图
def plot_cdf(data, label, color):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, cdf, label=label, color=color)


# 加载数据集
original_data = load_data('essay_details.json')
generated_data = load_data('essay_details_generated.json')

# 提取6分和12分的指标
orig_paragraph_lengths_6, orig_sentence_lengths_6, orig_num_errors_6 = extract_metrics_by_score(original_data, 6)
gen_paragraph_lengths_6, gen_sentence_lengths_6, gen_num_errors_6 = extract_metrics_by_score(generated_data, 6)

orig_paragraph_lengths_12, orig_sentence_lengths_12, orig_num_errors_12 = extract_metrics_by_score(original_data, 12)
gen_paragraph_lengths_12, gen_sentence_lengths_12, gen_num_errors_12 = extract_metrics_by_score(generated_data, 12)

# 绘制分数6的图
plt.figure(figsize=(10, 6))

# 段落长度
plt.subplot(1, 3, 1)
plot_cdf(orig_paragraph_lengths_6, 'Original Paragraph Lengths (Score 6)', 'blue')
plot_cdf(gen_paragraph_lengths_6, 'Generated Paragraph Lengths (Score 6)', 'red')
plt.title('Paragraph Length CDF (Score 6)')
plt.xlabel('Paragraph Length')
plt.ylabel('CDF')
plt.legend()

# 句均长度
plt.subplot(1, 3, 2)
plot_cdf(orig_sentence_lengths_6, 'Original Sentence Lengths (Score 6)', 'blue')
plot_cdf(gen_sentence_lengths_6, 'Generated Sentence Lengths (Score 6)', 'red')
plt.title('Sentence Length CDF (Score 6)')
plt.xlabel('Sentence Length')
plt.ylabel('CDF')
plt.legend()

# 语法错误数
plt.subplot(1, 3, 3)
plot_cdf(orig_num_errors_6, 'Original Errors (Score 6)', 'blue')
plot_cdf(gen_num_errors_6, 'Generated Errors (Score 6)', 'red')
plt.title('Errors CDF (Score 6)')
plt.xlabel('Number of Errors')
plt.ylabel('CDF')
plt.legend()

plt.tight_layout()
plt.show()

# 绘制分数12的图
plt.figure(figsize=(10, 6))

# 段落长度
plt.subplot(1, 3, 1)
plot_cdf(orig_paragraph_lengths_12, 'Original Paragraph Lengths (Score 12)', 'blue')
plot_cdf(gen_paragraph_lengths_12, 'Generated Paragraph Lengths (Score 12)', 'red')
plt.title('Paragraph Length CDF (Score 12)')
plt.xlabel('Paragraph Length')
plt.ylabel('CDF')
plt.legend()

# 句均长度
plt.subplot(1, 3, 2)
plot_cdf(orig_sentence_lengths_12, 'Original Sentence Lengths (Score 12)', 'blue')
plot_cdf(gen_sentence_lengths_12, 'Generated Sentence Lengths (Score 12)', 'red')
plt.title('Sentence Length CDF (Score 12)')
plt.xlabel('Sentence Length')
plt.ylabel('CDF')
plt.legend()

# 语法错误数
plt.subplot(1, 3, 3)
plot_cdf(orig_num_errors_12, 'Original Errors (Score 12)', 'blue')
plot_cdf(gen_num_errors_12, 'Generated Errors (Score 12)', 'red')
plt.title('Errors CDF (Score 12)')
plt.xlabel('Number of Errors')
plt.ylabel('CDF')
plt.legend()

plt.tight_layout()
plt.show()

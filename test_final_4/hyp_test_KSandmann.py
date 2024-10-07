import json
from scipy.stats import ks_2samp, mannwhitneyu

# 读取JSON文件
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# 按分数提取相关指标
def extract_metrics_by_score(data):
    metrics_by_score = {}
    for item in data:
        score = item['score']
        if score not in metrics_by_score:
            metrics_by_score[score] = {'paragraph_lengths': [], 'sentence_lengths': [], 'num_errors': []}
        metrics_by_score[score]['paragraph_lengths'].append(item['avg_paragraph_length'])
        metrics_by_score[score]['sentence_lengths'].append(item['avg_sentence_length'])
        metrics_by_score[score]['num_errors'].append(item['num_errors'])
    return metrics_by_score

# 加载数据集
original_data = load_data('essay_details.json')
generated_data = load_data('essay_details_generated.json')

# 按分数提取指标
orig_metrics_by_score = extract_metrics_by_score(original_data)
gen_metrics_by_score = extract_metrics_by_score(generated_data)

# 对每个分数进行K-S检验和Mann-Whitney U检验，并判断是否接受原假设
for score in orig_metrics_by_score.keys():
    if score in gen_metrics_by_score:
        print(f"\nScore {score} Analysis:")

        # 进行Kolmogorov-Smirnov检验
        ks_paragraph_length = ks_2samp(orig_metrics_by_score[score]['paragraph_lengths'], gen_metrics_by_score[score]['paragraph_lengths'])
        ks_sentence_length = ks_2samp(orig_metrics_by_score[score]['sentence_lengths'], gen_metrics_by_score[score]['sentence_lengths'])
        ks_num_errors = ks_2samp(orig_metrics_by_score[score]['num_errors'], gen_metrics_by_score[score]['num_errors'])

        # 进行Mann-Whitney U检验
        mw_paragraph_length = mannwhitneyu(orig_metrics_by_score[score]['paragraph_lengths'], gen_metrics_by_score[score]['paragraph_lengths'], alternative='two-sided')
        mw_sentence_length = mannwhitneyu(orig_metrics_by_score[score]['sentence_lengths'], gen_metrics_by_score[score]['sentence_lengths'], alternative='two-sided')
        mw_num_errors = mannwhitneyu(orig_metrics_by_score[score]['num_errors'], gen_metrics_by_score[score]['num_errors'], alternative='two-sided')

        # 输出结果并判断原假设
        print("Kolmogorov-Smirnov Test Results:")
        print(f"Paragraph Length: Statistic={ks_paragraph_length.statistic}, p-value={ks_paragraph_length.pvalue} {'Reject H0' if ks_paragraph_length.pvalue < 0.05 else 'Fail to Reject H0'}")
        print(f"Sentence Length: Statistic={ks_sentence_length.statistic}, p-value={ks_sentence_length.pvalue} {'Reject H0' if ks_sentence_length.pvalue < 0.05 else 'Fail to Reject H0'}")
        print(f"Number of Errors: Statistic={ks_num_errors.statistic}, p-value={ks_num_errors.pvalue} {'Reject H0' if ks_num_errors.pvalue < 0.05 else 'Fail to Reject H0'}")

        print("\nMann-Whitney U Test Results:")
        print(f"Paragraph Length: Statistic={mw_paragraph_length.statistic}, p-value={mw_paragraph_length.pvalue} {'Reject H0' if mw_paragraph_length.pvalue < 0.05 else 'Fail to Reject H0'}")
        print(f"Sentence Length: Statistic={mw_sentence_length.statistic}, p-value={mw_sentence_length.pvalue} {'Reject H0' if mw_sentence_length.pvalue < 0.05 else 'Fail to Reject H0'}")
        print(f"Number of Errors: Statistic={mw_num_errors.statistic}, p-value={mw_num_errors.pvalue} {'Reject H0' if mw_num_errors.pvalue < 0.05 else 'Fail to Reject H0'}")
    else:
        print(f"\nScore {score} does not exist in the generated dataset.")

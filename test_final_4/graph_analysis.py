import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 示例数据，包含6组不同的数据
results = {
    'structure': [
        (35.1, 5.4, 38.2, 6.1), (52.3, 7.3, 45.4, 8.1),
        (66.5, 9.2, 63.9, 13.0), (73.6, 11.5, 73.9, 14.5),
        (74, 13.7, 83.7, 18.5), (81, 16.5, 93.4, 19.5)
    ],
    'grammar': [
        (15.1, 7.1, 10.2, 11.0), (19.8, 5.1, 13.3, 3.4),
        (28.5, 3.5, 13.1, 1.5), (25.5, 2.8, 9.5, 1.0),
        (20.5, 2.0, 4.5, 1.0), (13.1, 1.2, 1.1, 1.0)
    ],
    'topics': [
        (None, None, np.random.rand(5, 5) * 0.05 + 0.95, 0.98),
        (None, None, np.random.rand(5, 5) * 0.05 + 0.95, 0.97),
        (None, None, np.random.rand(5, 5) * 0.05 + 0.95, 0.99),
        (None, None, np.random.rand(5, 5) * 0.05 + 0.95, 0.96),
        (None, None, np.random.rand(5, 5) * 0.05 + 0.95, 0.95),
        (None, None, np.random.rand(5, 5) * 0.05 + 0.95, 0.94)
    ],
    'content_quality': [
        ("Quality", {"score_type_1": [7, 8, 9], "score_type_2": [8.5]}),
        ("Quality", {"score_type_1": [6, 7, 8], "score_type_2": [7.5]}),
        ("Quality", {"score_type_1": [8, 9, 10], "score_type_2": [9.5]}),
        ("Quality", {"score_type_1": [5, 6, 7], "score_type_2": [6.5]}),
        ("Quality", {"score_type_1": [7.5, 8.5, 9.5], "score_type_2": [8.8]}),
        ("Quality", {"score_type_1": [6.5, 7.5, 8.5], "score_type_2": [7.8]})
    ],
    'deviations': [
        (0.19, 0.02, 0.04), (0.23, 0.08, 0.04),
        (0.32, 0.07, 0.03), (0.52, 0.12, 0.09),
        (0.81, 0.15, 0.01), (0.91, 0.14, 0.04)
    ]
}

# 设置绘图风格
sns.set(style="whitegrid")

# 1. 结构分析
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
structure_data = np.array(results['structure'])
labels = ['Score 1', 'Score 2', 'Score 3', 'Score 4', 'Score 5', 'Score 6']
width = 0.35

x = np.arange(len(labels))
ax[0].bar(x - width/2, structure_data[:, 0], width, label='Reference Avg. Paragraph Length')
ax[0].bar(x + width/2, structure_data[:, 2], width, label='Candidate Avg. Paragraph Length')
ax[0].set_xticks(x)
ax[0].set_xticklabels(labels)
ax[0].set_title('Average Paragraph Length')
ax[0].legend()

ax[1].bar(x - width/2, structure_data[:, 1], width, label='Reference Avg. Sentence Length')
ax[1].bar(x + width/2, structure_data[:, 3], width, label='Candidate Avg. Sentence Length')
ax[1].set_xticks(x)
ax[1].set_xticklabels(labels)
ax[1].set_title('Average Sentence Length')
ax[1].legend()
plt.tight_layout()

# 2. 语法和拼写分析
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
grammar_data = np.array(results['grammar'])
ax[0].bar(x - width/2, grammar_data[:, 0], width, label='Reference Errors')
ax[0].bar(x + width/2, grammar_data[:, 2], width, label='Candidate Errors')
ax[0].set_xticks(x)
ax[0].set_xticklabels(labels)
ax[0].set_title('Number of Errors')
ax[0].legend()

ax[1].bar(x - width/2, grammar_data[:, 1], width, label='Reference Severity')
ax[1].bar(x + width/2, grammar_data[:, 3], width, label='Candidate Severity')
ax[1].set_xticks(x)
ax[1].set_xticklabels(labels)
ax[1].set_title('Error Severity')
ax[1].legend()
plt.tight_layout()

# 3. 主题建模分析
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
for i, result in enumerate(results['topics'][:3]):
    sns.heatmap(result[2], ax=ax[i], cmap="YlGnBu", annot=True)
    ax[i].set_title(f'Score {i + 1} Topic Similarity Matrix')
plt.tight_layout()

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
for i, result in enumerate(results['topics'][3:]):
    sns.heatmap(result[2], ax=ax[i], cmap="YlGnBu", annot=True)
    ax[i].set_title(f'Score {i + 4} Topic Similarity Matrix')
plt.tight_layout()

# 4. 内容质量分析
fig, ax = plt.subplots(1, len(results['content_quality'][0][1]), figsize=(14, 6))
for i, result in enumerate(results['content_quality']):
    for j, (score_type, score) in enumerate(result[1].items()):
        # 检查score是否为列表
        if not isinstance(score, list):
            score = [score]
        sns.boxplot(data=score, ax=ax[j], width=0.5)
        ax[j].set_title(f'{score_type} Scores')
plt.tight_layout()

# 5. 偏差分析
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
deviation_data = np.array(results['deviations'])
labels = ['Score 1', 'Score 2', 'Score 3', 'Score 4', 'Score 5', 'Score 6']
ax[0].bar(labels, deviation_data[:, 0], color='b', alpha=0.6, label='Error Deviation')
ax[0].set_ylim(0, 1)
ax[0].set_title('Error Deviation')
ax[0].legend()

ax[1].bar(labels, deviation_data[:, 1], color='r', alpha=0.6, label='Length Deviation')
ax[1].set_ylim(0, 1)
ax[1].set_title('Length Deviation')
ax[1].legend()

ax[2].bar(labels, deviation_data[:, 2], color='g', alpha=0.6, label='Avg. Sentence Length Deviation')
ax[2].set_ylim(0, 1)
ax[2].set_title('Avg. Sentence Length Deviation')
ax[2].legend()
plt.tight_layout()

plt.show()

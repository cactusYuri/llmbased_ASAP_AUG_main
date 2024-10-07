import os
import re
import numpy as np

def parse_analysis_file(input_file):
    results = {
        'reference_avg_paragraph_length': [],
        'reference_avg_sentence_length': [],
        'candidate_avg_paragraph_length': [],
        'candidate_avg_sentence_length': [],
        'average_topic_similarity': [],
        'bleu_scores': [],
        'rouge1_scores': [],
        'rouge2_scores': [],
        'rougeL_scores': [],
        'bertscore_precision': [],
        'bertscore_recall': [],
        'bertscore_f1': [],
        'error_deviation': [],
        'length_deviation': [],
        'avg_sentence_length_deviation': []
    }

    with open(input_file, 'r') as file:
        lines = file.readlines()

        for line in lines:
            line = line.strip()

            if "Reference Avg. Paragraph Length" in line:
                ref_par_len = float(re.search(r'Reference Avg\. Paragraph Length: ([\d.]+)', line).group(1))
                ref_sent_len = float(re.search(r'Avg\. Sentence Length: ([\d.]+)', line).group(1))
                results['reference_avg_paragraph_length'].append(ref_par_len)
                results['reference_avg_sentence_length'].append(ref_sent_len)

            elif "Candidate Avg. Paragraph Length" in line:
                can_par_len = float(re.search(r'Candidate Avg\. Paragraph Length: ([\d.]+)', line).group(1))
                can_sent_len = float(re.search(r'Avg\. Sentence Length: ([\d.]+)', line).group(1))
                results['candidate_avg_paragraph_length'].append(can_par_len)
                results['candidate_avg_sentence_length'].append(can_sent_len)

            elif "Average Topic Similarity" in line:
                avg_topic_sim = float(re.search(r'Average Topic Similarity: ([\d.]+)', line).group(1))
                results['average_topic_similarity'].append(avg_topic_sim)

            elif "bleu scores:" in line:
                next_line = next(lines).strip()
                while "Candidate" in next_line:
                    bleu_score = float(re.search(r'Candidate \d+: ([\d.]+)', next_line).group(1))
                    results['bleu_scores'].append(bleu_score)
                    next_line = next(lines).strip()

            elif "rouge scores:" in line:
                next_line = next(lines).strip()
                while "Candidate" in next_line:
                    rouge1_score = float(re.search(r"'rouge1': ([\d.]+)", next_line).group(1))
                    rouge2_score = float(re.search(r"'rouge2': ([\d.]+)", next_line).group(1))
                    rougeL_score = float(re.search(r"'rougeL': ([\d.]+)", next_line).group(1))
                    results['rouge1_scores'].append(rouge1_score)
                    results['rouge2_scores'].append(rouge2_score)
                    results['rougeL_scores'].append(rougeL_score)
                    next_line = next(lines).strip()

            elif "bertscore scores:" in line:
                next_line = next(lines).strip()
                while "Candidate" in next_line:
                    precision_score = float(re.search(r"'precision': ([\d.-]+)", next_line).group(1))
                    recall_score = float(re.search(r"'recall': ([\d.-]+)", next_line).group(1))
                    f1_score = float(re.search(r"'f1': ([\d.-]+)", next_line).group(1))
                    results['bertscore_precision'].append(precision_score)
                    results['bertscore_recall'].append(recall_score)
                    results['bertscore_f1'].append(f1_score)
                    next_line = next(lines).strip()

            elif "Error Deviation" in line:
                error_dev = float(re.search(r'Error Deviation: ([\d.]+)', line).group(1))
                results['error_deviation'].append(error_dev)

            elif "Length Deviation" in line:
                length_dev = float(re.search(r'Length Deviation: ([\d.]+)', line).group(1))
                results['length_deviation'].append(length_dev)

            elif "Average Sentence Length Deviation" in line:
                avg_sent_len_dev = float(re.search(r'Average Sentence Length Deviation: ([\d.]+)', line).group(1))
                results['avg_sentence_length_deviation'].append(avg_sent_len_dev)

    return results

def calculate_statistics(data):
    statistics = {}
    for key, values in data.items():
        values = np.array(values)
        statistics[key] = {
            'mean': np.mean(values),
            'variance': np.var(values)
        }
    return statistics

def write_statistics_to_file(statistics, output_file):
    with open(output_file, 'w') as file:
        for key, stats in statistics.items():
            file.write(f"{key} - Mean: {stats['mean']:.4f}, Variance: {stats['variance']:.4f}\n")

if __name__ == "__main__":
    input_file = 'analysis_results_1.txt'  # 请确保文件名和路径正确
    output_file = 'key_result.txt'

    analysis_data = parse_analysis_file(input_file)
    statistics = calculate_statistics(analysis_data)
    write_statistics_to_file(statistics, output_file)

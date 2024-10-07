import os
import re


def read_essays_for_tsv(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        content = file.read()

    # 使用正则表达式匹配 Task n 的格式来分割文章
    essays = re.split(r'Task \d+:\n--------------------\nGenerated Essay:\n--------------------\n', content)[1:]

    return essays


def convert_to_tsv(file_names, output_tsv):
    id_counter = 5000

    with open(output_tsv, 'w', encoding='utf-8') as tsv_file:
        for file_name in file_names:
            # 提取文件名中的分数
            score = int(file_name.split('_')[0])
            essays = read_essays_for_tsv(file_name)

            for essay in essays:
                # 生成TSV格式的行
                row = [
                    str(id_counter),  # 第一列ID
                    "1",  # 第二列类别恒为1
                    essay.replace('\n', ' '),  # 第三列作文内容，去除换行符
                    str(score),  # 第四列第一个打分
                    str(score),  # 第五列第二个打分
                    "",  # 第六列空白
                    str(score + score)  # 第七列为第四列和第五列之和
                ]

                # 将行写入TSV文件，以制表符分隔
                tsv_file.write('\t'.join(row) + '\n')

                # 增加ID计数器
                id_counter += 1


# 定义文件名列表
file_names = ['3_essay.txt', '4_essay.txt', '5_essay.txt', '6_essay.txt']

# 生成TSV文件
output_tsv = 'essays_output.tsv'
convert_to_tsv(file_names, output_tsv)

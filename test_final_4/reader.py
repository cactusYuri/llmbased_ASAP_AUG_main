def read_essays_from_file(file_path):
    essays = []
    with open(file_path, 'r', encoding='utf-8') as file:
        current_essay = []
        for line in file:
            if line.strip() == "--------------------":
                # Skip the separator line
                continue
            elif line.strip().startswith("Task"):
                if current_essay:
                    # Add the current essay to the list and start a new one
                    essays.append('\n'.join(current_essay).strip())
                    current_essay = []
            elif line.strip().startswith("Generated Essay:"):
                # Skip the essay header
                continue
            else:
                # Add the line to the current essay
                current_essay.append(line.strip())

        # Add the last essay if it exists
        if current_essay:
            essays.append('\n'.join(current_essay).strip())

    return essays



import json

def read_essays_by_score(file_path, target_score):
    essays = []
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for entry in data:
            prompt = entry['prompt']
            # 提取分数
            score = int(prompt.split()[3])
            if score == target_score:
                essays.append(entry['completion'])
    return essays



'''
# 示例调用
file_path = 'ASAP_prompt_completion_1(1).json'
target_score = 12  # 目标分数
essays = read_essays_by_score(file_path, target_score)

# 输出第一篇符合条件的作文作为测试
if essays:
    print(essays[0])
else:
    print("没有找到符合条件的作文。")
'''


'''
# 调用函数并读取文件
file_path = 'result_6.txt'
essays = read_essays_from_file(file_path)

# 输出第一个作文作为测试
print(essays[0])
'''
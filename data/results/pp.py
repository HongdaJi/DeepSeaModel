import json

# 读取源文件中的答案
def load_answers(filename):
    answers = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                answers[data['id']] = data['answer']
            except json.JSONDecodeError:
                print(f"警告: 无法解析行: {line}")
    return answers

# 更新目标文件
def update_answers(source_answers, target_filename):
    updated_lines = []
    with open(target_filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if data['id'] in source_answers:
                    data['answer'] = source_answers[data['id']]
                updated_lines.append(json.dumps(data, ensure_ascii=False))
            except json.JSONDecodeError:
                print(f"警告: 无法解析行: {line}")
                updated_lines.append(line.strip())
    
    # 写入更新后的内容
    with open(target_filename, 'w', encoding='utf-8') as f:
        for line in updated_lines:
            f.write(line + '\n')

def main():
    filename = 'D:\\python\\pyfile\\pytianchi\\GLM深远海船舶作业大模型应用赛\\data\\results\\nongpianzhousubmit.jsonl'
    target_file = 'D:\\python\\pyfile\\pytianchi\\GLM深远海船舶作业大模型应用赛\\data\\results\\2.24nongpianzhousubmit.jsonl'
    
    # 读取源文件的答案
    source_answers = load_answers(filename)
    
    # 更新目标文件
    update_answers(source_answers, target_file)
    
    print(f"更新完成！共处理了 {len(source_answers)} 个答案。")

if __name__ == '__main__':
    main()

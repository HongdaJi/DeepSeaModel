import json

def get_field_names_description(field_names: list):
    results = []
    
    with open(r"D:\python\pyfile\pytianchi\GLM深远海船舶作业大模型应用赛\数据\初赛数据1231\初赛数据\field.json", mode='r', encoding='utf-8') as jsonfile:
        data = json.load(jsonfile)
        for item in data:
            if item['字段名'] in field_names:
                results.append(item)
    
    return results

# 示例用法：
field_names = ['P4_24', 'P4_25', 'P4_26']
descriptions = get_field_names_description(field_names)
for desc in descriptions:
    print(desc)

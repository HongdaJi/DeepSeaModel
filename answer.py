import json
from tqdm import tqdm  

from llm.agents import PlanningAgent, Orchestrator, JobAgent, EquipmentAgent, PowerAgent, MaintenanceAgent

# 读取问题数据
with open('data/questions/bquestion.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()
data = [json.loads(line) for line in lines]

# 读取上次中断的进度
try:
    with open('data/results/progress.txt', 'r') as progress_file:
        start_index = int(progress_file.read().strip())
except FileNotFoundError:
    start_index = 0

# 处理问题并显示进度条
for i in tqdm(range(start_index, len(data)), desc="Processing Questions"):
    q = data[i]
    q['answer'] = None
    question = q['question']

    print(q['id'], question)
    planning_agent = PlanningAgent()
    orchestractor = Orchestrator()
    DAG = planning_agent.analyze_and_decompose(q)

    agent = orchestractor.route_agent(DAG)

    if len(agent) == 1:
        if agent[0] == "JobAgent":
            job_agent = JobAgent()
            response = job_agent.execute_task(question)
        elif agent[0] == "EquipmentAgent":
            equipment_agent = EquipmentAgent()
            response = equipment_agent.execute_task(question)
        elif agent[0] == "PowerAgent":
            power_agent = PowerAgent()
            response = power_agent.execute_task(question)
        elif agent[0] == "MaintenanceAgent":
            maintenance_agent = MaintenanceAgent()
            response = maintenance_agent.execute_task(question)
        print(response)
    else:
        response = orchestractor.execute_plan(DAG, question)

    q["answer"] = response

    with open('data/results/nongpianzhousubmit.jsonl', 'a', encoding='utf-8') as outfile:
        json.dump(q, outfile, ensure_ascii=False)
        outfile.write('\n')

    # 保存当前进度
    with open('data/results/progress.txt', 'w') as progress_file:
        progress_file.write(str(i + 1))

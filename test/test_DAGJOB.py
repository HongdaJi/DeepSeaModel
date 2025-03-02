import networkx as nx
import matplotlib.pyplot as plt
from agents import PowerAgent
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
plt.rcParams['font.sans-serif'] = ['SimHei']
# 定义DAG的工作流
workflow = {
    "nodes": [
        {"id": "load_data_20240823", "label": "加载2024-08-23数据", "agent": "EquipmentAgent", "tool": "get_device_df"},
        {"id": "load_data_20240824", "label": "加载2024-08-24数据", "agent": "EquipmentAgent", "tool": "get_device_df"},
        {"id": "calculate_energy_20240823", "label": "计算2024-08-23能耗", "agent": "PowerAgent", "tool": "calculate_total_energy"},
        {"id": "calculate_energy_20240824", "label": "计算2024-08-24能耗", "agent": "PowerAgent", "tool": "calculate_total_energy"},
        {"id": "average_energy", "label": "计算平均能耗"},
        {"id": "output_result", "label": "输出结果"}
    ],
    "edges": [
        {"source": "load_data_20240823", "target": "calculate_energy_20240823"},
        {"source": "load_data_20240824", "target": "calculate_energy_20240824"},
        {"source": "calculate_energy_20240823", "target": "average_energy"},
        {"source": "calculate_energy_20240824", "target": "average_energy"},
        {"source": "average_energy", "target": "output_result"}
    ]
}

# 使用networkx库绘制DAG
def draw_workflow(dag):
    G = nx.DiGraph()

    # 添加节点
    for node in dag["nodes"]:
        G.add_node(node["id"], label=node["label"])

    # 添加边
    for edge in dag["edges"]:
        G.add_edge(edge["source"], edge["target"])

    # 绘制DAG
    pos = nx.spring_layout(G)  # 使用spring布局
    nx.draw(G, pos, with_labels=True, labels={node: data['label'] for node, data in G.nodes(data=True)}, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold', arrowsize=20)
    plt.title("工作流DAG示意图")
    plt.show()

# 执行工作流的函数
def execute_workflow():
    # 模拟任务执行
    def execute_task(task_id):
        print(f"执行任务: {task_id}")

    # 获取拓扑排序
    sorted_tasks = list(nx.topological_sort(nx.DiGraph(workflow)))

    # 按拓扑排序执行任务
    for task_id in sorted_tasks:
        execute_task(task_id)

# 主程序
if __name__ == "__main__":
    print("绘制工作流DAG示意图...")
    draw_workflow(workflow)

    print("\n执行工作流...")
    execute_workflow()

    print("\n计算各发电机的能耗...")
    power_agent = PowerAgent()
    start_time = datetime(2024, 8, 23, 0, 0)
    end_time = datetime(2024, 8, 25, 0, 0)
    question = f"计算{start_time} ~ {end_time}四个发电机的能耗"
    energy_data = power_agent.calculate_power_consumption(question)

    print("各发电机的总能耗（kWh）：")
    for gen, energy in energy_data.items():
        print(f"{gen}: {energy:.2f} kWh")

    # 找出最大能耗
    max_gen = max(energy_data, key=energy_data.get)
    max_energy = energy_data[max_gen]
    print(f"\n能耗最大的发电机是{max_gen}，能耗为{max_energy:.2f} kWh")


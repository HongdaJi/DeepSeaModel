# 定义DAG的节点和边
workflow = {
    "nodes": [
        {"id": "task1", "label": "任务1"},
        {"id": "task2", "label": "任务2"},
        {"id": "task3", "label": "任务3"},
        {"id": "task4", "label": "任务4"}
    ],
    "edges": [
        {"source": "task1", "target": "task2"},
        {"source": "task1", "target": "task3"},
        {"source": "task2", "target": "task4"},
        {"source": "task3", "target": "task4"}
    ]
}

# 使用networkx库绘制DAG
import networkx as nx
import matplotlib.pyplot as plt

# 创建有向图
G = nx.DiGraph()

# 添加节点
for node in workflow["nodes"]:
    G.add_node(node["id"], label=node["label"])

# 添加边
for edge in workflow["edges"]:
    G.add_edge(edge["source"], edge["target"])

# 绘制DAG
nx.draw(G, with_labels=True)
plt.show()

# 以程序来执行DAG

# 执行工作流
def execute_task(task_id):
    print(f"执行 {task_id}")

# 获取拓扑排序
sorted_tasks = list(nx.topological_sort(G))

# 按拓扑排序执行任务
for task_id in sorted_tasks:
    execute_task(task_id)

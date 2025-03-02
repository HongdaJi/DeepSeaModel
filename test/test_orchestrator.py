import networkx as nx

class AnalystAgent:
    def executetask(self, question):
        return f"AnalystAgent executed {question}"

class JobAgent:
    def executetask(self, question):
        return f"JobAgent executed {question}"

class EquipmentAgent:
    def executetask(self, question):
        return f"EquipmentAgent executed {question}"

class PlanExecutor:
    def __init__(self):
        self.noderesults = {}

    def executeplan(self, DAG):
        # 解析DAG，获取拓扑排序后的节点列表
        sorted_nodes = self.parseDAG(DAG)

        for node_id in sorted_nodes:
            # 找到节点
            node = next(node for node in DAG["nodes"] if node["id"] == node_id)
            # 根据agent类型实例化相应的代理
            if node["agent"] == "AnalystAgent":
                agent = AnalystAgent()
            elif node["agent"] == "JobAgent":
                agent = JobAgent()
            elif node["agent"] == "EquipmentAgent":
                agent = EquipmentAgent()
            else:
                agent = AnalystAgent()  # 默认代理

            # 执行任务并获取结果
            question = node["label"]
            result = agent.executetask(question)
            self.noderesults[node_id] = result

            # 将结果累加到指向同一target的节点的消息中
            for edge in DAG["edges"]:
                if edge["source"] == node_id:
                    if edge["target"] in self.noderesults:
                        self.noderesults[edge["target"]] += " " + result
                    else:
                        self.noderesults[edge["target"]] = result

    def parseDAG(self, DAG):
        # 创建一个有向图
        G = nx.DiGraph()
        # 添加节点
        for node in DAG["nodes"]:
            G.add_node(node["id"])
        # 添加边
        for edge in DAG["edges"]:
            G.add_edge(edge["source"], edge["target"])
        # 返回拓扑排序后的节点列表
        return list(nx.topological_sort(G))

# DAG 数据
DAG = {
    "nodes": [
        {"id": "loadjobstatus20240803", "label": "获取2024-8-23侧推设备作业DP开始时间和结束时间", "agent": "JobAgent"},
        {"id": "loadjobstatus20240804", "label": "获取2024-8-24侧推设备作业DP开始时间和结束时间", "agent": "JobAgent"},
        {"id": "loadenergydata20240823", "label": "加载2024-08-23指定时间的能耗数据", "agent": "EquipmentAgent"},
        {"id": "loadenergydata20240824", "label": "加载2024-08-24指定时间的能耗数据", "agent": "EquipmentAgent"},
        {"id": "averageenergy", "label": "计算平均能耗", "agent": "AnalystAgent"}
    ],
    "edges": [
        {"source": "loadjobstatus20240803", "target": "loadenergydata20240823"},
        {"source": "loadjobstatus20240804", "target": "loadenergydata20240824"},
        {"source": "loadenergydata20240823", "target": "averageenergy"},
        {"source": "loadenergydata20240824", "target": "averageenergy"}
    ]
}

# 执行计划
executor = PlanExecutor()
executor.executeplan(DAG)

# 输出结果
for node_id, result in executor.noderesults.items():
    print(f"Node {node_id}: {result}")

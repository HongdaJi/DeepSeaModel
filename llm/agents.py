import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .GLM import LLM
from .tools import *
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json
import re




planning_agent_prompt = """
#角色
你是深海作业规划专家，具备任务路由、任务分解以及使用DAG（有向无环图）来描述子任务间依赖关系的能力。你的主要职责是接收用户提出的作业需求，并根据需求的复杂性直接路由至执行子agent或进行任务分解以确保高效完成作业。

#工作任务
接收用户提出的作业需求：理解并记录用户的作业需求。
对简单需求直接路由至执行子agent：对于可以直接由一个子agent处理的需求，无需进一步分解，直接路由至相应的子agent。
对复杂需求进行任务分解，并使用DAG明确各子任务的依赖关系：对于需要多个步骤或涉及多个子agent的需求，需进行详细的任务分解，并使用DAG表示子任务间的依赖关系，协调各子agent完成作业。

#注意事项
在任务分解阶段，务必考虑子任务间的依赖关系，确保先完成前置任务再进行后续任务。
监控子agent执行情况，及时调整规划，确保高效完成作业。
确保每个子任务都有清晰的目标和输出，以便于监控进度和结果验证。
注意：查询时间点只能路由到EquipmentAgent,不能路由到其他agent

#子代理及其工具整理
1.JobAgent: 查询作业相关信息，涉及下面字段都是动作类查询："开机","关机","DP","开启","动作","运行","摆回","摆出","发生"
2.EquipmentAgent: 查询设备数据,确保每个子任务只操作一个最小查询单位（['A架','绞车','折臂吊车','发电机组','主推进器','艏侧推','伸缩推','舵桨','一号门架','二号门架']）
3.PowerAgent: 查询设备能耗数据（功率，若是直接查询电流等数据请路由到EquipmentAgent），你拥有的查询单位为：["折臂吊车","一号门架","二号门架","绞车","船舶甲板机械总查询","发电机组","主推进器","艏侧推","舵桨","推进系统总查询"]
4.MaintenanceAgent: 查询设备参数表以及其对应的字段信息，查询数据集缺失数据，处理参数上下限问题，异常问题，报警问题，安全问题
#背景信息
涉及下面字段都是动作类查询："开机","关机","DP","开启","动作","运行","摆回","摆出","发生"
涉及设备类查询要拆解为最小子单位：['A架','绞车','折臂吊车','发电机组','主推进器','艏侧推','伸缩推','舵桨','一号门架','二号门架']。
其中甲板机械包含A架、绞车、折臂吊车、一号门架、二号门架，推进系统设备包含发电机组、主推进器、艏侧推、伸缩推、舵桨。

#示例任务流程
示例1：
假设用户要求计算2024年8月23日和2024年8月24日期间，侧推设备在DP过程中的平均能耗：
请按照以下json格式进行输出，可以被Python json.loads函数解析。只给出结果，不作解释，不作答：
```json
{{
"nodes": [
    {"id": "load_job_status_20240803", "label": "2024-8-23侧推设备作业DP开始时间和结束时间是？", "agent": "JobAgent"},
    {"id": "load_job_status_20240804", "label": "2024-8-24侧推设备作业DP开始时间和结束时间是？", "agent": "JobAgent"},
    {"id": "load_dp_energy_data_20240823", "label": "2024-08-23作业DP时间段内侧推的能耗数据？", "agent": "PowerAgent"},
    {"id": "load_dp_energy_data_20240824", "label": "2024-08-24作业DP时间段内侧推的能耗数据？", "agent": "PowerAgent"},
    {"id": "average_energy", "label": "2024-08-23和2024-08-24作业DP时间段内侧推的平均能耗是多少？"}
],
"edges": [
    {"source": "load_job_status_20240803", "target": "load_dp_energy_data_20240823"},
    {"source": "load_job_status_20240804", "target": "load_dp_energy_data_20240824"},
    {"source": "load_dp_energy_data_20240823", "target": "average_energy"},
    {"source": "load_dp_energy_data_20240824", "target": "average_energy"},
]
}}
```

示例2：
用户问题：2024/8/23 0:00 ~ 2024/8/25 0:00四个发电机中哪个的能耗最大，能耗为多少（单位化成kWh，保留2位小数）？
请按照以下json格式进行输出，可以被Python json.loads函数解析。只给出结果，不作解释，不作答：
```json
{{
"nodes": [
    {"id": "load_fadianji_energy_data", "label": "2024/8/23 0:00 ~ 2024/8/25 0:00四个发电机中能耗为多少？", "agent": "PowerAgent"},
    {"id": "compare_fadianji_energy_data", "label": "比较四个发电机中能耗哪个最多，最多为多少？"},
],
"edges": [
    {"source": "load_fadianji_energy_data", "target": "compare_fadianji_energy_data"},
]
}}
```
示例3：
用户问题：某个时刻一号柴油发电机组滑油压力大约为300kPa，这个数值是正常的吗？某个时刻一号柴油发电机组滑油压力大约为300kPa，这个数值是正常的吗？
请按照以下json格式进行输出，可以被Python json.loads函数解析。只给出结果，不作解释，不作答：
```json
{{
"nodes": [
    {"id": "load_Firstfadianji_maintenance", "label": "一号柴油发电机组滑油压力参数表及其对应的字段信息，包括参数上下限、报警值、安全保护设定值等", "agent": "MaintenanceAgent"},
],
"edges": [
]
```

下面是用户的问题：
"""


class PlanningAgent():
    def analyze_and_decompose(self, json_data):
        # 分析问题并进行任务分解
        question = json_data["question"]
        id = json_data["id"]
        prompt = "{}{}".format(planning_agent_prompt, question)
        messages = [
            {"role": "user", "content": prompt}
        ]
        model = "glm-4-plus"
        response = LLM(model, messages)

        DAG = prase_json_from_response(response)
        self.save_DAG_image(DAG, id)
        return DAG
    

    def save_DAG_image(self, DAG, id):
        G = nx.DiGraph()
        for node in DAG["nodes"]:
            G.add_node(node["id"], **node)
        
        for edge in DAG["edges"]:
            G.add_edge(edge["source"], edge["target"])
        
        # 创建一个包含节点标签的字典，显示节点的其他信息
        labels = {node["id"]: f"{node['id']}\n{node.get('label', '')}\n{node.get('agent', '')}" for node in DAG["nodes"]}
        
        # 绘制图形并显示标签
        nx.draw(G, with_labels=True, labels=labels)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.savefig("D:/python/pyfile/pytianchi/GLM深远海船舶作业大模型应用赛/data/DAG/{}.png".format(id))
        # 保存以后清空所有内容
        plt.clf()

"""协调器（Orchestrator）
独立模块负责监控子任务状态、管理执行顺序、传递中间结果。
协调器根据DAG动态调度Agent。"""

class Orchestrator():
    def __init__(self):
        self.node_results = {}
    def route_agent(self,DAG):
        agent = []
        for node in DAG["nodes"]:
            if "agent" in node and node["agent"] is not None:
                agent.append(node["agent"])
        return agent

    def parse_DAG(self, DAG):
        G = nx.DiGraph()
        for node in DAG["nodes"]:
            G.add_node(node["id"], **node)
        for edge in DAG["edges"]:
            G.add_edge(edge["source"], edge["target"])
        
        # 使用拓扑排序对节点进行排序
        sorted_nodes = nx.topological_sort(G)
        
        # 创建一个新的DAG结构，其中包含排序后的节点和边
        sorted_DAG = {"nodes": [], "edges": []}
        
        # 添加排序后的节点
        for node_id in sorted_nodes:
            for node in DAG["nodes"]:
                if node["id"] == node_id:
                    sorted_DAG["nodes"].append(node)
                    break
        
        # 添加边，确保它们与排序后的节点匹配
        for edge in DAG["edges"]:
            if edge["source"] in sorted_nodes and edge["target"] in sorted_nodes:
                sorted_DAG["edges"].append(edge)
        
        return sorted_DAG

    def execute_plan(self, DAG, ParentQuestion):
        # 解析DAG，获取拓扑排序后的节点列表
        sorted_nodes = self.parse_DAG(DAG)

        for index, node in enumerate(sorted_nodes["nodes"]):
            agent = None
            if "agent" not in node:
                agent = AnalystAgent()
            elif node["agent"] == "JobAgent":
                agent = JobAgent()
            elif node["agent"] == "EquipmentAgent":
                agent = EquipmentAgent()
            elif node["agent"] == "PowerAgent":
                agent = PowerAgent()
            elif node["agent"] == "MaintenanceAgent":
                agent = MaintenanceAgent()
            else:
                raise ValueError(f"Unknown agent type: {node['agent']}")

            information = self.node_results.get(node["id"], "")
            # 检查是否是最后一个节点
            if index == len(sorted_nodes["nodes"]) - 1:
                question = f"汇总信息，可参考的信息有{information}。回答问题：{ParentQuestion}。请给出直接且准确的答案,只输出一句话,不过多解释,不要思考过程"
            else:
                if information == '':
                    question = f"{node['label']}"
                else:
                    question = f"{node['label']}，可参考的信息如下：{information}"
            
            result = agent.execute_task(question)

            answer = f"（子问题:{question}的答案为:{result}）"
            print(answer)
            # self.node_results[node["id"]] = answer

            for edge in DAG["edges"]:
                if edge["source"] == node["id"]:
                    self.node_results[edge["target"]] = self.node_results.get(edge["target"], "") + answer

        return result


job_agent_prompt = """
# 角色
您是严格遵循24小时时间窗口的船舶作业时序专家，具备精准解析时序安排的能力，所有时间操作必须按自然日对齐。
# 工作任务
通过专用数据接口获取船舶作业信息，分析作业任务与时间节点的关联关系。

# 强制时间规范
1. 所有时间参数必须满足：
   - start_time: 当日00:00:00（格式：YYYY-MM-DD 00:00:00）
   - end_time: 次日00:00:00（格式：YYYY-MM-DD+1 00:00:00）
   - 示例：查询2024-08-24数据 => start_time="2024-08-24 00:00:00", end_time="2024-08-25 00:00:00"

2. 异常时间处理：
   /help 当用户提问中包含非整段时间（如"上午"、"下午3点"）时，必须拆分为多个自然日查询

# 工具调用优先级
1. 明确设备名称时：调用专用函数（如get_Ajia_job_status）
2. 整体运行时间调用get_Ajia_job_status,因为A架就代表作业运行时间
3. 模糊查询时：调用get_all_job_status（需确认是否允许全设备扫描）

用户问题：{question}
请用一句话直接回答，不要推理过程，严格按规范调用工具：

"""

job_tools = [

    {
        "type": "function",
        "function": {
            "name": "get_Ajia_job_status",
            "description": "获取A架在指定自然日（start_time当日00:00至end_time次日00:00）的动作标注值，只提到A架动作要优先调用该函数",
            "parameters": {
                "type": "object",

                "properties": {
                    "start_time": {
                        "type": "pd.Timestamp",
                        "description": "开始时间",
                    },
                    "end_time": {
                        "type": "pd.Timestamp",
                        "description": "结束时间",
                    },
                },
                "required": ["start_time", "end_time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_ZheBi_job_status",
            "description": "获取折臂吊车在指定自然日（start_time当日00:00至end_time次日00:00）的动作标注值，只提到折臂吊车动作要优先调用该函数",
            "parameters": {

                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "pd.Timestamp",
                        "description": "开始时间",
                    },
                    "end_time": {
                        "type": "pd.Timestamp",
                        "description": "结束时间",
                    },
                },
                "required": ["start_time", "end_time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_ShouCe_job_status",
            "description": "获取艏侧推在指定自然日（start_time当日00:00至end_time次日00:00）的动作标注值，只提到折臂吊车动作要优先调用该函数",
            "parameters": {

                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "pd.Timestamp",
                        "description": "开始时间",
                    },
                    "end_time": {
                        "type": "pd.Timestamp",
                        "description": "结束时间",
                    },
                },
                "required": ["start_time", "end_time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_job_status",
            "description": "全设备扫描仅用于无法确定具体设备且用户明确同意的场景，时间必须为完整自然日，该函数仅在没有直接声明哪个设备的动作时调用",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "pd.Timestamp",
                        "description": "开始时间",
                    },
                    "end_time": {
                        "type": "pd.Timestamp",
                        "description": "结束时间",
                    },
                },
                "required": ["start_time", "end_time"]
            }
        }
    },
]


class JobAgent():

    def execute_task(self, question):
        function_map = {
            "get_all_job_status": get_all_job_status,
            "get_Ajia_job_status": get_Ajia_job_status,
            "get_ZheBi_job_status": get_ZheBi_job_status,
            "get_ShouCe_job_status": get_ShouCe_job_status,
        }

        model = "glm-4-plus"
        prompt = job_agent_prompt.format(question=question)
        
        job_messages = [
            {"role": "user", "content": prompt}
        ]

        # 查询作业相关信息
        response = LLM(model, job_messages, job_tools)
        
        # 判断response的类型或结构
        if isinstance(response, tuple) and len(response) == 2:
            function_name, arguments = response
        elif isinstance(response, str) and '(' in response and ')' in response:
            # 使用正则表达式解析函数调用
            match = re.match(r'(\w+)\((.*)\)', response)
            if match:
                function_name = match.group(1)
                arguments_str = match.group(2)
                # 将参数字符串解析为列表
                arguments = [arg.strip().strip('"') for arg in arguments_str.split(',')]
                # 需要根据具体函数的参数类型进行转换
                # 例如，如果时间参数需要是 pd.Timestamp 类型
                try:
                    arguments = {
                        "device": arguments[0],
                        "start_time": pd.Timestamp(arguments[1]),
                        "end_time": pd.Timestamp(arguments[2])
                    }
                except Exception as e:
                    return f"参数解析失败: {str(e)}"
            else:
                return response
        else:
            # 处理非函数调用的响应
            job_messages.append({"role": "tool", "content": response})
            return response

        if function_name in function_map:
            # 调用相应的工具函数
            result = function_map[function_name](**arguments)

            result = "以下是调用函数得到的真实数据" + str(result)
            # 处理结果，例如返回给用户或进一步处理
            job_messages.append({"role": "tool", "content": result})

        new_response = LLM(model, job_messages)
        return new_response



equipment_agent_prompt = """
#角色
您是深远海船舶智能数据解析专家，精准匹配设备数据字段是您的核心能力。

#工具调用规则（强制要求）
当问题涉及以下场景时必须立即调用工具：

设备实时状态查询 → get_device_df

时间段数据统计 → get_device_statistic


#数据规范（严格校验）
字段精确匹配：

设备名称必须从以下选取：['A架','绞车','折臂吊车','发电机组','主推进器','艏侧推','伸缩推','舵桨','门架']

时间参数规范：

格式强制：YYYY-MM-DD HH:MM:SS

默认时间范围：2024-05-14 00:00:00 至 2024-09-14 23:59:59


#执行流程（顺序不可变更）
解析用户问题中的关键要素：设备名称、时间范围、数据字段

自动补全缺失时间参数（当用户未明确时）

根据问题类型选择工具并传参：
device_name=设备名称
start_time=起始时间
end_time=结束时间
field_names=字段编码列表

#错误拦截机制
× 设备名错误："柴油机组"→修正为"发电机组"
× 字段名错误："功率"→修正为"P1_66"
× 时间格式错误："5月14日"→转换为"2024-05-14 00:00:00"
× 多工具调用：一次仅能输出一段，使用一个工具，不可输出多个工具
#当前设备库
{equipment_info} 

#最终响应要求
当检测到设备名、字段名、时间参数任一要素时，必须调用工具

响应仅包含工具返回的原生数据，不作额外解释

确保参数顺序：[device_name,start_time,end_time,field_names]
# 用户问题
{question}
请给出直接且准确的答案,只输出一句话,不过多解释,不要思考过程
"""

equipment_info = {
    "设备": {
        "甲板机械": {
            "A架": {
                "功能": "起重设备，用于深海作业设备（征服者）的发射以及回收",
                "相关数据": [
                    {"字段含义":"A架右舷角度", "字段名":"Ajia-0_v"},
                    {"字段含义":"A架左舷角度", "字段名":"Ajia-1_v"},
                    {"字段含义":"1启动柜电压", "字段名":"Ajia-2_v"},
                    {"字段含义":"1启动柜电流", "字段名":"Ajia-3_v"},
                    {"字段含义":"2启动柜电压", "字段名":"Ajia-4_v"},
                    {"字段含义":"2启动柜电流", "字段名":"Ajia-5_v"}
                ]
            },
            "绞车": {
                "功能": "起重、牵引和拖动",
                "相关数据": [
                    {"字段含义":"绞车A放缆长度", "字段名":"PLC_point0_value"},
                    {"字段含义":"绞车A放缆速度", "字段名":"PLC_point1_value"},
                    {"字段含义":"绞车A张力", "字段名":"PLC_point2_value"},
                    {"字段含义":"绞车B放缆长度", "字段名":"PLC_point3_value"},
                    {"字段含义":"绞车B放缆速度", "字段名":"PLC_point4_value"},
                    {"字段含义":"绞车B张力", "字段名":"PLC_point5_value"},
                    {"字段含义":"绞车C放缆长度", "字段名":"PLC_point6_value"},
                    {"字段含义":"绞车C放缆速度", "字段名":"PLC_point7_value"},
                    {"字段含义":"绞车C张力", "字段名":"PLC_point8_value"},
                    {"字段含义":"绞车变频器-Ua电压", "字段名":"1-15-0_v"},
                    {"字段含义":"绞车变频器-Ub电压", "字段名":"1-15-1_v"},
                    {"字段含义":"绞车变频器-Uc电压", "字段名":"1-15-2_v"},
                    {"字段含义":"绞车变频器-Ia电流", "字段名":"1-15-3_v"},
                    {"字段含义":"绞车变频器-Ib电流", "字段名":"1-15-4_v"},
                    {"字段含义":"绞车变频器-Ic电流", "字段名":"1-15-5_v"},
                    {"字段含义":"绞车变频器-Pt有功功率", "字段名":"1-15-6_v"},
                    {"字段含义":"绞车变频器-Qt无功功率", "字段名":"1-15-7_v"},
                    {"字段含义":"绞车变频器-St视在功率", "字段名":"1-15-8_v"},
                    {"字段含义":"绞车变频器-总功率因素", "字段名":"1-15-9_v"},
                    {"字段含义":"绞车变频器-频率", "字段名":"1-15-10_v"},
                    {"字段含义":"绞车变频器-电压不平衡度", "字段名":"1-15-11_v"},
                    {"字段含义":"绞车变频器-电流不平衡度", "字段名":"1-15-12_v"},
                    {"字段含义":"绞车变频器-相序状态", "字段名":"1-15-13_v"}
                ]
            },
            "折臂吊车": {
                "功能": "用于小艇的下放和回收等，相对灵活",
                "相关数据": [
                    {"字段含义":"折臂吊车液压-Ua电压", "字段名":"13-11-0_v"},
                    {"字段含义":"折臂吊车液压-Ub电压", "字段名":"13-11-1_v"},
                    {"字段含义":"折臂吊车液压-Uc电压", "字段名":"13-11-2_v"},
                    {"字段含义":"折臂吊车液压-Ia电流", "字段名":"13-11-3_v"},
                    {"字段含义":"折臂吊车液压-Ib电流", "字段名":"13-11-4_v"},
                    {"字段含义":"折臂吊车液压-Ic电流", "字段名":"13-11-5_v"},
                    {"字段含义":"折臂吊车液压-Pt有功功率", "字段名":"13-11-6_v"},
                    {"字段含义":"折臂吊车液压-Qt无功功率", "字段名":"13-11-7_v"},
                    {"字段含义":"折臂吊车液压-St视在功率", "字段名":"13-11-8_v"},
                    {"字段含义":"折臂吊车液压-总功率因素", "字段名":"13-11-9_v"},
                    {"字段含义":"折臂吊车液压-频率", "字段名":"13-11-10_v"},
                    {"字段含义":"折臂吊车液压-电压不平衡度", "字段名":"13-11-11_v"},
                    {"字段含义":"折臂吊车液压-电流不平衡度", "字段名":"13-11-12_v"},
                    {"字段含义":"折臂吊车液压-相序状态", "字段名":"13-11-13_v"}
                ]
            },
            "门架": {
                "功能": "用于小艇的下放和回收等，相对灵活",
                "相关数据": [
                    {"字段含义":"一号门架主液压泵-Ua电压", "字段名":"1-5-0_v"},
                    {"字段含义":"一号门架主液压泵-Ub电压", "字段名":"1-5-1_v"},
                    {"字段含义":"一号门架主液压泵-Uc电压", "字段名":"1-5-2_v"},
                    {"字段含义":"一号门架主液压泵-Ia电流", "字段名":"1-5-3_v"},
                    {"字段含义":"一号门架主液压泵-Ib电流", "字段名":"1-5-4_v"},
                    {"字段含义":"一号门架主液压泵-Ic电流", "字段名":"1-5-5_v"},
                    {"字段含义":"一号门架主液压泵-Pt有功功率", "字段名":"1-5-6_v"},
                    {"字段含义":"一号门架主液压泵-Qt无功功率", "字段名":"1-5-7_v"},
                    {"字段含义":"一号门架主液压泵-St视在功率", "字段名":"1-5-8_v"},
                    {"字段含义":"一号门架主液压泵-总功率因素", "字段名":"1-5-9_v"},
                    {"字段含义":"一号门架主液压泵-频率", "字段名":"1-5-10_v"},
                    {"字段含义":"一号门架主液压泵-电压不平衡度", "字段名":"1-5-11_v"},
                    {"字段含义":"一号门架主液压泵-电流不平衡度", "字段名":"1-5-12_v"},
                    {"字段含义":"一号门架主液压泵-相序状态", "字段名":"1-5-13_v"},
                    {"字段含义":"二号门架主液压泵-Ua电压", "字段名":"13-14-0_v"},
                    {"字段含义":"二号门架主液压泵-Ub电压", "字段名":"13-14-1_v"},
                    {"字段含义":"二号门架主液压泵-Uc电压", "字段名":"13-14-2_v"},
                    {"字段含义":"二号门架主液压泵-Ia电流", "字段名":"13-14-3_v"},
                    {"字段含义":"二号门架主液压泵-Ib电流", "字段名":"13-14-4_v"},
                    {"字段含义":"二号门架主液压泵-Ic电流", "字段名":"13-14-5_v"},
                    {"字段含义":"二号门架主液压泵-Pt有功功率", "字段名":"13-14-6_v"},
                    {"字段含义":"二号门架主液压泵-Qt无功功率", "字段名":"13-14-7_v"},
                    {"字段含义":"二号门架主液压泵-St视在功率", "字段名":"13-14-8_v"},
                    {"字段含义":"二号门架主液压泵-总功率因素", "字段名":"13-14-9_v"},
                    {"字段含义":"二号门架主液压泵-频率", "字段名":"13-14-10_v"},
                    {"字段含义":"二号门架主液压泵-电压不平衡度", "字段名":"13-14-11_v"},
                    {"字段含义":"二号门架主液压泵-电流不平衡度", "字段名":"13-14-12_v"},
                    {"字段含义":"二号门架主液压泵-相序状态", "字段名":"13-14-13_v"}
                ]
            }
        },
        "推进系统": {
            "发电机组": {
                "组成": "四台柴油发电机组以及一台应急柴油发电机组",
                "功能": "全船的电力来源",
                "相关数据": [
                    {"字段含义":"一号柴油发电机组有功功率测量", "字段名":"P1_66"},
                    {"字段含义":"一号柴油发电机组转速", "字段名":"P1_2"},
                    {"字段含义":"一号柴油发电机组燃油消耗率", "字段名":"P1_3"},
                    {"字段含义":"二号柴油发电机组有功功率测量", "字段名":"P1_75"},
                    {"字段含义":"二号柴油发电机组转速", "字段名":"P1_24"},
                    {"字段含义":"二号柴油发电机组燃油消耗率", "字段名":"P1_25"},
                    {"字段含义":"三号柴油发电机组有功功率测量", "字段名":"P2_51"},
                    {"字段含义":"三号柴油发电机组转速", "字段名":"P2_2"},
                    {"字段含义":"三号柴油发电机组燃油消耗率", "字段名":"P2_3"},
                    {"字段含义":"四号柴油发电机组有功功率测量", "字段名":"P2_60"},
                    {"字段含义":"四号柴油发电机组转速", "字段名":"P2_24"},
                    {"字段含义":"四号柴油发电机组燃油消耗率", "字段名":"P2_25"},
                    {"字段含义":"停泊/应急发电机组转速", "字段名":"P1_46"},
                    {"字段含义":"停泊/应急发电机组燃油消耗率", "字段名":"P1_47"}
                ]
            },
            "推进器": {
                "主推进器": {
                    "功能": "负责船舶的主要移动",
                    "数量": 2,
                    "相关数据": [
                        {"字段含义":"一号推进变频器功率允许", "字段名":"P3_32"},
                        {"字段含义":"一号推进变频器功率反馈", "字段名":"P3_15"},
                        {"字段含义":"二号推进变频器功率允许", "字段名":"P4_15"},
                        {"字段含义":"二号推进变频器功率反馈", "字段名":"P4_16"}
                    ]
                },
                "艏侧推": {
                    "功能": "精确保持船头的位置，用于停靠码头或者动力定位(DP)",
                    "相关数据": [
                        {"字段含义":"艏推主开关电流", "字段名":"P1_80"},
                        {"字段含义":"艏侧推螺旋桨螺距命令", "字段名":"P3_16"},
                        {"字段含义":"艏侧推螺旋桨螺距反馈", "字段名":"P3_17"},
                        {"字段含义":"艏推功率反馈", "字段名":"P3_18"},
                        {"字段含义":"艏推启动功率允许", "字段名":"P3_33"}
                    ]
                },
                "可伸缩推": {
                    "功能": "360°旋转，提供灵活操作性",
                    "相关数据": [
                        {"字段含义":"伸缩推主开关电流", "字段名":"P4_22"},
                        {"字段含义":"可伸缩推螺距命令", "字段名":"P4_17"},
                        {"字段含义":"可伸缩推螺距反馈", "字段名":"P4_18"},
                        {"字段含义":"可伸缩推方位命令", "字段名":"P4_19"},
                        {"字段含义":"可伸缩推方位反馈", "字段名":"P4_20"},
                        {"字段含义":"可伸缩推功率反馈", "字段名":"P4_21"},
                        {"字段含义":"可伸缩推功率允许", "字段名":"P4_22"}
                    ]
                }
            },
            "舵桨": {
                "功能": "改变航向",
                "数量": 2,
                "每个舵桨舵机数量": 2,
                "相关数据": [
                    {"字段含义":"左舵桨主开关电流", "字段名":"P1_81"},
                    {"字段含义":"右舵桨主开关电流", "字段名":"P2_64"},
                    {"字段含义":"一号舵桨转速命令", "字段名":"P3_21"},
                    {"字段含义":"一号舵桨转速反馈", "字段名":"P3_22"},
                    {"字段含义":"一号舵桨方位命令", "字段名":"P3_19"},
                    {"字段含义":"一号舵桨方位反馈", "字段名":"P3_20"},
                    {"字段含义":"一号舵桨转舵A-Ua电压", "字段名":"1-2-0_v"},
                    {"字段含义":"一号舵桨转舵A-Ub电压", "字段名":"1-2-1_v"},
                    {"字段含义":"一号舵桨转舵A-Uc电压", "字段名":"1-2-2_v"},
                    {"字段含义":"一号舵桨转舵A-Ia电流", "字段名":"1-2-3_v"},
                    {"字段含义":"一号舵桨转舵A-Ib电流", "字段名":"1-2-4_v"},
                    {"字段含义":"一号舵桨转舵A-Ic电流", "字段名":"1-2-5_v"},
                    {"字段含义":"一号舵桨转舵A-Pt有功功率", "字段名":"1-2-6_v"},
                    {"字段含义":"一号舵桨转舵A-Qt无功功率", "字段名":"1-2-7_v"},
                    {"字段含义":"一号舵桨转舵A-St视在功率", "字段名":"1-2-8_v"},
                    {"字段含义":"一号舵桨转舵A-总功率因素", "字段名":"1-2-9_v"},
                    {"字段含义":"一号舵桨转舵A-频率", "字段名":"1-2-10_v"},
                    {"字段含义":"一号舵桨转舵A-电压不平衡度", "字段名":"1-2-11_v"},
                    {"字段含义":"一号舵桨转舵A-电流不平衡度", "字段名":"1-2-12_v"},
                    {"字段含义":"一号舵桨转舵A-相序状态", "字段名":"1-2-13_v"},
                    {"字段含义":"一号舵桨转舵B-Ua电压", "字段名":"1-3-0_v"},
                    {"字段含义":"一号舵桨转舵B-Ub电压", "字段名":"1-3-1_v"},
                    {"字段含义":"一号舵桨转舵B-Uc电压", "字段名":"1-3-2_v"},
                    {"字段含义":"一号舵桨转舵B-Ia电流", "字段名":"1-3-3_v"},
                    {"字段含义":"一号舵桨转舵B-Ib电流", "字段名":"1-3-4_v"},
                    {"字段含义":"一号舵桨转舵B-Ic电流", "字段名":"1-3-5_v"},
                    {"字段含义":"一号舵桨转舵B-Pt有功功率", "字段名":"1-3-6_v"},
                    {"字段含义":"一号舵桨转舵B-Qt无功功率", "字段名":"1-3-7_v"},
                    {"字段含义":"一号舵桨转舵B-St视在功率", "字段名":"1-3-8_v"},
                    {"字段含义":"一号舵桨转舵B-总功率因素", "字段名":"1-3-9_v"},
                    {"字段含义":"一号舵桨转舵B-频率", "字段名":"1-3-10_v"},
                    {"字段含义":"一号舵桨转舵B-电压不平衡度", "字段名":"1-3-11_v"},
                    {"字段含义":"一号舵桨转舵B-电流不平衡度", "字段名":"1-3-12_v"},
                    {"字段含义":"一号舵桨转舵B-相序状态", "字段名":"1-3-13_v"},
                    {"字段含义":"二号舵桨转速命令", "字段名":"P4_25"},
                    {"字段含义":"二号舵桨转速反馈", "字段名":"P4_26"},
                    {"字段含义":"二号舵桨方位命令", "字段名":"P4_23"},
                    {"字段含义":"二号舵桨方位反馈", "字段名":"P4_24"},
                    {"字段含义":"二号舵桨转舵A-Ua电压", "字段名":"13-2-0_v"},
                    {"字段含义":"二号舵桨转舵A-Ub电压", "字段名":"13-2-1_v"},
                    {"字段含义":"二号舵桨转舵A-Uc电压", "字段名":"13-2-2_v"},
                    {"字段含义":"二号舵桨转舵A-Ia电流", "字段名":"13-2-3_v"},
                    {"字段含义":"二号舵桨转舵A-Ib电流", "字段名":"13-2-4_v"},
                    {"字段含义":"二号舵桨转舵A-Ic电流", "字段名":"13-2-5_v"},
                    {"字段含义":"二号舵桨转舵A-Pt有功功率", "字段名":"13-2-6_v"},
                    {"字段含义":"二号舵桨转舵A-Qt无功功率", "字段名":"13-2-7_v"},
                    {"字段含义":"二号舵桨转舵A-St视在功率", "字段名":"13-2-8_v"},
                    {"字段含义":"二号舵桨转舵A-总功率因素", "字段名":"13-2-9_v"},
                    {"字段含义":"二号舵桨转舵A-频率", "字段名":"13-2-10_v"},
                    {"字段含义":"二号舵桨转舵A-电压不平衡度", "字段名":"13-2-11_v"},
                    {"字段含义":"二号舵桨转舵A-电流不平衡度", "字段名":"13-2-12_v"},
                    {"字段含义":"二号舵桨转舵A-相序状态", "字段名":"13-2-13_v"},
                    {"字段含义":"二号舵桨转舵B-Ua电压", "字段名":"13-3-0_v"},
                    {"字段含义":"二号舵桨转舵B-Ub电压", "字段名":"13-3-1_v"},
                    {"字段含义":"二号舵桨转舵B-Uc电压", "字段名":"13-3-2_v"},
                    {"字段含义":"二号舵桨转舵B-Ia电流", "字段名":"13-3-3_v"},
                    {"字段含义":"二号舵桨转舵B-Ib电流", "字段名":"13-3-4_v"},
                    {"字段含义":"二号舵桨转舵B-Ic电流", "字段名":"13-3-5_v"},
                    {"字段含义":"二号舵桨转舵B-Pt有功功率", "字段名":"13-3-6_v"},
                    {"字段含义":"二号舵桨转舵B-Qt无功功率", "字段名":"13-3-7_v"},
                    {"字段含义":"二号舵桨转舵B-St视在功率", "字段名":"13-3-8_v"},
                    {"字段含义":"二号舵桨转舵B-总功率因素", "字段名":"13-3-9_v"},
                    {"字段含义":"二号舵桨转舵B-频率", "字段名":"13-3-10_v"},
                    {"字段含义":"二号舵桨转舵B-电压不平衡度", "字段名":"13-3-11_v"},
                    {"字段含义":"二号舵桨转舵B-电流不平衡度", "字段名":"13-3-12_v"},
                    {"字段含义":"二号舵桨转舵B-相序状态", "字段名":"13-3-13_v"}
                ]
            }
        }
    }
}

equipment_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_device_df",
            "description": "给出时间段(start_time,end_time)和设备名称（device_name）及设备字段（field_names），获取开始时间到结束时间之间设备的具体数据",
            "parameters": {
                "type": "object",
                "properties": {

                    "device_name": {
                        "type": "str",
                        "description": "设备名称，支持以下值：'A架'、'绞车'、'折臂吊车'、'发电机组'、'主推进器'、'艏侧推'、'伸缩推'、'舵桨'、'门架'",
                        "enum": ['A架','绞车','折臂吊车','发电机组','主推进器','艏侧推','伸缩推','舵桨','门架'],
                    },
                    "start_time": {
                        "type": "pd.Timestamp",
                        "description": "开始时间",
                    },
                    "end_time": {
                        "type": "pd.Timestamp",
                        "description": "结束时间",
                    },
                    "field_names": {
                        "type": "array",
                        "items": {
                            "type": "str",
                            "description": "字段名称",  
                        },
                        "description": "含有系列字段名称的数组",
                    },
                },
                "required": ["device_name","start_time", "end_time","field_names"]
            }

        }
    },
        {
        "type": "function",
        "function": {
            "name": "get_device_statistic",
            "description": "给出时间段(start_time,end_time)和设备名称(device_name)及设备字段(field_names)，获取开始时间到结束时间之间设备的统计学数据(最大值，最小值和求和值)",
            "parameters": {
                "type": "object",

                "properties": {
                    "device_name": {
                        "type": "str",
                        "description": "设备名称，支持以下值：'A架'、'绞车'、'折臂吊车'、'发电机组'、'主推进器'、'艏侧推'、'伸缩推'、'舵桨'、'门架'",
                        "enum": ['A架','绞车','折臂吊车','发电机组','主推进器','艏侧推','伸缩推','舵桨','门架'],
                    },
                    "start_time": {
                        "type": "pd.Timestamp",
                        "description": "开始时间",
                    },
                    "end_time": {
                        "type": "pd.Timestamp",
                        "description": "结束时间",
                    },
                    "field_names": {
                        "type": "array",
                        "items": {
                            "type": "str",
                            "description": "字段名称",  
                        },
                        "description": "含有系列字段名称的数组",
                    },
                },
                "required": ["device_name","start_time", "end_time","field_names"]
            }

        }
    },
]



class EquipmentAgent():
    def execute_task(self, question):
        function_map = {
            "get_device_df": get_device_df,
            "get_device_statistic": get_device_statistic,    
        }
        functioncall_description_map = {
            "get_device_df": "获取指定设备指定字段开始时间到结束时间之间的值，返回带有指定字段的DataFrame",
            "get_device_statistic": "获取指定设备指定字段开始时间到结束时间之间的统计学指标，返回每个字段的最大值、最小值和sum值",
        }
        model = "glm-4-plus"

        prompt = equipment_agent_prompt.format(question=question, equipment_info=equipment_info)
        equipment_messages = [
            {"role": "user", "content": prompt}
        ]
        # 查询设备相关信息
        # equipment_messages.append({"role": "system", "content": "不要假设工具和模拟，要调用真实提供的tools列表，我只有三个tools，分别是get_device_df和get_device_statistic和get_device_energy"})
        # equipment_messages.append({"role": "system", "content": "对照工具函数字段名称传入的正确字段名称和数量，不要使用错误字段名称，如device_name而不是device,field_names而不是field,严格按照[device_name,start_time,end_time,field_names]格式传入"})
        # print(prompt)
        # 如何规范使其必然调用工具？

        response = LLM(model, equipment_messages, equipment_tools)
        # 判断response的类型或结构
        if isinstance(response, tuple) and len(response) == 2:
            function_name, arguments = response
        elif isinstance(response, str) and '\n' in response:
            # 解析字符串形式的响应
            function_name, json_str = response.split('\n', 1)
            try:
                arguments = json.loads(json_str)
            except json.JSONDecodeError:
                return response
        else:
            return response
        
        if function_name in function_map:
            # 调用相应的工具函数
            try:
                result = function_map[function_name](**arguments)
            except Exception as e:
                return f"函数调用失败: {str(e)}"
            functioncall_description = functioncall_description_map[function_name]
            field_names = arguments.get('field_names', [])
            if field_names:
                data_field_description = str(get_field_names_description(field_names))
                equipment_messages.append({"role": "system", "content": data_field_description})

            functioncall_description = f"以下是调用{function_name}函数以及得到的真实数据。函数描述{functioncall_description}。" + str(result)
            equipment_messages.append({"role": "tool", "content": functioncall_description})

        new_response = LLM(model, equipment_messages)
        return new_response
    
power_agent_prompt = """
#角色
你是深海作业能耗专家，具备精准计算和分析船舶各系统设备能耗的能力。

#工作任务
接收用户的能耗查询需求，通过调用合适的工具函数获取准确的能耗数据。

#注意事项
1. 严格遵循工具函数的调用规范
2. 确保时间参数格式正确(YYYY-MM-DD HH:MM:SS)
3. 一次仅能调用一个工具函数

#可用工具函数
1. get_device_energy: 计算单个设备能耗
   支持设备：["折臂吊车","一号门架","二号门架","绞车","发电机组","主推进器","艏侧推","舵桨"]

2. get_total_deck_machinery_energy: 计算甲板机械总能耗
   包含设备：折臂吊车、一号门架、二号门架、绞车

3. get_total_tuijinxitong_energy: 计算推进系统总能耗
   包含设备：发电机组、主推进器、艏侧推、舵桨

#用户问题
{question}

请给出直接且准确的答案，只输出一句话，不过多解释，不要思考过程。
"""

power_tools = [
        {
        "type": "function",
        "function": {
            "name": "get_device_energy",
            "description": "计算指定时间段内指定设备的总能耗。返回值为总能耗（kWh，float 类型）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "pd.Timestamp",
                        "description": "开始时间",
                    },
                    "end_time": {
                        "type": "pd.Timestamp",
                        "description": "结束时间",
                    },
                    "device": {
                        "type": "string",
                        "description": "设备名称，支持以下值其中之一：'折臂吊车'、'一号门架'、'二号门架'、'绞车'、'发电机组'、'主推进器'、'艏侧推'、'舵桨'",
                        "enum": ["折臂吊车", "一号门架", "二号门架", "绞车", "发电机组", "主推进器", "艏侧推", "舵桨"],
                    },
                },
                "required": ["start_time", "end_time", "device"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_total_deck_machinery_energy",
            "description": "计算甲板机械设备在指定时间范围内的总能耗。返回值为总能耗（kWh，float 类型）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "pd.Timestamp",
                        "description": "开始时间",
                    },
                    "end_time": {
                        "type": "pd.Timestamp",
                        "description": "结束时间",
                    },
                },
                "required": ["start_time", "end_time"],
            },
        },
    },  
    {
        "type": "function",
        "function": {
            "name": "get_total_tuijinxitong_energy",
            "description": "计算推进系统在指定时间范围内的总能耗。返回值为总能耗（kWh，float 类型）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "pd.Timestamp",
                        "description": "开始时间",
                    },
                    "end_time": {
                        "type": "pd.Timestamp",
                        "description": "结束时间",
                    },
                },
                "required": ["start_time", "end_time"],
            },
        },
    },
]


class PowerAgent():

    def execute_task(self, question):
        function_map = {
            "get_device_energy": get_device_energy,
            "get_total_deck_machinery_energy": get_total_deck_machinery_energy,
            "get_total_tuijinxitong_energy":get_total_tuijinxitong_energy,
        }

        model = "glm-4-plus"
        prompt = power_agent_prompt.format(question=question)
        
        power_messages = [
            {"role": "user", "content": prompt}
        ]

        # 查询作业相关信息
        response = LLM(model, power_messages, job_tools)
        
        # 判断response的类型或结构
        if isinstance(response, tuple) and len(response) == 2:
            function_name, arguments = response

        elif isinstance(response, str) and '\n' in response:
            # 解析字符串形式的响应
            function_name, json_str = response.split('\n', 1)
            try:
                arguments = json.loads(json_str)
            except json.JSONDecodeError:
                return response
        else:
            return response

        if function_name in function_map:
            # 调用相应的工具函数
            result = function_map[function_name](**arguments)

            result = "以下是调用函数得到的真实数据" + str(result)
            # 处理结果，例如返回给用户或进一步处理
            power_messages.append({"role": "tool", "content": result})

        new_response = LLM(model, power_messages)
        return new_response


Maintenance_agent_prompt = """
# 角色
您是船舶作业异常专家，具备根据所给设备发现异常并给出解决方案的能力。

下面是字段含义：
字段	含义	备注
Channel_Text	参数名	
Channel_Text_CN	参数中文名	
Alarm_Information_Range_Low	参数下限	
Alarm_Information_Range_High	参数上限	
Alarm_Information_Unit	报警值的单位（压力、转速、温度等）	
Alarm_Information_Kind_Of_Singal	报警信号类型（开关量、模拟量等）	
Parameter_Information_Alarm	报警值	这个值是指当监测参数达到或超过这个设定值时，系统会发出报警信号。报警值通常用于指示设备运行状态是否在安全范围内。例如，温度、压力等参数的上限或下限报警值。
Parameter_Information_Inhibit	屏蔽值	屏蔽值用于在特定条件下抑制报警信号的发出。例如，在船舶停港期间，某些与航行无关的报警可以被屏蔽，以避免不必要的报警干扰。
Parameter_Information_Delayed	延迟值	延迟值用于设置报警信号的延迟时间，以避免因短暂的参数波动而产生误报警。延迟时间可以根据监测参数的不同而设定为长延时或短延时。
Safety_Protection_Set_Value	安全保护设定值	安全保护设定值是指当监测参数达到这个值时，系统将采取安全保护措施，如自动停机，以防止设备损坏或事故发生。
Remarks	附注	备注设备超过安全保护设定值之后动作

以下是字段名：
一号柴油发电机组负载
一号柴油发电机组转速
一号柴油发电机组燃油消耗率
一号柴油发电机组蓄电池电压
一号柴油发电机组缸套水温度
一号柴油发电机组缸套水压力
一号柴油发电机组缸套水液位
一号柴油发电机组左排气温度
一号柴油发电机组右排气温度
一号柴油发电机组滑油压力
一号柴油发电机组滑油温度
一号柴油发电机组启动空气压力
一号柴油发电机组燃油压力
一号柴油发电机组应急停止
一号柴油发电机组安全系统故障
一号柴油发电机组控制系统故障
一号柴油发电机组发动机控制模块通用报警
一号柴油发电机组启动故障
一号柴油发电机组冷却液温度
一号柴油发电机组转速传感器故障
一号柴油发电机组缸套水温度传感器故障
一号柴油发电机组滑油压力传感器故障
一号柴油发电机组应急停止传感器故障
一号柴油发电机组超速停车
一号柴油发电机组缸套水高温停车
一号柴油发电机组滑油压力低停车
一号柴油发电机组海水压力低
一号柴油发电机组滑油滤器压差
一号柴油发电机组膨胀柜液位低
一号柴油发电机组MAC报警板24V电源故障
一号柴油发电机组冷风温度
一号柴油发电机组热风温度
一号柴油发电机组非驱动轴轴承温度
一号柴油发电机组驱动轴轴承温度
一号柴油发电机组U相绕组温度显示
一号柴油发电机组V相绕组温度显示
一号柴油发电机组W相绕组温度显示
一号柴油发电机组冷却水泄漏报警
一号柴油发电机报警系统失效
一号柴油发电机组膨胀水柜低位报警
二号柴油发电机组负载
二号柴油发电机组转速
二号柴油发电机组燃油消耗率
二号柴油发电机组蓄电池电压
二号柴油发电机组缸套水温度
二号柴油发电机组缸套水压力
二号柴油发电机组缸套水液位
二号柴油发电机组左排气温度
二号柴油发电机组右排气温度
二号柴油发电机组滑油压力
二号柴油发电机组滑油温度
二号柴油发电机组启动空气压力
二号柴油发电机组燃油压力
二号柴油发电机组应急停止
二号柴油发电机组安全系统故障
二号柴油发电机组控制系统故障
二号柴油发电机组发动机控制模块通用报警
二号柴油发电机组启动故障
二号柴油发电机组冷却液温度
二号柴油发电机组转速传感器故障
二号柴油发电机组缸套水温度传感器故障
二号柴油发电机组滑油压力传感器故障
二号柴油发电机组应急停止传感器故障
二号柴油发电机组超速停车
二号柴油发电机组缸套水高温停车
二号柴油发电机组滑油压力低停车
二号柴油发电机组海水压力低
二号柴油发电机组滑油滤器压差
二号柴油发电机组膨胀柜液位低
二号柴油发电机组MAC报警板24V电源故障
二号柴油发电机组冷风温度
二号柴油发电机组热风温度
二号柴油发电机组非驱动轴轴承温度
二号柴油发电机组驱动轴轴承温度
二号柴油发电机组U相绕组温度显示
二号柴油发电机组V相绕组温度显示
二号柴油发电机组W相绕组温度显示
二号柴油发电机组冷却水泄漏报警
二号柴油发电机报警系统失效
二号柴油发电机组膨胀水柜低位报警
停泊/应急发电机组负载
停泊/应急发电机组转速
停泊/应急发电机组燃油消耗率
停泊/应急发电机组蓄电池电压
停泊/应急发电机组缸套水温度
停泊/应急发电机组缸套水压力
停泊/应急发电机组缸套水液位
停泊/应急发电机组滑油压力
停泊/应急发电机组滑油温度
停泊/应急发电机组燃油压力
停泊/应急发电机组应急停止
停泊/应急发电机组启动故障
停泊/应急发电机组就地/遥控指示
停泊/应急发电机组运行
停泊/应急发电机组冷却液液位
停泊/应急发电机组报警系统失效
一号柴油发电机组U-V相电压测量
一号柴油发电机组V-W相电压测量
一号柴油发电机组W-U相电压测量
一号柴油发电机组U相电流测量
一号柴油发电机组V相电流测量
一号柴油发电机组W相电流测量
一号柴油发电机组有功功率测量
一号柴油发电机组频率测量
一号柴油发电机组励磁电流
一号柴油发电机组主开关闭合反馈
一号柴油发电机组主开关开启反馈
一号柴油发电机组主开关合闸命令
一号柴油发电机组主开关分闸命令
一号柴油发电机组主开关脱扣报警
一号柴油发电机组主开关远程控制反馈
一号柴油发电机组远程控制
一号柴油发电机组就绪
一号柴油发电机组遥控启动
一号柴油发电机组遥控停止
一号柴油发电机组升速
一号柴油发电机组减速
一号柴油发电机组卸载
一号柴油发电机组功率分配模式转换
一号柴油发电机组额定转速运行
一号柴油发电机组公共停车
一号柴油发电机组就地/遥控指示
一号柴油发电机组备车完毕
一号柴油发电机组起动失败
一号柴油发电机组负载分配线路故障
一号柴油发电机组负荷分配器公共报警
一号柴油发电机组AVR功率监控
二号柴油发电机组U-V相电压测量
二号柴油发电机组V-W相电压测量
二号柴油发电机组W-U相电压测量
二号柴油发电机组U相电流测量
二号柴油发电机组V相电流测量
二号柴油发电机组W相电流测量
二号柴油发电机组有功功率测量
二号柴油发电机组频率测量
二号柴油发电机组励磁电流
二号柴油发电机组主开关闭合反馈
二号柴油发电机组主开关开启反馈
二号柴油发电机组主开关合闸命令
二号柴油发电机组主开关分闸命令
二号柴油发电机组主开关脱扣报警
二号柴油发电机组主开关远程控制反馈
二号柴油发电机组远程控制
二号柴油发电机组就绪
二号柴油发电机组遥控启动
二号柴油发电机组遥控停止
二号柴油发电机组升速
二号柴油发电机组减速
二号柴油发电机组卸载
二号柴油发电机组功率分配模式转换
二号柴油发电机组额定转速运行
二号柴油发电机组公共停车
二号柴油发电机组就地/遥控指示
二号柴油发电机组备车完毕
二号柴油发电机组起动失败
二号柴油发电机组负载分配线路故障
二号柴油发电机组负荷分配器公共报警
二号柴油发电机组AVR功率监控
停泊/应急发电机开关合闸命令
停泊/应急发电机开关分闸命令
停泊/应急发电机遥控起动
停泊/应急发电机遥控停止
停泊/应急发电机减速
停泊/应急发电机升速
停泊/应急发电机电压测量
停泊/应急发电机电流测量
停泊/应急发电机开关合闸反馈
停泊/应急发电机开关分闸反馈
停泊/应急发电机开关远程控制模式反馈
停泊/应急发电机开关脱扣报警
停泊/应急模式指示
停泊/应急发电机公共停车
停泊/应急发电机遥控备车完毕
停泊/应急发电机运行指示
停泊/应急发电机应急切断电源故障
停泊/应急发电机应急切断断线报警
一号外消防功率请求
艏推主开关合闸命令
艏推主开关分闸命令
艏推主开关闭合反馈
艏推主开关开启反馈
艏推主开关电流
艏推主开关远程控制模式反馈
艏推主开关脱扣报警
左舵桨开关脱扣
左舵桨开关闭合反馈
左舵桨开关开启反馈
左舵桨应急切断电源故障
左舵桨应急切断断线报警
左舵桨主开关电流
MCC左屏主开关合闸命令
MCC左屏主开关分闸命令
MCC左屏主开关脱扣报警
MCC左屏主开关闭合反馈
MCC左屏主开关开启反馈
MCC左屏主开关远程控制模式反馈
MCC左屏主开关电流
ROV1主开关合闸命令
ROV1主开关分闸命令
ROV1主开关脱扣报警
ROV1主开关闭合反馈
ROV1主开关开启反馈
ROV1主开关电流
ROV1主开关远程控制模式反馈
一号外消防主开关合闸命令
一号外消防主开关分闸命令
一号外消防主开关脱扣报警
一号外消防主开关闭合反馈
一号外消防主开关开启反馈
一号外消防主开关远程控制模式反馈
一号外消防主开关电流
690V主配电板联络开关合闸命令
690V主配电板联络开关分闸命令
690V主配电板联络开关合闸反馈
690V主配电板联络开关分闸反馈
690V主配电板联络开关远程控制反馈
690V主配电板联络开关脱扣报警
690V主配电板联络开关应急切断电源故障
690V主配电板联络开关应急切断断线报警
690V主配电板联络开关24V控制电压故障
690V主配电板A排电压测量
690V主配电板A排频率测量
690V主配电板A排绝缘故障
690V主配电板A排电源故障
690V主配电板联络开关A汇流排一级设备优先脱扣
690V主配电板联络开关A汇流排二级设备优先脱扣
一号690V/230V主变压器原边开关合闸反馈
一号690V/230V主变压器原边开关分闸反馈
主应配左舷联络开关（应配侧）合闸反馈
主应配左舷联络开关（应配侧）分闸反馈
主应配左舷联络开关（主配侧）合闸反馈
主应配左舷联络开关（主配侧）分闸反馈
主应配左舷联络开关（主配侧）脱扣报警
一号690V/230V主变压器副边开关合闸命令
一号690V/230V主变压器副边开关分闸命令
一号690V/230V主变压器副边开关合闸反馈
一号690V/230V主变压器副边开关分闸反馈
一号690V/230V主变压器副边开关远程遥控反馈
一号690V/230V主变压器副边开关脱扣报警
230V主配电板联络开关合闸命令
230V主配电板联络开关分闸命令
230V主配电板联络开关合闸反馈
230V主配电板联络开关分闸反馈
230V主配电板联络开关远程遥控反馈
230V主配电板联络开关脱扣报警
230V主配电板联络开关应急切断电源故障
230V主配电板联络开关应急切断断线故障
230V主配电板A排绝缘故障
230V主配电板B排绝缘故障
一号应急变压器原边开关脱扣报警
一号应急变压器原边开关合闸反馈
一号应急变压器原边开关分闸反馈
一号应急变压器副边开关合闸反馈
一号应急变压器副边开关分闸反馈
690V应急配电板电压测量
690V应急配电板绝缘故障
230V应急配电板汇流排绝缘故障
三号柴油发电机组负载
三号柴油发电机组转速
三号柴油发电机组燃油消耗率
三号柴油发电机组蓄电池电压
三号柴油发电机组缸套水温度
三号柴油发电机组缸套水压力
三号柴油发电机组缸套水液位
三号柴油发电机组左排气温度
三号柴油发电机组右排气温度
三号柴油发电机组滑油压力
三号柴油发电机组滑油温度
三号柴油发电机组启动空气压力
三号柴油发电机组燃油压力
三号柴油发电机组应急停止
三号柴油发电机组安全系统故障
三号柴油发电机组控制系统故障
三号柴油发电机组发动机控制模块通用报警
三号柴油发电机组启动故障
三号柴油发电机组冷却液温度
三号柴油发电机组转速传感器故障
三号柴油发电机组缸套水温度传感器故障
三号柴油发电机组滑油压力传感器故障
三号柴油发电机组应急停止传感器故障
三号柴油发电机组超速停车
三号柴油发电机组缸套水高温停车
三号柴油发电机组滑油压力低停车
三号柴油发电机组海水压力低
三号柴油发电机组滑油滤器压差
三号柴油发电机组膨胀柜液位低
三号柴油发电机组MAC报警板24V电源故障
三号柴油发电机组冷风温度
三号柴油发电机组热风温度
三号柴油发电机组非驱动轴轴承温度
三号柴油发电机组驱动轴轴承温度
三号柴油发电机组U相绕组温度显示
三号柴油发电机组V相绕组温度显示
三号柴油发电机组W相绕组温度显示
三号柴油发电机组冷却水泄漏报警
三号柴油发电机报警系统失效
三号柴油发电机组膨胀水柜低位报警
四号柴油发电机组负载
四号柴油发电机组转速
四号柴油发电机组燃油消耗率
四号柴油发电机组蓄电池电压
四号柴油发电机组缸套水温度
四号柴油发电机组缸套水压力
四号柴油发电机组缸套水液位
四号柴油发电机组左排气温度
四号柴油发电机组右排气温度
四号柴油发电机组滑油压力
四号柴油发电机组滑油温度
四号柴油发电机组启动空气压力
四号柴油发电机组燃油压力
四号柴油发电机组应急停止
四号柴油发电机组安全系统故障
四号柴油发电机组控制系统故障
四号柴油发电机组发动机控制模块通用报警
四号柴油发电机组启动故障
四号柴油发电机组冷却液温度
四号柴油发电机组转速传感器故障
四号柴油发电机组缸套水温度传感器故障
四号柴油发电机组滑油压力传感器故障
四号柴油发电机组应急停止传感器故障
四号柴油发电机组超速停车
四号柴油发电机组缸套水高温停车
四号柴油发电机组滑油压力低停车
四号柴油发电机组海水压力低
四号柴油发电机组滑油滤器压差
四号柴油发电机组膨胀柜液位低
四号柴油发电机组MAC报警板24V电源故障
四号柴油发电机组冷风温度
四号柴油发电机组热风温度
四号柴油发电机组非驱动轴轴承温度
四号柴油发电机组驱动轴轴承温度
四号柴油发电机组U相绕组温度显示
四号柴油发电机组V相绕组温度显示
四号柴油发电机组W相绕组温度显示
四号柴油发电机组冷却水泄漏报警
四号柴油发电机报警系统失效
四号柴油发电机组膨胀水柜低位报警

用户问题：{question}
请用一句话直接回答，不要推理过程，严格按规范调用工具：

"""

maintenance_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_maintenance_data",
            "description": "获取指定字段的维护数据",
            "parameters": {
                "type": "object",
                "properties": {
                    "field_name": {
                        "type": "str",
                        "description": "字段名称"
                    }
                },
                "required": ["field_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_data_sum",
            "description": "获取指定数据表在指定日期内的数据条数,csv_name只能为以下字段名称:Ajia_plc, Jiaoche_plc, Port1_ksbg_1",
            "parameters": {
                "type": "object",
                "properties": {
                    "day_time": {
                        "type": "str",
                        "description": "日期"
                    },
                    "csv_name": {
                        "type": "str",
                        "description": "字段名称",
                        "enum": ["Ajia_plc", "Jiaoche_plc", "Port1_ksbg_1"]
                    }
                },
                "required": ["day_time", "csv_name"]
            }
        }
    }
]


class MaintenanceAgent():

    def execute_task(self, question):
        function_map = {
            "get_maintenance_data": get_maintenance_data,
            "get_data_sum": get_data_sum, 
        }

        model = "glm-4-plus"
        prompt = Maintenance_agent_prompt.format(question=question)
        
        maintenance_messages = [
            {"role": "user", "content": prompt}
        ]

        # 查询作业相关信息
        response = LLM(model, maintenance_messages, maintenance_tools)
        
        # 判断response的类型或结构
        if isinstance(response, tuple) and len(response) == 2:
            function_name, arguments = response
        else:
            # 处理非函数调用的响应
            maintenance_messages.append({"role": "tool", "content": response})
            return response

        if function_name in function_map:
            # 调用相应的工具函数
            result = function_map[function_name](**arguments)

            result = "以下是调用函数得到的真实信息" + str(result)
            # 处理结果，例如返回给用户或进一步处理
            maintenance_messages.append({"role": "tool", "content": result})

        new_response = LLM(model, maintenance_messages)
        return new_response

class AnalystAgent():
    def execute_task(self,question):
        model = "glm-4-plus"
 
        prompt = "你是深远海船舶专家，回答用户问题:{}.请给出直接且准确的答案,只输出一句话,不过多解释,不要思考过程".format(question)
        # 准备消息
        messages = [{"role": "user", "content": prompt}]
        
        # 生成响应
        response = LLM(model, messages)
        return response
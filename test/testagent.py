from agents import JobAgent,EquipmentAgent,PlanningAgent,PowerAgent
from GLM import InformationSharing
# job_agent = JobAgent()

# question = "2024/8/19 下午A架的第一次开机时间（请以XX:XX输出）？"

# response = job_agent.query_job(question)


# question = "一号柴油发电机组有功功率测量的范围是多少kW到多少kW？"

# question = "假设柴油的密度为0.8448kg/L，柴油热值为42.6MJ/kg，请计算2024/8/23 0:00 ~ 2024/8/25 0:00的理论发电量（单位化成kWh，保留2位小数）？"
# EquipmentAgent = EquipmentAgent()
# df = EquipmentAgent.get_equipment_data(question)
# PowerAgent = PowerAgent()
# PowerAgent.calculate_power_consumption(question,df)


question = {"id": "gysxdmx_00001", "question": "2024/8/19 下午A架的第一次开机时间（请以XX:XX输出）？", "answer": ""}
# question = {"id": "gysxdmx_00034", "question": "2024/8/23 0:00 ~ 2024/8/25 0:00四个发电机中哪个的能耗最大，能耗为多少（单位化成kWh，保留2位小数）？", "answer": ""}
planning_agent = PlanningAgent()
response = planning_agent.analyze_and_decompose(question)
print(response)
# planning_agent.execute_workflow(question,response)

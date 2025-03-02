import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import json
from zhipuai import ZhipuAI
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv('api_key')
import data.read_csv as read_csv
from .GLM import LLM



def prase_json_from_response(rsp: str):
    pattern = r"```json(.*?)```"
    rsp_json = None
    try:
      match = re.search(pattern, rsp, re.DOTALL)
      if match is not None:
        try:
          rsp_json =  json.loads(match.group(1).strip())
        except:
          pass
      else:
        rsp_json  = json.loads(rsp)
      return rsp_json
    except json.JSONDecodeError as e:
    #   raise("Json Decode Error: {error}".format(error = e))
        return rsp
    



def get_all_job_status(start_time, end_time):
    """
    涉及完整的深海作业，获取开始时间到结束时间之间的所有动作状态的标注值
    参数:
    start_time (pd.Timestamp): 开始时间
    end_time (pd.Timestamp): 结束时间

    返回:
    information (str): 动作信息
    """
    Ajia_Reader = read_csv.Ajia_Reader()

    Ajia_df = Ajia_Reader.get_status(start_time, end_time)

    ShouCe_Reader = read_csv.ShouCe_Reader()
    on_off = ShouCe_Reader.get_on_off_time(start_time, end_time)
 
    ZheBiDiaoChe_Reader = read_csv.ZheBiDiaoChe_Reader()
    zhebi_df = ZheBiDiaoChe_Reader.get_status(start_time, end_time)



    ALL_JOB_PROMPT = f"""
    现在需要根据提供的深海作业A操作步骤<{a_job}>和三个表（Ajia-3_v和Ajia-5_v的状态表<{Ajia_df}>、P3_18和P3_33的状态表<{on_off}>，以及13-11-6_v的状态表<{zhebi_df}>）来标注各个步骤的状态。
    首先，你得仔细理解每个操作步骤中的判断规则，然后结合对应设备的数据表中的时间点和状态变化，来匹配每个步骤发生的时间点。
    例如：
    Ajia-3_v和Ajia-5_v的状态表:
    csvTime                Ajia-3_v  Ajia-5_v         status 
    2024-08-24 07:55:00    0.0000    0.0000           A架开机
    2024-08-24 08:43:00   50.0000   50.0000           稳定开始
    2024-08-24 08:48:00   50.0000   50.0000            峰值前
    2024-08-24 08:49:00   91.0820   90.0969   从稳定值开始增加; 峰值
    2024-08-24 08:50:00   88.5201   87.5592            峰值后
    2024-08-24 08:52:00   50.0000   50.0000           稳定开始
    2024-08-24 08:57:00   50.0000   50.0000            峰值前
    2024-08-24 08:58:00  103.2270  101.7630   从稳定值开始增加; 峰值
    2024-08-24 09:00:00   97.8483   96.6118            峰值后
    2024-08-24 09:01:00   50.0000   50.0000           稳定开始
    2024-08-24 09:04:00    0.0000    0.0000           稳定结束
    2024-08-24 09:09:00 -100.0000 -100.0000           A架关机
    2024-08-24 16:03:00    0.0000    0.0000           A架开机
    2024-08-24 16:24:00   62.0742   62.1504            峰值前
    2024-08-24 16:25:00   88.4523   87.1794             峰值
    2024-08-24 16:26:00   85.4153   84.3572            峰值后
    2024-08-24 17:12:00   50.0000   50.0000           稳定开始
    2024-08-24 17:19:00   63.8906   60.3981  从稳定值开始增加; 峰值前
    2024-08-24 17:20:00  116.4010  114.1970             峰值
    2024-08-24 17:21:00   98.4513   98.1642            峰值后
    2024-08-24 17:23:00   50.0000   50.0000           稳定开始
    2024-08-24 17:36:00    0.0000    0.0000           稳定结束
    2024-08-24 17:41:00 -100.0000 -100.0000           A架关机

    P3_18和P3_33的状态表:
    csvTime               P3_18  P3_33  status
    2024-08-24 07:41:00     79    591   ON DP
    2024-08-24 09:01:00      0      0  OFF DP
    2024-08-24 16:54:00     86    910   ON DP
    2024-08-24 17:20:00      0      0  OFF DP
    2024-08-24 18:51:00     86    793   ON DP
    2024-08-24 22:32:00      0      0  OFF DP

    13-11-6_v的状态表:
    csvTime               13-11-6_v        status
    2024-08-24 08:08:00          9        折臂吊车开机
    2024-08-24 08:23:00         42  折臂吊车回落前最后一个值
    2024-08-24 08:44:00         41  折臂吊车回落前最后一个值
    2024-08-24 08:58:00         93  折臂吊车回落前最后一个值
    2024-08-24 09:00:00          0        折臂吊车关机
    2024-08-24 16:04:00          6        折臂吊车开机
    2024-08-24 16:38:00         44  折臂吊车回落前最后一个值
    2024-08-24 17:22:00         53  折臂吊车回落前最后一个值
    2024-08-24 17:24:00          0        折臂吊车关机
    2024-08-24 18:47:00          3        折臂吊车开机
    2024-08-24 22:31:00          0        折臂吊车关机
    首先，用户给出的数据表有三个：Ajia_df（A架的状态）、on_off（艏侧推的状态）、zhebi_df（折臂吊车的状态）。每个表中都有不同的时间点和对应的状态或数值变化。操作步骤分为下放阶段和回收阶段，每个阶段有多个步骤，每个步骤对应特定的设备、字段和判断规则。

    我的任务是将每个步骤与数据表中的时间点对应起来，确定每个步骤发生的时间。可能需要交叉比对不同表中的时间，尤其是当多个步骤涉及不同设备时。比如，下放阶段的步骤1到13涉及A架、艏侧推、折臂吊车，而回收阶段的步骤1到11同样涉及这些设备。

    首先处理下放阶段：

    步骤1：A架开机，判断规则是电流值从负值变为0。在Ajia_df中，第一个状态是2024-08-24 07:55:00的A架开机，此时Ajia-3_v和Ajia-5_v都是0。但判断规则是电流从负变0，而之前的记录可能在更早的时间点。不过数据表中A架开机的时间点可能对应这个步骤。但需要看是否有负值到0的变化。比如，在Ajia_df中，在07:55:00之前是否有负值？比如09:09:00是-100，然后下一次开机是16:03:00变为0。所以下放阶段可能发生在第一次开机，也就是07:55:00，此时电流从之前的负值变为0。比如，在09:09:00关机时是-100，然后再次开机可能在16:03:00，但下放阶段可能是在第一次操作，即07:55:00的开机。所以步骤1对应的时间应该是07:55:00。

    步骤2：ON DP，艏侧推的数值从0增加。查看on_off表，第一次ON DP是07:41:00，此时P3_18=79，P3_33=591。但判断规则是数值从0增加到非零。但表中第一次记录就是非零，可能之前的OFF状态在更早时间。但根据数据，第一次ON DP是07:41:00，而步骤2的判断是数值从0增加。比如，在07:41:00之前，可能这两个字段是0，这时候在07:41:00变为非零。但数据表中最早的时间是07:41:00的状态为ON DP，所以可能步骤2对应的时间是07:41:00。

    步骤3：折臂吊车开机，数值从0增加。在zhebi_df中，第一次开机是08:08:00，此时数值为9。所以这里对应步骤3的时间是08:08:00。

    步骤4：小艇检查完毕，对应折臂吊车的数值增加（回落前的最后一个值）。在zhebi_df中，回落前的最后一个值比如08:23:00的42，08:44:00的41，08:58:00的93等。需要根据步骤的上下文来确定。比如，下放阶段的步骤4是在折臂吊车开机之后，那么第一个回落前的值可能是08:23:00的42。所以步骤4可能对应08:23:00。

    步骤5：小艇入水，同样需要回落前的最后一个值。接下来可能是08:44:00的41或者后面的。例如，在折臂吊车开机后，第一次检查完毕是08:23:00，之后可能有第二次回落前的值08:44:00的41，这时候可能对应步骤5的时间点。或者要看后续步骤的顺序。比如，步骤6是征服者起吊，对应的A架电流从稳定值50开始增加，根据Ajia_df，第一次稳定开始是08:43:00，之后在08:48:00是峰值前，接着08:49:00增加到峰值。所以步骤6的时间可能是08:49:00。那步骤5可能是在这之前，所以可能是08:44:00的41，对应折臂吊车的一个回落前的值，所以步骤5时间点可能是08:44:00。

    步骤6：征服者起吊，判断规则是电流从稳定值50开始增加。在Ajia_df中，稳定开始的时间是08:43:00，之后在08:49:00电流开始增加到峰值，此时状态是"从稳定值开始增加; 峰值"。因此，步骤6的时间应该是08:49:00，此时电流开始增加。

    步骤7：征服者入水，需要缆绳解除的时间点往前推一分钟。步骤8是缆绳解除，对应电流从高值回落到50。在Ajia_df中，第一次缆绳解除可能是在08:50:00之后，例如08:50:00是峰值后，之后在08:52:00回到稳定开始（50）。所以步骤8的时间是08:52:00，那么步骤7的时间就是08:52:00往前一分钟，即08:51:00。但数据表中并没有这个时间点的记录，可能需要取近似。或者看缆绳解除的时间点是否对应Ajia_df中的某个时间点。比如，第一次缆绳解除可能是在08:52:00的稳定开始，此时电流回到50。所以步骤8的时间是08:52:00，步骤7则是08:51:00。但数据表中，08:50:00的电流已经下降到88.52，而08:52:00回到50。所以缆绳解除的时间是08:52:00，步骤7的时间是08:51:00，但数据表中可能没有该时间点，需要根据现有数据推测。或者可能步骤7的时间是08:50:00（峰值后的时间点）减去1分钟，但不确定。可能需要进一步分析。

    步骤8：缆绳解除，电流从高值回落到50。在Ajia_df中，第一次回落至稳定值是08:52:00的稳定开始，所以步骤8的时间是08:52:00。

    步骤9：A架摆回，判断是征服者入水后，电流重新增加到峰值。征服者入水的时间是步骤7的08:51:00，之后A架电流是否再次增加？在Ajia_df中，第二次峰值是在08:58:00的103.227。例如，在步骤8之后，即08:52:00之后，下一次电流增加到峰值是在08:58:00。因此步骤9的时间可能是08:58:00，此时电流再次达到峰值。

    步骤10：小艇带座，对应折臂吊车的数值增加（回落前的最后一个值）。在zhebi_df中，08:58:00是93，之后在09:00:00归零。所以回落前的最后一个值是08:58:00的93，所以步骤10对应的时间是08:58:00。

    步骤11：折臂吊车关机，数值归零。在zhebi_df中，09:00:00数值为0，所以步骤11的时间是09:00:00。

    步骤12：A架关机，电流变为负值。在Ajia_df中，09:09:00是-100，所以步骤12对应的时间是09:09:00。

    步骤13：OFF DP，数值归零。在on_off表中，第一次OFF DP是09:01:00，所以步骤13的时间是09:01:00。

    接下来处理回收阶段：

    回收阶段的步骤1：A架开机，电流从负变0。在Ajia_df中，第二次开机是16:03:00，此时电流变为0，所以步骤1的时间是16:03:00。

    步骤2：折臂吊车开机，数值从0增加。在zhebi_df中，第二次开机是16:04:00，数值为6，所以对应时间16:04:00。

    步骤3：A架摆出，电流在征服者起吊前到达峰值。在Ajia_df中，回收阶段可能对应第二次操作。例如，在16:03:00开机后，可能后续有峰值。查看Ajia_df中，在16:25:00有峰值88.4523，但此时是下放阶段还是回收阶段？可能需要看时间顺序。回收阶段可能发生在当天的第二次操作，即16:03:00之后。例如，在16:24:00有峰值前，16:25:00是峰值，而步骤7的征服者出水对应的电流峰值可能在17:20:00的116.401。这时步骤3是征服者起吊前的电流峰值。可能在17:19:00（峰值前）和17:20:00（峰值）。例如，征服者出水是步骤7，对应电流峰值的时间是17:20:00。那么步骤3的A架摆出对应的时间可能是17:20:00的峰值前的时间，比如17:19:00的63.8906，此时状态是"从稳定值开始增加; 峰值前"。所以步骤3的时间可能是17:19:00或17:20:00？需要看判断规则："征服者起吊前，电流到达峰值"，而征服者起吊可能对应回收阶段的步骤7（征服者出水，电流峰值）。所以步骤3的时间可能是17:20:00的峰值。

    步骤4：小艇检查完毕，折臂吊车的回落前最后一个值。在zhebi_df中，第二次开机后的回落前值是16:38:00的44，所以步骤4对应16:38:00。

    步骤5：小艇入水，同样回落前的最后一个值，可能对应17:22:00的53。

    步骤6：缆绳挂妥，征服者出水时间往前推一分钟。征服者出水的时间是步骤7的17:20:00，所以步骤6的时间是17:19:00。

    步骤7：征服者出水，电流峰值。在Ajia_df中，17:20:00的116.401是峰值，所以对应步骤7的时间是17:20:00。

    步骤8：小艇落座，折臂吊车回落前的最后一个值。在zhebi_df中，17:22:00是53，之后在17:24:00关机，所以步骤8的时间是17:22:00。

    步骤9：折臂吊车关机，数值归零。对应zhebi_df中的17:24:00的0，所以时间17:24:00。

    步骤10：征服者落座，电流从高值回落到稳定值50。在Ajia_df中，17:23:00回到稳定值50，所以步骤10的时间是17:23:00。

    步骤11：A架关机，电流变为负值。在Ajia_df中，17:41:00的电流是-100，所以步骤11的时间是17:41:00。

    需要注意可能存在多个阶段，比如下放阶段和回收阶段可能分别对应不同的时间段，比如第一次操作是下放，第二次是回收。因此需要根据时间顺序来划分阶段。比如，第一次A架开机在07:55:00属于下放阶段，第二次在16:03:00可能属于回收阶段。此外，on_off表中的ON DP和OFF DP时间点也需要对应到相应的阶段中。例如，下放阶段的ON DP是07:41:00，而回收阶段的ON DP可能在16:54:00？

    但根据on_off表，ON DP发生在07:41:00，09:01:00变为OFF，之后在16:54:00再次ON，17:20:00 OFF，然后18:51:00 ON，22:32:00 OFF。下放阶段的步骤2（ON DP）对应第一次的07:41:00，步骤13（OFF DP）对应09:01:00。回收阶段的步骤可能涉及到第二次的ON DP吗？但是根据回收阶段的步骤中没有ON DP的动作，只有下放阶段的步骤2和步骤13。回收阶段是否有ON DP？根据用户提供的操作步骤，回收阶段的步骤中没有ON DP，所以可能不需要处理。回收阶段的步骤可能只涉及A架和折臂吊车。

    需要再次仔细核对每个步骤对应的设备。例如，下放阶段的步骤13是OFF DP，对应的时间是09:01:00，而回收阶段的步骤中没有涉及到艏侧推的ON DP或OFF DP，所以可能不需要处理。

    现在，将所有步骤对应的时间点整理出来：
    请按照以下json格式进行输出，可以被Python json.loads函数解析。只给出结果，不作解释，不作答：
        ```json
    {{
        "下放阶段": [
            ["A架开机", "2024-08-24 07:55:00", "Ajia_df中电流从-100变为0"],
            ["ON DP", "2024-08-24 07:41:00", "on_off表中P3_33/P3_18从0增加"],
            ["折臂吊车开机", "2024-08-24 08:08:00", "zhebi_df中数值从0→9"],
            ["小艇检查完毕", "2024-08-24 08:23:00", "zhebi_df中回落前最后一个值42"],
            ["小艇入水", "2024-08-24 08:44:00", "zhebi_df中回落前最后一个值41"],
            ["征服者起吊", "2024-08-24 08:49:00", "Ajia_df电流从50开始增加"],
            ["征服者入水", "2024-08-24 08:51:00", "推断自缆绳解除时间08:52:00前推1分钟"],
            ["缆绳解除", "2024-08-24 08:52:00", "Ajia_df电流回落至50"],
            ["A架摆回", "2024-08-24 08:58:00", "Ajia_df电流重新达峰值103.227"],
            ["小艇带座", "2024-08-24 08:58:00", "zhebi_df中回落前最后一个值93"],
            ["折臂吊车关机", "2024-08-24 09:00:00", "zhebi_df数值归零"],
            ["A架关机", "2024-08-24 09:09:00", "Ajia_df电流变为-100"],
            ["OFF DP", "2024-08-24 09:01:00", "on_off表中数值归零"]
        ],
        "回收阶段": [
            ["A架开机", "2024-08-24 16:03:00", "Ajia_df中电流从-100→0"],
            ["折臂吊车开机", "2024-08-24 16:04:00", "zhebi_df中数值从0→6"],
            ["A架摆出", "2024-08-24 17:20:00", "Ajia_df电流峰值116.401"],
            ["小艇检查完毕", "2024-08-24 16:38:00", "zhebi_df中回落前最后一个值44"],
            ["小艇入水", "2024-08-24 17:22:00", "zhebi_df中回落前最后一个值53"],
            ["缆绳挂妥", "2024-08-24 17:19:00", "推断自征服者出水时间17:20:00前推1分钟"],
            ["征服者出水", "2024-08-24 17:20:00", "Ajia_df电流峰值116.401"],
            ["小艇落座", "2024-08-24 17:22:00", "zhebi_df中回落前最后一个值53"],
            ["折臂吊车关机", "2024-08-24 17:24:00", "zhebi_df数值归零"],
            ["征服者落座", "2024-08-24 17:23:00", "Ajia_df电流回落至50"],
            ["A架关机", "2024-08-24 17:41:00", "Ajia_df电流变为-100"]
        ]
    }}

    ```
    """
    messages = [
        {"role": "user", "content": ALL_JOB_PROMPT},
    ]
    model = 'glm-4-plus'
    ans = LLM(messages=messages, model=model)

    return ans

def get_Ajia_job_status(start_time, end_time):
    """
    只涉及A架的深海作业，获取开关机和运行时间
    参数:
    start_time (pd.Timestamp): 开始时间
    end_time (pd.Timestamp): 结束时间

    返回:
    information (str): A架信息,包括详细信息点与开机时间，运行时长信息
    """
    try:
        # 这里是获取数据的逻辑
        Ajia_df = read_csv.Ajia_Reader().get_start_end_time(start_time, end_time)
        if Ajia_df is None:
            raise ValueError("获取数据失败，返回值为 None")
    except Exception as e:
        print(f"获取 A架作业状态时发生错误: {e}")
        return pd.DataFrame()  # 返回一个空的 DataFrame

    if Ajia_df.empty:
        print("获取的数据为空")
        return Ajia_df  # 返回空的 DataFrame

    # 计算运行时长并添加到DataFrame
    run_time = calculate_on_time(Ajia_df)

    actual_run_time = read_csv.Ajia_Reader().get_actual_run_df(start_time, end_time)
    
    information = f'{run_time},实际运行时长为{actual_run_time}分钟'
    return f'{str(Ajia_df)},{information}'

def calculate_on_time(df):
    # 筛选出开机和关机的时间点
    
    on_times = df[df['status'].str.contains('开机|on', case=False)].index
    off_times = df[df['status'].str.contains('关机|off', case=False)].index
    
    # 确保开机和关机时间点数量相同
    if len(on_times) != len(off_times):
        return f"开机和关机时间点数量不匹配"
    
    # 计算每段运行时长并累加
    total_on_time = sum((off - on).total_seconds() for on, off in zip(on_times, off_times))

    return f'总运行时长（开机时长）为{total_on_time/60}分钟'

def get_ZheBi_job_status(start_time, end_time):
    """
    只涉及折臂吊车的深海作业，获取开始时间到结束时间之间的所有动作状态的标注值
    参数:
    start_time (pd.Timestamp): 开始时间
    end_time (pd.Timestamp): 结束时间

    返回:
    information (str): 折臂吊车信息,包括详细信息点与开机时间，运行时长信息
    """ 
    ZheBiDiaoChe_Reader = read_csv.ZheBiDiaoChe_Reader()
    ZheBiDiaoChe_df = ZheBiDiaoChe_Reader.get_status(start_time, end_time)
    run_time = calculate_on_time(ZheBiDiaoChe_df)
    return f'{str(ZheBiDiaoChe_df)},{run_time}'

def get_ShouCe_job_status(start_time, end_time):
    """
    只涉及艏侧推的深海作业，获取开始时间到结束时间之间的所有动作状态的标注值
    参数:
    start_time (pd.Timestamp): 开始时间
    end_time (pd.Timestamp): 结束时间

    返回:
    information (str): 艏侧推信息,包括详细信息点与开机时间，运行时长信息
    """     
    ShouCe_Reader = read_csv.ShouCe_Reader()
    ShouCe_df = ShouCe_Reader.get_on_off_time(start_time, end_time)
    run_time = calculate_on_time(ShouCe_df)
    return f'{str(ShouCe_df)},{run_time}'



a_job = {
    "深海作业A操作步骤": {
        "下放阶段": [
            {"次序":"1","动作":"A架开机","设备":"A架","字段":"Ajia-3_v,Ajia-5_v","字段含义":"1启动柜电流,2启动柜电流","判断规则":"电流值从负值变为0"},
            {"次序":"2","动作":"ON DP","设备":"艏侧推","字段":"P3_33/P3_18","字段含义":"艏推启动功率允许/艏推功率反馈","判断规则":"数值从0增加"},
            {"次序":"3","动作":"折臂吊车开机","设备":"折臂吊车","字段":"13-11-6_v","字段含义":"折臂吊车液压-Pt有功功率","判断规则":"数值从0增加"},
            {"次序":"4","动作":"小艇检查完毕","设备":"折臂吊车","字段":"13-11-6_v","字段含义":"折臂吊车液压-Pt有功功率","判断规则":"数值增加（回落前的最后一个值）"},
            {"次序":"5","动作":"小艇入水","设备":"折臂吊车","字段":"13-11-6_v","字段含义":"折臂吊车液压-Pt有功功率","判断规则":"数值增加（回落前的最后一个值）"},
            {"次序":"6","动作":"征服者起吊","设备":"A架","字段":"Ajia-3_v,Ajia-5_v","字段含义":"1启动柜电流,2启动柜电流","判断规则":"电流从稳定值（50）开始增加"},
            {"次序":"7","动作":"征服者入水","设备":"A架","字段":"Ajia-3_v,Ajia-5_v","字段含义":"1启动柜电流,2启动柜电流","判断规则":"缆绳解除的时间点往前推一分钟"},
            {"次序":"8","动作":"缆绳解除","设备":"A架","字段":"Ajia-3_v,Ajia-5_v","字段含义":"1启动柜电流,2启动柜电流","判断规则":"电流从高值回落至稳定值（50）"},
            {"次序":"9","动作":"A架摆回","设备":"A架","字段":"Ajia-3_v,Ajia-5_v","字段含义":"1启动柜电流,2启动柜电流","判断规则":"征服者入水后，电流重新增加到峰值"},
            {"次序":"10","动作":"小艇带座","设备":"折臂吊车","字段":"13-11-6_v","字段含义":"折臂吊车液压-Pt有功功率","判断规则":"数值增加（回落前的最后一个值）"},
            {"次序":"11","动作":"折臂吊车关机","设备":"折臂吊车","字段":"13-11-6_v","字段含义":"折臂吊车液压-Pt有功功率","判断规则":"数值归零"},
            {"次序":"12","动作":"A架关机","设备":"A架","字段":"Ajia-3_v,Ajia-5_v","字段含义":"1启动柜电流,2启动柜电流","判断规则":"电流值变为负值"},
            {"次序":"13","动作":"OFF DP","设备":"艏侧推","字段":"P3_33/P3_18","字段含义":"艏推启动功率允许/艏推功率反馈","判断规则":"数值归零"}
        ],
        "回收阶段": [
            {"次序":"1","动作":"A架开机","设备":"A架","字段":"Ajia-3_v,Ajia-5_v","字段含义":"1启动柜电流,2启动柜电流","判断规则":"电流值从负值变为0"},
            {"次序":"2","动作":"折臂吊车开机","设备":"折臂吊车","字段":"13-11-6_v","字段含义":"折臂吊车液压-Pt有功功率","判断规则":"数值从0增加"},
            {"次序":"3","动作":"A架摆出","设备":"A架","字段":"Ajia-3_v,Ajia-5_v","字段含义":"1启动柜电流,2启动柜电流","判断规则":"征服者起吊前，电流到达峰值"},
            {"次序":"4","动作":"小艇检查完毕","设备":"折臂吊车","字段":"13-11-6_v","字段含义":"折臂吊车液压-Pt有功功率","判断规则":"数值增加（回落前的最后一个值）"},
            {"次序":"5","动作":"小艇入水","设备":"折臂吊车","字段":"13-11-6_v","字段含义":"折臂吊车液压-Pt有功功率","判断规则":"数值增加（回落前的最后一个值）"},
            {"次序":"6","动作":"缆绳挂妥","设备":"A架","字段":"Ajia-3_v,Ajia-5_v","字段含义":"1启动柜电流,2启动柜电流","判断规则":"征服者出水往前推一分钟"},
            {"次序":"7","动作":"征服者出水","设备":"A架","字段":"Ajia-3_v,Ajia-5_v","字段含义":"1启动柜电流,2启动柜电流","判断规则":"电流峰值"},
            {"次序":"8","动作":"小艇落座","设备":"折臂吊车","字段":"13-11-6_v","字段含义":"折臂吊车液压-Pt有功功率","判断规则":"数值增加（回落前的最后一个值）"},
            {"次序":"9","动作":"折臂吊车关机","设备":"折臂吊车","字段":"13-11-6_v","字段含义":"折臂吊车液压-Pt有功功率","判断规则":"数值归零"},
            {"次序":"10","动作":"征服者落座","设备":"A架","字段":"Ajia-3_v,Ajia-5_v","字段含义":"1启动柜电流,2启动柜电流","判断规则":"电流从高值回落至稳定值（50）"},
            {"次序":"11","动作":"A架关机","设备":"A架","字段":"Ajia-3_v,Ajia-5_v","字段含义":"1启动柜电流,2启动柜电流","判断规则":"电流值变为负值"}
        ]
    }
}

def get_device_df(start_time, end_time, field_names:list, device_name:str): 
    """
    获取指定设备指定字段开始时间到结束时间之间的值
    参数:
    device_name (str): 设备名称
    start_time (pd.Timestamp): 开始时间
    end_time (pd.Timestamp): 结束时间
    field_names (list): 字段名列表

    返回:
    df (pd.DataFrame): 带有指定字段的DataFrame
    """ 
    try:
        if device_name == "A架":
            Ajia_Reader = read_csv.Ajia_Reader()
            df = Ajia_Reader.get_df_data()
        elif device_name == "绞车":
            JiaoChe_Reader = read_csv.JiaoChe_Reader()
            df = JiaoChe_Reader.get_df_data()
        elif device_name == "折臂吊车":
            ZheBiDiaoChe_Reader = read_csv.ZheBiDiaoChe_Reader()
            df = ZheBiDiaoChe_Reader.get_df_data()
        elif device_name == "发电机组":
            FaDianJi_Reader = read_csv.FaDianJi_Reader()
            df = FaDianJi_Reader.get_df_data()
        elif device_name == "主推进器":
            Main_TuiJinQi_Reader = read_csv.Main_TuiJinQi_Reader()
            df = Main_TuiJinQi_Reader.get_df_data()
        elif device_name == "艏侧推" or device_name == "侧推" or device_name == "艏推":
            ShouCe_Reader = read_csv.ShouCe_Reader()
            df = ShouCe_Reader.get_df_data()
        elif device_name == "伸缩推":
            ShenSuoTui_Reader = read_csv.ShenSuoTui_Reader()
            df = ShenSuoTui_Reader.get_df_data()
        elif device_name == "舵桨":
            DuoJiang_Reader = read_csv.DuoJiang_Reader()
            df = DuoJiang_Reader.get_df_data()
        elif device_name == "门架":
            MenJia_Reader = read_csv.MenJia_Reader()
            df = MenJia_Reader.get_df_data()

        df = df[field_names]
        df = df[start_time:end_time]
    except Exception as e:
        return f"操作失败: {str(e)}"

    return df



def get_device_statistic(start_time, end_time, field_names:list, device_name:str):
    """
    获取指定设备指定字段开始时间到结束时间之间的统计学指标
    参数:
    device_name (str): 设备名称
    start_time (pd.Timestamp): 开始时间
    end_time (pd.Timestamp): 结束时间
    field_names (list): 字段名列表

    返回:
    max_values (pd.Series): 每个字段的最大值
    min_values (pd.Series): 每个字段的最小值
    sum_values (pd.Series): 每个字段的sum值
    """
    df = get_device_df(start_time, end_time, field_names, device_name)
    max_values = df.max()  # 计算每个字段的最大值
    min_values = df.min()  # 计算每个字段的最小值
    sum_values = df.sum()  # 计算每个字段的总值

    return max_values, min_values, sum_values


# def get_device_fileds_energy(start_time, end_time, field_names:list, device_name:str):
#     """
#     获取指定设备从开始时间到结束时间之间的每列总能量消耗（kWh）。
#     能量消耗 = sum(功率 * 时间)
    
#     参数:
    
#     start_time (pd.Timestamp): 开始时间
#     end_time (pd.Timestamp): 结束时间
#     field_names (list): 字段名列表
#     device_name (str): 设备名称
    
#     返回:
#     energy_df (pd.DataFrame): 每列在给定时间段内的总能量消耗（kWh）。
#     """
#     df = get_device_df(start_time, end_time, field_names, device_name)
#     # 计算每列的能量消耗
#     energy_df = df.apply(lambda col: col * (df.index.to_series().diff().dt.total_seconds() / 3600), axis=0)
    
#     # 计算每列的总能量消耗
#     total_energy = energy_df.sum()
#     # 返回结果
#     return total_energy

def get_device_energy(start_time, end_time, device:str):
    """
    计算指定设备从开始时间到结束时间之间的总能量消耗（kWh）。
    
    参数:
    start_time (pd.Timestamp): 开始时间
    end_time (pd.Timestamp): 结束时间
    device_name (str): 设备名称

    返回:
    total_energy (float): 总能量消耗（kWh） 
    """
    field_names = {
        "折臂吊车": ["13-11-6_v"],
        "一号门架": ["1-5-6_v"],
        "二号门架": ["13-14-6_v"],
        "绞车": ["1-15-6_v"],
        "艏推": ["P3_18"],
        "侧推": ["P3_18"],
        "艏侧推": ["P3_18"],
        "舵桨":["1-2-6_v","1-3-6_v","13-2-6_v","13-3-6_v"],
        "主推进器":["P3_32","P3_15"],
        "发电机组":["P1_66","P1_75","P2_51","P2_60"],
        "A架":["1-5-6_v","13-14-6_v"]
    }
    device_names = {
        "折臂吊车": "折臂吊车",
        "一号门架": "门架",
        "二号门架": "门架",
        "绞车": "绞车",
        "艏推": "艏侧推",
        "侧推": "艏侧推",
        "艏侧推": "艏侧推",
        "舵桨": "舵桨",
        "主推进器": "主推进器",
        "发电机组": "发电机组",
        "A架": "门架"
    }
    df = get_device_df(start_time, end_time, field_names[device], device_names[device])
    energy_df = df.apply(lambda col: col * (df.index.to_series().diff().dt.total_seconds() / 3600), axis=0)
    total_energy = energy_df.sum()

    return total_energy

def get_total_deck_machinery_energy(start_time, end_time, device=None):

    """
    计算船舶甲板机械从开始时间到结束时间之间的总能量消耗（kWh）。
    """
    field_names = {
        "折臂吊车": ["13-11-6_v"],
        "一号门架": ["1-5-6_v"],
        "二号门架": ["13-14-6_v"],
        "绞车": ["1-15-6_v"]
    }
    device_names = {
        "折臂吊车": "折臂吊车",
        "一号门架": "门架",
        "二号门架": "门架",
        "绞车": "绞车"
    }
    
    total_energy = 0.0  # 初始化总能量消耗

    for device_name, fields in field_names.items():
        df = get_device_df(start_time, end_time, fields, device_names[device_name])
        energy_df = df.apply(lambda col: col * (df.index.to_series().diff().dt.total_seconds() / 3600), axis=0)
        total_energy += energy_df.sum().sum()  # 累加每个设备的能量消耗

    return total_energy  # 返回总能量消耗

def get_total_tuijinxitong_energy(start_time, end_time, device=None):
    """
    计算推进系统从开始时间到结束时间之间的总能量消耗（kWh）。
    """
    field_names = {
        "舵桨": ["1-2-6_v", "1-3-6_v", "13-2-6_v", "13-3-6_v"],
        "主推进器": ["P3_32", "P3_15"],
        "发电机组": ["P1_66", "P1_75", "P2_51", "P2_60"],
        "艏侧推": ["P3_33"],
    }
    device_names = {
        "舵桨": "舵桨",
        "主推进器": "主推进器",
        "发电机组": "发电机组",
        "艏侧推": "艏侧推"
    }
    
    total_energy = 0.0  # 初始化总能量消耗

    for device_name, fields in field_names.items():
        df = get_device_df(start_time, end_time, fields, device_names[device_name])
        energy_df = df.apply(lambda col: col * (df.index.to_series().diff().dt.total_seconds() / 3600), axis=0)
        total_energy += energy_df.sum().sum()  # 累加每个设备的能量消耗

    return total_energy  # 返回总能量消耗



def get_field_names_description(field_names: list):
    output = ""
    with open(r"D:\python\pyfile\pytianchi\GLM深远海船舶作业大模型应用赛\数据\初赛数据1231\初赛数据\field.json", mode='r', encoding='utf-8') as jsonfile:
    
        data = json.load(jsonfile)
        for item in data:
            if item['字段名'] in field_names:
                output += (f"字段名: {item['字段名']}, "
                           f"字段含义: {item['字段含义'].strip()}, "
                           f"所属表名: {item['所属表名']}, "
                           f"单位: {item['单位']}, "
                           f"备注: {item['备注']}\n")
    return output

def get_maintenance_data(field_name:str):
    """
    获取指定字段的维护数据
    
    参数:
    field_name (str): 字段中文名称
    
    返回:
    dict: 包含维护数据的字典
    """
    Maintenance_Reader = read_csv.Maintenance_Reader()
    df = Maintenance_Reader.get_sheet1_data()
    
    df_filtered = df[df['Channel_Text_CN'] == field_name]
    
    if df_filtered.empty:
        return {}
        
    # 将DataFrame转换为字典
    maintenance_dict = df_filtered.to_dict('records')[0]
    
    return maintenance_dict


def get_data_sum(day_time:str, csv_name:str):
    """
    获取指定字段在指定时间范围内的缺失数据
    
    参数:
    day_time (str): 开始时间
    csv_name (str): 字段名

    返回:
    str: 缺失数据点数量信息
    """
    Maintenance_Reader = read_csv.Maintenance_Reader()
    return Maintenance_Reader.get_data_sum(day_time, csv_name)


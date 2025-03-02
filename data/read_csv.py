import pandas as pd
import os
import numpy as np
from abc import ABC

file_dir = 'D://python//pyfile//pytianchi//GLM深远海船舶作业大模型应用赛//数据//初赛数据1231//初赛数据'


class DataReader(ABC):
    def __init__(self, file_dir):
        self.file_dir = file_dir

    #纵向连接
    def concat_data(self, dirs):
        first_file = os.path.join(self.file_dir, dirs[0])
        df = pd.read_csv(first_file).iloc[:, 1:]
        df['csvTime'] = pd.to_datetime(df['csvTime'])
        for col in df.columns:
            if col != 'csvTime':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.set_index('csvTime')

        all_dfs = [df]
        for dir in dirs[1:]:
            csv_dir = os.path.join(self.file_dir, dir)
            data = pd.read_csv(csv_dir).iloc[:, 1:]
            data['csvTime'] = pd.to_datetime(data['csvTime'])
            for col in data.columns:
                if col != 'csvTime':
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            data = data.set_index('csvTime')
            all_dfs.append(data)

        df = pd.concat(all_dfs, axis=0)
        df.index = df.index.floor('1min')
        # 处理时间长
        # df.index = df.index.map(lambda dt: pd.to_datetime(np.round(dt.timestamp() / 60) * 60, unit='s'))
        df = df.sort_index()
        return df
    
    #横向连接
    def merge_data(self, *dfs):
        if not dfs:
            return None
        
        result = dfs[0]
        for df in dfs[1:]:
            result = pd.merge(result, df, on='csvTime', how='outer').fillna(0)
        result = result.sort_index()
        return result
    


class Ajia_Reader(DataReader):
    def __init__(self):
        super().__init__(file_dir)

    def get_df_data(self):
        self.Ajia_dirs = ['Ajia_plc_1_1.csv','Ajia_plc_1_2.csv']
        self.Ajia_df = self.concat_data(self.Ajia_dirs)
        return self.Ajia_df[["Ajia-0_v", "Ajia-1_v","Ajia-2_v","Ajia-3_v","Ajia-4_v","Ajia-5_v"]]


    def get_actual_run_df(self, start_time, end_time):
        df = self.get_df_data()
        df = df[["Ajia-3_v", "Ajia-5_v"]]
        df = df.loc[start_time:end_time]
        df = df.dropna()  
        actual_run_time = len(df)  # 计算实际运行时间的总和
        return actual_run_time
    
    def get_job_data(self):
        df = self.get_df_data()
        df = df[["Ajia-3_v","Ajia-5_v"]]
        
        df.fillna(-100, inplace=True)

        # 如果两个值差40，则把低值赋给高值
        for row in df.itertuples():
            if abs(row[1]-row[2]) > 40:
                if row[1] > row[2]:
                    df.loc[row.Index, "Ajia-3_v"] = row[2]
                else:
                    df.loc[row.Index, "Ajia-5_v"] = row[1]
        # 将50到60之间的值视为50
        df = df.map(lambda x: 50 if 50 <= x <= 60 else x)
        return df
    
    # 获取开机和关机的时间
    def get_start_end_time(self,start_time, end_time):
        
        Ajia_df = self.get_job_data()

        df = Ajia_df.loc[start_time:end_time]

        df['status'] = ''
        shifted_df = df[1:] 

        for row1, row2 in zip(df.itertuples(), shifted_df.itertuples()):
            if (row1[1]< 0  and row2[1] >= 0) and (row1[2]< 0  and row2[2] >= 0):
                df.loc[row2.Index, 'status'] = 'A架开机'


            if (row1[1] >=0 and row2[1]< 0) and (row1[2] >=0 and row2[2]< 0):
                df.loc[row2.Index, 'status'] = 'A架关机'


        df = df[df['status'] != '']

        if df.empty:
            return None  
        return df
    
    def get_peaks(self, start_time, end_time):
        Ajia_df = self.get_job_data()
        df = Ajia_df.loc[start_time:end_time]
        df['status'] = ''
        for i in range(1, len(df) - 1): 
            if df.iloc[i, 0] > df.iloc[i-1, 0] and df.iloc[i, 0] > df.iloc[i+1, 0] and df.iloc[i, 0] and df.iloc[i, 1] > df.iloc[i-1, 1] and df.iloc[i, 1] > df.iloc[i+1, 1] and df.iloc[i, 1] > 78:  # 判断是否为峰值
                df.at[df.index[i-1], 'status'] = '峰值前'
                df.at[df.index[i], 'status'] = '峰值'
                df.at[df.index[i+1], 'status'] = '峰值后'


        df = df[df['status'] != '']
        if df.empty:
            return None  
        return df
    

    def get_stables(self, start_time, end_time):
        Ajia_df = self.get_job_data()
        df = Ajia_df.loc[start_time:end_time]

        df['status'] = ''
        df.index = pd.to_datetime(df.index)

        stable_flag = False
        for i in range(len(df)-1):
            if df["Ajia-3_v"].iloc[i] == 50 and df["Ajia-5_v"].iloc[i] == 50:
                if stable_flag == False:
                    if df["Ajia-3_v"].iloc[i+1] == 50 and df["Ajia-5_v"].iloc[i+1] == 50:
                        df.at[df.index[i], 'status'] = '稳定开始'
                        stable_flag = True
                else:
                    if df["Ajia-3_v"].iloc[i+1] != 50 and df["Ajia-5_v"].iloc[i+1] != 50:
                        if df["Ajia-3_v"].iloc[i+1] > 50 and df["Ajia-5_v"].iloc[i+1] > 50:
                            df.at[df.index[i+1], 'status'] = '从稳定值开始增加'
                            stable_flag = False
                            continue
                        df.at[df.index[i+1], 'status'] = '稳定结束'
                        stable_flag = False

        # 返回标记了状态的行
        return df[df['status'] != '']
    
    def get_status(self, start_time, end_time):
        start_end = self.get_start_end_time(start_time, end_time)
        peaks = self.get_peaks(start_time, end_time)
        stable = self.get_stables(start_time, end_time)

        merged_df = pd.concat([start_end, peaks, stable]).sort_index()

        # 按时间分组聚合（电压取第一个值，状态合并）
        merged_df = merged_df.groupby(merged_df.index).agg({
            'Ajia-3_v': 'first',
            'Ajia-5_v': 'first',
            'status': lambda s: '; '.join(sorted(s.unique(), key=lambda x: -len(x)))
        })
        return merged_df

        
class JiaoChe_Reader(DataReader):
    def __init__(self):
        super().__init__(file_dir)

    def JiaoChe_data(self):
        self.JiaoChe_dirs = ['JiaoChe_plc_1_1.csv','JiaoChe_plc_1_2.csv']
        self.JiaoChe_df = self.concat_data(self.JiaoChe_dirs)
        return self.JiaoChe_df[["PLC_point0_value","PLC_point1_value","PLC_point2_value","PLC_point3_value","PLC_point4_value","PLC_point5_value","PLC_point6_value","PLC_point7_value","PLC_point8_value"]]

    def JiaoChe_Trans_data(self):
        self.JiaoChe_Trans_dirs = ['device_1_15_meter_115_1.csv','device_1_15_meter_115_2.csv']
        self.JiaoChe_Trans_df = self.concat_data(self.JiaoChe_Trans_dirs)
        return self.JiaoChe_Trans_df[["1-15-0_v","1-15-1_v","1-15-2_v","1-15-3_v","1-15-4_v","1-15-5_v","1-15-6_v","1-15-7_v","1-15-8_v","1-15-9_v","1-15-10_v","1-15-11_v","1-15-12_v","1-15-13_v"]]

    
    def get_df_data(self):
        JiaoChe_df = self.JiaoChe_data()
        JiaoChe_Trans_df = self.JiaoChe_Trans_data()
        return self.merge_data(JiaoChe_df, JiaoChe_Trans_df)

class ZheBiDiaoChe_Reader(DataReader):
    def __init__(self):
        super().__init__(file_dir)

    def get_df_data(self):
        self.ZheBiDiaoChe_dirs = ['device_13_11_meter_1311_1.csv','device_13_11_meter_1311_2.csv']
        self.ZheBiDiaoChe_df = self.concat_data(self.ZheBiDiaoChe_dirs)
        return self.ZheBiDiaoChe_df[["13-11-0_v","13-11-1_v","13-11-2_v","13-11-3_v","13-11-4_v","13-11-5_v","13-11-6_v","13-11-7_v","13-11-8_v","13-11-9_v","13-11-10_v","13-11-11_v","13-11-12_v","13-11-13_v"]]

    def get_job_data(self):
        df = self.get_df_data()
        return df[["13-11-6_v"]]
    
    def get_start_end_time(self,start_time, end_time):
        df = self.get_job_data()
        df = df.loc[start_time:end_time]
        df['status'] = ''
        for i in range(1, len(df) - 1): 
            if df.iloc[i,0] > df.iloc[i-1,0] and df.iloc[i-1,0] ==0:
                df.at[df.index[i], 'status'] = '折臂吊车开机'
            elif df.iloc[i-1,0] > df.iloc[i,0] and df.iloc[i,0] ==0:
                df.at[df.index[i], 'status'] = '折臂吊车关机'

        return df[df['status'] != '']

    def get_peaks(self, start_time, end_time):
        Ajia_df = self.get_job_data()
        df = Ajia_df.loc[start_time:end_time]
        df['status'] = ''
        for i in range(1, len(df) - 1): 
            if df.iloc[i, 0] > df.iloc[i-1, 0] and df.iloc[i, 0] > df.iloc[i+1, 0] and df.iloc[i, 0] > 30: 
                df.at[df.index[i], 'status'] = '折臂吊车回落前最后一个值'
        df = df[df['status'] != '']  
        return df
    
    def get_status(self, start_time, end_time):
        start_end = self.get_start_end_time(start_time, end_time)
        peaks = self.get_peaks(start_time, end_time)

        merged_df = pd.concat([start_end, peaks]).sort_index()

        return merged_df

class FaDianJi_Reader(DataReader):
    def __init__(self):
        super().__init__(file_dir)
        self.Port1_1_FaDianJi_dirs = ['Port1_ksbg_1_1.csv','Port1_ksbg_1_2.csv']
        self.Port1_2_FaDianJi_dirs = ['Port1_ksbg_2_1.csv','Port1_ksbg_2_2.csv']
        self.Port1_3_FaDianJi_dirs = ['Port1_ksbg_3_1.csv','Port1_ksbg_3_2.csv']
        self.Port2_1_FaDianJi_dirs = ['Port2_ksbg_1_1.csv','Port2_ksbg_1_2.csv']
        self.Port2_2_FaDianJi_dirs = ['Port2_ksbg_2_1.csv','Port2_ksbg_2_2.csv']
        self.Port2_3_FaDianJi_dirs = ['Port2_ksbg_3_1.csv','Port2_ksbg_3_2.csv']

    def First_FaDianJi_data(self):
        self.Port1_1_FaDianJi_df = self.concat_data(self.Port1_1_FaDianJi_dirs)
        self.Port1_3_FaDianJi_df = self.concat_data(self.Port1_3_FaDianJi_dirs)
        self.First_FaDianJi_df = self.merge_data(self.Port1_1_FaDianJi_df, self.Port1_3_FaDianJi_df)
        return self.First_FaDianJi_df[['P1_2', 'P1_3', 'P1_66']]
    
    
    def Second_FaDianJi_data(self):
        self.Port1_1_FaDianJi_df = self.concat_data(self.Port1_1_FaDianJi_dirs)
        self.Port1_3_FaDianJi_df = self.concat_data(self.Port1_3_FaDianJi_dirs)
        self.Second_FaDianJi_df = self.merge_data(self.Port1_1_FaDianJi_df, self.Port1_3_FaDianJi_df)
        return self.Second_FaDianJi_df[['P1_24', 'P1_25', 'P1_75']]

    def Third_FaDianJi_data(self):
        self.Port2_1_FaDianJi_df = self.concat_data(self.Port2_1_FaDianJi_dirs)
        self.Port2_2_FaDianJi_df = self.concat_data(self.Port2_2_FaDianJi_dirs)
        self.Third_FaDianJi_df = self.merge_data(self.Port2_1_FaDianJi_df, self.Port2_2_FaDianJi_df)
        return self.Third_FaDianJi_df[['P2_2', 'P2_3', 'P2_51']]

    def Fourth_FaDianJi_data(self):
        self.Port2_1_FaDianJi_df = self.concat_data(self.Port2_1_FaDianJi_dirs)
        self.Port2_3_FaDianJi_df = self.concat_data(self.Port2_3_FaDianJi_dirs)
        self.Fourth_FaDianJi_df = self.merge_data(self.Port2_1_FaDianJi_df, self.Port2_3_FaDianJi_df)
        return self.Fourth_FaDianJi_df[['P2_24', 'P2_25', 'P2_60']]

    def Emergency_FaDianJi_data(self):
        self.Port1_2_FaDianJi_df = self.concat_data(self.Port1_2_FaDianJi_dirs)
        self.Emergency_FaDianJi_df = self.merge_data(self.Port1_1_FaDianJi_df, self.Port1_2_FaDianJi_df)
        return self.Emergency_FaDianJi_df[['P1_46', 'P1_47']]

    def get_df_data(self):
        First_FaDianJi_df = self.First_FaDianJi_data()
        Second_FaDianJi_df = self.Second_FaDianJi_data()
        Third_FaDianJi_df = self.Third_FaDianJi_data()
        Fourth_FaDianJi_df = self.Fourth_FaDianJi_data()
        Emergency_FaDianJi_df = self.Emergency_FaDianJi_data()
        return self.merge_data(First_FaDianJi_df, Second_FaDianJi_df, Third_FaDianJi_df, Fourth_FaDianJi_df, Emergency_FaDianJi_df)

class Main_TuiJinQi_Reader(DataReader):
    def __init__(self):
        super().__init__(file_dir)

    def First_TuiJinQi_data(self):
        self.First_TuiJinQi_dirs = ['Port3_ksbg_8_1.csv','Port3_ksbg_8_2.csv']
        self.First_TuiJinQi_df = self.concat_data(self.First_TuiJinQi_dirs)
        return self.First_TuiJinQi_df[['P3_32', 'P3_15']]
    
    def Second_TuiJinQi_data(self):
        self.Second_TuiJinQi_dirs = ['Port4_ksbg_7_1.csv','Port4_ksbg_7_2.csv']
        self.Second_TuiJinQi_df = self.concat_data(self.Second_TuiJinQi_dirs)
        return self.Second_TuiJinQi_df[['P4_15', 'P4_16']]
    
    def get_df_data(self):
        First_TuiJinQi_df = self.First_TuiJinQi_data()
        Second_TuiJinQi_df = self.Second_TuiJinQi_data()
        return self.merge_data(First_TuiJinQi_df, Second_TuiJinQi_df)
    
class ShouCe_Reader(DataReader):
    def __init__(self):
        super().__init__(file_dir)

    def ShouCe_A_data(self):
        self.ShouCe_dirs = ['Port1_ksbg_4_1.csv','Port1_ksbg_4_2.csv']
        self.ShouCe_A_df = self.concat_data(self.ShouCe_dirs)
        return self.ShouCe_A_df[["P1_80"]]

    def ShouCe_Other_data(self):
        self.ShouCe_Other_dirs = ['Port3_ksbg_9_1.csv','Port3_ksbg_9_2.csv']
        self.ShouCe_Other_df = self.concat_data(self.ShouCe_Other_dirs)
        return self.ShouCe_Other_df[["P3_16", "P3_17", "P3_18", "P3_33"]]

    def get_df_data(self):
        ShouCe_A_df = self.ShouCe_A_data()
        ShouCe_Other_df = self.ShouCe_Other_data()
        return self.merge_data(ShouCe_A_df, ShouCe_Other_df)

    def get_job_data(self):
        ShouCe_df = self.ShouCe_Other_data()
        return ShouCe_df[["P3_18", "P3_33"]]

    def get_on_off_time(self,start_time, end_time):
        df = self.get_job_data()
        df = df.loc[start_time:end_time]
        df['status'] = ''
        for i in range(1, len(df) - 1): 
            if df.iloc[i,0] > df.iloc[i-1,0] and df.iloc[i-1,0] ==0 or df.iloc[i,1] > df.iloc[i-1,1] and df.iloc[i-1,1] ==0:
                df.at[df.index[i], 'status'] = 'ON DP'
            elif df.iloc[i-1,0] > df.iloc[i,0] and df.iloc[i,0] ==0 or df.iloc[i-1,1] > df.iloc[i,1] and df.iloc[i,1] ==0:
                df.at[df.index[i], 'status'] = 'OFF DP'

        return df[df['status'] != '']
    

class ShenSuoTui_Reader(DataReader):
    def __init__(self):
        super().__init__(file_dir)

    def get_df_data(self):
        self.ShenSuoTui_dirs = ['Port4_ksbg_8_1.csv','Port4_ksbg_8_2.csv']
        self.ShenSuoTui_df = self.concat_data(self.ShenSuoTui_dirs)
        return self.ShenSuoTui_df[["P4_22", "P4_17", "P4_18", "P4_19", "P4_20", "P4_21", "P4_22"]]

class DuoJiang_Reader(DataReader):
    def __init__(self):
        super().__init__(file_dir)

    def LeftDuoJiang_data(self):
        self.LeftDuoJiang_dirs = ['Port1_ksbg_5_1.csv','Port1_ksbg_5_2.csv']
        self.LeftDuoJiang_df = self.concat_data(self.LeftDuoJiang_dirs)
        return self.LeftDuoJiang_df[["P1_81"]]

    def RightDuoJiang_data(self):
        self.RightDuoJiang_dirs = ['Port2_ksbg_4_1.csv','Port2_ksbg_4_2.csv']
        self.RightDuoJiang_df = self.concat_data(self.RightDuoJiang_dirs)
        return self.RightDuoJiang_df[["P2_64"]]

    def AllDuoJiang_data(self):
        LeftDuoJiang_df = self.LeftDuoJiang_data()
        RightDuoJiang_df = self.RightDuoJiang_data()
        return self.merge_data(LeftDuoJiang_df, RightDuoJiang_df)
    
    def First_DuoJiang_Control_data(self):
        self.First_DuoJiang_Control_dirs = ['Port3_ksbg_10_1.csv','Port3_ksbg_10_2.csv']
        self.First_DuoJiang_Control_df = self.concat_data(self.First_DuoJiang_Control_dirs)
        return self.First_DuoJiang_Control_df[["P3_19","P3_20","P3_21", "P3_22"]]

    def First_DuoJiangA_Feedback_data(self):
        self.First_DuoJiangA_Feedback_dirs = ['device_1_2_meter_102_1.csv','device_1_2_meter_102_2.csv']
        self.First_DuoJiangA_Feedback_df = self.concat_data(self.First_DuoJiangA_Feedback_dirs)
        return self.First_DuoJiangA_Feedback_df[["1-2-0_v","1-2-1_v","1-2-2_v","1-2-3_v","1-2-4_v","1-2-5_v","1-2-6_v","1-2-7_v","1-2-8_v","1-2-9_v","1-2-10_v","1-2-11_v","1-2-12_v","1-2-13_v"]]


    def First_DuoJiangB_Feedback_data(self):
        self.First_DuoJiangB_Feedback_dirs = ['device_1_3_meter_103_1.csv','device_1_3_meter_103_2.csv']
        self.First_DuoJiangB_Feedback_df = self.concat_data(self.First_DuoJiangB_Feedback_dirs)
        return self.First_DuoJiangB_Feedback_df[["1-3-0_v","1-3-1_v","1-3-2_v","1-3-3_v","1-3-4_v","1-3-5_v","1-3-6_v","1-3-7_v","1-3-8_v","1-3-9_v","1-3-10_v","1-3-11_v","1-3-12_v","1-3-13_v"]]
    
    def First_DuoJiang_data(self):
        First_DuoJiang_Control_df = self.First_DuoJiang_Control_data()
        First_DuoJiangA_Feedback_df = self.First_DuoJiangA_Feedback_data()
        First_DuoJiangB_Feedback_df = self.First_DuoJiangB_Feedback_data()
        return self.merge_data(First_DuoJiang_Control_df, First_DuoJiangA_Feedback_df, First_DuoJiangB_Feedback_df)

    def Second_DuoJiang_Control_data(self):
        self.Second_DuoJiang_Control_dirs = ['Port4_ksbg_9_1.csv','Port4_ksbg_9_2.csv']
        self.Second_DuoJiang_Control_df = self.concat_data(self.Second_DuoJiang_Control_dirs)
        return self.Second_DuoJiang_Control_df[["P4_23","P4_24","P4_25","P4_26"]]

    def Second_DuoJiangA_Feedback_data(self):
        self.Second_DuoJiangA_Feedback_dirs = ['device_13_2_meter_1302_1.csv','device_13_2_meter_1302_2.csv']
        self.Second_DuoJiangA_Feedback_df = self.concat_data(self.Second_DuoJiangA_Feedback_dirs)
        return self.Second_DuoJiangA_Feedback_df[["13-2-0_v","13-2-1_v","13-2-2_v","13-2-3_v","13-2-4_v","13-2-5_v","13-2-6_v","13-2-7_v","13-2-8_v","13-2-9_v","13-2-10_v","13-2-11_v","13-2-12_v","13-2-13_v"]]

    def Second_DuoJiangB_Feedback_data(self):
        self.Second_DuoJiangB_Feedback_dirs = ['device_13_3_meter_1303_1.csv','device_13_3_meter_1303_2.csv']
        self.Second_DuoJiangB_Feedback_df = self.concat_data(self.Second_DuoJiangB_Feedback_dirs)
        return self.Second_DuoJiangB_Feedback_df[["13-3-0_v","13-3-1_v","13-3-2_v","13-3-3_v","13-3-4_v","13-3-5_v","13-3-6_v","13-3-7_v","13-3-8_v","13-3-9_v","13-3-10_v","13-3-11_v","13-3-12_v","13-3-13_v"]]
    
    def Second_DuoJiang_data(self):
        Second_DuoJiang_Control_df = self.Second_DuoJiang_Control_data()
        Second_DuoJiangA_Feedback_df = self.Second_DuoJiangA_Feedback_data()
        Second_DuoJiangB_Feedback_df = self.Second_DuoJiangB_Feedback_data()
        return self.merge_data(Second_DuoJiang_Control_df, Second_DuoJiangA_Feedback_df, Second_DuoJiangB_Feedback_df)

    def get_df_data(self):
        AllDuoJiang_df = self.AllDuoJiang_data()
        First_DuoJiang_df = self.First_DuoJiang_data()
        Second_DuoJiang_df = self.Second_DuoJiang_data()
        return self.merge_data(AllDuoJiang_df, First_DuoJiang_df, Second_DuoJiang_df)
    

class MenJia_Reader(DataReader):
    def __init__(self):
        super().__init__(file_dir)

    def First_MenJia_data(self):
        self.First_MenJia_dirs = ['device_1_5_meter_105_1.csv','device_1_5_meter_105_2.csv']
        self.First_MenJia_df = self.concat_data(self.First_MenJia_dirs)
        return self.First_MenJia_df[["1-5-0_v","1-5-1_v","1-5-2_v","1-5-3_v","1-5-4_v","1-5-5_v","1-5-6_v","1-5-7_v","1-5-8_v","1-5-9_v","1-5-10_v","1-5-11_v","1-5-12_v","1-5-13_v"]]
    
    def Second_MenJia_data(self):
        self.Second_MenJia_dirs = ['device_13_14_meter_1314_1.csv','device_13_14_meter_1314_2.csv']
        self.Second_MenJia_df = self.concat_data(self.Second_MenJia_dirs)
        return self.Second_MenJia_df[["13-14-0_v","13-14-1_v","13-14-2_v","13-14-3_v","13-14-4_v","13-14-5_v","13-14-6_v","13-14-7_v","13-14-8_v","13-14-9_v","13-14-10_v","13-14-11_v","13-14-12_v","13-14-13_v"]]
    
    def get_df_data(self):
        First_MenJia_df = self.First_MenJia_data()
        Second_MenJia_df = self.Second_MenJia_data()
        return self.merge_data(First_MenJia_df, Second_MenJia_df)

class Maintenance_Reader(DataReader):
    def __init__(self):
        super().__init__(file_dir)

    def get_sheet1_data(self):
        # 读取设备参数表
        file_path = os.path.join(self.file_dir, '设备参数详情.xlsx')
        self.sheet1_df = pd.read_excel(file_path, sheet_name='Sheet1')
        return self.sheet1_df

    def get_field_desc_data(self):
        # 读取字段释义表
        file_path = os.path.join(self.file_dir, '设备参数详情.xlsx')
        self.field_desc_df = pd.read_excel(file_path, sheet_name='字段释义')
        return self.field_desc_df

    def get_First_FaDianJi_data(self):
        # 读取发电机组参数表
        df = self.get_sheet1_data()
        # 使用正则表达式筛选一号柴油发电机相关的行
        first_generator_df = df[df['Channel_Text_CN'].str.contains('^一号柴油发电机', regex=True)]
        return first_generator_df
    
    def get_Second_FaDianJi_data(self):
        # 读取发电机组参数表
        df = self.get_sheet1_data()
        # 使用正则表达式筛选二号柴油发电机相关的行
        second_generator_df = df[df['Channel_Text_CN'].str.contains('^二号柴油发电机', regex=True)]
        return second_generator_df
    
    def get_Third_FaDianJi_data(self):
        # 读取发电机组参数表
        df = self.get_sheet1_data()
        # 使用正则表达式筛选三号柴油发电机相关的行
        third_generator_df = df[df['Channel_Text_CN'].str.contains('^三号柴油发电机', regex=True)]
        return third_generator_df
    
    def get_Fourth_FaDianJi_data(self):
        # 读取发电机组参数表
        df = self.get_sheet1_data()
        # 使用正则表达式筛选四号柴油发电机相关的行
        fourth_generator_df = df[df['Channel_Text_CN'].str.contains('^四号柴油发电机', regex=True)]
        return fourth_generator_df
    
    def get_Emergency_FaDianJi_data(self):
        # 读取发电机组参数表
        df = self.get_sheet1_data()
        # 使用正则表达式筛选应急柴油发电机相关的行
        emergency_generator_df = df[df['Channel_Text_CN'].str.contains('^停泊/应急柴油发电机', regex=True)]
        return emergency_generator_df
    
    def get_Other_data(self):
        # 读取设备参数表
        df = self.get_sheet1_data()
        # 使用正则表达式筛选其他设备相关的行
        other_df = df[~df['Channel_Text_CN'].str.contains('^一号柴油发电机|二号柴油发电机|三号柴油发电机|四号柴油发电机|停泊/应急柴油发电机', regex=True)]
        return other_df
    
    def get_data_sum(self, day_time, csv_name):
        if csv_name == 'Ajia_plc' or csv_name == 'A架':
            csv_name = 'Ajia_plc_1'
        elif csv_name == 'Jiaoche_plc' or csv_name == '绞车':
            csv_name = 'Jiaoche_plc_1'
            
        csv_name_1 = csv_name + '_1.csv'
        csv_name_2 = csv_name + '_2.csv'
        
        # 将day_time转换为当天的起始时间和结束时间
        start_time = pd.Timestamp(day_time).normalize()  # 设置为当天00:00:00
        end_time = start_time + pd.Timedelta(days=1)  # 设置为下一天00:00:00
        
        df_1 = pd.read_csv(os.path.join(self.file_dir, csv_name_1))
        df_2 = pd.read_csv(os.path.join(self.file_dir, csv_name_2))
        df = pd.concat([df_1, df_2])

        df['csvTime'] = pd.to_datetime(df['csvTime'])
        df = df.set_index('csvTime')
        df = df.loc[start_time:end_time] 
        
        return f"总共数据点数量: {len(df)},理想数据量为1440"
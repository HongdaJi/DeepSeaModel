import read_csv
import pandas as pd

Ajia_Reader = read_csv.Ajia_Reader()

df = Ajia_Reader.get_df_data()

# df按时间查询
start = pd.Timestamp('2024-09-14 14:00')
end = pd.Timestamp('2024-09-14 15:00')  # 查询到17:00
data_in_range = df.loc[start:end]
print(data_in_range)


# # 查询特定日期
# specific_date = pd.Timestamp('2024-05-16')
# data_on_date = df.loc[specific_date]
# print(data_on_date)

# print(Ajia_Reader.Ajia_data())
# Ajia_Reader = read_csv.Ajia_Reader()
# print(Ajia_Reader.csvtime_merge())
# JiaoChe_Reader = read_csv.JiaoChe_Reader()
# print(JiaoChe_Reader.JiaoChe_data())

# JiaoChe_Trans_Reader = read_csv.JiaoChe_Reader()
# print(JiaoChe_Trans_Reader.JiaoChe_Trans_data())
# JiaoChe_Reader = read_csv.JiaoChe_Reader()
# print(JiaoChe_Reader.JiaoChe_data())

# JiaoChe_ALL_Reader = read_csv.JiaoChe_Reader()

# df = JiaoChe_ALL_Reader.JiaoChe_ALL_data()
# df.to_csv('jiaoche.csv', index=True)

# ZheBiDiaoChe_Reader = read_csv.ZheBiDiaoChe_Reader()
# print(ZheBiDiaoChe_Reader.ZheBiDiaoChe_data())

# FaDianJi_Reader = read_csv.FaDianJi_Reader()

# print(FaDianJi_Reader.First_FaDianJi_data())
# print(FaDianJi_Reader.Second_FaDianJi_data())
# print(FaDianJi_Reader.Third_FaDianJi_data())
# print(FaDianJi_Reader.Fourth_FaDianJi_data())
# print(FaDianJi_Reader.Emergency_FaDianJi_data())

# df = FaDianJi_Reader.Fourth_FaDianJi_data()
# df.to_csv('test.csv', index=False)

# Main_TuiJinQi_Reader = read_csv.Main_TuiJinQi_Reader()
# print(Main_TuiJinQi_Reader.First_TuiJinQi_data())
# print(Main_TuiJinQi_Reader.Second_TuiJinQi_data())
# print(Main_TuiJinQi_Reader.Main_TuiJinQi_ALL_data())

# ShouCe_Reader = read_csv.ShouCe_Reader()
# print(ShouCe_Reader.ShouCe_A_data())
# print(ShouCe_Reader.ShouCe_Other_data())
# print(ShouCe_Reader.ShouCe_ALL_data())

# ShenSuoTui_Reader = read_csv.ShenSuoTui_Reader()
# print(ShenSuoTui_Reader.ShenSuoTui_data())

# DuoJiang_Reader = read_csv.DuoJiang_Reader()

# print(DuoJiang_Reader.LeftDuoJiang_data())
# print(DuoJiang_Reader.RightDuoJiang_data())
# print(DuoJiang_Reader.AllDuoJiang_data())
# print(DuoJiang_Reader.First_DuoJiang_Control_data())
# print(DuoJiang_Reader.Second_DuoJiang_Control_data())
# print(DuoJiang_Reader.First_DuoJiangA_Feedback_data())
# print(DuoJiang_Reader.First_DuoJiangB_Feedback_data())
# print(DuoJiang_Reader.Second_DuoJiangA_Feedback_data())
# print(DuoJiang_Reader.Second_DuoJiangB_Feedback_data())

# print(DuoJiang_Reader.First_DuoJiang_data())
# print(DuoJiang_Reader.Second_DuoJiang_data())


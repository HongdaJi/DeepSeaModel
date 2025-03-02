from llm.tools import *
from data.read_csv import *
start_time = pd.Timestamp('2024-08-24 00:00')
end_time = pd.Timestamp('2024-08-25 00:00')

print(get_maintenance_data("一号柴油发电机组滑油压力"))
# print(get_lack_data(start_time, end_time, "Ajia_plc"))
# print(get_maintenance_Other_data())
# print(get_device_energy(start_time, end_time, "折臂吊车"))
# print(get_total_tuijinxitong_energy(start_time, end_time))
# print(get_total_deck_machinery_energy(start_time, end_time))
# Ajia_Reader = read_csv.Ajia_Reader()

# print(Ajia_Reader.get_status(start_time, end_time))

print(get_all_job_status(start_time, end_time))
# print(get_Ajia_job_status(start_time, end_time))

# print(get_ShouCe_job_status(start_time, end_time))
# print(get_ZheBi_job_status(start_time, end_time))
# Ajia_Reader = Ajia_Reader()
# print(Ajia_Reader.get_df_data())


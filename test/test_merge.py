import pandas as pd

# 创建DataFrame1
df1 = pd.DataFrame({
    'csvTime': ['2024/5/16 16:00', '2024/5/16 16:01', '2024/5/16 16:02',
                '2024/5/16 16:03', '2024/5/17 16:04'],
    'P1_24': [1799, 1801, 1801, 1801, 1798],
    'P1_3': [0, 0, 0, 0, 0],
    'P1_2': [0, 0, 0, 0, 0]
})

# 创建DataFrame2
df2 = pd.DataFrame({
    'csvTime': ['2024/5/16 16:00', '2024/5/16 16:00', '2024/5/16 16:01',
               '2024/5/16 16:02', '2024/5/16 16:03', '2024/5/16 16:04',
               '2024/5/16 16:05', '2024/5/16 16:06', '2024/5/16 16:08',
               '2024/5/16 16:09'],
    'P1_66': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
})

# 使用merge函数按csvTime列合并DataFrame
# merged_df = pd.merge(df1, df2, on='csvTime')
result_df = pd.merge(df1, df2, on='csvTime', how='outer')

# result2_df = pd.merge(df1, df2,left_index=True, right_index=True, how='outer')
print(result_df)
# print(result2_df)
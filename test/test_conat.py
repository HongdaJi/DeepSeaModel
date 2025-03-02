import pandas as pd

def merge_dataframes(df1, df2, key_column):
    """
    合并两个 DataFrame，并用 0 填充缺失值。

    Args:
        df1: 第一个 DataFrame。
        df2: 第二个 DataFrame。
        key_column: 用作主键的列名。

    Returns:
        合并后的 DataFrame。
    """

    merged_df = pd.merge(df1, df2, on='csvTime', how='outer').fillna(0)
    return merged_df

# 更新后的 data1
data1 = {
    'csvTime': ['2024-05-16 16:00', '2024-05-16 16:01', '2024-05-16 16:02', '2024-05-16 16:03'],
    'col1': [6942, 6942, 6940, 8999],
    'col2': [6947, 6947, 6945, 1111],
    'col3': [4, 9, 11, 22]
}

data2 = {
    'csvTime': ['2024-05-16 16:00', '2024-05-16 16:02', '2024-05-16 16:03'],
    'PLC_point0_value': [float('nan'), float('nan'), float('nan')],
    'PLC_point1_value': [float('nan'), float('nan'), float('nan')]
}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

# 合并 DataFrame
merged_df = merge_dataframes(df1, df2, 'csvTime')
print(merged_df)
import pandas as pd
import numpy as np

df = pd.DataFrame({'value': [1, 2, 3]}, index=pd.to_datetime(['2024-10-27 10:00:29', '2024-10-27 10:00:15', '2024-10-27 10:00:45']))
df.index = df.index.map(lambda dt: pd.to_datetime(np.round(dt.timestamp() / 60) * 60, unit='s'))
print(df)
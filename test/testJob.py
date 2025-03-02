import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.job import Job

job = Job()

start = pd.Timestamp('2024-05-16 16:00')
end = pd.Timestamp('2024-05-17 16:00') 
df = job.get_state(start,end,'Ajia')
picture_path = job.Data_visualization(start,end,'Ajia')
print(df)

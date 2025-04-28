import kagglehub
from kagglehub import KaggleDatasetAdapter
import seaborn as sns
import pandas as pd
import os

path = kagglehub.dataset_download("aiaiaidavid/the-big-dataset-of-ultra-marathon-running")
filename = "TWO_CENTURIES_OF_UM_RACES.csv"
path = os.path.join(path, filename)
df = pd.read_csv(path)
print(df.head(10))
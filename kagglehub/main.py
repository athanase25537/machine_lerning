import kagglehub
from kagglehub import KaggleDatasetAdapter
import seaborn as sns
import pandas as pd
import os

path = "/home/andriamasy/.cache/kagglehub/datasets/aiaiaidavid/the-big-dataset-of-ultra-marathon-running/versions/2"
filename = "TWO_CENTURIES_OF_UM_RACES.csv"
path = os.path.join(path, filename)
df = pd.read_csv(path)
# print(df.head())
print(df.columns)
# clean up the data
# only want USA, 50k or 50Mi, 2020
df = df[(df['Event distance/length'].isin(['50km', '50mi'])) & (df[df['Athlete country'] == 'USA']) & (df[df['Year of event'] == 2020])]
print(df.head())
print(df.shape)
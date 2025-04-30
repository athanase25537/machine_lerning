import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

dir = "/home/andriamasy/.cache/kagglehub/datasets/arunjangir245/boston-housing-dataset/versions/2"
filename = "BostonHousing.csv"
path = os.path.join(dir, filename)

# Read the dataset
df = pd.read_csv(path)

# Clear the dataset
df = df.dropna(axis=0)

y = df['medv']
X = df.drop(columns=['medv'])

model3 = RandomForestRegressor(random_state=1)

model3.fit(X, y)

# Save the models
joblib.dump(model3, "boston_housing_model.pkl")
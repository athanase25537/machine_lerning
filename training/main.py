import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Download latest version
# path = kagglehub.dataset_download("arunjangir245/boston-housing-dataset")
dir = "/home/andriamasy/.cache/kagglehub/datasets/arunjangir245/boston-housing-dataset/versions/2"
filename = "BostonHousing.csv"
path = os.path.join(dir, filename)
# print("Path to dataset files:", path)

# Read the dataset
df = pd.read_csv(path)

# Clear the dataset
df = df.dropna(axis=0)

# print("Dataset shape:", df.shape)
# print("Dataset columns:", df.columns)
# print("Dataset head:\n", df.head())

y = df['medv']
X = df.drop(columns=['medv'])
# print("X shape:", X.shape)
# print("y shape:", y.shape)

# Check for missing values
# print("Missing values in X:\n", X.isnull().sum())

# Check for missing values in y
# print("Missing values in y:\n", y.isnull().sum())

# Check for duplicates
# print("Duplicates in X:", X.duplicated().sum())
# print("Duplicates in y:", y.duplicated().sum())

model1 = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# train the model
model1.fit(X_train, y_train)

# make predictions
y_pred = model1.predict(X_test)
score1 = model1.score(X_test, y_test) * 100
mean1 = mean_absolute_error(y_test, y_pred)

model2 = DecisionTreeRegressor(random_state=1)
# train the model
model2.fit(X_train, y_train)
# make predictions
y_pred = model2.predict(X_test)
score2 = model2.score(X_test, y_test) * 100
mean2 = mean_absolute_error(y_test, y_pred)

model3 = RandomForestRegressor(random_state=1)
# train the model
model3.fit(X_train, y_train)
# make predictions
y_pred = model3.predict(X_test)
score3 = model3.score(X_test, y_test) * 100
mean3= mean_absolute_error(y_test, y_pred)

print("Scores:")
print(f"Linear Regression: {score1:.2f}%")
print(f"Decision Tree Regressor: {score2:.2f}%")
print(f"Random Forest Regressor: {score3:.2f}%\n")

print("Mean Absolute Errors:")
print(f"Linear Regression: {mean1*1000:.2f}$")
print(f"Decision Tree Regressor: {mean2*1000:.2f}$")
print(f"Random Forest Regressor: {mean3*1000:.2f}$")
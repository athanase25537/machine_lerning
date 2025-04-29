from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = 'datasets/test.csv'
data_frame = pd.read_csv(file_path)

X = data_frame['x'].values.reshape(-1, 1)
y = data_frame['y'].values.reshape(-1, 1)

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
print("Coefficients:", model.coef_)

x_test = np.array([[15]])
y_pred = model.predict(x_test)
print(f"Prediction for x={x_test[0][0]}:", y_pred[0][0])

plt.figure()
plt.scatter(X, y, color='blue', label='Data points')
plt.scatter(x_test, y_pred, color='yellow', label=f"Prediction for x={x_test[0][0]}")
plt.plot(X, predictions, color='red', label='Regression line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter plot of data points')
plt.show()
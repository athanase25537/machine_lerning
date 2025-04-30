import joblib
import pandas as pd
model = joblib.load("boston_housing_model.pkl")

features = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']
data_pred = [0.00632, 18.0, 2.31, 0.0, 0.538, 6.575, 65.2, 4.09, 1, 296.0, 15.3, 396.9, 4.98]
X_pred = pd.DataFrame(
    [data_pred],
    columns=features
)
y_pred = model.predict(X_pred)
print("Predicted value:", y_pred[0])
print(model.score(X_pred, y_pred) * 100)


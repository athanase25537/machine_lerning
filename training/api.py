from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel

class HouseFeatures(BaseModel):
    crim: float
    zn: float
    indus: float
    chas: int
    nox: float
    rm: float
    age: float
    dis: float
    rad: int
    tax: float
    ptratio: float
    b: float
    lstat: float

app = FastAPI()

@app.get("/")
def main():
    return {"message": "Welcome to the Boston Housing predict price application!"}

@app.post("/predict")
def predict(data: HouseFeatures):
    """
    Predict the price of a house in Boston based on the input features.
    """
    # Load the model
    import joblib
    model = joblib.load("boston_housing_model.pkl")

    # Define the features
    features = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']

    # Check if all required features are present in the input data
    missing_features = [f for f in model.feature_names_in_ if f not in data.dict()]
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {', '.join(missing_features)}"
        )
    
    
    # Create a DataFrame from the input data
    X_pred = pd.DataFrame([data.dict()], columns=features)

    # Make prediction
    y_pred = model.predict(X_pred)

    return {
        "predicted_price": round(y_pred[0], 2)*1000,
        "unit": "USD",
        "score": "98.03%"
        }

@app.get("/health")
def health():
    """
    Check the health of the application.
    """
    return {"status": "healthy"}

# run application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8083)
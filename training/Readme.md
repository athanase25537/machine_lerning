# Boston Housing Price Prediction API

This FastAPI application provides a machine learning model to predict housing prices in Boston based on various features like crime rate, number of rooms, and proximity to employment centers.
## Table of Contents

###    API Features

###    Installation

###    Usage

###    API Endpoints

###    Input Features Documentation

###    Example Requests

## API Features

###    Predict Boston housing prices using a pre-trained RandomForestRegressor model.

###    Input validation with Pydantic to ensure correct data types.

###    Automatic docs with Swagger UI (/docs) and ReDoc (/redoc).

###    Error handling for missing/invalid features.

## Installation
1. Clone the repository
bash

git clone https://github.com/athanase25537/boston-housing-api.git
cd boston-housing-api

2. Install dependencies
bash

    pip install -r requirements.txt

3. Download the pre-trained model

Ensure boston_housing_model.pkl is in the project root.
## Usage
Run the API locally
bash

uvicorn main:app --reload

    Access:

        API Docs (Swagger UI): http://127.0.0.1:8000/docs

        ReDoc: http://127.0.0.1:8000/redoc


## API Endpoints
Endpoint	Method	Description	Input	Output
/predict	POST	Predict house price	JSON (see below)	{"predicted_price": float, "unit": "USD", "score": 98,03%}

## Input Features Documentation
Feature	Type	Description	Example
crim	float	Crime rate per capita	0.027
zn	float	Residential land zoned for large lots	0.0
indus	float	Non-retail business acres per town	7.07
chas	int	Near Charles River? (1=Yes, 0=No)	0
nox	float	Nitric oxide concentration	0.469
rm	float	Average rooms per dwelling	6.421
age	float	% of homes built before 1940	78.9
dis	float	Distance to employment centers	4.967
rad	float	Highway accessibility index	2
tax	float	Property tax rate per $100K	242
ptratio	float	Pupil-teacher ratio	17.8
b	float	Demographic proportion (historical)	396.9
lstat	float	% of lower-status population	9.14

Output:
{
    "predicted_price": 25.934,
    "unit": "USD", 
    "score": 98,03%
}
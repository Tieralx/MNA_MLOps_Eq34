# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd
import numpy as np


# Load the model
with open("Deploy\cervical_cancer_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the target names (class labels)
data = pd.read_csv("sobar-72.csv") 
#data = genfromtxt("sobar-72.csv",delimiter=',')
target_names = ['ca_cervix']
#feature_names = data.columns
  
        

# Define the input data format for prediction
class cervicalcancer(BaseModel):
    
    features: list[int]
    
# Initialize FastAPI
app = FastAPI()

# Prediction endpoint
@app.post("/predict")
def predict(cervicalcancer_data: cervicalcancer):
    if len(cervicalcancer_data.features) != model.n_features_in_:
        raise HTTPException(
            status_code=400,
            detail=f"Input must contain {model.n_features_in_} features."
        )
    # Predict
    prediction = model.predict([cervicalcancer_data.features])[0]
    prediction_name = target_names
    return {"prediction": int(prediction), "prediction_name": prediction_name}

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Cervical cancer classification model API"}
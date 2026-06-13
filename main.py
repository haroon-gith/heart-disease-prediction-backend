from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
 
app = FastAPI()
 
# Load model files
model   = joblib.load("KNN_heart.pkl")
scaler  = joblib.load("Scaler.pkl")
columns = joblib.load("Columns.pkl")
 
# Input data ka format
class HeartInput(BaseModel):
    Age: int
    RestingBP: float
    Cholesterol: float
    MaxHR: float
    Oldpeak: float
    Sex: str           # "M" ya "F"
    ChestPainType: str # "ATA", "NAP", "ASY", "TA"
    RestingECG: str    # "Normal", "ST", "LVH"
    ExerciseAngina: str # "Y" ya "N"
    ST_Slope: str      # "Up", "Flat", "Down"
    FastingBS: int     # 0 ya 1
 
@app.get("/")
def home():
    return {"message": "Heart Disease API chal rahi hai!"}
 
@app.post("/predict")
def predict(data: HeartInput):
    # Step 1: DataFrame banao
    input_dict = {
        "Age": data.Age,
        "RestingBP": data.RestingBP,
        "Cholesterol": data.Cholesterol,
        "MaxHR": data.MaxHR,
        "Oldpeak": data.Oldpeak,
        "FastingBS": data.FastingBS,
        "Sex": data.Sex,
        "ChestPainType": data.ChestPainType,
        "RestingECG": data.RestingECG,
        "ExerciseAngina": data.ExerciseAngina,
        "ST_Slope": data.ST_Slope,
    }
    df = pd.DataFrame([input_dict])
 
    # Step 2: Encoding karo
    df_encoded = pd.get_dummies(df, drop_first=True)
    df_encoded = df_encoded.astype(int)
 
    # Step 3: Training columns se align karo
    df_aligned = df_encoded.reindex(columns=columns, fill_value=0)
 
    # Step 4: Numeric columns scale karo
    numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    df_aligned[numeric_cols] = scaler.transform(df_aligned[numeric_cols])
 
    # Step 5: Predict karo
    prediction = model.predict(df_aligned)[0]
 
    if prediction == 1:
        result = "Heart Disease hai ⚠️"
    else:
        result = "Heart Disease nahi hai ✅"
 
    return {
        "prediction": int(prediction),
        "result": result
    }
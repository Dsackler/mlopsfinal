from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from fastapi import Body

# run uvicorn deployment:app --reload

class Input(BaseModel):
    data: list

app = FastAPI()
model_no_drift = pickle.load(open("model_no_drift.pkl", 'rb'))
model_drift = pickle.load(open("model_drift.pkl", 'rb'))

columns = ["brand","year","engine_size","fuel_type","transmission","mileage","condition","model"]

@app.get("/")
def read_root():
    return {"message": "Hello! This is the prediction API."}

@app.post("/predict_no_drift")
def predict(data: list = Body(...)):
    df = pd.DataFrame([data], columns=columns)
    prediction = model_no_drift.predict(df)
    return {"prediction": prediction.tolist()}

@app.post("/predict_drift")
def predict(data: list = Body(...)):
    df = pd.DataFrame([data], columns=columns)
    prediction = model_drift.predict(df)
    return {"prediction": prediction.tolist()}
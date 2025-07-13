from fastapi import FastAPI
from app.schemas.tx import TxItem
from app.core.utils import prepare_features
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("app/model/model.pkl")

@app.get("/")
def root():
    return {"message": "Budget Predictor API - OK"}

@app.post("/predict")
def predict(transactions: list[TxItem]):
    df = pd.DataFrame([t.dict() for t in transactions])
    X_pred = prepare_features(df)
    y_pred = model.predict(X_pred)

    return [{"category": cat, "prediction": float(pred)} for cat, pred in zip(X_pred["category"], y_pred)]

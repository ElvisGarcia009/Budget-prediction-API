from fastapi import FastAPI
from typing import List

# Importa tu nuevo esquema de features
from app.schemas.feature import FeatureItem  
# Importa la función que hace la predicción
from app.model.predictor import predict_category  

app = FastAPI()

# Carga el modelo (ya en predictor.py)
# model = joblib.load("app/model/model.pkl")  ← se queda en predictor.py

@app.get("/")
def root():
    return {"message": "Budget Predictor API - OK"}

@app.post("/predict")
def predict(features: List[FeatureItem]):
    # Convierte la lista de FeatureItem a lista de dicts
    data = [f.dict() for f in features]
    # Llama a la función que retorna [{"category": ..., "prediction": ...}, ...]
    results = predict_category(data)
    return {"predictions": results}

# app/model/predictor.py

import joblib
import os
import pandas as pd

# Ruta al modelo entrenado
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(MODEL_PATH)

def predict_category(data: list[dict]) -> list[dict]:
    """
    data: lista de dicts con keys:
      - category
      - partial_sum
      - day_of_fortnight
      - percent_of_fortnight
      - avg_daily_spending_so_far
      - days_left_in_fortnight

    Retorna una lista de dicts:
      [{"category": <str>, "prediction": <float>}, ...]
    """
    # Convertimos a DataFrame
    df = pd.DataFrame(data)

    # Ejecutamos la predicción
    preds = model.predict(df)

    # Empaquetamos la salida
    return [
        {"category": row["category"], "prediction": float(pred)}
        for row, pred in zip(data, preds)
    ]

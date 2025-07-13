import pandas as pd
import joblib
from datetime import date

MODEL_FILE = "forest_by_cat.pkl"

def prepare_features(transactions):
    """
    transactions: list of (date_string, category, amount)
    Devuelve un DataFrame con una fila por categoría.
    """
    df = pd.DataFrame(transactions, columns=["date", "category", "amount"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["category"] = df["category"].str.strip()

    ref_date = df["date"].min()
    total_days = 15 if ref_date.day <= 15 else ref_date.days_in_month - 15
    day_idx = (df["date"].max() - ref_date).days + 1
    days_left = total_days - day_idx
    percent   = day_idx / total_days

    grouped = df.groupby("category")["amount"].agg(["sum", "count"]).reset_index()
    grouped.rename(columns={"sum": "partial_sum", "count": "_tmp"}, inplace=True)

    # construir DataFrame final
    grouped["day_of_fortnight"]          = day_idx
    grouped["percent_of_fortnight"]      = percent
    grouped["avg_daily_spending_so_far"] = grouped["partial_sum"] / day_idx
    grouped["days_left_in_fortnight"]    = days_left
    return grouped.drop(columns=["_tmp"])

# ejemplo: transacciones acumuladas hasta hoy
transactions = [
("2025-07-01","transporte", 300.0),
    ("2025-07-02","transporte", 300.0),
    ("2025-07-03","transporte", 300.0),
    ("2025-07-04","transporte", 300.0),
    ("2025-07-05","transporte", 300.0),
    ("2025-07-06","transporte", 300.0),
    ("2025-07-08","transporte", 300.0),
    ("2025-07-09","transporte", 300.0),
    ("2025-07-10","transporte", 300.0),
    ("2025-07-11","transporte", 300.0),
    ("2025-07-12","transporte", 300.0),
    ("2025-07-13","transporte", 300.0),
]

X_pred = prepare_features(transactions)

model = joblib.load(MODEL_FILE)
y_pred = model.predict(X_pred)

print("──────── Predicción por categoría ────────")
for cat, pred in zip(X_pred["category"], y_pred):
    print(f"{cat:<12s}: RD$ {pred:,.2f}")

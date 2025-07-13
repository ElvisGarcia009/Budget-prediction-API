import pandas as pd
from pathlib import Path
from datetime import date
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ──────────────────────────────
CSV_PATH   = "transactions.csv"  # ← tu archivo con date,category,amount
SEP        = ","                 # usa "\t" si está tab-separado
MODEL_FILE = "forest_by_cat.pkl"
RANDOM_SEED = 42
# ──────────────────────────────

def fortnight_key(ts: pd.Timestamp) -> str:
    """Devuelve un identificador YYYY-MM-{1|2} para quincenas."""
    return f"{ts.year}-{ts.month:02d}-{1 if ts.day <= 15 else 2}"

def fortnight_total_days(ts: pd.Timestamp) -> int:
    return 15 if ts.day <= 15 else (ts.days_in_month - 15)

def build_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Genera un ejemplo por DÍA y CATEGORÍA dentro de cada quincena."""
    samples = []
    for (fkey, cat), g in df.groupby(["fortnight", "category"]):
        g = g.sort_values("date")
        total_spent_cat = g["amount"].sum()

        start_day  = 1 if g["date"].iloc[0].day <= 15 else 16
        total_days = fortnight_total_days(g["date"].iloc[0])

        # Resampleo diario por categoría
        daily = g.set_index("date")["amount"].resample("D").sum()
        cum   = daily.cumsum()

        for d, cum_sum in cum.items():
            day_idx   = (d.date() - date(d.year, d.month, start_day)).days + 1
            days_left = total_days - day_idx
            samples.append({
                "category": cat,
                "partial_sum": cum_sum,
                "day_of_fortnight": day_idx,
                "percent_of_fortnight": day_idx / total_days,
                "avg_daily_spending_so_far": cum_sum / day_idx,
                "days_left_in_fortnight": days_left,
                "total_spent": total_spent_cat,
            })
    return pd.DataFrame(samples)

def main():
    # 1. Leer CSV
    df = pd.read_csv(CSV_PATH, sep=SEP, header=0, names=["date", "category", "amount"])
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S %z", utc=True)
    df["amount"] = df["amount"].astype(float)
    df["category"] = df["category"].str.strip()   # limpia espacios
    df["fortnight"] = df["date"].apply(fortnight_key)

    # 2. Construir dataset día-a-día por categoría
    data = build_samples(df)
    print(f"🔧  Ejemplos generados: {len(data):,}")

    X = data.drop(columns=["total_spent"])
    y = data["total_spent"]

    # 3. Pipeline: One-hot de categoría + RandomForest
    categorical = ["category"]
    numeric = [c for c in X.columns if c != "category"]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ], remainder="passthrough")

    model = Pipeline([
        ("pre",   pre),
        ("rf",    RandomForestRegressor(
                    n_estimators=400,
                    min_samples_leaf=2,
                    random_state=RANDOM_SEED,
                    n_jobs=-1))
    ])

    model.fit(X, y)

    # 4. Evaluación rápida (train set)
    preds = model.predict(X)
    print("🎯 MAE:", f"{mean_absolute_error(y, preds):,.2f}")
    print("🎯 R² :", f"{r2_score(y, preds):.4f}")

    # 5. Guardar pipeline
    joblib.dump(model, MODEL_FILE)
    print(f"✅ Modelo guardado en «{MODEL_FILE}»")

if __name__ == "__main__":
    main()

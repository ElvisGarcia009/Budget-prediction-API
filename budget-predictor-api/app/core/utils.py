import pandas as pd
from datetime import date

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["category"] = df["category"].str.strip()

    ref_date = df["date"].min()
    total_days = 15 if ref_date.day <= 15 else ref_date.days_in_month - 15
    day_idx = (df["date"].max() - ref_date).days + 1
    days_left = total_days - day_idx
    percent = day_idx / total_days

    grouped = df.groupby("category")["amount"].agg(["sum", "count"]).reset_index()
    grouped.rename(columns={"sum": "partial_sum", "count": "_tmp"}, inplace=True)

    grouped["day_of_fortnight"] = day_idx
    grouped["percent_of_fortnight"] = percent
    grouped["avg_daily_spending_so_far"] = grouped["partial_sum"] / day_idx
    grouped["days_left_in_fortnight"] = days_left

    return grouped.drop(columns=["_tmp"])

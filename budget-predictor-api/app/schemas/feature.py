# app/schemas/feature.py
from pydantic import BaseModel

class FeatureItem(BaseModel):
    category: str
    partial_sum: float
    day_of_fortnight: int
    percent_of_fortnight: float
    avg_daily_spending_so_far: float
    days_left_in_fortnight: int

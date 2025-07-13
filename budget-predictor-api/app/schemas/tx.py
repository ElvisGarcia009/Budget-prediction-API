from pydantic import BaseModel

class TxItem(BaseModel):
    date: str
    category: str
    amount: float

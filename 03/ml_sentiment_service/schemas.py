from pydantic import BaseModel
from typing import List

class Item(BaseModel):
    text: str

class PredictRequest(BaseModel):
    items: List[Item]

class PredictResponseItem(BaseModel):
    text: str
    label: str

class PredictResponse(BaseModel):
    items: List[PredictResponseItem]

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


class PredictRequest(BaseModel):
    """
    Prosta wersja: wysyłasz cechy jako dict.
    Dzięki temu nie musisz ręcznie deklarować 20+ pól datasetu.
    """
    features: Dict[str, Any] = Field(..., description="Raw input features (column -> value)")


class Reason(BaseModel):
    feature: str
    contribution: float


class PredictResponse(BaseModel):
    score: float
    decision: str
    threshold: float
    reasons: List[Reason] = []
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    tracking_uri: str
    source: str  # "mlflow" albo "local"

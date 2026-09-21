"""
API FastAPI pour servir les prévisions de consommation énergétique.

Lancement:
    uvicorn src.api.app:app --reload --port 8000
"""

from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.models.predict import predict_one

app = FastAPI(
    title="DriftGuard API",
    description="API de prévision de consommation énergétique avec monitoring de drift intégré.",
    version="0.1.0",
)


class PredictionRequest(BaseModel):
    datetime: str  # format ISO, ex: "2018-01-15T08:00:00"


class PredictionResponse(BaseModel):
    datetime: str
    predicted_consumption_mw: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        datetime.fromisoformat(request.datetime)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="Format de date invalide. Utilise le format ISO, ex: 2018-01-15T08:00:00",
        )

    result = predict_one(request.datetime)
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)

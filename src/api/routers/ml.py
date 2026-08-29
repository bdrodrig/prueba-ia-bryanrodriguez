"""Endpoints directos a los modelos de ML/DL (sin pasar por la base de datos)."""
import os
from fastapi import APIRouter, Depends, HTTPException
from src.api.auth import get_current_user
from src.api.models_db import User
from src.api.schemas.ml import (
    ChurnPredictRequest, SentimentRequest, SentimentResponse, ModelsInfoResponse,
)
from src.api.schemas.ticket import TicketClassifyRequest, TicketClassifyResponse
from src.ml.churn_predictor import ChurnPredictor
from src.ml.ticket_classifier import TicketClassifier
from src.dl.sentiment_model import SentimentClassifier

router = APIRouter(prefix="/api/v1/ml", tags=["Modelos ML"])

_models_cache = {}


def _get(name: str):
    if name not in _models_cache:
        if name == "churn":
            _models_cache[name] = ChurnPredictor()
        elif name == "ticket":
            _models_cache[name] = TicketClassifier()
        elif name == "sentiment":
            _models_cache[name] = SentimentClassifier()
    return _models_cache[name]


@router.post("/predict-churn", summary="Predecir churn a partir de datos crudos (sin cliente en BD)")
def predict_churn(data: ChurnPredictRequest, current_user: User = Depends(get_current_user)):
    return _get("churn").predict(data.model_dump())


@router.post("/classify-ticket", response_model=TicketClassifyResponse, summary="Clasificar un ticket")
def classify_ticket(data: TicketClassifyRequest, current_user: User = Depends(get_current_user)):
    try:
        return _get("ticket").predict(data.description)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/analyze-sentiment", response_model=SentimentResponse, summary="Analizar sentimiento de un texto")
def analyze_sentiment(data: SentimentRequest, current_user: User = Depends(get_current_user)):
    return _get("sentiment").predict(data.text)


@router.get("/models/info", response_model=ModelsInfoResponse, summary="Información de los modelos disponibles")
def models_info(current_user: User = Depends(get_current_user)):
    def _exists(path):
        return os.path.exists(path)

    return ModelsInfoResponse(models=[
        {
            "name": "ticket_classifier", "type": "sklearn (LogisticRegression + TF-IDF)",
            "task": "Clasificación de tickets (5 categorías)",
            "loaded": _exists("models/ticket_classifier.joblib"),
        },
        {
            "name": "churn_predictor", "type": "sklearn (RandomForestClassifier)",
            "task": "Predicción de churn (binaria, con probabilidad)",
            "loaded": _exists("models/churn_predictor.joblib"),
        },
        {
            "name": "sentiment_model", "type": "TensorFlow/Keras (Embedding + LSTM)",
            "task": "Clasificación de sentimiento (3 clases)",
            "loaded": _exists("models/sentiment_model.keras"),
        },
        {
            "name": "resolution_time_model", "type": "TensorFlow/Keras (Functional API, inputs mixtos)",
            "task": "Regresión de tiempo de resolución (horas)",
            "loaded": _exists("models/resolution_time_model.keras"),
        },
    ])

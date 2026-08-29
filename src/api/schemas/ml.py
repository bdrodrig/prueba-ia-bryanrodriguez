"""Schemas de Pydantic para los endpoints de ML/DL y autenticación."""
from typing import Optional
from pydantic import BaseModel, field_validator, ConfigDict


class ChurnPredictRequest(BaseModel):
    monthly_charge: float
    tenure_months: int
    total_charges: float
    num_tickets: int
    avg_satisfaction: float
    plan_type: str
    contract_type: str
    payment_method: str

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "monthly_charge": 18.0, "tenure_months": 2, "total_charges": 36.0,
            "num_tickets": 4, "avg_satisfaction": 1.8,
            "plan_type": "basic", "contract_type": "month-to-month", "payment_method": "cash",
        }
    })


class SentimentRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def validar_longitud(cls, v: str) -> str:
        if len(v.strip()) < 3:
            raise ValueError("El texto es demasiado corto para analizar")
        return v


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float


class ModelsInfoResponse(BaseModel):
    models: list


class LoginRequest(BaseModel):
    username: str
    password: str

    model_config = ConfigDict(json_schema_extra={
        "example": {"username": "agente1", "password": "password123"}
    })


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class AgentChatRequest(BaseModel):
    session_id: str
    message: str
    customer_id: Optional[int] = None

    model_config = ConfigDict(json_schema_extra={
        "example": {"session_id": "sesion-abc123", "message": "Hola, tengo un problema con mi internet"}
    })


class AgentChatResponse(BaseModel):
    response: str
    intent: Optional[str]
    escalate: bool

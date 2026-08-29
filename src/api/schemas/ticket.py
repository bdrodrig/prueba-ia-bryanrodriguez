"""Schemas de Pydantic para tickets."""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, field_validator, ConfigDict

CATEGORIAS_VALIDAS = {"TECH", "BILL", "PLAN", "CNCL", "OTHR"}
PRIORIDADES_VALIDAS = {"low", "medium", "high"}


class TicketCreate(BaseModel):
    customer_id: int
    description: str
    priority: str = "medium"
    category: Optional[str] = None  # si no se envía, se infiere con el clasificador

    @field_validator("description")
    @classmethod
    def validar_descripcion(cls, v: str) -> str:
        """Descripción: mínimo 20, máximo 500 caracteres (requisito del enunciado)."""
        if len(v) < 20:
            raise ValueError(f"La descripción debe tener al menos 20 caracteres (recibido: {len(v)})")
        if len(v) > 500:
            raise ValueError(f"La descripción no puede exceder 500 caracteres (recibido: {len(v)})")
        return v

    @field_validator("priority")
    @classmethod
    def validar_prioridad(cls, v: str) -> str:
        if v not in PRIORIDADES_VALIDAS:
            raise ValueError(f"priority debe ser uno de: {PRIORIDADES_VALIDAS}")
        return v

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "customer_id": 1,
            "description": "Mi internet lleva 3 días sin funcionar y nadie me ayuda",
            "priority": "high",
        }
    })


class TicketUpdate(BaseModel):
    status: Optional[str] = None
    priority: Optional[str] = None
    satisfaction: Optional[float] = None


class TicketResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    ticket_id: int
    customer_id: int
    category: str
    description: str
    priority: str
    status: str
    is_active: bool
    created_at: datetime
    resolved_at: Optional[datetime] = None
    satisfaction: Optional[float] = None


class TicketClassifyRequest(BaseModel):
    description: str

    @field_validator("description")
    @classmethod
    def validar_longitud_minima(cls, v: str) -> str:
        if len(v) < 10:
            raise ValueError("El texto debe tener al menos 10 caracteres")
        return v

    model_config = ConfigDict(json_schema_extra={
        "example": {"description": "Mi internet está muy lento desde hace días"}
    })


class TicketClassifyResponse(BaseModel):
    category: str
    probabilities: dict

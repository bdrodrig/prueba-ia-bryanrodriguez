"""Schemas de Pydantic para clientes -- validan lo que entra y sale de la API."""
import re
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, field_validator, ConfigDict


class CustomerBase(BaseModel):
    name: str
    email: EmailStr
    phone: str
    plan_type: str
    monthly_charge: float
    contract_type: str
    payment_method: str

    @field_validator("phone")
    @classmethod
    def validar_telefono(cls, v: str) -> str:
        """Teléfono: mínimo 10 dígitos, solo números, debe empezar con 09
        (requisito explícito del enunciado -- formato celular en Ecuador)."""
        if not v.isdigit():
            raise ValueError("El teléfono debe contener solo números")
        if len(v) < 10:
            raise ValueError("El teléfono debe tener mínimo 10 dígitos")
        if not v.startswith("09"):
            raise ValueError("El teléfono debe empezar con 09")
        return v

    @field_validator("plan_type")
    @classmethod
    def validar_plan(cls, v: str) -> str:
        permitidos = {"basic", "standard", "premium"}
        if v not in permitidos:
            raise ValueError(f"plan_type debe ser uno de: {permitidos}")
        return v


class CustomerCreate(CustomerBase):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "name": "María Fernanda Rojas",
            "email": "maria.rojas@example.com",
            "phone": "0991234567",
            "plan_type": "premium",
            "monthly_charge": 55.0,
            "contract_type": "two_year",
            "payment_method": "credit_card",
        }
    })


class CustomerUpdate(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    plan_type: Optional[str] = None
    monthly_charge: Optional[float] = None
    contract_type: Optional[str] = None
    payment_method: Optional[str] = None

    @field_validator("phone")
    @classmethod
    def validar_telefono(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not v.isdigit() or len(v) < 10 or not v.startswith("09"):
            raise ValueError("Teléfono inválido: mínimo 10 dígitos, solo números, debe empezar con 09")
        return v


class CustomerResponse(CustomerBase):
    model_config = ConfigDict(from_attributes=True)

    customer_id: int
    tenure_months: int
    total_charges: float
    num_tickets: int
    avg_satisfaction: float
    is_active: bool
    created_at: datetime


class ChurnPredictionResponse(BaseModel):
    customer_id: int
    churn_probability: float
    risk_level: str

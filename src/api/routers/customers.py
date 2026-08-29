"""Endpoints de clientes: CRUD + predicción de churn."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.database import get_db
from src.api.models_db import Customer, Prediction, User
from src.api.auth import get_current_user, require_role
from src.api.schemas.customer import (
    CustomerCreate, CustomerUpdate, CustomerResponse, ChurnPredictionResponse,
)
from src.ml.churn_predictor import ChurnPredictor

router = APIRouter(prefix="/api/v1/customers", tags=["Clientes"])

_churn_predictor = None


def _get_churn_predictor() -> ChurnPredictor:
    global _churn_predictor
    if _churn_predictor is None:
        _churn_predictor = ChurnPredictor()
    return _churn_predictor


@router.get("", response_model=list[CustomerResponse], summary="Listar clientes activos")
def listar_clientes(
    skip: int = 0, limit: int = 50,
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
):
    return (
        db.query(Customer)
        .filter(Customer.is_active == True)  # noqa: E712
        .offset(skip).limit(limit).all()
    )


@router.get("/{customer_id}", response_model=CustomerResponse, summary="Obtener un cliente")
def obtener_cliente(
    customer_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
):
    customer = db.query(Customer).filter(
        Customer.customer_id == customer_id, Customer.is_active == True  # noqa: E712
    ).first()
    if not customer:
        raise HTTPException(status_code=404, detail=f"Cliente {customer_id} no encontrado")
    return customer


@router.post(
    "", response_model=CustomerResponse, status_code=status.HTTP_201_CREATED,
    summary="Crear un cliente",
)
def crear_cliente(
    data: CustomerCreate, db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin", "agent")),
):
    existente = db.query(Customer).filter(Customer.email == data.email).first()
    if existente:
        raise HTTPException(status_code=409, detail=f"Ya existe un cliente con el email {data.email}")

    customer = Customer(**data.model_dump())
    db.add(customer)
    db.commit()
    db.refresh(customer)
    return customer


@router.put("/{customer_id}", response_model=CustomerResponse, summary="Actualizar un cliente")
def actualizar_cliente(
    customer_id: int, data: CustomerUpdate, db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin", "agent")),
):
    customer = db.query(Customer).filter(
        Customer.customer_id == customer_id, Customer.is_active == True  # noqa: E712
    ).first()
    if not customer:
        raise HTTPException(status_code=404, detail=f"Cliente {customer_id} no encontrado")

    for campo, valor in data.model_dump(exclude_unset=True).items():
        setattr(customer, campo, valor)

    db.commit()
    db.refresh(customer)
    return customer


@router.delete("/{customer_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Eliminar un cliente (lógico)")
def eliminar_cliente(
    customer_id: int, db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    customer = db.query(Customer).filter(
        Customer.customer_id == customer_id, Customer.is_active == True  # noqa: E712
    ).first()
    if not customer:
        raise HTTPException(status_code=404, detail=f"Cliente {customer_id} no encontrado")

    customer.is_active = False   # eliminación LÓGICA, nunca db.delete()
    db.commit()


@router.get(
    "/{customer_id}/churn-prediction", response_model=ChurnPredictionResponse,
    summary="Predecir probabilidad de abandono de un cliente",
)
def predecir_churn_cliente(
    customer_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
):
    customer = db.query(Customer).filter(
        Customer.customer_id == customer_id, Customer.is_active == True  # noqa: E712
    ).first()
    if not customer:
        raise HTTPException(status_code=404, detail=f"Cliente {customer_id} no encontrado")

    predictor = _get_churn_predictor()
    resultado = predictor.predict({
        "monthly_charge": customer.monthly_charge,
        "tenure_months": customer.tenure_months,
        "total_charges": customer.total_charges,
        "num_tickets": customer.num_tickets,
        "avg_satisfaction": customer.avg_satisfaction,
        "plan_type": customer.plan_type,
        "contract_type": customer.contract_type,
        "payment_method": customer.payment_method,
    })

    # Registramos la predicción (trazabilidad -- tabla `predictions` del esquema)
    db.add(Prediction(
        customer_id=customer_id,
        churn_prob=resultado["churn_probability"],
        risk_level=resultado["risk_level"],
    ))
    db.commit()

    return ChurnPredictionResponse(customer_id=customer_id, **resultado)

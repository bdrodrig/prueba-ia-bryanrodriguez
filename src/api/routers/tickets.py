"""Endpoints de tickets: CRUD + clasificación automática."""
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.database import get_db
from src.api.models_db import Ticket, Customer, User
from src.api.auth import get_current_user
from src.api.schemas.ticket import (
    TicketCreate, TicketUpdate, TicketResponse,
    TicketClassifyRequest, TicketClassifyResponse,
)
from src.ml.ticket_classifier import TicketClassifier

router = APIRouter(prefix="/api/v1/tickets", tags=["Tickets"])

_ticket_classifier = None


def _get_classifier() -> TicketClassifier:
    global _ticket_classifier
    if _ticket_classifier is None:
        _ticket_classifier = TicketClassifier()
    return _ticket_classifier


@router.get("", response_model=list[TicketResponse], summary="Listar tickets activos")
def listar_tickets(
    skip: int = 0, limit: int = 50,
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
):
    return (
        db.query(Ticket)
        .filter(Ticket.is_active == True)  # noqa: E712
        .offset(skip).limit(limit).all()
    )


@router.get("/{ticket_id}", response_model=TicketResponse, summary="Obtener un ticket")
def obtener_ticket(
    ticket_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
):
    ticket = db.query(Ticket).filter(
        Ticket.ticket_id == ticket_id, Ticket.is_active == True  # noqa: E712
    ).first()
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} no encontrado")
    return ticket


@router.post(
    "", response_model=TicketResponse, status_code=status.HTTP_201_CREATED,
    summary="Crear un ticket",
    description="Si no se especifica `category`, se infiere automáticamente "
                "con el clasificador de la Parte 1.1.",
)
def crear_ticket(
    data: TicketCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
):
    cliente = db.query(Customer).filter(
        Customer.customer_id == data.customer_id, Customer.is_active == True  # noqa: E712
    ).first()
    if not cliente:
        raise HTTPException(status_code=404, detail=f"Cliente {data.customer_id} no encontrado")

    categoria = data.category
    if categoria is None:
        categoria = _get_classifier().predict(data.description)["category"]

    ticket = Ticket(
        customer_id=data.customer_id,
        category=categoria,
        description=data.description,
        priority=data.priority,
    )
    db.add(ticket)
    cliente.num_tickets = (cliente.num_tickets or 0) + 1
    db.commit()
    db.refresh(ticket)
    return ticket


@router.put("/{ticket_id}", response_model=TicketResponse, summary="Actualizar un ticket")
def actualizar_ticket(
    ticket_id: int, data: TicketUpdate, db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    ticket = db.query(Ticket).filter(
        Ticket.ticket_id == ticket_id, Ticket.is_active == True  # noqa: E712
    ).first()
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} no encontrado")

    for campo, valor in data.model_dump(exclude_unset=True).items():
        setattr(ticket, campo, valor)

    if data.status == "resolved" and ticket.resolved_at is None:
        ticket.resolved_at = datetime.utcnow()

    db.commit()
    db.refresh(ticket)
    return ticket


@router.post(
    "/classify", response_model=TicketClassifyResponse,
    summary="Clasificar la descripción de un ticket",
    description="No persiste nada en base de datos -- solo ejecuta el modelo "
                "de la Parte 1.1 y devuelve la categoría más probable.",
)
def clasificar_ticket(
    data: TicketClassifyRequest, current_user: User = Depends(get_current_user),
):
    try:
        return _get_classifier().predict(data.description)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

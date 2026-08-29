"""
Modelos ORM (tablas). Todas las tablas con eliminación tienen `is_active`
para eliminación lógica -- nunca se hace DELETE físico (requisito del
enunciado: "Eliminaciones: Siempre lógicas").
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text,
)
from sqlalchemy.orm import relationship
from src.api.database import Base


class Customer(Base):
    __tablename__ = "customers"

    customer_id = Column(Integer, primary_key=True, index=True)
    name = Column(String(120), nullable=False)
    email = Column(String(120), unique=True, nullable=False, index=True)
    phone = Column(String(15), nullable=False)
    plan_type = Column(String(20), nullable=False)
    monthly_charge = Column(Float, nullable=False)
    tenure_months = Column(Integer, default=0)
    total_charges = Column(Float, default=0.0)
    contract_type = Column(String(20), nullable=False)
    payment_method = Column(String(20), nullable=False)
    num_tickets = Column(Integer, default=0)
    avg_satisfaction = Column(Float, default=3.0)
    is_active = Column(Boolean, default=True)   # <- eliminación lógica
    created_at = Column(DateTime, default=datetime.utcnow)

    tickets = relationship("Ticket", back_populates="customer")


class Ticket(Base):
    __tablename__ = "tickets"

    ticket_id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.customer_id"), nullable=False)
    category = Column(String(10), nullable=False)   # TECH, BILL, PLAN, CNCL, OTHR
    description = Column(Text, nullable=False)
    priority = Column(String(10), default="medium")
    status = Column(String(20), default="open")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)
    satisfaction = Column(Float, nullable=True)

    customer = relationship("Customer", back_populates="tickets")
    interactions = relationship("Interaction", back_populates="ticket")


class Interaction(Base):
    __tablename__ = "interactions"

    interaction_id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(Integer, ForeignKey("tickets.ticket_id"), nullable=False)
    agent_response = Column(Text, nullable=True)
    customer_msg = Column(Text, nullable=False)
    sentiment = Column(String(15), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    resolution_time = Column(Float, nullable=True)

    ticket = relationship("Ticket", back_populates="interactions")


class Prediction(Base):
    __tablename__ = "predictions"

    prediction_id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.customer_id"), nullable=False)
    churn_prob = Column(Float, nullable=False)
    risk_level = Column(String(10), nullable=False)
    model_version = Column(String(20), default="v1")
    created_at = Column(DateTime, default=datetime.utcnow)


class AgentSession(Base):
    __tablename__ = "agent_sessions"

    session_id = Column(String(50), primary_key=True, index=True)
    customer_id = Column(Integer, ForeignKey("customers.customer_id"), nullable=True)
    conversation = Column(Text, default="[]")   # JSON serializado del historial
    tokens_used = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)


class User(Base):
    """Usuarios de la API (para autenticación) -- separado de Customer,
    que representa a los clientes de telecomunicaciones."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    hashed_password = Column(String(200), nullable=False)
    role = Column(String(20), default="customer")   # admin, agent, customer
    is_active = Column(Boolean, default=True)

"""
Parte 4.2 - Protocolo MCP (Model Context Protocol).

Expone las mismas capacidades del sistema (clasificar tickets, predecir churn,
consultar clientes, crear tickets, chatear con el agente) pero en el formato
que otros agentes de IA esperan: JSON-RPC 2.0.
"""
from typing import Any, Optional
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.api.database import get_db
from src.api.models_db import Customer, Ticket
from src.ml.ticket_classifier import TicketClassifier
from src.ml.churn_predictor import ChurnPredictor
from src.agent.graph import get_agent_instance, SIMULATED_CUSTOMERS

router = APIRouter(prefix="/mcp", tags=["MCP"])

_ticket_classifier = None
_churn_predictor = None


def _get_ticket_classifier() -> TicketClassifier:
    global _ticket_classifier
    if _ticket_classifier is None:
        _ticket_classifier = TicketClassifier()
    return _ticket_classifier


def _get_churn_predictor() -> ChurnPredictor:
    global _churn_predictor
    if _churn_predictor is None:
        _churn_predictor = ChurnPredictor()
    return _churn_predictor


# ---------------------------------------------------------------------------
# Definición de las herramientas (tools) que expone este servidor MCP
# ---------------------------------------------------------------------------

TOOLS_SCHEMA = [
    {
        "name": "predict_churn",
        "description": "Predice la probabilidad de abandono de un cliente a partir de sus datos.",
        "input_schema": {
            "type": "object",
            "properties": {
                "monthly_charge": {"type": "number"}, "tenure_months": {"type": "integer"},
                "total_charges": {"type": "number"}, "num_tickets": {"type": "integer"},
                "avg_satisfaction": {"type": "number"}, "plan_type": {"type": "string"},
                "contract_type": {"type": "string"}, "payment_method": {"type": "string"},
            },
            "required": ["monthly_charge", "tenure_months", "plan_type", "contract_type", "payment_method"],
        },
    },
    {
        "name": "classify_ticket",
        "description": "Clasifica la descripción de un ticket de soporte en una de 5 categorías.",
        "input_schema": {
            "type": "object",
            "properties": {"description": {"type": "string"}},
            "required": ["description"],
        },
    },
    {
        "name": "get_customer_info",
        "description": "Obtiene la información de un cliente por su customer_id.",
        "input_schema": {
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"],
        },
    },
    {
        "name": "create_ticket",
        "description": "Crea un nuevo ticket de soporte para un cliente existente.",
        "input_schema": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "integer"}, "description": {"type": "string"},
                "priority": {"type": "string"},
            },
            "required": ["customer_id", "description"],
        },
    },
    {
        "name": "chat_with_agent",
        "description": "Envía un mensaje al agente conversacional y obtiene su respuesta.",
        "input_schema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"}, "message": {"type": "string"},
                "customer_id": {"type": "string"},
            },
            "required": ["session_id", "message"],
        },
    },
]


# ---------------------------------------------------------------------------
# Modelos de request/response en formato JSON-RPC 2.0
# ---------------------------------------------------------------------------

class MCPToolExecuteRequest(BaseModel):
    id: Optional[str] = None
    tool: str
    arguments: dict = {}


def _mcp_result(request_id: Optional[str], content: Any, is_error: bool = False) -> dict:
    """Envuelve cualquier resultado en el formato JSON-RPC 2.0 que pide el enunciado."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"content": [content] if not isinstance(content, list) else content, "isError": is_error},
    }


# ---------------------------------------------------------------------------
# Endpoints MCP
# ---------------------------------------------------------------------------

@router.get("/capabilities", summary="Lista las capacidades del servidor MCP")
def get_capabilities():
    return {
        "jsonrpc": "2.0",
        "result": {
            "server": "sistema-atencion-cliente-mcp",
            "version": "1.0.0",
            "tools": TOOLS_SCHEMA,
        },
    }


@router.post("/tools/execute", summary="Ejecuta una herramienta MCP")
def execute_tool(payload: MCPToolExecuteRequest, db: Session = Depends(get_db)):
    tool = payload.tool
    args = payload.arguments

    try:
        if tool == "predict_churn":
            resultado = _get_churn_predictor().predict(args)
            return _mcp_result(payload.id, resultado)

        elif tool == "classify_ticket":
            resultado = _get_ticket_classifier().predict(args["description"])
            return _mcp_result(payload.id, resultado)

        elif tool == "get_customer_info":
            customer_id = args["customer_id"]
            info = SIMULATED_CUSTOMERS.get(customer_id)
            if info is None:
                return _mcp_result(payload.id, {"error": f"Cliente {customer_id} no encontrado"}, is_error=True)
            return _mcp_result(payload.id, info)

        elif tool == "create_ticket":
            cliente = db.query(Customer).filter(
                Customer.customer_id == args["customer_id"], Customer.is_active == True  # noqa: E712
            ).first()
            if not cliente:
                return _mcp_result(payload.id, {"error": "Cliente no encontrado"}, is_error=True)

            categoria = _get_ticket_classifier().predict(args["description"])["category"]
            ticket = Ticket(
                customer_id=args["customer_id"], description=args["description"],
                category=categoria, priority=args.get("priority", "medium"),
            )
            db.add(ticket)
            db.commit()
            db.refresh(ticket)
            return _mcp_result(payload.id, {"ticket_id": ticket.ticket_id, "category": categoria})

        elif tool == "chat_with_agent":
            agent = get_agent_instance()
            resultado = agent.chat(args["session_id"], args["message"], args.get("customer_id"))
            return _mcp_result(payload.id, resultado)

        else:
            return _mcp_result(payload.id, {"error": f"Herramienta '{tool}' no reconocida"}, is_error=True)

    except (KeyError, ValueError) as e:
        return _mcp_result(payload.id, {"error": f"Argumentos inválidos: {e}"}, is_error=True)


@router.get("/resources", summary="Lista los recursos disponibles")
def list_resources():
    return {
        "jsonrpc": "2.0",
        "result": {
            "resources": [
                {"id": "ticket_categories", "name": "Categorías de tickets", "type": "static"},
                {"id": "customer:{id}", "name": "Información de un cliente (dinámico)", "type": "dynamic"},
            ]
        },
    }


@router.get("/resources/{resource_id}", summary="Obtiene un recurso específico")
def get_resource(resource_id: str):
    if resource_id == "ticket_categories":
        return _mcp_result(None, {
            "TECH": "Problemas técnicos", "BILL": "Consultas de facturación",
            "PLAN": "Cambio de plan o servicios", "CNCL": "Cancelación de servicio", "OTHR": "Otros",
        })

    if resource_id.startswith("customer:"):
        customer_id = resource_id.split(":", 1)[1]
        info = SIMULATED_CUSTOMERS.get(customer_id)
        if info is None:
            return _mcp_result(None, {"error": f"Cliente {customer_id} no encontrado"}, is_error=True)
        return _mcp_result(None, info)

    return _mcp_result(None, {"error": f"Recurso '{resource_id}' no encontrado"}, is_error=True)

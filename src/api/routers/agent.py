"""Endpoints del agente conversacional (Parte 3, integrado vía LangGraph)."""
import json
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.api.database import get_db
from src.api.models_db import AgentSession, User
from src.api.auth import get_current_user
from src.api.schemas.ml import AgentChatRequest, AgentChatResponse
from src.agent.graph import get_agent_instance

router = APIRouter(prefix="/api/v1/agent", tags=["Agente conversacional"])


@router.post("/chat", response_model=AgentChatResponse, summary="Enviar un mensaje al agente")
def chat(
    data: AgentChatRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
):
    agent = get_agent_instance()
    resultado = agent.chat(data.session_id, data.message, str(data.customer_id) if data.customer_id else None)

    # Persistimos la sesión en BD (además de la memoria en RAM del objeto agent)
    session_row = db.query(AgentSession).filter(AgentSession.session_id == data.session_id).first()
    conversacion = agent.sessions[data.session_id]["messages"]
    if session_row is None:
        session_row = AgentSession(
            session_id=data.session_id,
            customer_id=data.customer_id,
            conversation=json.dumps(conversacion, ensure_ascii=False),
        )
        db.add(session_row)
    else:
        session_row.conversation = json.dumps(conversacion, ensure_ascii=False)
    db.commit()

    return AgentChatResponse(
        response=resultado["response"], intent=resultado["intent"], escalate=resultado["escalate"],
    )


@router.get("/sessions/{session_id}", summary="Consultar el historial de una sesión")
def obtener_sesion(
    session_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
):
    session_row = db.query(AgentSession).filter(AgentSession.session_id == session_id).first()
    if not session_row:
        raise HTTPException(status_code=404, detail=f"Sesión {session_id} no encontrada")
    return {
        "session_id": session_row.session_id,
        "customer_id": session_row.customer_id,
        "conversation": json.loads(session_row.conversation),
        "is_active": session_row.is_active,
        "started_at": session_row.started_at,
        "ended_at": session_row.ended_at,
    }


@router.delete("/sessions/{session_id}", summary="Cerrar una sesión del agente")
def cerrar_sesion(
    session_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
):
    session_row = db.query(AgentSession).filter(AgentSession.session_id == session_id).first()
    if not session_row:
        raise HTTPException(status_code=404, detail=f"Sesión {session_id} no encontrada")

    session_row.is_active = False
    session_row.ended_at = datetime.utcnow()
    db.commit()

    agent = get_agent_instance()
    agent.sessions.pop(session_id, None)   # libera también la memoria en RAM

    return {"detail": f"Sesión {session_id} cerrada correctamente"}

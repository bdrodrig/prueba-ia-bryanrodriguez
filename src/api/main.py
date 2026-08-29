"""
Punto de entrada de la API. Arma la aplicación FastAPI, registra todos los
routers, configura el manejo de errores estandarizado y crea la base de
datos + usuarios de prueba al arrancar.
"""
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.database import Base, engine, SessionLocal
from src.api.models_db import User
from src.api.auth import hash_password
from src.api.routers import auth, customers, tickets, ml, agent
from src.mcp.server import router as mcp_router

app = FastAPI(
    title="Sistema Inteligente de Atención al Cliente",
    description=(
        "API para una empresa de telecomunicaciones: clasificación de tickets, "
        "predicción de churn, agente conversacional (LangGraph) y protocolo MCP. "
        "Prueba técnica -- Desarrollador de IA/ML."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Manejo de errores estandarizado (requisito del enunciado)
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Errores de validación de Pydantic -> respuesta consistente y legible."""
    errores = [
        {"campo": ".".join(str(x) for x in err["loc"][1:]), "mensaje": err["msg"]}
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Error de validación", "detalles": errores},
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Unifica el formato de cualquier HTTPException lanzada en los routers."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Red de seguridad: cualquier error no controlado devuelve 500 estandarizado
    en vez de un traceback crudo al cliente."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Error interno del servidor", "detalle": str(exc)},
    )


# ---------------------------------------------------------------------------
# Registro de routers
# ---------------------------------------------------------------------------

app.include_router(auth.router)
app.include_router(customers.router)
app.include_router(tickets.router)
app.include_router(ml.router)
app.include_router(agent.router)
app.include_router(mcp_router)


# ---------------------------------------------------------------------------
# Arranque: crear tablas + usuarios de prueba
# ---------------------------------------------------------------------------

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            usuarios_demo = [
                ("admin1", "admin123", "admin"),
                ("agente1", "agente123", "agent"),
                ("cliente1", "cliente123", "customer"),
            ]
            for username, password, role in usuarios_demo:
                db.add(User(username=username, hashed_password=hash_password(password), role=role))
            db.commit()
            print("Usuarios de prueba creados: admin1/admin123, agente1/agente123, cliente1/cliente123")
    finally:
        db.close()


@app.get("/", tags=["Root"], summary="Health check")
def root():
    return {"status": "ok", "service": "Sistema Inteligente de Atención al Cliente"}

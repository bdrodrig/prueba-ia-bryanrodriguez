"""Endpoints de autenticación: POST /login, POST /refresh."""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from src.api.database import get_db
from src.api.models_db import User
from src.api.schemas.ml import TokenResponse, RefreshRequest
from src.api.auth import (
    verify_password, create_access_token, create_refresh_token, decode_token,
)

router = APIRouter(prefix="/api/v1/auth", tags=["Autenticación"])


@router.post(
    "/login", response_model=TokenResponse,
    summary="Iniciar sesión",
    description="Autentica con username/password (form-data, estándar OAuth2) y "
                "devuelve un access token (30 min) y un refresh token (7 días).",
)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username, User.is_active == True).first()  # noqa: E712
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Usuario o contraseña incorrectos",
        )
    return TokenResponse(
        access_token=create_access_token(user.username, user.role),
        refresh_token=create_refresh_token(user.username),
    )


@router.post(
    "/refresh", response_model=TokenResponse,
    summary="Renovar access token",
    description="Recibe un refresh token válido y devuelve un nuevo par de tokens, "
                "sin que el usuario tenga que volver a ingresar su contraseña.",
)
def refresh(payload: RefreshRequest, db: Session = Depends(get_db)):
    data = decode_token(payload.refresh_token)
    if data.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Se requiere un refresh token válido")

    user = db.query(User).filter(User.username == data.get("sub"), User.is_active == True).first()  # noqa: E712
    if not user:
        raise HTTPException(status_code=401, detail="Usuario no encontrado o inactivo")

    return TokenResponse(
        access_token=create_access_token(user.username, user.role),
        refresh_token=create_refresh_token(user.username),
    )

"""
Configuración de la base de datos. Usa SQLite para desarrollo (por defecto),
pero al fijar DATABASE_URL vía variable de entorno puedes apuntar a
PostgreSQL sin cambiar una sola línea del resto del código -- esa es
la ventaja de pasar por un ORM (SQLAlchemy) en vez de SQL crudo.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./customer_service.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependency de FastAPI: abre una sesión por request y la cierra al final."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

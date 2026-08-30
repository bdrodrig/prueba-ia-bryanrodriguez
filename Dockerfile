# Python 3.11 (no 3.12+/3.14) por la misma razón que en el entorno local:
# TensorFlow todavía no publica wheels estables para versiones más nuevas.
FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema necesarias para compilar psycopg2 y bcrypt
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
# Sistema Inteligente de Atención al Cliente — Prueba Técnica IA/ML

**Nombre completo:** Bryan Rodríguez

## Tiempo dedicado a cada sección

| Sección | Tiempo dedicado |
|---|---|
| Setup inicial del proyecto | 40 min |
| Parte 1 — ML con scikit-learn | 1 hora |
| Parte 2 — Deep Learning con TensorFlow/Keras | 1 hora |
| Parte 3 — Agente conversacional con LangGraph | 1 hora |
| Parte 4 — API REST con FastAPI y MCP | 1 hora |
| Docker, SQL, documentación | 30 min |
| **Total** | 5 horas, 10 min |

## Decisiones técnicas tomadas y justificación

### Parte 1 — ML
- **Comparación de 2 modelos** (Naive Bayes vs Regresión Logística) para el clasificador de tickets, elegidos por ser rápidos y soportar `predict_proba`.
- **`StratifiedKFold`** en vez de `KFold` simple, para mantener la proporción de clases en cada fold — crítico en el dataset de churn, desbalanceado.
- **`RandomForestClassifier`** para churn, por dar `feature_importances_` nativamente sin pasos adicionales.
- Los modelos de entrenamiento (`scripts/train_*.py`) están **separados** de los módulos de lógica (`src/ml/*.py`) para evitar un bug de `pickling` de `joblib` con funciones definidas en `__main__` (ver Dificultades).

### Parte 2 — Deep Learning
- **`mask_zero=True`** en la capa `Embedding` del modelo de sentimiento — sin esto, con `MAX_LEN=200` y frases reales de 5-13 palabras, el padding domina la secuencia y el modelo no aprende nada (ver Dificultades).
- El modelo de tiempo de resolución usa la **Functional API de Keras** (no `Sequential`) para combinar una rama de texto (Embedding + pooling) con una rama de datos estructurados (categoría, prioridad, hora, día) — así se cumple el requisito de "inputs mixtos" de forma nativa en la arquitectura, no concatenando vectores a mano.

### Parte 3 — Agente con LangGraph
- El agente **no depende de un LLM externo** (no requiere API key de pago) — usa lógica determinística combinada con los 3 modelos entrenados (clasificador de tickets, predictor de churn, clasificador de sentimiento). Esto lo hace reproducible y gratuito de ejecutar, cumpliendo igualmente el requisito de integración con los modelos ML/DL.
- Los 3 modelos se cargan con un patrón de **carga perezosa (lazy loading)** (`_get_models()`) — no se cargan al importar el módulo, sino la primera vez que se usan, evitando cargar TensorFlow/sklearn innecesariamente.
- Se implementó un **singleton compartido** (`get_agent_instance()`) para que la misma instancia del agente (y sus sesiones en memoria) sea usada tanto por la API REST (`/api/v1/agent/chat`) como por el protocolo MCP (`chat_with_agent`) — sin esto, una conversación no podría continuar de un canal al otro.

### Parte 4 — FastAPI y MCP
- **Arquitectura por capas**: `routers/` (HTTP) → `schemas/` (validación Pydantic) → `models_db.py` (ORM) → `database.py` (conexión), sin mezclar responsabilidades.
- **Eliminación siempre lógica** (`is_active = False`), nunca `DELETE` físico, en clientes y tickets.
- **JWT con access token (30 min) + refresh token (7 días)** y 3 roles (`admin`, `agent`, `customer`) vía `require_role()`.
- **SQLite para desarrollo** (cero configuración) y **PostgreSQL para Docker/producción**, controlado por una sola variable de entorno (`DATABASE_URL`) — el resto del código no cambia, gracias a usar SQLAlchemy como ORM.
- El **stored procedure** (`sp_close_ticket`) vive en PostgreSQL (SQLite no soporta procedimientos reales) y resuelve una operación atómica: cerrar un ticket y recalcular el promedio de satisfacción del cliente sobre todos sus tickets resueltos, en un solo paso.
- Manejo de errores centralizado con `@app.exception_handler` para devolver siempre el mismo formato de error (validación, HTTP, y errores no controlados).

## Dificultades encontradas

Estas son las dificultades técnicas reales que surgieron durante el desarrollo (con su causa raíz):

1. **`ModuleNotFoundError: No module named 'src'`** al ejecutar scripts de entrenamiento — causado por ejecutar el archivo directamente en vez de como módulo (`python -m scripts.train_x` en vez de `python scripts/train_x.py`), necesario para que los imports `from src...` funcionen.
2. **Bug de pickling con `joblib`**: al entrenar un modelo desde un archivo ejecutado como `__main__`, las funciones personalizadas (como el preprocesador de texto) se guardan como pertenecientes a `__main__`, y no se pueden cargar desde otro script. Solución: separar el script de entrenamiento (`scripts/`) de la lógica reutilizable (`src/`).
3. **`UnicodeDecodeError` al leer los CSV**: causado por abrir/guardar los archivos con Excel, que en configuración regional en español re-guarda el CSV en otra codificación (y a veces con `;` como separador en vez de `,`). Solución: nunca editar los CSV con Excel, solo con editor de texto plano.
4. **El modelo de sentimiento no aprendía** (accuracy estancado en 33%, equivalente a azar en 3 clases): causado por usar `MAX_LEN=200` de padding con frases reales de solo 5-13 palabras, sin decirle a la LSTM que ignore el padding. Solución: `mask_zero=True` en la capa `Embedding`.
5. **Conflicto de dependencias `numpy`/`tensorflow`**: `tensorflow==2.18.0` exige `numpy<2.1.0`, pero se había fijado `numpy==2.1.3` en `requirements.txt` sin verificar la compatibilidad conjunta. Solución: bajar a `numpy==2.0.2` y validar el árbol completo de dependencias con `pip install --dry-run` antes de fijar versiones.
6. **`AttributeError: 'LogisticRegression' object has no attribute 'multi_class'`**: un modelo `.joblib` fue entrenado con una versión de scikit-learn distinta a la instalada al momento de cargarlo. Solución: reentrenar los modelos con el entorno final ya fijado en `requirements.txt`, y nunca reinstalar dependencias sin reentrenar los modelos que dependen de ellas.
7. **Conflicto `passlib`/`bcrypt`**: `passlib==1.7.4` no reconoce la API de `bcrypt>=5.0`. Solución: fijar `bcrypt==4.0.1` explícitamente en `requirements.txt`.
8. **Python 3.14 incompatible con TensorFlow**: TensorFlow aún no publica wheels para versiones muy recientes de Python. Solución: usar Python 3.11 en un entorno virtual dedicado para este proyecto, sin afectar la instalación global.

## Instrucciones para ejecutar el proyecto

### Requisitos previos
- Python 3.11 (NO 3.12+/3.14 por incompatibilidad con TensorFlow)
- Docker Desktop (para la opción con PostgreSQL)

### 1. Entorno virtual e instalación
```bash
py -3.11 -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Entrenar todos los modelos
```bash
python -m scripts.train_ticket_classifier
python -m scripts.train_churn_predictor
python -m scripts.train_sentiment_model
python -m scripts.train_resolution_time_model
```

### 3. Probar el agente conversacional de forma aislada
```bash
python -m src.agent.graph
```

### 4. Levantar la API (opción A: local con SQLite)
```bash
uvicorn src.api.main:app --reload
```
Abre `http://127.0.0.1:8000/docs` para la documentación interactiva (Swagger UI).

### 5. Levantar todo con Docker (opción B: con PostgreSQL)
```bash
docker-compose up
```
Esto levanta la API (`http://localhost:8000`) y PostgreSQL, ejecutando
`sql/init_postgress.sql` automáticamente la primera vez.

### 6. Correr los tests
```bash
pytest tests/ -v
```

## Credenciales de prueba

Se crean automáticamente al arrancar la API por primera vez:

| Usuario | Contraseña | Rol |
|---|---|---|
| `admin1` | `admin123` | admin |
| `agente1` | `agente123` | agent |
| `cliente1` | `cliente123` | customer |

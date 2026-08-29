"""
Punto de entrada para entrenar los modelos. Se ejecuta como script,
pero IMPORTA la lógica desde src/ -- así joblib guarda las referencias
de función correctamente (evita el bug de pickling con __main__).

Uso: python scripts/train_ticket_classifier.py
"""
from src.ml.ticket_classifier import entrenar_y_evaluar

if __name__ == "__main__":
    entrenar_y_evaluar()
"""
Parte 1.1 - Clasificador de tickets de soporte.

"""
import re
import unicodedata
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

MODEL_PATH = "models/ticket_classifier.joblib"
MIN_TEXT_LENGTH = 10


def clean_text(text: str) -> str:
    """
    Limpieza de texto en español. 
    """
    if not isinstance(text, str):
        raise ValueError("El texto de entrada debe ser una cadena de texto")

    text = text.strip()
    if len(text) < MIN_TEXT_LENGTH:
        raise ValueError(
            f"El texto debe tener al menos {MIN_TEXT_LENGTH} caracteres "
            f"(recibido: {len(text)})"
        )

    text = text.lower()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^\w\sáéíóúñü]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_pipeline(model) -> Pipeline:
    """Pipeline completo: TF-IDF -> modelo. Encapsula todo el preprocesamiento
    para que el mismo objeto se use en train y en predicción (evita
    desincronización entre cómo se entrenó y cómo se predice)."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,
            ngram_range=(1, 2),   
            min_df=2,
            max_features=3000,
        )),
        ("clf", model),
    ])


def entrenar_y_evaluar(csv_path: str = "data/tickets_train.csv"):
    df = pd.read_csv(csv_path)

    X = df["description"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    candidatos = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    resultados = {}

    print("=" * 60)
    print("COMPARACIÓN DE MODELOS (validación cruzada, 5 folds)")
    print("=" * 60)

    for nombre, modelo in candidatos.items():
        pipeline = build_pipeline(modelo)
        scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring="f1_macro")
        resultados[nombre] = scores
        print(f"{nombre}: f1_macro = {scores.mean():.4f} (+/- {scores.std():.4f})")

    mejor_nombre = max(resultados, key=lambda k: resultados[k].mean())
    print(f"\nMejor modelo: {mejor_nombre}")

    # Reentrenamos el mejor modelo con todo el set de entrenamiento
    mejor_pipeline = build_pipeline(candidatos[mejor_nombre])
    mejor_pipeline.fit(X_train, y_train)

    y_pred = mejor_pipeline.predict(X_test)

    print("\n" + "=" * 60)
    print(f"EVALUACIÓN FINAL EN TEST SET ({mejor_nombre})")
    print("=" * 60)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1-macro: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print("\nReporte por categoría:")
    print(classification_report(y_test, y_pred))

    print("Matriz de confusión:")
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)

    joblib.dump(mejor_pipeline, MODEL_PATH)
    print(f"\nModelo guardado en: {MODEL_PATH}")

    return mejor_pipeline, cm_df


class TicketClassifier:
    """Wrapper de inferencia -- esto es lo que importará la API y el agente."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.pipeline = joblib.load(model_path)

    def predict(self, text: str) -> dict:
        clean_text(text)  # valida longitud mínima y lanza ValueError si no cumple
        categoria = self.pipeline.predict([text])[0]
        probabilidades = self.pipeline.predict_proba([text])[0]
        clases = self.pipeline.classes_
        prob_por_clase = {
            clase: round(float(prob), 4)
            for clase, prob in zip(clases, probabilidades)
        }
        return {
            "category": categoria,
            "probabilities": prob_por_clase,
        }


if __name__ == "__main__":
    entrenar_y_evaluar()
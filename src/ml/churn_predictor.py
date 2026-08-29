"""
Parte 1.2 - Predicción de churn de clientes.

Expone:
- entrenar_y_evaluar(): EDA, feature engineering, entrena, evalúa, guarda.
- ChurnPredictor: clase reutilizable para predecir en producción.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    classification_report,
)

MODEL_PATH = "models/churn_predictor.joblib"

NUMERIC_FEATURES = [
    "monthly_charge", "tenure_months", "total_charges",
    "num_tickets", "avg_satisfaction",
    "tickets_per_month", "is_new_customer",   # <- features derivados
]
CATEGORICAL_FEATURES = ["plan_type", "contract_type", "payment_method"]


def analisis_exploratorio(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("ANÁLISIS EXPLORATORIO")
    print("=" * 60)
    print(f"Total de clientes: {len(df)}")
    print(f"\nDistribución de churn:")
    print(df["churn_status"].value_counts(normalize=True).round(3))
    print(f"\nValores nulos por columna:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print(f"\nChurn promedio por tipo de contrato:")
    print(df.groupby("contract_type")["churn_status"].mean().round(3))
    print(f"\nCorrelación de variables numéricas con churn:")
    corr = df[["monthly_charge", "tenure_months", "total_charges",
               "num_tickets", "avg_satisfaction", "churn_status"]].corr()["churn_status"]
    print(corr.round(3).sort_values(ascending=False))


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea 2 features derivados:
    - tickets_per_month: intensidad de quejas relativa a la antigüedad
      (un cliente con 3 tickets en 2 meses es una señal MUY distinta
      a 3 tickets en 5 años).
    - is_new_customer: bandera de riesgo temprano (los clientes nuevos
      suelen tener más probabilidad de cancelar -- "buyer's remorse").
    """
    df = df.copy()
    df["tickets_per_month"] = df["num_tickets"] / (df["tenure_months"] + 1)
    df["is_new_customer"] = (df["tenure_months"] < 6).astype(int)
    return df


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    modelo = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",   # <- maneja el desbalanceo (22% churn) sin necesitar SMOTE
        random_state=42,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("clf", modelo),
    ])


def entrenar_y_evaluar(csv_path: str = "data/customers.csv"):
    df = pd.read_csv(csv_path)

    analisis_exploratorio(df)

    df = feature_engineering(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["churn_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    print("\n" + "=" * 60)
    print("EVALUACIÓN DEL MODELO DE CHURN")
    print("=" * 60)

    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Average Precision (área bajo precision-recall): {ap:.4f}")
    print("\nReporte de clasificación (umbral 0.5):")
    print(classification_report(y_test, y_pred))

    # Puntos clave de la curva ROC y precision-recall (para graficarlos si se desea)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    print("\n" + "=" * 60)
    print("EXPLICABILIDAD: FEATURE IMPORTANCE")
    print("=" * 60)
    feature_names = (
        NUMERIC_FEATURES
        + list(pipeline.named_steps["preprocessor"]
               .named_transformers_["cat"]
               .named_steps["onehot"]
               .get_feature_names_out(CATEGORICAL_FEATURES))
    )
    importances = pipeline.named_steps["clf"].feature_importances_
    importancia_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)
    print(importancia_df.to_string(index=False))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModelo (con preprocesador incluido) guardado en: {MODEL_PATH}")

    return pipeline, {"auc": auc, "average_precision": ap,
                       "roc_curve": (fpr, tpr), "pr_curve": (precision, recall),
                       "feature_importance": importancia_df}


class ChurnPredictor:
    """Wrapper de inferencia para la API y el agente."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.pipeline = joblib.load(model_path)

    def predict(self, customer: dict) -> dict:
        df = pd.DataFrame([customer])
        df = feature_engineering(df)
        df = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

        churn_prob = float(self.pipeline.predict_proba(df)[0, 1])
        if churn_prob >= 0.7:
            risk_level = "alto"
        elif churn_prob >= 0.4:
            risk_level = "medio"
        else:
            risk_level = "bajo"

        return {"churn_probability": round(churn_prob, 4), "risk_level": risk_level}


if __name__ == "__main__":
    entrenar_y_evaluar()
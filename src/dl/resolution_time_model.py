"""
Parte 2.2 - Predicción de tiempo de resolución de tickets (regresión).
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

VOCAB_SIZE = 2000
TEXT_MAX_LEN = 30   # las descripciones son cortas -- distinto al MAX_LEN=200 de sentimiento
MODEL_PATH = "models/resolution_time_model.keras"
TOKENIZER_PATH = "models/resolution_tokenizer.pkl"
PREPROCESSOR_PATH = "models/resolution_preprocessor.pkl"

NUMERIC_FEATURES = ["created_hour"]
CATEGORICAL_FEATURES = ["category", "priority", "created_dow"]


def build_structured_preprocessor() -> ColumnTransformer:
    """One-hot para categoría/prioridad/día, escalado para la hora."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def build_model(n_structured_features: int) -> keras.Model:
    text_input = keras.Input(shape=(TEXT_MAX_LEN,), name="text_input")
    x_text = keras.layers.Embedding(VOCAB_SIZE, 32, mask_zero=True)(text_input)
    x_text = keras.layers.GlobalAveragePooling1D()(x_text)
    x_text = keras.layers.Dense(16, activation="relu")(x_text)

    # Rama estructurada
    structured_input = keras.Input(shape=(n_structured_features,), name="structured_input")
    x_struct = keras.layers.Dense(32, activation="relu")(structured_input)
    x_struct = keras.layers.Dense(16, activation="relu")(x_struct)

    # Combinamos ambas ramas
    combined = keras.layers.Concatenate()([x_text, x_struct])
    x = keras.layers.Dense(32, activation="relu")(combined)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    output = keras.layers.Dense(1, activation="linear", name="resolution_time")(x)

    model = keras.Model(inputs=[text_input, structured_input], outputs=output)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def entrenar_y_evaluar(csv_path: str = "data/tickets_detail.csv"):
    df = pd.read_csv(csv_path)

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        df, df["resolution_time_hours"], test_size=0.2, random_state=42
    )

    # --- Preprocesamiento de texto ---
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_df["description"])
    X_train_text = pad_sequences(
        tokenizer.texts_to_sequences(X_train_df["description"]),
        maxlen=TEXT_MAX_LEN, padding="post", truncating="post"
    )
    X_test_text = pad_sequences(
        tokenizer.texts_to_sequences(X_test_df["description"]),
        maxlen=TEXT_MAX_LEN, padding="post", truncating="post"
    )

    # --- Preprocesamiento de estructurados ---
    preprocessor = build_structured_preprocessor()
    X_train_struct = preprocessor.fit_transform(X_train_df).astype("float32")
    X_test_struct = preprocessor.transform(X_test_df).astype("float32")
    if hasattr(X_train_struct, "toarray"):
        X_train_struct = X_train_struct.toarray()
        X_test_struct = X_test_struct.toarray()

    model = build_model(n_structured_features=X_train_struct.shape[1])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]

    history = model.fit(
        {"text_input": X_train_text, "structured_input": X_train_struct},
        y_train,
        validation_split=0.2,
        epochs=60,
        batch_size=32,
        callbacks=callbacks,
        verbose=2,
    )

    y_pred = model.predict({"text_input": X_test_text, "structured_input": X_test_struct}).flatten()

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("EVALUACIÓN DEL MODELO DE TIEMPO DE RESOLUCIÓN")
    print("=" * 60)
    print(f"MAE:  {mae:.3f} horas (error promedio absoluto)")
    print(f"RMSE: {rmse:.3f} horas (penaliza más los errores grandes)")
    print(f"R²:   {r2:.3f} (proporción de varianza explicada, 1.0 = perfecto)")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(history.history["loss"], label="train")
    ax.plot(history.history["val_loss"], label="validation")
    ax.set_title("MSE loss por epoch")
    ax.set_xlabel("Epoch")
    ax.legend()
    plt.tight_layout()
    plt.savefig("models/resolution_time_training_curve.png")
    plt.close()

    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(PREPROCESSOR_PATH, "wb") as f:
        pickle.dump(preprocessor, f)
    model.save(MODEL_PATH)
    print(f"\nModelo guardado en: {MODEL_PATH}")

    return model, {"mae": mae, "rmse": rmse, "r2": r2}


class ResolutionTimePredictor:
    """Wrapper de inferencia para la API y el agente."""

    def __init__(self):
        self.model = keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            self.tokenizer = pickle.load(f)
        with open(PREPROCESSOR_PATH, "rb") as f:
            self.preprocessor = pickle.load(f)

    def predict(self, ticket: dict) -> dict:
        df = pd.DataFrame([ticket])
        text_seq = pad_sequences(
            self.tokenizer.texts_to_sequences(df["description"]),
            maxlen=TEXT_MAX_LEN, padding="post", truncating="post"
        )
        struct = self.preprocessor.transform(df).astype("float32")
        if hasattr(struct, "toarray"):
            struct = struct.toarray()
        pred_hours = float(self.model.predict(
            {"text_input": text_seq, "structured_input": struct}, verbose=0
        )[0, 0])
        return {"estimated_resolution_hours": round(max(pred_hours, 0.1), 2)}


if __name__ == "__main__":
    entrenar_y_evaluar()
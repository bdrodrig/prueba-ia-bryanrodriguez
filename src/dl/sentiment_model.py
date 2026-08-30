"""
Parte 2.1 - Red neuronal para clasificación de sentimiento
(interacciones cliente-agente: positivo, neutral, negativo).
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin interfaz gráfica (necesario para correr desde terminal/Docker)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

VOCAB_SIZE = 10_000
MAX_LEN = 200
MODEL_PATH = "models/sentiment_model.keras"
TOKENIZER_PATH = "models/sentiment_tokenizer.pkl"
LABEL_ENCODER_PATH = "models/sentiment_label_encoder.pkl"


def build_model(num_classes: int) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(MAX_LEN,)),
        keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=64, mask_zero=True),
        keras.layers.LSTM(64, return_sequences=False),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def entrenar_y_evaluar(csv_path: str = "data/interactions.csv"):
    df = pd.read_csv(csv_path)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["sentiment"])  # positivo/neutral/negativo -> 0/1/2

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["customer_msg"], y, test_size=0.2, stratify=y, random_state=42
    )

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_text)

    X_train_seq = tokenizer.texts_to_sequences(X_train_text)
    X_test_seq = tokenizer.texts_to_sequences(X_test_text)

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

    model = build_model(num_classes=len(label_encoder.classes_))
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    history = model.fit(
        X_train_pad, y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=16,
        callbacks=callbacks,
        verbose=2,
    )

    # Visualización: curvas de entrenamiento (requisito del enunciado)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="validation")
    axes[0].set_title("Loss por epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy por epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig("models/sentiment_training_curves.png")
    plt.close()
    print("Curvas de entrenamiento guardadas en: models/sentiment_training_curves.png")

    y_pred_proba = model.predict(X_test_pad)
    y_pred = np.argmax(y_pred_proba, axis=1)

    print("\n" + "=" * 60)
    print("EVALUACIÓN EN TEST SET")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    print("Matriz de confusión:")
    print(cm_df)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(label_encoder.classes_)))
    ax.set_yticks(range(len(label_encoder.classes_)))
    ax.set_xticklabels(label_encoder.classes_)
    ax.set_yticklabels(label_encoder.classes_)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig("models/sentiment_confusion_matrix.png")
    plt.close()
    print("Matriz de confusión guardada en: models/sentiment_confusion_matrix.png")

    # Guardar tokenizer y label encoder (necesarios para inferencia futura)
    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    model.save(MODEL_PATH)
    print(f"\nModelo guardado en: {MODEL_PATH}")
    print(f"Tokenizer guardado en: {TOKENIZER_PATH}")

    return model, history, cm_df


class SentimentClassifier:
    """Wrapper de inferencia para la API y el agente de LangGraph."""

    def __init__(self):
        self.model = keras.models.load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, "rb") as f:
            self.tokenizer = pickle.load(f)
        with open(LABEL_ENCODER_PATH, "rb") as f:
            self.label_encoder = pickle.load(f)

    def predict(self, text: str) -> dict:
        seq = self.tokenizer.texts_to_sequences([text])
        pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        proba = self.model.predict(pad, verbose=0)[0]
        clase_idx = int(np.argmax(proba))
        sentimiento = self.label_encoder.inverse_transform([clase_idx])[0]
        return {
            "sentiment": sentimiento,
            "confidence": round(float(proba[clase_idx]), 4),
        }


if __name__ == "__main__":
    entrenar_y_evaluar()

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import os

print("Loading data...")

signals = np.load("data/ecg_signals.npy")
labels = np.load("data/ecg_labels.npy")

# Normalize each signal to 0-1 range
def normalize(sig):
    mn = np.min(sig)
    mx = np.max(sig)
    if mx - mn == 0:
        return sig
    return (sig - mn) / (mx - mn)

print("Normalizing signals...")
normalized = np.array([normalize(s) for s in signals])

# Reshape for LSTM: (samples, timesteps, features)
X = normalized.reshape(normalized.shape[0], normalized.shape[1], 1)
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the autoencoder ONLY on normal signals
# This is the key idea: it learns what normal looks like
normal_idx = np.where(y_train == 0)[0]
X_train_normal = X_train[normal_idx]

print("Training on", len(X_train_normal), "normal signals only")

# Build the LSTM Autoencoder
# Encoder: compress the signal
# Decoder: reconstruct it
# High reconstruction error = anomaly

timesteps = X.shape[1]

encoder_input = keras.Input(shape=(timesteps, 1))
encoded = keras.layers.LSTM(64, return_sequences=False)(encoder_input)
encoded = keras.layers.RepeatVector(timesteps)(encoded)
decoded = keras.layers.LSTM(64, return_sequences=True)(encoded)
decoded = keras.layers.TimeDistributed(keras.layers.Dense(1))(decoded)

autoencoder = keras.Model(encoder_input, decoded)
autoencoder.compile(optimizer="adam", loss="mae")

autoencoder.summary()

print("")
print("Training LSTM Autoencoder...")

history = autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    shuffle=True,
    verbose=1
)

# Calculate reconstruction error on test set
X_test_reconstructed = autoencoder.predict(X_test, verbose=0)
reconstruction_errors = np.mean(np.abs(X_test - X_test_reconstructed), axis=(1, 2))

# Save errors for threshold experiment
np.save("results/lstm_reconstruction_errors.npy", reconstruction_errors)
np.save("results/lstm_test_labels.npy", y_test)

# Pick threshold at mean + 2 std of errors on normal training data
train_reconstructed = autoencoder.predict(X_train_normal, verbose=0)
train_errors = np.mean(np.abs(X_train_normal - train_reconstructed), axis=(1, 2))

threshold = np.mean(train_errors) + 2 * np.std(train_errors)
print("")
print("Threshold (mean + 2 std):", round(threshold, 6))

preds = (reconstruction_errors > threshold).astype(int)

precision = round(precision_score(y_test, preds), 4)
recall = round(recall_score(y_test, preds), 4)
f1 = round(f1_score(y_test, preds), 4)
auc = round(roc_auc_score(y_test, reconstruction_errors), 4)

print("")
print("=== LSTM Autoencoder Results ===")
print("Precision:", precision)
print("Recall:   ", recall)
print("F1 Score: ", f1)
print("ROC-AUC:  ", auc)

os.makedirs("results", exist_ok=True)

result = {
    "model": ["LSTM Autoencoder"],
    "precision": [precision],
    "recall": [recall],
    "f1": [f1],
    "roc_auc": [auc]
}
df = pd.DataFrame(result)
df.to_csv("results/lstm_results.csv", index=False)

autoencoder.save("models/lstm_autoencoder.keras")

print("")
print("Model saved to models/lstm_autoencoder.keras")
print("Results saved to results/lstm_results.csv")

import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import pandas as pd

print("Loading saved model...")
autoencoder = keras.models.load_model("models/lstm_autoencoder.keras")

signals = np.load("data/ecg_signals.npy")

def normalize(sig):
    mn = np.min(sig)
    mx = np.max(sig)
    if mx - mn == 0:
        return sig
    return (sig - mn) / (mx - mn)

normalized = np.array([normalize(s) for s in signals])
X = normalized.reshape(normalized.shape[0], normalized.shape[1], 1)

# Simulate IoT inference: one window at a time
# This is what would happen on a wearable device
print("Running latency simulation (100 windows, one at a time)...")

latencies = []

for i in range(100):
    window = X[i].reshape(1, X.shape[1], 1)
    start = time.time()
    prediction = autoencoder.predict(window, verbose=0)
    error = np.mean(np.abs(window - prediction))
    end = time.time()

    ms = (end - start) * 1000
    latencies.append(ms)

latencies = np.array(latencies)

avg_ms = round(np.mean(latencies), 2)
median_ms = round(np.median(latencies), 2)
p95_ms = round(np.percentile(latencies, 95), 2)
max_ms = round(np.max(latencies), 2)

print("")
print("=== IoT Latency Results (per ECG window) ===")
print("Average latency:  ", avg_ms, "ms")
print("Median latency:   ", median_ms, "ms")
print("95th percentile:  ", p95_ms, "ms")
print("Max latency:      ", max_ms, "ms")

# Is this real-time feasible?
# PTB-XL 100Hz recording is 10 seconds = 1000 samples
# If inference is well under 10,000ms, we can keep up
print("")
if avg_ms < 500:
    print("Result: IoT deployment is FEASIBLE (avg latency well under 500ms)")
else:
    print("Result: Latency may be too high for edge deployment without optimization")

result = {
    "avg_ms": [avg_ms],
    "median_ms": [median_ms],
    "p95_ms": [p95_ms],
    "max_ms": [max_ms]
}
df = pd.DataFrame(result)
df.to_csv("results/latency_results.csv", index=False)

print("Saved to results/latency_results.csv")

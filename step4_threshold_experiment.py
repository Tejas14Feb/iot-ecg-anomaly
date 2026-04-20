import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os

print("Loading reconstruction errors and labels...")

errors = np.load("results/lstm_reconstruction_errors.npy")
labels = np.load("results/lstm_test_labels.npy")

# Load training errors to compute threshold base stats
# We need the training error mean and std
# If you saved them separately this would be cleaner, but we can estimate from test normal errors
normal_errors = errors[labels == 0]
base_mean = np.mean(normal_errors)
base_std = np.std(normal_errors)

print("Normal error mean:", round(base_mean, 6))
print("Normal error std:", round(base_std, 6))

# Try thresholds at mean + 1std, 2std, 3std
thresholds = {
    "mean + 1 std": base_mean + 1 * base_std,
    "mean + 2 std": base_mean + 2 * base_std,
    "mean + 3 std": base_mean + 3 * base_std
}

rows = []
for name, thresh in thresholds.items():
    preds = (errors > thresh).astype(int)
    p = round(precision_score(labels, preds, zero_division=0), 4)
    r = round(recall_score(labels, preds, zero_division=0), 4)
    f = round(f1_score(labels, preds, zero_division=0), 4)
    rows.append({"threshold": name, "precision": p, "recall": r, "f1": f})
    print(name, "-> Precision:", p, "  Recall:", r, "  F1:", f)

df = pd.DataFrame(rows)
df.to_csv("results/threshold_sensitivity.csv", index=False)

# Plot precision vs recall for different thresholds
precisions = [r["precision"] for r in rows]
recalls = [r["recall"] for r in rows]
names = [r["threshold"] for r in rows]

plt.figure(figsize=(7, 5))
plt.plot(recalls, precisions, marker="o", color="steelblue", linewidth=2)
for i, name in enumerate(names):
    plt.annotate(name, (recalls[i], precisions[i]), textcoords="offset points", xytext=(5, 5), fontsize=9)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Tradeoff at Different Thresholds")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/threshold_precision_recall.png", dpi=150)
plt.close()

print("")
print("Threshold sensitivity results saved")
print("Plot saved to results/threshold_precision_recall.png")

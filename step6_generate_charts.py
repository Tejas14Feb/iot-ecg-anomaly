import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

print("Generating result charts...")

# --- Chart 1: Model Comparison Bar Chart ---
# Load both result files
if_df = pd.read_csv("results/isolation_forest_results.csv")
lstm_df = pd.read_csv("results/lstm_results.csv")

models = ["Isolation Forest", "LSTM Autoencoder"]
precisions = [float(if_df["precision"]), float(lstm_df["precision"])]
recalls = [float(if_df["recall"]), float(lstm_df["recall"])]
f1s = [float(if_df["f1"]), float(lstm_df["f1"])]
aucs = [float(if_df["roc_auc"]), float(lstm_df["roc_auc"])]

x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar(x - 1.5 * width, precisions, width, label="Precision", color="steelblue")
bars2 = ax.bar(x - 0.5 * width, recalls, width, label="Recall", color="darkorange")
bars3 = ax.bar(x + 0.5 * width, f1s, width, label="F1 Score", color="green")
bars4 = ax.bar(x + 1.5 * width, aucs, width, label="ROC-AUC", color="purple")

ax.set_ylabel("Score")
ax.set_title("Model Comparison: Isolation Forest vs LSTM Autoencoder")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)

for bar in [bars1, bars2, bars3, bars4]:
    for rect in bar:
        height = rect.get_height()
        ax.annotate(str(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

plt.tight_layout()
plt.savefig("results/model_comparison.png", dpi=150)
plt.close()
print("Saved model_comparison.png")


# --- Chart 2: ROC Curve for LSTM ---
errors = np.load("results/lstm_reconstruction_errors.npy")
test_labels = np.load("results/lstm_test_labels.npy")

fpr, tpr, thresholds = roc_curve(test_labels, errors)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (AUC = " + str(round(roc_auc, 3)) + ")")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - LSTM Autoencoder")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/roc_curve.png", dpi=150)
plt.close()
print("Saved roc_curve.png")


# --- Chart 3: Reconstruction Error Distribution ---
normal_errors = errors[test_labels == 0]
abnormal_errors = errors[test_labels == 1]

plt.figure(figsize=(8, 5))
plt.hist(normal_errors, bins=40, alpha=0.6, color="steelblue", label="Normal ECG")
plt.hist(abnormal_errors, bins=40, alpha=0.6, color="red", label="Abnormal ECG")
plt.xlabel("Reconstruction Error (MAE)")
plt.ylabel("Count")
plt.title("Reconstruction Error Distribution")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("results/error_distribution.png", dpi=150)
plt.close()
print("Saved error_distribution.png")


# --- Print final summary table ---
print("")
print("========================================")
print("       FINAL RESULTS SUMMARY")
print("========================================")
print("")
print("Model                Precision  Recall   F1      ROC-AUC")
print("Isolation Forest    ", precisions[0], "    ", recalls[0], "  ", f1s[0], "  ", aucs[0])
print("LSTM Autoencoder    ", precisions[1], "    ", recalls[1], "  ", f1s[1], "  ", aucs[1])
print("")
print("All charts saved in results/")

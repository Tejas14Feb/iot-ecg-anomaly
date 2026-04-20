import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import os

print("Loading data...")

signals = np.load("data/ecg_signals.npy")
labels = np.load("data/ecg_labels.npy")

print("Total records:", len(signals))
print("Normal:", int(np.sum(labels == 0)), "  Abnormal:", int(np.sum(labels == 1)))

# Extract simple statistical features per signal
# Mean, std, min, max - same as described in the slides
def extract_features(signal_array):
    features = []
    for sig in signal_array:
        mean = np.mean(sig)
        std = np.std(sig)
        minimum = np.min(sig)
        maximum = np.max(sig)
        features.append([mean, std, minimum, maximum])
    return np.array(features)

print("Extracting features...")
X = extract_features(signals)
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train on all training data (unsupervised - Isolation Forest doesnt use labels)
print("Training Isolation Forest...")
model = IsolationForest(contamination=0.3, random_state=42, n_estimators=100)
model.fit(X_train)

# Predict: isolation forest returns 1 for normal, -1 for anomaly
# We convert to 0=normal, 1=anomaly to match our labels
raw_preds = model.predict(X_test)
preds = np.where(raw_preds == -1, 1, 0)

# Scores for ROC-AUC (more negative = more anomalous)
scores = -model.score_samples(X_test)

precision = round(precision_score(y_test, preds), 4)
recall = round(recall_score(y_test, preds), 4)
f1 = round(f1_score(y_test, preds), 4)
auc = round(roc_auc_score(y_test, scores), 4)

print("")
print("=== Isolation Forest Results ===")
print("Precision:", precision)
print("Recall:   ", recall)
print("F1 Score: ", f1)
print("ROC-AUC:  ", auc)

os.makedirs("results", exist_ok=True)

result = {
    "model": ["Isolation Forest"],
    "precision": [precision],
    "recall": [recall],
    "f1": [f1],
    "roc_auc": [auc]
}
df = pd.DataFrame(result)
df.to_csv("results/isolation_forest_results.csv", index=False)

print("")
print("Results saved to results/isolation_forest_results.csv")

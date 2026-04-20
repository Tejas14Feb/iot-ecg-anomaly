# ML-Driven Anomaly Detection in Wearable IoT Health Monitoring
**Tejas Budharamu • CSC 444 Internet of Things • University of South Dakota**

---

## Project Overview

This project builds a machine learning pipeline to detect cardiac anomalies in ECG signals from wearable IoT devices. We compare two approaches:

- **LSTM Autoencoder** (primary model): learns what normal ECG looks like, flags high reconstruction error as anomalous
- **Isolation Forest** (baseline): classical unsupervised anomaly detection using statistical features

Dataset: PTB-XL (public ECG dataset, 21,799 records from 18,869 patients)

---

## Setup

```bash
pip install -r requirements.txt
```

---

## How to Run (in order)

### Step 1: Download data
```bash
python step1_download_data.py
```
Downloads 500 ECG records from PTB-XL and saves them locally.
Takes a few minutes depending on internet speed.

### Step 2: Train Isolation Forest baseline
```bash
python step2_isolation_forest.py
```
Extracts statistical features (mean, std, min, max) and trains the baseline model.
Prints and saves Precision, Recall, F1, ROC-AUC.

### Step 3: Train LSTM Autoencoder
```bash
python step3_lstm_autoencoder.py
```
Trains the LSTM autoencoder on normal ECG signals only.
Saves the model and test results.

### Step 4: Threshold sensitivity experiment
```bash
python step4_threshold_experiment.py
```
Tests three thresholds (mean + 1σ, 2σ, 3σ) and shows the Precision-Recall tradeoff.

### Step 5: IoT latency simulation
```bash
python step5_latency_simulation.py
```
Measures inference time per ECG window to check if deployment on wearables is feasible.

### Step 6: Generate all charts
```bash
python step6_generate_charts.py
```
Produces three publication-ready charts in the `results/` folder.

---

## Results Files

After running all steps, the `results/` folder contains:

| File | Description |
|------|-------------|
| `isolation_forest_results.csv` | Precision, Recall, F1, ROC-AUC for baseline |
| `lstm_results.csv` | Same metrics for LSTM Autoencoder |
| `threshold_sensitivity.csv` | Precision/Recall at 1σ, 2σ, 3σ thresholds |
| `latency_results.csv` | Inference time in milliseconds |
| `model_comparison.png` | Side-by-side bar chart of both models |
| `roc_curve.png` | ROC curve for LSTM Autoencoder |
| `error_distribution.png` | Reconstruction error for normal vs abnormal |
| `threshold_precision_recall.png` | Precision-Recall tradeoff plot |

---

## Project Structure

```
iot-ecg-anomaly/
├── step1_download_data.py
├── step2_isolation_forest.py
├── step3_lstm_autoencoder.py
├── step4_threshold_experiment.py
├── step5_latency_simulation.py
├── step6_generate_charts.py
├── requirements.txt
├── README.md
├── data/
├── models/
└── results/
```

---

## Notes

- The LSTM autoencoder is trained exclusively on **normal signals** (semi-supervised)
- Anomaly detection threshold is set at `mean + 2σ` of training reconstruction error
- All experiments use the PTB-XL dataset accessed through the `wfdb` Python library

# 🌿 GreenTravel Intelligence Challenge

A machine learning pipeline to predict **high-carbon business trips** using process mining features extracted from Celonis travel event data.

---

## 📋 Problem Statement

Corporate travel is a significant source of CO₂ emissions. This challenge (part of the **Celonis Hackathon**) asks us to predict whether a given business trip will be classified as high-carbon — *without* using any direct emission columns — enabling companies to proactively identify and reduce their carbon footprint before travel occurs.

---

## 📁 Dataset

The following files are used (not committed due to size):

| File | Description |
|------|-------------|
| `public_trip_data.csv` | Labeled training data with trip details and target `HighCarbon` |
| `public_trip_event_log.csv` | Process event log for training trips (timestamps, event names) |
| `public_trip_event_attributes.csv` | Supplementary attributes per training trip |
| `private_trip_data.csv` | Unlabeled test data for inference |
| `private_trip_event_log.csv` | Process event log for test trips |
| `private_trip_event_attributes.csv` | Supplementary attributes for test trips |

> **Note:** All raw CSV files are excluded from this repository via `.gitignore` due to their size. Only `submission.csv` (model predictions) is committed.

---

## 🚫 Prohibited Features

The following columns are direct emissions data and were **strictly excluded** from training to avoid data leakage:

- `Departure_CO2e`
- `Return_CO2e`
- `Hotel_CO2e`
- `Spend_CO2e`
- `TotalCO2e`

---

## 🛠️ Approach

### 1. Data Ingestion & Merging
- Loaded all six CSV files and merged them on `TripID`.
- Combined trip data with event attributes and aggregated event log features.

### 2. Process Mining Feature Engineering
From the event logs, the following features were engineered per trip:

| Feature | Description |
|---------|-------------|
| `num_events` | Total number of booking/process events |
| `num_unique_events` | Number of distinct event types |
| `duration_seconds` | Total time from first to last event |

### 3. Feature Alignment
- Dynamically identified columns present in both public and private datasets to avoid feature mismatch at inference time.
- Categorical columns were cast to `category` dtype for LightGBM's native handling.

### 4. Model: LightGBM Classifier
**LightGBM** was chosen for its:
- ✅ Native handling of categorical features and missing values
- ✅ Efficient gradient boosting on tabular data
- ✅ Built-in early stopping to prevent overfitting

**Training Configuration:**
```
n_estimators    : 1000 (with early stopping)
learning_rate   : 0.05
max_depth       : 6
subsample       : 0.8
colsample_bytree: 0.8
early_stopping  : 50 rounds
```

### 5. Validation Strategy
- 80/20 stratified train/validation split on the public dataset.
- Final model retrained on the **full public dataset** using the best iteration count determined during early stopping.

---

## 📊 Results

Evaluated on a held-out 20% validation set from `public_trip_data.csv`:

| Metric | Score |
|--------|-------|
| **ROC-AUC** | **0.9994** |
| **F1 Score** | **0.9871** |
| **Precision** | **0.9898** |
| **Recall** | **0.9844** |

**Confusion Matrix (Validation Set):**
```
[[9760   33]
 [  51 3214]]
```

---

## 📂 Repository Structure

```
.
├── solve.py           # End-to-end ML pipeline script
├── submission.csv     # Final predictions for private test set
├── .gitignore         # Excludes raw CSV data files
└── README.md          # Project documentation
```

---

## 🚀 How to Run

1. **Install dependencies:**
   ```bash
   pip install lightgbm pandas scikit-learn
   ```

2. **Place all 6 raw CSV files in the same directory as `solve.py`.**

3. **Run the pipeline:**
   ```bash
   python solve.py
   ```

4. The final predictions will be saved to `submission.csv`.

---

## 📤 Output Format

The `submission.csv` file contains probabilistic predictions:

```csv
TripID,HighCarbon
2,0.9911545620986708
4,0.010575798939762103
5,0.8659537446627063
...
```

---

## 🏷️ Tech Stack

- **Python 3.13**
- **LightGBM 4.6.0**
- **Pandas 2.3.3**
- **Scikit-learn 1.8.0**

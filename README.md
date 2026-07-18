# 🌿 GreenTravel Intelligence Challenge

A machine learning pipeline to predict **high-carbon business trips** using process mining features extracted from Celonis travel event data — built for the **Celonis Hackathon**.

---

## 📋 Problem Statement

Corporate travel is a significant source of CO₂ emissions. This challenge asks us to predict whether a given business trip will be classified as **high-carbon** — *without* using any direct emission columns — enabling companies to proactively identify and reduce their carbon footprint before travel occurs.

---

## 📁 Dataset

The following files are used (not committed due to size):

| File | Description |
|------|-------------|
| `public_trip_data.csv` | Labeled training data (65,289 trips) with `HighCarbon` target |
| `public_trip_event_log.csv` | Process event log for training trips (timestamps, event names) |
| `public_trip_event_attributes.csv` | Supplementary attributes per training trip |
| `private_trip_data.csv` | Unlabeled test data (21,764 trips) for inference |
| `private_trip_event_log.csv` | Process event log for test trips |
| `private_trip_event_attributes.csv` | Supplementary attributes for test trips |

> **Note:** All raw CSV files are excluded from this repository via `.gitignore` due to their size. Only the final submission file is committed.

---

## 🚫 Prohibited Features

The following columns are direct emission leakage features and were **strictly excluded** from all training:

- `Departure_CO2e`
- `Return_CO2e`
- `Hotel_CO2e`
- `Spend_CO2e`
- `TotalCO2e`

---

## 🛠️ Approach (v2)

### 1. Data Ingestion & Assembly
- Loaded all six CSV files and assembled them by merging on `TripID`.
- Dynamically aligned features between public and private datasets to avoid any train/test column mismatch.

### 2. Process Mining Feature Engineering
From the raw event logs, the following features were extracted per trip:

| Feature | Description |
|---------|-------------|
| `event_count` | Total number of booking/process events |
| `unique_event_count` | Number of distinct event types triggered |
| `process_duration_sec` | Total time span from first to last event |
| `total_disruptions` | Sum of all disruption-type events |
| `evt_*` (18 features) | Individual counts of each disruption event type (e.g. `evt_Flight_Cancellation`, `evt_Hotel_Change`, etc.) |

**Disruption event types tracked:**
`Trip Extension`, `Itinerary Edit`, `Hotel Change`, `Mode of Transportation Change`, `Ticket Reissued`, `Flight Change`, `Flight Delay`, `Flight Cancellation`, `Vehicle Change`, `Missed Flight`, `Travel Delay`, `Rental Cancellation`, `Expense Request Edit`, `Expense Request Denied`, `Missed Pickup`, `Train Change`, `Train Cancellation`, `Train Delay`, `Missed Train`

### 3. Attribute Feature Engineering
From event attributes, binary presence flags were created for all reason columns:
`has_ExpenseDenialReason`, `has_ReasonForTransportCancellation`, `has_NewHotelSelection`, etc.

Numerical attributes extracted: `ExpenseReimbursementAmount`, `TransportationPriceDifference`, `ExtensionLength`, `DaysPreapproved`.

### 4. Model: LightGBM + CatBoost Ensemble (5-Fold CV)

The v2 solution uses a **5-fold stratified cross-validation ensemble** of two gradient boosting frameworks:

| Model | Strength |
|-------|----------|
| **LightGBM** | Fast training, native categorical support, excellent on tabular data |
| **CatBoost** | Superior handling of categorical features without preprocessing |

Final predictions = **0.5 × LightGBM + 0.5 × CatBoost** (averaged across all 5 folds)

**Training Configuration:**
```
n_estimators    : up to 2000 (with early stopping @ 50 rounds)
learning_rate   : 0.05
max_depth       : 6
num_leaves      : 31 (LightGBM)
subsample       : 0.8
colsample_bytree: 0.8
cv_folds        : 5 (StratifiedKFold)
random_seed     : 42
```

**Total features used:** 49 (10 categorical + 39 numerical)

---

## 📊 Results

### Per-Fold Cross-Validation AUC

| Fold | LightGBM AUC | CatBoost AUC | Ensemble AUC |
|------|-------------|--------------|--------------|
| 1 | 0.99931 | 0.99930 | 0.99931 |
| 2 | 0.99922 | 0.99928 | 0.99923 |
| 3 | 0.99952 | 0.99952 | 0.99954 |
| 4 | 0.99951 | 0.99944 | 0.99952 |
| 5 | 0.99931 | 0.99923 | 0.99933 |

### Overall OOF (Out-of-Fold) Metrics

| Model | CV AUC | Std Dev |
|-------|--------|---------|
| LightGBM | 0.99938 | ±0.00012 |
| CatBoost | 0.99935 | ±0.00011 |
| **Ensemble** | **0.99938** | **±0.00012** |

| Metric | Score |
|--------|-------|
| **OOF ROC-AUC** | **0.99936** |
| **OOF F1 Score** | **0.9863** |
| **OOF Precision** | **0.9895** |
| **OOF Recall** | **0.9832** |

**OOF Confusion Matrix (full public set, 65,289 trips):**
```
               Predicted 0   Predicted 1
Actual 0          48,793          170
Actual 1             275       16,051
```

### Calibration
The model is extremely well-calibrated — predictions cluster near 0 or 1 with only the boundary region (~0.5) showing any discrepancy:
```
predicted~0.000  actual=0.000
predicted~0.494  actual=0.501  ← boundary region, well-calibrated
predicted~0.999  actual=1.000
```

---

## 📂 Repository Structure

```
.
├── solve_v2.py          # v2 end-to-end ML pipeline (LightGBM + CatBoost ensemble, 5-fold CV)
├── submission_v2.csv    # Final predictions for private test set (21,764 trips)
├── .gitignore           # Excludes raw CSV data files
└── README.md            # Project documentation
```

---

## 🚀 How to Run

1. **Install dependencies:**
   ```bash
   pip install lightgbm catboost pandas scikit-learn
   ```

2. **Place all 6 raw CSV files in the same directory as `solve_v2.py`.**

3. **Run the pipeline:**
   ```bash
   python solve_v2.py
   ```

4. Predictions are saved to `submission_v2.csv`.

---

## 📤 Output Format

```csv
TripID,HighCarbon
2,0.9985123...
4,0.0009871...
5,0.8734562...
...
```

`HighCarbon` values are **probabilities** (0–1), where values closer to 1 indicate a high-carbon trip.

**Private test set prediction summary (21,764 trips):**
| Stat | Value |
|------|-------|
| Mean predicted probability | 0.2487 |
| Median | 0.0017 |
| Min | 0.0004 |
| Max | 0.9990 |

---

## 🏷️ Tech Stack

- **Python 3.13**
- **LightGBM 4.6.0**
- **CatBoost**
- **Pandas 2.3.3**
- **Scikit-learn 1.8.0**

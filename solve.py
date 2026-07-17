import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
# Load trip data
train_df = pd.read_csv('public_trip_data.csv')
test_df = pd.read_csv('private_trip_data.csv')

# Load event attributes
train_attr = pd.read_csv('public_trip_event_attributes.csv')
test_attr = pd.read_csv('private_trip_event_attributes.csv')

# Load event logs
train_log = pd.read_csv('public_trip_event_log.csv')
test_log = pd.read_csv('private_trip_event_log.csv')

print("Preprocessing event logs...")
# Process timestamps
def process_logs(log_df):
    log_df['EventTimestamp'] = pd.to_datetime(log_df['EventTimestamp'], errors='coerce')
    agg = log_df.groupby('TripID').agg(
        num_events=('EventName', 'count'),
        num_unique_events=('EventName', 'nunique'),
        first_event=('EventTimestamp', 'min'),
        last_event=('EventTimestamp', 'max')
    )
    agg['duration_seconds'] = (agg['last_event'] - agg['first_event']).dt.total_seconds()
    agg.drop(columns=['first_event', 'last_event'], inplace=True)
    return agg.reset_index()

train_log_agg = process_logs(train_log)
test_log_agg = process_logs(test_log)

print("Merging datasets...")
# Merge
train = train_df.merge(train_attr, on='TripID', how='left').merge(train_log_agg, on='TripID', how='left')
test = test_df.merge(test_attr, on='TripID', how='left').merge(test_log_agg, on='TripID', how='left')

print("Feature engineering...")
# Drop prohibited features
prohibited = ['Departure_CO2e', 'Return_CO2e', 'Hotel_CO2e', 'Spend_CO2e', 'TotalCO2e']
for col in prohibited:
    if col in train.columns:
        train.drop(columns=[col], inplace=True)
    if col in test.columns:
        test.drop(columns=[col], inplace=True)

target = 'HighCarbon'
y = train[target]
train.drop(columns=[target], inplace=True)

# Ensure only columns present in both train and test are kept
common_cols = [c for c in train.columns if c in test.columns]
train_features = train[common_cols].copy()
test_features = test[common_cols].copy()

# Prepare categorical features
cat_cols = train_features.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Categorical columns: {cat_cols}")
for col in cat_cols:
    train_features[col] = train_features[col].astype('category')
    test_features[col] = test_features[col].astype('category')
    
# Keep TripID for output but don't use it as a feature
train_features.drop(columns=['TripID'], inplace=True, errors='ignore')
test_features_for_pred = test_features.drop(columns=['TripID'], errors='ignore')

print("Splitting validation set...")
X_train, X_val, y_train, y_val = train_test_split(train_features, y, test_size=0.2, random_state=42, stratify=y)

print("Training LightGBM model...")
model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

print("Evaluating model...")
val_preds_prob = model.predict_proba(X_val)[:, 1]
val_preds = model.predict(X_val)

roc_auc = roc_auc_score(y_val, val_preds_prob)
f1 = f1_score(y_val, val_preds)
precision = precision_score(y_val, val_preds)
recall = recall_score(y_val, val_preds)

print(f"ROC-AUC: {roc_auc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_val, val_preds))

print("Training final model on full public data...")
final_model = lgb.LGBMClassifier(
    n_estimators=model.best_iteration_,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
final_model.fit(train_features, y)

print("Generating predictions for private test set...")
test_preds_prob = final_model.predict_proba(test_features_for_pred)[:, 1]

submission = pd.DataFrame({
    'TripID': test['TripID'],
    'HighCarbon': test_preds_prob
})

submission.to_csv('submission.csv', index=False)
print("Saved submission to submission.csv")

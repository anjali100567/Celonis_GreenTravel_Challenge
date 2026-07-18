import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.calibration import calibration_curve

UP = '/mnt/user-data/uploads'
SEED = 42
N_FOLDS = 5

# ---------- Load ----------
pub_trip = pd.read_csv(f'{UP}/public_trip_data.csv')
priv_trip = pd.read_csv(f'{UP}/private_trip_data.csv')
pub_attr = pd.read_csv(f'{UP}/public_trip_event_attributes.csv')
priv_attr = pd.read_csv(f'{UP}/private_trip_event_attributes.csv')
pub_log = pd.read_csv(f'{UP}/public_trip_event_log.csv', parse_dates=['EventTimestamp'])
priv_log = pd.read_csv(f'{UP}/private_trip_event_log.csv', parse_dates=['EventTimestamp'])
sample_sub = pd.read_csv(f'{UP}/sample_submission.csv')

BANNED = ['Departure_CO2e', 'Return_CO2e', 'Hotel_CO2e', 'Spend_CO2e', 'TotalCO2e']

DISRUPTION_EVENTS = [
    'Trip Extension', 'Itinerary Edit', 'Hotel Change', 'Mode of Transportation Change',
    'Ticket Reissued', 'Flight Change', 'Flight Delay', 'Flight Cancellation',
    'Vehicle Change', 'Missed Flight', 'Travel Delay', 'Rental Cancellation',
    'Expense Request Edit', 'Expense Request Denied', 'Missed Pickup',
    'Train Change', 'Train Cancellation', 'Train Delay', 'Missed Train'
]


def build_log_features(log):
    g = log.groupby('TripID')
    feats = pd.DataFrame(index=g.size().index)
    feats['event_count'] = g.size()
    feats['unique_event_count'] = g['EventName'].nunique()
    feats['process_duration_sec'] = g['EventTimestamp'].agg(lambda x: (x.max() - x.min()).total_seconds())
    pivot = log[log['EventName'].isin(DISRUPTION_EVENTS)].groupby(['TripID', 'EventName']).size().unstack(fill_value=0)
    pivot.columns = ['evt_' + c.replace(' ', '_') for c in pivot.columns]
    feats = feats.join(pivot, how='left')
    evt_cols = [c for c in feats.columns if c.startswith('evt_')]
    feats[evt_cols] = feats[evt_cols].fillna(0)
    feats['total_disruptions'] = feats[evt_cols].sum(axis=1)
    return feats.reset_index()


def build_attr_features(attr):
    a = attr.copy()
    out = pd.DataFrame({'TripID': a['TripID']})
    reason_cols = [
        'ExpenseDenialReason', 'ReasonForTransportCancellation', 'NewTransportSelection',
        'ReasonForNewTransport', 'NewHotelSelection', 'ReasonForNewHotel',
        'NewModeOfTransportation', 'ReasonForTransportationChange', 'ReasonForDelay'
    ]
    for c in reason_cols:
        out[f'has_{c}'] = a[c].notna().astype(int)
    out['ExpenseReimbursementAmount'] = a['ExpenseReimbursementAmount'].fillna(0)
    out['TransportationPriceDifference'] = a['TransportationPriceDifference'].fillna(0)
    out['ExtensionLength'] = a['ExtensionLength'].fillna(0)
    out['DaysPreapproved'] = a['DaysPreapproved'].fillna(0)
    out['ProcessCode'] = a['ProcessCode'].astype('category')
    out['ExpenseReimbursementReason'] = a['ExpenseReimbursementReason'].fillna('None').astype('category')
    return out


def assemble(trip, attr_feats, log_feats):
    return trip.merge(attr_feats, on='TripID', how='left').merge(log_feats, on='TripID', how='left')


pub_log_feats = build_log_features(pub_log)
priv_log_feats = build_log_features(priv_log)
pub_attr_feats = build_attr_features(pub_attr)
priv_attr_feats = build_attr_features(priv_attr)

pub = assemble(pub_trip, pub_attr_feats, pub_log_feats)
priv = assemble(priv_trip, priv_attr_feats, priv_log_feats)

drop_cols = BANNED + ['HighCarbon', 'EmployeeNumber', 'ShippingType']
target = pub_trip.set_index('TripID')['HighCarbon']

X_pub = pub.drop(columns=[c for c in drop_cols if c in pub.columns]).set_index('TripID')
X_priv = priv.drop(columns=[c for c in drop_cols if c in priv.columns]).set_index('TripID')
y_pub = target.loc[X_pub.index]

common_cols = [c for c in X_pub.columns if c in X_priv.columns]
X_pub = X_pub[common_cols].copy()
X_priv = X_priv[common_cols].copy()

# log-derived numeric cols may have NaN for trips w/ no log rows -> fill 0
log_num_cols = ['event_count', 'unique_event_count', 'process_duration_sec', 'total_disruptions'] + \
               [c for c in common_cols if c.startswith('evt_')]
for c in log_num_cols:
    X_pub[c] = X_pub[c].fillna(0)
    X_priv[c] = X_priv[c].fillna(0)

cat_cols = X_pub.select_dtypes(include=['object', 'str', 'category']).columns.tolist()
for c in cat_cols:
    X_pub[c] = X_pub[c].astype(str).fillna('NA').astype('category')
    X_priv[c] = X_priv[c].astype(str).fillna('NA').astype('category')
    cats = pd.api.types.union_categoricals([X_pub[c], X_priv[c]]).categories
    X_pub[c] = X_pub[c].cat.set_categories(cats)
    X_priv[c] = X_priv[c].cat.set_categories(cats)

print('Feature columns:', len(common_cols), '| Categorical:', len(cat_cols))
print('X_pub:', X_pub.shape, 'X_priv:', X_priv.shape)

cat_idx = [X_pub.columns.get_loc(c) for c in cat_cols]  # for catboost

# ---------- 5-fold CV: LightGBM + CatBoost ensemble ----------
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_lgb = np.zeros(len(X_pub))
oof_cat = np.zeros(len(X_pub))
priv_pred_lgb = np.zeros(len(X_priv))
priv_pred_cat = np.zeros(len(X_priv))

lgb_params = dict(objective='binary', metric='auc', learning_rate=0.05,
                   max_depth=6, num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                   seed=SEED, verbose=-1)

fold_aucs_lgb, fold_aucs_cat, fold_aucs_ens = [], [], []

X_pub_cat_str = X_pub.copy()
for c in cat_cols:
    X_pub_cat_str[c] = X_pub_cat_str[c].astype(str)
X_priv_cat_str = X_priv.copy()
for c in cat_cols:
    X_priv_cat_str[c] = X_priv_cat_str[c].astype(str)

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_pub, y_pub)):
    X_tr, X_val = X_pub.iloc[tr_idx], X_pub.iloc[val_idx]
    y_tr, y_val = y_pub.iloc[tr_idx], y_pub.iloc[val_idx]

    # --- LightGBM ---
    train_set = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    val_set = lgb.Dataset(X_val, label=y_val, categorical_feature=cat_cols, reference=train_set)
    model_lgb = lgb.train(lgb_params, train_set, num_boost_round=2000,
                           valid_sets=[val_set],
                           callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    val_pred_lgb = model_lgb.predict(X_val, num_iteration=model_lgb.best_iteration)
    oof_lgb[val_idx] = val_pred_lgb
    priv_pred_lgb += model_lgb.predict(X_priv, num_iteration=model_lgb.best_iteration) / N_FOLDS

    # --- CatBoost ---
    X_tr_c, X_val_c = X_pub_cat_str.iloc[tr_idx], X_pub_cat_str.iloc[val_idx]
    train_pool = Pool(X_tr_c, y_tr, cat_features=cat_idx)
    val_pool = Pool(X_val_c, y_val, cat_features=cat_idx)
    model_cat = CatBoostClassifier(iterations=2000, learning_rate=0.05, depth=6,
                                    loss_function='Logloss', eval_metric='AUC',
                                    random_seed=SEED, verbose=False,
                                    early_stopping_rounds=50)
    model_cat.fit(train_pool, eval_set=val_pool, use_best_model=True)
    val_pred_cat = model_cat.predict_proba(X_val_c)[:, 1]
    oof_cat[val_idx] = val_pred_cat
    priv_pred_cat += model_cat.predict_proba(X_priv_cat_str)[:, 1] / N_FOLDS

    auc_lgb = roc_auc_score(y_val, val_pred_lgb)
    auc_cat = roc_auc_score(y_val, val_pred_cat)
    auc_ens = roc_auc_score(y_val, 0.5 * val_pred_lgb + 0.5 * val_pred_cat)
    fold_aucs_lgb.append(auc_lgb)
    fold_aucs_cat.append(auc_cat)
    fold_aucs_ens.append(auc_ens)
    print(f'Fold {fold+1}: LGB AUC={auc_lgb:.5f}  CatBoost AUC={auc_cat:.5f}  Ensemble AUC={auc_ens:.5f}')

print()
print(f'LightGBM  CV AUC: {np.mean(fold_aucs_lgb):.5f} +/- {np.std(fold_aucs_lgb):.5f}')
print(f'CatBoost  CV AUC: {np.mean(fold_aucs_cat):.5f} +/- {np.std(fold_aucs_cat):.5f}')
print(f'Ensemble  CV AUC: {np.mean(fold_aucs_ens):.5f} +/- {np.std(fold_aucs_ens):.5f}')

# OOF-based overall metrics (more robust than single split)
oof_ens = 0.5 * oof_lgb + 0.5 * oof_cat
oof_auc = roc_auc_score(y_pub, oof_ens)
oof_pred_binary = (oof_ens > 0.5).astype(int)
oof_f1 = f1_score(y_pub, oof_pred_binary)
oof_prec = precision_score(y_pub, oof_pred_binary)
oof_rec = recall_score(y_pub, oof_pred_binary)
cm = confusion_matrix(y_pub, oof_pred_binary)

print()
print(f'OOF Ensemble ROC-AUC: {oof_auc:.5f}')
print(f'OOF Ensemble F1: {oof_f1:.4f}  Precision: {oof_prec:.4f}  Recall: {oof_rec:.4f}')
print('OOF Confusion matrix:\n', cm)

# ---------- Calibration check ----------
frac_pos, mean_pred = calibration_curve(y_pub, oof_ens, n_bins=10, strategy='quantile')
print('\nCalibration (predicted prob bin -> actual positive rate):')
for mp, fp in zip(mean_pred, frac_pos):
    print(f'  predicted~{mp:.3f}  actual={fp:.3f}  diff={fp-mp:+.3f}')

# ---------- Final ensemble prediction on private set ----------
final_priv_pred = 0.5 * priv_pred_lgb + 0.5 * priv_pred_cat

sub = pd.DataFrame({'TripID': X_priv.index, 'HighCarbon': final_priv_pred})
sub = sample_sub[['TripID']].merge(sub, on='TripID', how='left')
sub.to_csv('/home/claude/greentravel/submission.csv', index=False)
print('\nSaved submission.csv. Shape:', sub.shape)
print(sub.describe())

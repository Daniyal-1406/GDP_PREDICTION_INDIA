import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

DATA_PATH   = 'data/India_1960_2025_-_india_complete_1960_2025.csv'
SAVE_DIR    = 'preprocessed'
WINDOW_SIZE = 5
TRAIN_END   = 2014   # train 1960-2014, test 2015-2025

os.makedirs(SAVE_DIR, exist_ok=True)

print("=" * 60)
print("STEP 2 — PREPROCESSING  (1960-2014 train | 2015-2025 test)")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
df = df.rename(columns={
    'year'               : 'Year',
    'gdp_usd'            : 'GDP_USD_Raw',
    'investment_pct_gdp' : 'Investment',
    'agri_pct_gdp'       : 'Agriculture',
    'imports_pct_gdp'    : 'Import',
    'exports_pct_gdp'    : 'Export',
    'literacy_rate'      : 'Education',
    'total_power_gen_mu' : 'Power_Generation',
    'mobile_per100'      : 'Mobile_Penetration',
    'internet_pct'       : 'Internet_Penetration',
    'Sentiment'          : 'Sentiment_Score',
    'gdp_growth'         : 'GDP_Growth_Rate',
    'log_gdp'            : 'log_GDP',
})

# GDP in Billion USD
df['GDP_Billion_USD'] = df['GDP_USD_Raw'] / 1e9

years   = df['Year'].values.copy()

# ── Crisis Dummy ──────────────────────────────────────────────────────────────
crisis_years = {
    1965, 1966,   # Indo-Pak war, drought
    1971,         # Bangladesh war
    1974,         # oil shock
    1979, 1980,   # second oil shock
    1991,         # BoP crisis
    1999,         # Kargil War
    2001,         # 9/11 effects
    2008, 2009,   # Global Financial Crisis
    2016, 2017,   # Demonetisation + GST
    2020,         # COVID-19
}
df['Crisis_Dummy'] = df['Year'].isin(crisis_years).astype(np.float32)

# ── Log GDP lags ──────────────────────────────────────────────────────────────
df['log_GDP_lag1'] = df['log_GDP'].shift(1).bfill()
df['log_GDP_lag2'] = df['log_GDP'].shift(2).bfill()

# ── Target: next year's GDP growth rate ───────────────────────────────────────
last_3_avg = df['GDP_Growth_Rate'].iloc[-4:-1].mean()
df['Target_Growth'] = df['GDP_Growth_Rate'].shift(-1).fillna(last_3_avg)

# ── Verify zero NaNs ──────────────────────────────────────────────────────────
check_cols = ['GDP_Growth_Rate','log_GDP_lag1','log_GDP_lag2',
              'Target_Growth','Sentiment_Score','Crisis_Dummy']
print("\n[0] NaN check:")
all_ok = True
for col in check_cols:
    n = df[col].isna().sum()
    status = "✓" if n == 0 else f"✗ {n} NaNs"
    print(f"    {col:<22}: {status}")
    if n > 0: all_ok = False
assert all_ok, "NaNs found — fix before proceeding"
print("    All clean ✓")

# ── Feature columns — 13 total ────────────────────────────────────────────────
FEATURE_COLS = [
    'Investment', 'Agriculture', 'Import', 'Export', 'Education',
    'Power_Generation', 'Mobile_Penetration', 'Internet_Penetration',
    'Sentiment_Score', 'Crisis_Dummy',
    'GDP_Growth_Rate', 'log_GDP_lag1', 'log_GDP_lag2',
]
TARGET_COL = 'Target_Growth'

print(f"\n[1] Dataset: {len(df)} rows | {df['Year'].min()}–{df['Year'].max()}")
print(f"    Features : {len(FEATURE_COLS)}")
print(f"    Target   : next year GDP growth rate %")
print(f"    Growth range: {df['Target_Growth'].min():.2f}% to {df['Target_Growth'].max():.2f}%")

X_all   = df[FEATURE_COLS].values.astype(np.float32)
y_all   = df[TARGET_COL].values.astype(np.float32).reshape(-1, 1)
gdp_raw = df['GDP_Billion_USD'].values.astype(np.float32)
log_gdp = df['log_GDP'].values.astype(np.float32)

# ── Split BEFORE scaling ──────────────────────────────────────────────────────
train_mask = years <= TRAIN_END
test_mask  = years >  TRAIN_END

X_train_raw = X_all[train_mask]
X_test_raw  = X_all[test_mask]

print(f"\n[2] Train: {X_train_raw.shape[0]} rows ({years[train_mask][0]}–{years[train_mask][-1]})")
print(f"    Test : {X_test_raw.shape[0]}  rows ({years[test_mask][0]}–{years[test_mask][-1]})")

# ── Scale X — fit on train only ───────────────────────────────────────────────
scaler_X       = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler_X.fit_transform(X_train_raw)
X_test_scaled  = scaler_X.transform(X_test_raw)

X_full = np.vstack([X_train_scaled, X_test_scaled])
y_full = y_all

print(f"[3] X scaled on train only | y (growth %) kept raw")

# ── Sliding window sequences ──────────────────────────────────────────────────
def make_sequences(X, y, years_arr, window, cutoff, mode):
    Xs, ys, yrs = [], [], []
    for i in range(window, len(X)):
        yr = years_arr[i]
        if mode == 'train' and yr > cutoff:  continue
        if mode == 'test'  and yr <= cutoff: continue
        Xs.append(X[i - window : i])
        ys.append(y[i])
        yrs.append(yr)
    return np.array(Xs, np.float32), np.array(ys, np.float32), np.array(yrs)

X_train_seq, y_train_seq, train_years = make_sequences(
    X_full, y_full, years, WINDOW_SIZE, TRAIN_END, 'train')
X_test_seq,  y_test_seq,  test_years  = make_sequences(
    X_full, y_full, years, WINDOW_SIZE, TRAIN_END, 'test')

# Verify
assert not np.isnan(X_train_seq).any(), "NaN in X_train_seq"
assert not np.isnan(y_train_seq).any(), "NaN in y_train_seq"
assert not np.isnan(X_test_seq).any(),  "NaN in X_test_seq"
assert not np.isnan(y_test_seq).any(),  "NaN in y_test_seq"

print(f"\n[4] Sequences (window={WINDOW_SIZE})")
print(f"    X_train_seq : {X_train_seq.shape}  (samples, timesteps, features)")
print(f"    y_train_seq : {y_train_seq.shape}")
print(f"    Train years : {train_years[0]}–{train_years[-1]}")
print(f"    X_test_seq  : {X_test_seq.shape}")
print(f"    Test  years : {test_years}")
print(f"    y_train range: [{y_train_seq.min():.2f}%, {y_train_seq.max():.2f}%]")
print(f"    y_test  range: [{y_test_seq.min():.2f}%, {y_test_seq.max():.2f}%]")
print(f"    No NaNs ✓")

# ── Save ──────────────────────────────────────────────────────────────────────
np.save(f'{SAVE_DIR}/X_train_seq.npy',  X_train_seq)
np.save(f'{SAVE_DIR}/y_train_seq.npy',  y_train_seq)
np.save(f'{SAVE_DIR}/X_test_seq.npy',   X_test_seq)
np.save(f'{SAVE_DIR}/y_test_seq.npy',   y_test_seq)
np.save(f'{SAVE_DIR}/train_years.npy',  train_years)
np.save(f'{SAVE_DIR}/test_years.npy',   test_years)
np.save(f'{SAVE_DIR}/log_gdp_all.npy',  log_gdp)
np.save(f'{SAVE_DIR}/gdp_raw_all.npy',  gdp_raw)
np.save(f'{SAVE_DIR}/years_all.npy',    years.astype(np.int32))

with open(f'{SAVE_DIR}/scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)

print(f"\n[5] Saved to preprocessed/")
print(f"{'='*60}")
print(f"STEP 2 COMPLETE")
print(f"  Train: {X_train_seq.shape[0]} sequences | Test: {X_test_seq.shape[0]} sequences")
print(f"  window={WINDOW_SIZE} | 13 features | target=growth rate %")
print(f"{'='*60}")

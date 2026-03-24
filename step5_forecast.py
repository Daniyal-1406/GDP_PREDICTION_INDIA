import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import os, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Model ─────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])

class GDPTransformer(nn.Module):
    def __init__(self, n_features=13):
        super().__init__()
        self.input_projection    = nn.Linear(n_features, 64)
        self.pos_encoder         = PositionalEncoding(64, dropout=0.1)
        enc = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=128,
            dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc, num_layers=2)
        self.output_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(0.1),   nn.Linear(32, 1))
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.output_head(x)

# ── Config ────────────────────────────────────────────────────────────────────
PREPROCESSED_DIR = 'preprocessed'
MODELS_DIR       = 'models'
OUTPUTS_DIR      = 'outputs'
DATA_PATH        = 'data/India_1960_2025_-_india_complete_1960_2025.csv'
FORECAST_YEARS   = [2026, 2027, 2028, 2029, 2030]
MC_SAMPLES       = 200
WINDOW_SIZE      = 5
FEATURE_COLS = [
    'Investment', 'Agriculture', 'Import', 'Export', 'Education',
    'Power_Generation', 'Mobile_Penetration', 'Internet_Penetration',
    'Sentiment_Score', 'Crisis_Dummy',
    'GDP_Growth_Rate', 'log_GDP_lag1', 'log_GDP_lag2',
]

os.makedirs(OUTPUTS_DIR, exist_ok=True)

print("=" * 60)
print("STEP 5 — FORECAST (2026–2030) + MC Confidence Intervals")
print("=" * 60)

# ── Load ──────────────────────────────────────────────────────────────────────
gdp_raw_all = np.load(f'{PREPROCESSED_DIR}/gdp_raw_all.npy')
years_all   = np.load(f'{PREPROCESSED_DIR}/years_all.npy')
test_years  = np.load(f'{PREPROCESSED_DIR}/test_years.npy')
train_years = np.load(f'{PREPROCESSED_DIR}/train_years.npy')
X_test_seq  = np.load(f'{PREPROCESSED_DIR}/X_test_seq.npy')
y_test_seq  = np.load(f'{PREPROCESSED_DIR}/y_test_seq.npy')
X_train_seq = np.load(f'{PREPROCESSED_DIR}/X_train_seq.npy')
y_mean      = float(np.load(f'{PREPROCESSED_DIR}/y_gr_mean.npy')[0])
y_std       = float(np.load(f'{PREPROCESSED_DIR}/y_gr_std.npy')[0])

with open(f'{PREPROCESSED_DIR}/scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)

df = pd.read_csv(DATA_PATH)
df = df.rename(columns={
    'year':'Year','gdp_usd':'GDP_USD_Raw',
    'investment_pct_gdp':'Investment','agri_pct_gdp':'Agriculture',
    'imports_pct_gdp':'Import','exports_pct_gdp':'Export',
    'literacy_rate':'Education','total_power_gen_mu':'Power_Generation',
    'mobile_per100':'Mobile_Penetration','internet_pct':'Internet_Penetration',
    'Sentiment':'Sentiment_Score','gdp_growth':'GDP_Growth_Rate','log_gdp':'log_GDP',
})
df['GDP_Billion_USD'] = df['GDP_USD_Raw'] / 1e9
crisis_years = {1965,1966,1971,1974,1979,1980,1991,1999,2001,2008,2009,2016,2017,2020}
df['Crisis_Dummy'] = df['Year'].isin(crisis_years).astype(float)
df['log_GDP_lag1'] = df['log_GDP'].shift(1).bfill()
df['log_GDP_lag2'] = df['log_GDP'].shift(2).bfill()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load model ────────────────────────────────────────────────────────────────
model = GDPTransformer(n_features=13)
state = torch.load(f'{MODELS_DIR}/best_model.pth', map_location=device,
                   weights_only=True)
model.load_state_dict(state)
model.to(device)
print(f"\n[1] Model loaded | forecast years: {FORECAST_YEARS}")

# ── Build scaled feature matrix ───────────────────────────────────────────────
X_known        = df[FEATURE_COLS].values.astype(np.float32)
X_known_scaled = scaler_X.transform(X_known)

# Seed window = last WINDOW_SIZE rows of known data (2021-2025)
window_buffer  = list(X_known_scaled[-WINDOW_SIZE:])

gdp_hist_years = list(years_all.astype(int))
gdp_hist_vals  = list(gdp_raw_all)

print(f"[2] Seed window: years {years_all[-WINDOW_SIZE:]} | GDP seed: {gdp_hist_vals[-1]:.2f}B")

# ── Feature row builder for future years ──────────────────────────────────────
def build_feature_row(yr, df_hist, hist_years, hist_vals):
    last = df_hist.iloc[-1]
    def get_gdp(y):
        if y in hist_years: return hist_vals[hist_years.index(y)]
        return hist_vals[-1]

    gdp_prev  = get_gdp(yr - 1)
    gdp_prev2 = get_gdp(yr - 2)
    gdp_growth_cur = (gdp_prev - gdp_prev2) / gdp_prev2 * 100

    return {
        'Investment'           : float(last['Investment']),
        'Agriculture'          : max(float(last['Agriculture']) - 0.2, 10.0),
        'Import'               : float(last['Import']),
        'Export'               : float(last['Export']),
        'Education'            : min(float(last['Education']) + 0.2, 80.0),
        'Power_Generation'     : float(last['Power_Generation']) * 1.05,
        'Mobile_Penetration'   : min(float(last['Mobile_Penetration']), 95.0),
        'Internet_Penetration' : min(float(last['Internet_Penetration']) * 1.07, 95.0),
        'Sentiment_Score'      : 0.35,
        'Crisis_Dummy'         : 0.0,
        'GDP_Growth_Rate'      : gdp_growth_cur,
        'log_GDP_lag1'         : np.log(gdp_prev),
        'log_GDP_lag2'         : np.log(gdp_prev2),
    }

# ── Monte Carlo forecast ──────────────────────────────────────────────────────
def mc_forecast(model, window_buf, n_samples, df_hist,
                hist_years, hist_vals, device, y_mean, y_std):
    all_paths = []
    for _ in range(n_samples):
        model.train()   # dropout ON → uncertainty
        buf        = [r.copy() for r in window_buf]
        h_years    = hist_years.copy()
        h_vals     = hist_vals.copy()
        path       = []
        for yr in FORECAST_YEARS:
            feat = build_feature_row(yr, df_hist, h_years, h_vals)
            feat_arr    = np.array([[feat[c] for c in FEATURE_COLS]], np.float32)
            feat_scaled = scaler_X.transform(feat_arr)[0]
            seq = np.array([buf], np.float32)
            with torch.no_grad():
                pred_norm   = model(torch.FloatTensor(seq).to(device)).item()
            pred_growth = pred_norm * y_std + y_mean
            prev_gdp    = h_vals[h_years.index(yr-1)] if (yr-1) in h_years else h_vals[-1]
            pred_gdp    = prev_gdp * (1 + pred_growth/100)
            path.append(pred_gdp)
            h_years.append(yr); h_vals.append(pred_gdp)
            buf.pop(0); buf.append(feat_scaled)
        all_paths.append(path)
    return np.array(all_paths)

all_paths = mc_forecast(model, window_buffer, MC_SAMPLES, df,
                        gdp_hist_years, gdp_hist_vals, device, y_mean, y_std)

gdp_mean   = all_paths.mean(axis=0)
gdp_median = np.median(all_paths, axis=0)
gdp_p10    = np.percentile(all_paths, 10, axis=0)
gdp_p25    = np.percentile(all_paths, 25, axis=0)
gdp_p75    = np.percentile(all_paths, 75, axis=0)
gdp_p90    = np.percentile(all_paths, 90, axis=0)

forecast_growth = []
for i, yr in enumerate(FORECAST_YEARS):
    prev = gdp_hist_vals[gdp_hist_years.index(yr-1)] if (yr-1) in gdp_hist_years \
           else gdp_mean[i-1]
    forecast_growth.append((gdp_mean[i] - prev) / prev * 100)

print(f"\n[3] Forecast (2026–2030)")
print(f"    {'Year':>6}  {'Mean GDP':>12}  {'P10':>10}  {'P90':>10}  {'Growth%':>10}")
print(f"    {'-'*6}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")
for i, yr in enumerate(FORECAST_YEARS):
    print(f"    {yr:>6}  {gdp_mean[i]:>10.2f}B  "
          f"{gdp_p10[i]:>8.2f}B  {gdp_p90[i]:>8.2f}B  "
          f"{forecast_growth[i]:>8.2f}%")

# ── Get test predictions for combined plot ────────────────────────────────────
model.eval()
with torch.no_grad():
    test_pn = model(torch.FloatTensor(X_test_seq).to(device)).cpu().numpy().flatten()
test_pg  = test_pn * y_std + y_mean
test_gdp_p, test_gdp_a = [], []
for i, yr in enumerate(test_years):
    pi = np.where(years_all==int(yr)-1)[0][0]
    ci = np.where(years_all==int(yr))[0][0]
    test_gdp_p.append(gdp_raw_all[pi]*(1+test_pg[i]/100))
    test_gdp_a.append(gdp_raw_all[ci])
test_gdp_p = np.array(test_gdp_p); test_gdp_a = np.array(test_gdp_a)

rmse = np.sqrt(np.mean((test_gdp_p-test_gdp_a)**2))
mape = np.mean(np.abs((test_gdp_a-test_gdp_p)/test_gdp_a))*100
r2   = 1-np.sum((test_gdp_a-test_gdp_p)**2)/np.sum((test_gdp_a-test_gdp_a.mean())**2)

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left — full timeline + forecast
ax = axes[0]
hist_mask = years_all <= 2014
ax.plot(years_all[hist_mask], gdp_raw_all[hist_mask], 'o-',
        color='#378ADD', linewidth=1.5, markersize=3, label='Historical (train)')
ax.plot(test_years, test_gdp_a,  'o-',
        color='#0F6E56', linewidth=2, markersize=7, label='Actual (2015–25)')
ax.plot(test_years, test_gdp_p,  's--',
        color='#D85A30', linewidth=2, markersize=7, label='Predicted (2015–25)')
ax.plot(FORECAST_YEARS, gdp_mean, 'D-',
        color='#854F0B', linewidth=2.5, markersize=8, label='Forecast (2026–30)')
ax.fill_between(FORECAST_YEARS, gdp_p10, gdp_p90,
                alpha=0.15, color='#854F0B', label='80% CI')
ax.fill_between(FORECAST_YEARS, gdp_p25, gdp_p75,
                alpha=0.30, color='#854F0B', label='50% CI')
ax.axvline(x=2014.5, color='gray', linestyle=':', linewidth=1.5)
ax.axvline(x=2025.5, color='#854F0B', linestyle=':', linewidth=1.5)
ax.set_xlabel('Year'); ax.set_ylabel('GDP (Billion USD)')
ax.set_title('India GDP — Full Timeline & Forecast (1965–2030)')
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Right — forecast zoom
ax2 = axes[1]
for path in all_paths[::5]:
    ax2.plot(FORECAST_YEARS, path, color='#EF9F27', linewidth=0.4, alpha=0.25)
ax2.plot(FORECAST_YEARS, gdp_mean,   'D-',
         color='#854F0B', linewidth=2.5, markersize=9, label='Mean', zorder=5)
ax2.plot(FORECAST_YEARS, gdp_median, 's--',
         color='#633806', linewidth=1.5, markersize=7, label='Median', zorder=5)
ax2.fill_between(FORECAST_YEARS, gdp_p10, gdp_p90,
                 alpha=0.2, color='#854F0B', label='80% CI')
ax2.fill_between(FORECAST_YEARS, gdp_p25, gdp_p75,
                 alpha=0.35, color='#854F0B', label='50% CI')
for i, yr in enumerate(FORECAST_YEARS):
    ax2.annotate(f"{gdp_mean[i]:.0f}B\n±{(gdp_p90[i]-gdp_p10[i])/2:.0f}B",
                 xy=(yr, gdp_mean[i]), xytext=(0,16),
                 textcoords='offset points', ha='center',
                 fontsize=8, color='#412402', fontweight='bold')
ax2.set_xlabel('Year'); ax2.set_ylabel('GDP (Billion USD)')
ax2.set_title(f'Forecast 2026–2030\n({MC_SAMPLES} Monte Carlo dropout passes)')
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

plt.suptitle(
    f'India GDP Transformer | Test RMSE={rmse:.1f}B | MAPE={mape:.1f}% | R²={r2:.3f}',
    fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUTS_DIR}/forecast_2026_2030.png', dpi=150)
plt.close()
print(f"\n[4] Forecast plot → outputs/forecast_2026_2030.png")

print(f"\n[5] Final Summary")
print(f"    Test (2015–2025) — RMSE: {rmse:.2f}B | MAPE: {mape:.2f}% | R²: {r2:.4f}")
print(f"\n    Forecast (2026–2030):")
print(f"    {'Year':>6}  {'Mean GDP':>12}  {'Low P10':>12}  {'High P90':>12}  {'Growth%':>10}")
print(f"    {'-'*6}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*10}")
for i, yr in enumerate(FORECAST_YEARS):
    print(f"    {yr:>6}  {gdp_mean[i]:>10.2f}B  "
          f"{gdp_p10[i]:>10.2f}B  {gdp_p90[i]:>10.2f}B  "
          f"{forecast_growth[i]:>8.2f}%")

print(f"\n{'='*60}")
print(f"STEP 5 COMPLETE")
print(f"  2030 GDP mean   : {gdp_mean[-1]:.0f}B USD")
print(f"  2030 GDP range  : {gdp_p10[-1]:.0f}B – {gdp_p90[-1]:.0f}B (80% CI)")
print(f"{'='*60}")

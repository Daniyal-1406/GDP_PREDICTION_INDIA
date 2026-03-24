import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

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

# ── Augmentation ──────────────────────────────────────────────────────────────
def augment_sequences(X, y, n_augmented=500, noise_std=0.015, seed=42):
    rng = np.random.default_rng(seed)
    aug_X, aug_y = [X.copy()], [y.copy()]
    for _ in range(n_augmented // len(X)):
        aug_X.append(X + rng.normal(0, noise_std, X.shape).astype(np.float32))
        aug_y.append(y + rng.normal(0, 0.04, y.shape).astype(np.float32))
    X_aug = np.vstack(aug_X); y_aug = np.vstack(aug_y)
    idx = rng.permutation(len(X_aug))
    return X_aug[idx], y_aug[idx]

# ── Config ────────────────────────────────────────────────────────────────────
PREPROCESSED_DIR = 'preprocessed'
MODELS_DIR       = 'models'
OUTPUTS_DIR      = 'outputs'
TRAIN_END        = 2014
EPOCHS           = 1000
BATCH_SIZE       = 32
LEARNING_RATE    = 1e-3
PATIENCE         = 100
WARMUP_EPOCHS    = 20
N_AUGMENTED      = 500
VAL_SPLIT        = 5    # hold out last 5 train years as val

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

print("=" * 60)
print("STEP 4 — TRAINING  (1960-2014 train | 2015-2025 test)")
print("=" * 60)

# ── Load ──────────────────────────────────────────────────────────────────────
X_train_seq = np.load(f'{PREPROCESSED_DIR}/X_train_seq.npy')
y_train_seq = np.load(f'{PREPROCESSED_DIR}/y_train_seq.npy')
X_test_seq  = np.load(f'{PREPROCESSED_DIR}/X_test_seq.npy')
y_test_seq  = np.load(f'{PREPROCESSED_DIR}/y_test_seq.npy')
train_years = np.load(f'{PREPROCESSED_DIR}/train_years.npy')
test_years  = np.load(f'{PREPROCESSED_DIR}/test_years.npy')
gdp_raw_all = np.load(f'{PREPROCESSED_DIR}/gdp_raw_all.npy')
years_all   = np.load(f'{PREPROCESSED_DIR}/years_all.npy')

# z-score normalise target — fit on train only
y_mean = float(y_train_seq.mean())
y_std  = float(y_train_seq.std())
y_train_norm = ((y_train_seq - y_mean) / y_std).astype(np.float32)
y_test_norm  = ((y_test_seq  - y_mean) / y_std).astype(np.float32)

np.save(f'{PREPROCESSED_DIR}/y_gr_mean.npy', np.array([y_mean]))
np.save(f'{PREPROCESSED_DIR}/y_gr_std.npy',  np.array([y_std]))

print(f"\n[1] Loaded")
print(f"    X_train_seq : {X_train_seq.shape}  | X_test_seq: {X_test_seq.shape}")
print(f"    y mean={y_mean:.3f}%  std={y_std:.3f}%")
print(f"    Train years : {train_years[0]}–{train_years[-1]}")
print(f"    Test  years : {test_years}")
print(f"    y_test range: [{y_test_seq.min():.2f}%, {y_test_seq.max():.2f}%]")

# ── Val split → augment train ─────────────────────────────────────────────────
val_idx      = len(X_train_seq) - VAL_SPLIT
X_tr_orig    = X_train_seq[:val_idx]
y_tr_orig    = y_train_norm[:val_idx]
X_val        = X_train_seq[val_idx:]
y_val        = y_train_norm[val_idx:]

X_tr_aug, y_tr_aug = augment_sequences(X_tr_orig, y_tr_orig,
                                        n_augmented=N_AUGMENTED,
                                        noise_std=0.015, seed=42)

print(f"\n[2] Train: {len(X_tr_aug)} augmented ({len(X_tr_orig)} original)")
print(f"    Val  : {len(X_val)} samples — years {train_years[val_idx:train_years[-1]+1]}")

# ── Tensors ───────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[3] Device: {device}")

to_t = lambda a: torch.FloatTensor(a).to(device)
X_tr_t,  y_tr_t  = to_t(X_tr_aug), to_t(y_tr_aug)
X_val_t, y_val_t = to_t(X_val),    to_t(y_val)
X_test_t          = to_t(X_test_seq)
X_orig_t          = to_t(X_train_seq)

loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                    batch_size=BATCH_SIZE, shuffle=True)

# ── Model + optimizer ─────────────────────────────────────────────────────────
torch.manual_seed(42)
model     = GDPTransformer(n_features=13).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                             betas=(0.9, 0.999), weight_decay=1e-4)

def lr_fn(e):
    if e < WARMUP_EPOCHS: return (e+1) / WARMUP_EPOCHS
    p = (e - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
    return max(0.02, 0.5*(1 + math.cos(math.pi*p)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)
n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[4] Params: {n_params:,} | LR: {LEARNING_RATE} | Epochs: {EPOCHS} | Patience: {PATIENCE}")

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\n[5] Training...")
print(f"    {'Epoch':>6}  {'Train MSE':>12}  {'Val MSE':>12}  {'LR':>12}")
print(f"    {'-'*6}  {'-'*12}  {'-'*12}  {'-'*12}")

tr_losses, vl_losses       = [], []
best_vl, best_ep, pat_ctr  = float('inf'), 0, 0
best_state                  = None

for ep in range(1, EPOCHS+1):
    model.train()
    ep_tr = 0.0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ep_tr += loss.item()
    ep_tr /= len(loader)

    model.eval()
    with torch.no_grad():
        ep_vl = criterion(model(X_val_t), y_val_t).item()

    scheduler.step()
    tr_losses.append(ep_tr); vl_losses.append(ep_vl)

    if ep % 100 == 0 or ep == 1:
        print(f"    {ep:>6}  {ep_tr:>12.6f}  {ep_vl:>12.6f}  "
              f"{optimizer.param_groups[0]['lr']:>12.8f}")

    if ep_vl < best_vl:
        best_vl, best_ep, pat_ctr = ep_vl, ep, 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    else:
        pat_ctr += 1
        if pat_ctr >= PATIENCE:
            print(f"\n    Early stop ep {ep} (best: ep {best_ep}, val: {best_vl:.6f})")
            break

# ── Save ──────────────────────────────────────────────────────────────────────
model.load_state_dict(best_state)
torch.save(best_state, f'{MODELS_DIR}/best_model.pth')
print(f"\n[6] Saved → models/best_model.pth  (epoch {best_ep})")

# ── Loss curve ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(tr_losses, label='Train MSE (augmented)', color='#378ADD', linewidth=2)
ax.plot(vl_losses, label='Val MSE (original)',    color='#D85A30', linewidth=2)
ax.axvline(x=best_ep-1, color='#639922', linestyle='--',
           linewidth=1.5, label=f'Best epoch ({best_ep})')
mv = int(np.argmin(vl_losses))
ax.annotate(f'Min val\nep {mv+1}\n{vl_losses[mv]:.4f}',
            xy=(mv, vl_losses[mv]),
            xytext=(min(mv+50, len(vl_losses)-100),
                    vl_losses[mv]+(max(vl_losses)-min(vl_losses))*0.3),
            arrowprops=dict(arrowstyle='->', color='#333'), fontsize=9)
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss (normalised growth)')
ax.set_title('Train vs Val Loss — GDP Transformer (1960–2014)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUTS_DIR}/loss_curve.png', dpi=150)
plt.close()
print(f"[7] Loss curve → outputs/loss_curve.png")

# ── Evaluate ──────────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    preds_norm = model(X_test_t).cpu().numpy().flatten()

preds_growth  = preds_norm * y_std + y_mean
actual_growth = y_test_seq.flatten()

preds_gdp, actuals_gdp = [], []
for i, yr in enumerate(test_years):
    prev_idx = np.where(years_all == int(yr)-1)[0][0]
    prev_gdp = gdp_raw_all[prev_idx]
    preds_gdp.append(prev_gdp * (1 + preds_growth[i]/100))
    curr_idx = np.where(years_all == int(yr))[0][0]
    actuals_gdp.append(gdp_raw_all[curr_idx])

preds_gdp   = np.array(preds_gdp)
actuals_gdp = np.array(actuals_gdp)

rmse_gr  = np.sqrt(np.mean((preds_growth - actual_growth)**2))
r2_gr    = 1 - np.sum((actual_growth-preds_growth)**2)/np.sum((actual_growth-actual_growth.mean())**2)
rmse_gdp = np.sqrt(np.mean((preds_gdp-actuals_gdp)**2))
mape_gdp = np.mean(np.abs((actuals_gdp-preds_gdp)/actuals_gdp))*100
r2_gdp   = 1 - np.sum((actuals_gdp-preds_gdp)**2)/np.sum((actuals_gdp-actuals_gdp.mean())**2)

print(f"\n[8] Test results (2015–2025)")
print(f"\n    Growth rate:")
print(f"    {'Year':>6}  {'Actual%':>10}  {'Pred%':>8}  {'Err%':>8}")
print(f"    {'-'*6}  {'-'*10}  {'-'*8}  {'-'*8}")
for i, yr in enumerate(test_years):
    print(f"    {int(yr):>6}  {actual_growth[i]:>8.2f}%  "
          f"{preds_growth[i]:>6.2f}%  "
          f"{abs(actual_growth[i]-preds_growth[i]):>6.2f}%")

print(f"\n    Recovered GDP:")
print(f"    {'Year':>6}  {'Actual':>12}  {'Predicted':>12}  {'Error':>10}  {'Err%':>8}")
print(f"    {'-'*6}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*8}")
for i, yr in enumerate(test_years):
    a = actuals_gdp[i]; p = preds_gdp[i]
    print(f"    {int(yr):>6}  {a:>10.2f}B  {p:>10.2f}B  "
          f"{abs(a-p):>8.2f}B  {abs(a-p)/a*100:>7.1f}%")

print(f"\n    Growth — RMSE: {rmse_gr:.2f}%  |  R²: {r2_gr:.4f}")
print(f"    GDP    — RMSE: {rmse_gdp:.2f}B  |  R²: {r2_gdp:.4f}  |  MAPE: {mape_gdp:.2f}%")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.plot(test_years, actual_growth, 'o-', label='Actual',
        color='#378ADD', linewidth=2, markersize=8)
ax.plot(test_years, preds_growth,  's--', label='Predicted',
        color='#D85A30', linewidth=2, markersize=8)
for i, yr in enumerate(test_years):
    ax.annotate(f"{preds_growth[i]:.1f}%", xy=(yr, preds_growth[i]),
                xytext=(0,12), textcoords='offset points',
                ha='center', fontsize=7, color='#993C1D')
ax.set_xlabel('Year'); ax.set_ylabel('Growth Rate (%)')
ax.set_title(f'Growth Rate (2015–2025)\nRMSE:{rmse_gr:.2f}% | R²:{r2_gr:.3f}')
ax.legend(); ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(test_years, actuals_gdp, 'o-', label='Actual',
         color='#378ADD', linewidth=2, markersize=8)
ax2.plot(test_years, preds_gdp,   's--', label='Predicted',
         color='#D85A30', linewidth=2, markersize=8)
for i, yr in enumerate(test_years):
    ax2.annotate(f"{preds_gdp[i]:.0f}B", xy=(yr, preds_gdp[i]),
                 xytext=(0,12), textcoords='offset points',
                 ha='center', fontsize=7, color='#993C1D')
ax2.set_xlabel('Year'); ax2.set_ylabel('GDP (Billion USD)')
ax2.set_title(f'Recovered GDP (2015–2025)\nRMSE:{rmse_gdp:.1f}B | R²:{r2_gdp:.3f}')
ax2.legend(); ax2.grid(True, alpha=0.3)

with torch.no_grad():
    tr_norm = model(X_orig_t).cpu().numpy().flatten()
tr_gr = tr_norm * y_std + y_mean
tr_gdp_p, tr_gdp_a = [], []
for i, yr in enumerate(train_years):
    pi = np.where(years_all==int(yr)-1)[0][0]
    ci = np.where(years_all==int(yr))[0][0]
    tr_gdp_p.append(gdp_raw_all[pi]*(1+tr_gr[i]/100))
    tr_gdp_a.append(gdp_raw_all[ci])
tr_gdp_p = np.array(tr_gdp_p); tr_gdp_a = np.array(tr_gdp_a)

ax3 = axes[2]
ax3.plot(train_years, tr_gdp_a, 'o-',
         color='#378ADD', linewidth=1.5, markersize=3, label='Actual (train)')
ax3.plot(train_years, tr_gdp_p, '--',
         color='#85B7EB', linewidth=1.5, label='Fit (train)', alpha=0.8)
ax3.plot(test_years, actuals_gdp, 'o-',
         color='#0F6E56', linewidth=2, markersize=7, label='Actual (test)')
ax3.plot(test_years, preds_gdp, 's--',
         color='#D85A30', linewidth=2, markersize=7, label='Predicted (test)')
ax3.axvline(x=2014.5, color='gray', linestyle=':', linewidth=1.5, label='Split')
ax3.set_xlabel('Year'); ax3.set_ylabel('GDP (Billion USD)')
ax3.set_title('Full Timeline (1965–2025)')
ax3.legend(fontsize=7); ax3.grid(True, alpha=0.3)

plt.suptitle(
    f'GDP Transformer | 1960-2025 | RMSE={rmse_gdp:.1f}B | MAPE={mape_gdp:.1f}% | R²={r2_gdp:.3f}',
    fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUTS_DIR}/actual_vs_predicted.png', dpi=150)
plt.close()
print(f"[9] Plot → outputs/actual_vs_predicted.png")

print(f"\n{'='*60}")
print(f"STEP 4 COMPLETE")
print(f"  Growth RMSE : {rmse_gr:.2f}%  |  R² : {r2_gr:.4f}")
print(f"  GDP RMSE    : {rmse_gdp:.2f}B  |  R² : {r2_gdp:.4f}")
print(f"  GDP MAPE    : {mape_gdp:.2f}%")
if   r2_gdp >= 0.93: print(f"  STATUS: EXCELLENT — proceed to step5")
elif r2_gdp >= 0.87: print(f"  STATUS: GOOD — proceed to step5")
elif r2_gdp >= 0.80: print(f"  STATUS: FAIR — try noise_std=0.01")
else:                print(f"  STATUS: POOR — try different seed")
print(f"{'='*60}")

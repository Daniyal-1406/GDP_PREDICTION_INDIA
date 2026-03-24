import torch
import torch.nn as nn
import math

INPUT_FEATURES = 13
D_MODEL        = 64    # bigger now — 50 train samples can support it
N_HEADS        = 4
N_LAYERS       = 2     # 2 layers with more data
FFN_DIM        = 128
DROPOUT        = 0.1
SEQ_LEN        = 5

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
    """
    Transformer encoder for India GDP growth rate forecasting.
    Input  : (batch, 5, 13)  — 5-year window, 13 features
    Output : (batch, 1)      — next year GDP growth rate (normalised)
    Recovery: GDP(t) = GDP(t-1) * (1 + pred_growth/100)

    Features [0-9] : Investment, Agriculture, Import, Export, Education,
                     Power_Generation, Mobile_Penetration, Internet_Penetration,
                     Sentiment_Score, Crisis_Dummy
    Features [10]  : GDP_Growth_Rate  (current year %)
    Features [11]  : log_GDP_lag1     (log GDP last year)
    Features [12]  : log_GDP_lag2     (log GDP 2 years ago)
    """
    def __init__(self, n_features=INPUT_FEATURES):
        super().__init__()
        self.input_projection    = nn.Linear(n_features, D_MODEL)
        self.pos_encoder         = PositionalEncoding(D_MODEL, dropout=DROPOUT)
        enc = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=FFN_DIM,
            dropout=DROPOUT, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(enc, num_layers=N_LAYERS)
        self.output_head = nn.Sequential(
            nn.Linear(D_MODEL, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.input_projection(x)       # (batch, 5, 64)
        x = self.pos_encoder(x)            # (batch, 5, 64)
        x = self.transformer_encoder(x)    # (batch, 5, 64)
        x = x.mean(dim=1)                  # (batch, 64)
        return self.output_head(x)         # (batch, 1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("=" * 60)
    print("STEP 3 — MODEL DEFINITION  (d_model=64, 2 layers)")
    print("=" * 60)

    model = GDPTransformer()
    print(f"\n{model}")
    total = count_parameters(model)
    print(f"\nTrainable parameters: {total:,}")

    dummy_in  = torch.randn(8, SEQ_LEN, INPUT_FEATURES)
    model.eval()
    with torch.no_grad():
        dummy_out = model(dummy_in)

    print(f"\nInput  shape : {dummy_in.shape}  (batch, seq_len, features)")
    print(f"Output shape : {dummy_out.shape}  (batch, 1) — growth rate")

    print(f"\n{'='*60}")
    print(f"STEP 3 COMPLETE")
    print(f"  Params : {total:,}  (was 9,537 — bigger model, more data)")
    print(f"{'='*60}")

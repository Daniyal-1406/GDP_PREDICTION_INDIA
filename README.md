# India GDP Prediction (1960–2030)

This project predicts India's GDP from 2026 to 2030 using a Transformer model trained on 65 years of macroeconomic data. It also generates confidence intervals using Monte Carlo Dropout to account for uncertainty in the forecast.

---

## Why this project?

India's economy has gone through a lot — wars, oil shocks, liberalisation, demonetisation, COVID. Standard regression models don't capture these complex patterns well. This project uses a Transformer encoder (the same attention mechanism behind modern LLMs) to learn from sequences of economic indicators and predict where the GDP is heading.

---

## Data

- **Source**: World Bank + custom sentiment index
- **Range**: 1960–2025 (65 years)
- **Train**: 1960–2014
- **Test**: 2015–2025
- **Forecast**: 2026–2030

### Features used (13 total)

- Investment, Agriculture, Imports, Exports (all as % of GDP)
- Literacy rate, Power generation, Mobile & Internet penetration
- Sentiment score, Crisis dummy flag
- GDP growth rate, Log GDP lag 1 & 2

Crisis years are flagged explicitly: 1965–66, 1971, 1974, 1979–80, 1991, 1999, 2001, 2008–09, 2016–17, 2020.

---

## Model

A Transformer encoder with:
- 5-year sliding window as input
- 4 attention heads, 2 encoder layers
- Output: next year's GDP growth rate (%)
- GDP is then recovered as: `GDP(t) = GDP(t-1) × (1 + growth / 100)`

For the 2026–2030 forecast, the model runs 200 Monte Carlo passes (dropout kept on) to produce 50% and 80% confidence intervals.

---

## Project Structure

```
GDP_PREDICTION_INDIA/
├── data/                        # Raw dataset (1960–2025)
├── step2_preprocessing.py       # Cleaning, feature engineering, sequence creation
├── step3_model.py               # Transformer architecture
├── step4_train.py               # Training loop
├── step5_forecast.py            # Forecast + confidence interval plots
├── requirements.txt
└── .gitignore
```

---

## How to Run

```bash
git clone https://github.com/Daniyal-1406/GDP_PREDICTION_INDIA.git
cd GDP_PREDICTION_INDIA
pip install -r requirements.txt

python step2_preprocessing.py
python step3_model.py
python step4_train.py
python step5_forecast.py
```

Forecast charts will be saved in the `outputs/` folder.

---

## Stack

Python · PyTorch · scikit-learn · pandas · matplotlib

---

## Author

Daniyal — [GitHub](https://github.com/Daniyal-1406)

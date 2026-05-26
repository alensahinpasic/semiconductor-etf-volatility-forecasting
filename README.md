# Replication Notebook — README
## Forecasting Short-Term Semiconductor ETF Volatility
### A Multi-Layer HAR Model with Microstructure Indicators and NLP Sentiment

**Author:** Alen Sahinpasic | Central European University | May 2026

---

## What This Is

This repository contains the full replication pipeline for a BA thesis on short-term volatility forecasting for semiconductor ETFs. A single Jupyter notebook (`replication_notebook.ipynb`) takes two inputs and reproduces every table, figure, and result in the thesis from scratch.

**Two inputs required:**
1. `data/LSEG_headlines_clean.csv` — provided in this repository
2. Public OHLCV market data — downloaded automatically from Yahoo Finance when you run the notebook

Everything else is computed inside the notebook.

---

## Repository Structure

```
.
├── data/
│   └── LSEG_headlines_clean.csv        ← Place this here before running
├── replication_notebook.ipynb          ← Run this
├── variable_dictionary.csv             ← Description of all variables
└── README.md                           ← This file
```

After running the notebook, the following folders are created automatically:

```
data/
│   ├── ohlcv_SMH.csv                   ← Raw OHLCV for SMH
│   ├── ohlcv_SOXX.csv                  ← Raw OHLCV for SOXX
│   ├── gk_volatility.csv               ← Garman-Klass daily volatility
│   ├── forecast_targets.csv            ← h-day average volatility targets (h=1..5)
│   ├── microstructure.csv              ← Log range and relative volume
│   ├── headlines_scored.csv            ← Per-headline FinBERT scores
│   └── daily_sentiment.csv             ← Daily aggregated sentiment S_t

results/
│   ├── forecasts_SMH.csv               ← OOS forecasts for SMH (M1, M2, M3)
│   ├── forecasts_SOXX.csv              ← OOS forecasts for SOXX
│   ├── loss_metrics.csv                ← RMSE, MAE, QLIKE for all models
│   ├── dm_tests.csv                    ← Diebold-Mariano test results
│   ├── coefficients.csv                ← Expanding-window OLS coefficients
│   ├── conditional_microstructure_gains.csv
│   │
│   ├── tables/
│   │   ├── table1_summary_stats_clean.csv
│   │   ├── table2_main_microstructure_vs_har.csv
│   │   ├── table3_main_dm_microstructure_vs_har.csv
│   │   ├── table4_conditional_microstructure_gains.csv
│   │   ├── appendix_table_A1_sentiment_vs_har.csv
│   │   ├── appendix_table_A2_three_model_loss.csv
│   │   └── appendix_table_A3_all_dm_tests.csv
│   │
│   └── figures/
│       ├── fig1.1_data_pipeline.png
│       ├── fig1.2_gk_series.png
│       ├── fig2_oos_forecast.png
│       ├── fig3_rolling_rmse.png
│       ├── fig4_coef_decay.png
│       ├── figA_news_coverage.png
│       ├── figB_news_by_year.png
│       ├── figC_sentiment_series.png
│       ├── figD_sentiment_distribution.png
│       ├── figE_sentiment_vs_vol.png
│       ├── figF_rmse_comparison.png
│       └── figG_conditional_microstructure_gains.png
```

---

## Setup

### Requirements

```bash
pip install pandas numpy yfinance transformers torch scikit-learn statsmodels matplotlib seaborn scipy
```

Python 3.9 or above recommended.

### Hardware

FinBERT scoring (Section 8, ~23,000 headlines) is the only compute-intensive step:

| Hardware | Estimated time |
|----------|---------------|
| Apple Silicon (MPS) | 5–10 min |
| CUDA GPU | 3–5 min |
| CPU only | 20–40 min |

All other sections run in under 15 minutes on any hardware.

---

## How to Run

1. Clone or unzip this repository
2. Place `LSEG_headlines_clean.csv` inside the `data/` folder
3. Open `replication_notebook.ipynb` in Jupyter
4. Run all cells top to bottom — **Kernel → Restart & Run All**
5. All outputs save automatically to `results/tables/` and `results/figures/`

Do not skip sections — each depends on variables created by the previous one.

---

## Notebook Sections

| Section | What it does | Key output saved |
|---------|-------------|-----------------|
| 2 | Imports and project paths | Creates folder structure |
| 3 | Loads LSEG headlines CSV | — |
| 4 | Downloads OHLCV from Yahoo Finance | `ohlcv_SMH.csv`, `ohlcv_SOXX.csv` |
| 5 | Computes Garman-Klass volatility | `gk_volatility.csv` |
| 6 | Computes h-day forecast targets | `forecast_targets.csv` |
| 7 | Computes log range and relative volume | `microstructure.csv` |
| 8 | Scores headlines with ProsusAI/finbert | `headlines_scored.csv` |
| 9 | Aggregates to daily sentiment S_t | `daily_sentiment.csv` |
| 10 | Merges all series into modelling dataset | — |
| 11 | Defines OLS and Newey-West functions | — |
| 12 | Expanding-window OLS for M1, M2, M3 | `forecasts_SMH.csv`, `forecasts_SOXX.csv`, `coefficients.csv` |
| 13 | Computes RMSE, MAE, QLIKE | `loss_metrics.csv` |
| 14 | Bonferroni-corrected DM tests | `dm_tests.csv` |
| 14B | Conditional RMSE by market condition | `conditional_microstructure_gains.csv` |
| 15 | Generates all thesis tables | `results/tables/*.csv` |
| 16 | Generates all thesis figures | `results/figures/*.png` |
| 17 | Verifies all files and key numerical values | Prints PASS/FAIL report |
| 18 | Prints full file summary with sizes | — |

---

## Models

| Label | Specification |
|-------|--------------|
| **M1** | HAR: `GK_{t,h} = β₀ + β_d·GK_{t-1} + β_w·GK_{t-5:t-1} + β_m·GK_{t-22:t-1} + ε` |
| **M2** | HAR + FinBERT sentiment: M1 + `β_s·S_t` |
| **M3** | HAR + microstructure: M1 + `β_r·Range_t + β_v·RelVol_t` |

- **Estimation:** Expanding-window OLS with Newey-West HAC standard errors (lag = h)
- **In-sample:** 2020-01-02 to 2024-04-30 (1,089 trading days)
- **Out-of-sample:** 2024-05-01 to 2025-12-31 (419 trading days at h=1)
- **ETFs:** SMH, SOXX
- **Horizons:** h = 1, 2, 3, 4, 5 trading days

---

## Input Data: LSEG_headlines_clean.csv

| Column | Type | Description |
|--------|------|-------------|
| `date` | string YYYY-MM-DD | Trading day |
| `time` | string HH:MM:SS | Publication time UTC; null on no-news days |
| `source` | string | News provider: RTRS=Reuters, ASSPR=AP |
| `ticker` | string | Refinitiv RIC ticker(s); null for sector-level headlines |
| `headline` | string | Full headline text; null on no-news placeholder rows |

**Coverage:** 23,430 headlines, January 2020 – December 2025  
**Note:** 42 rows have null headlines — these are placeholder rows for trading days with zero semiconductor news, intentionally retained to maintain calendar alignment. Section 9 assigns S_t = 0 to these days automatically.

---

## Key Results (verified by Section 17)

| ETF | h | Model | RMSE |
|-----|---|-------|------|
| SMH | 1 | M1 HAR | 0.007474 |
| SMH | 1 | M3 Microstructure | 0.006845 |
| SOXX | 1 | M1 HAR | 0.007691 |
| SOXX | 1 | M3 Microstructure | 0.007051 |

RMSE improvement from M3 vs M1: **5.8% to 9.9%** across all ETF-horizon combinations.

---

## Reproducibility Notes

- Market data is re-downloaded each run; results are stable within the fixed 2020–2025 date range
- FinBERT scores are deterministic given the same model weights (ProsusAI/finbert, HuggingFace)
- Section 17 runs automated verification with tolerance 1e-4 and prints a full PASS/FAIL report

---

## Citation

> Sahinpasic, Alen. 2026. "Forecasting Short-Term Semiconductor ETF Volatility: A Multi-Layer HAR Model with Microstructure Indicators and NLP Sentiment." BA Thesis, Central European University, Vienna.

## Licence

Code and derived data: MIT licence. News headline text: sourced from Refinitiv/LSEG, subject to their terms of use.

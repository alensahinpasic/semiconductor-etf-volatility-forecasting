# Variable Dictionary
## Semiconductor ETF Volatility Forecasting Dataset

**Author:** Alen Sahinpasic | Central European University | May 2026

This document describes every variable across all dataset components produced by the replication notebook.

---

## 1. Input News Data — `LSEG_headlines_clean.csv`

| Variable | Type | Description | Notes |
|----------|------|-------------|-------|
| `date` | string (YYYY-MM-DD) | Trading day of the headline | Aligned to NYSE/NASDAQ calendar |
| `time` | string (HH:MM:SS) | Publication time in UTC | Null for no-news placeholder rows |
| `source` | string | News provider code | RTRS = Reuters, ASSPR = AP, ASINIK, FT |
| `ticker` | string | Refinitiv RIC ticker(s) for the headline | Null for sector-level headlines; multiple tickers separated by space |
| `headline` | string | Full headline text as published | Null for no-news placeholder rows (42 rows total) |

**Coverage:** 23,430 headlines, January 2020 – December 2025  
**No-news rows:** 42 rows have null headlines — these are placeholder rows for trading days with zero semiconductor news, retained to maintain trading calendar alignment.

---

## 2. Market Data — `data/ohlcv_SMH.csv`, `data/ohlcv_SOXX.csv`

Downloaded from Yahoo Finance via yfinance. One file per ETF.

| Variable | Type | Units | Description |
|----------|------|-------|-------------|
| `Date` | datetime | — | Trading day |
| `Open` | float | USD | Daily opening price |
| `High` | float | USD | Daily high price |
| `Low` | float | USD | Daily low price |
| `Close` | float | USD | Daily closing (adjusted) price |
| `Volume` | integer | shares | Daily trading volume |

---

## 3. Volatility Target — `data/gk_volatility.csv`

| Variable | Type | Units | Description | Formula |
|----------|------|-------|-------------|---------|
| `Date` | datetime | — | Trading day | — |
| `SMH` | float | dimensionless | Garman-Klass daily volatility for SMH | See below |
| `SOXX` | float | dimensionless | Garman-Klass daily volatility for SOXX | See below |

**Garman-Klass estimator:**

$$GK_t = \sqrt{0.5 \left[\ln\frac{H_t}{L_t}\right]^2 - (2\ln 2 - 1)\left[\ln\frac{C_t}{O_t}\right]^2}$$

Uses both the intraday high-low range and the open-to-close price movement. More efficient than squared close-to-close returns. Values floored at zero.

---

## 4. Forecast Targets — `data/forecast_targets.csv`

| Variable | Type | Description | Formula |
|----------|------|-------------|---------|
| `SMH_h1` | float | 1-day ahead average GK volatility for SMH | (1/1) × GK_{t+1} |
| `SMH_h2` | float | 2-day ahead average GK volatility for SMH | (1/2) × (GK_{t+1} + GK_{t+2}) |
| `SMH_h3` | float | 3-day ahead average GK volatility for SMH | (1/3) × Σ GK_{t+j}, j=1..3 |
| `SMH_h4` | float | 4-day ahead average GK volatility for SMH | (1/4) × Σ GK_{t+j}, j=1..4 |
| `SMH_h5` | float | 5-day ahead average GK volatility for SMH | (1/5) × Σ GK_{t+j}, j=1..5 |
| `SOXX_h1` … `SOXX_h5` | float | Same as above for SOXX | — |

These are the dependent variables in all regression models.

---

## 5. HAR Components

Constructed inside the modelling dataset. Used as regressors in all three models.

| Variable | Description | Formula |
|----------|-------------|---------|
| `gk_d` | Daily HAR component | GK_{t-1} |
| `gk_w` | Weekly HAR component | (1/5) × Σ GK_{t-j}, j=1..5 |
| `gk_m` | Monthly HAR component | (1/22) × Σ GK_{t-j}, j=1..22 |

---

## 6. Microstructure Variables — `data/microstructure.csv`

| Variable | Type | Units | Description | Formula | Notes |
|----------|------|-------|-------------|---------|-------|
| `SMH_range` | float | dimensionless | Log high-low range for SMH | ln(H_t / L_t) | Always positive; structurally related to GK_t since both use H and L |
| `SOXX_range` | float | dimensionless | Log high-low range for SOXX | ln(H_t / L_t) | — |
| `SMH_relvol` | float | ratio | Relative volume for SMH | Volume_t / MA_20(Volume_t) | First 19 observations null due to rolling window; values near 1.0 = normal volume |
| `SOXX_relvol` | float | ratio | Relative volume for SOXX | Volume_t / MA_20(Volume_t) | — |

**Caveat:** `Range_t` and `GK_t` are both derived from the daily high-low spread. This creates structural overlap between the M3 predictor and the volatility target. The range coefficient should be interpreted as capturing persistence in intraday price dispersion, not as a fully independent information source.

---

## 7. NLP Sentiment Variables

### Headline-level — `data/headlines_scored.csv`

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `headline` | string | — | Original headline text |
| `date` | datetime | — | Trading day |
| `positive` | float | [0, 1] | FinBERT probability of positive tone |
| `negative` | float | [0, 1] | FinBERT probability of negative tone |
| `neutral` | float | [0, 1] | FinBERT probability of neutral tone |
| `net_sentiment` | float | [-1, 1] | Net sentiment score: P(positive) − P(negative) |

Model: `ProsusAI/finbert` (Araci 2019), a BERT model fine-tuned on financial text.

### Daily aggregated — `data/daily_sentiment.csv`

| Variable | Type | Range | Description | Notes |
|----------|------|-------|-------------|-------|
| `date` | datetime | — | Trading day | — |
| `S_t` | float | [-1, 1] | Daily average net sentiment | Mean of net_sentiment across all headlines on day t; set to 0 on no-news days |
| `S_pos_t` | float | [0, 1] | Daily average positive probability | — |
| `S_neg_t` | float | [0, 1] | Daily average negative probability | — |
| `n_headlines` | integer | ≥ 0 | Number of headlines on day t | 0 on no-news days |

**Design choice:** S_t is a simple unweighted daily average. It does not account for source importance, firm-level relevance, event type, or publication timing. This is intentional — the variable is designed to be a transparent and reproducible baseline sentiment signal.

---

## 8. Forecast Outputs — `results/forecasts_SMH.csv`, `results/forecasts_SOXX.csv`

| Variable | Description |
|----------|-------------|
| `date` | Out-of-sample forecast date |
| `realized_h{1..5}` | Realised average GK volatility over next h days |
| `M1_h{1..5}_forecast` | M1 (HAR) forecast for horizon h |
| `M2_h{1..5}_forecast` | M2 (HAR + sentiment) forecast for horizon h |
| `M3_h{1..5}_forecast` | M3 (HAR + microstructure) forecast for horizon h |

---

## 9. Loss Metrics — `results/loss_metrics.csv`

| Variable | Description |
|----------|-------------|
| `ETF` | SMH or SOXX |
| `horizon` | Forecast horizon h (1–5) |
| `model` | M1, M2, or M3 |
| `RMSE` | Root mean squared error over out-of-sample period |
| `MAE` | Mean absolute error over out-of-sample period |
| `QLIKE` | QLIKE loss: mean(log(forecast) + realized/forecast) |
| `N_forecasts` | Number of out-of-sample forecasts used |

---

## 10. Diebold-Mariano Tests — `results/dm_tests.csv`

| Variable | Description |
|----------|-------------|
| `ETF` | SMH or SOXX |
| `horizon` | Forecast horizon h (1–5) |
| `comparison` | Model comparison label (e.g. M3_vs_M1) |
| `DM_stat` | Diebold-Mariano test statistic (positive = challenger better) |
| `p_raw` | Raw one-sided p-value |
| `p_bonf` | Bonferroni-corrected p-value (multiplied by 5 horizons) |
| `significant_5pct` | Boolean: True if p_bonf < 0.05 |
| `N` | Number of forecast pairs used |

Long-run variance estimated using Newey-West correction with lag = h.

---

## 11. Conditional Analysis — `results/conditional_microstructure_gains.csv`

| Variable | Description |
|----------|-------------|
| `ETF` | SMH or SOXX |
| `Condition` | Conditioning dimension (e.g. Realized volatility, News volume) |
| `Group` | Subgroup label (e.g. High-volatility days, Quiet-news days) |
| `N` | Number of out-of-sample days in this subgroup |
| `M3_vs_M1_pct` | RMSE improvement (%): 100 × (RMSE_M1 − RMSE_M3) / RMSE_M1 |

Positive values mean M3 has lower RMSE than M1 within that subgroup. Groups are defined ex-post using realised out-of-sample data — they are diagnostic, not real-time trading signals.

---

## References

- Araci, Dogu. 2019. "FinBERT: Financial Sentiment Analysis with Pre-Trained Language Models." arXiv:1908.10063.
- Corsi, F. 2008. "A Simple Approximate Long-Memory Model of Realized Volatility." Journal of Financial Econometrics 7(2): 174–196.
- Garman, Mark B., and Michael J. Klass. 1980. "On the Estimation of Security Price Volatilities from Historical Data." Journal of Business 53(1): 67.
- Diebold, Francis X., and Roberto S. Mariano. 1995. "Comparing Predictive Accuracy." Journal of Business & Economic Statistics 13(3): 253–263.
- Newey, Whitney K., and Kenneth D. West. 1987. "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix." Econometrica 55(3): 703.

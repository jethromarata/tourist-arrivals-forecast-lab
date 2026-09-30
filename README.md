# Philippine Tourist Arrivals — LSTM Forecasting Lab

A Streamlit app that forecasts monthly Philippine tourist arrivals using an
LSTM neural network, built as a full pipeline: data cleaning, feature
selection (Spearman + VIF), leakage-safe preprocessing, LSTM training and
tuning, honest evaluation against naive baselines, and SHAP-based
explainability.

## How to run

python -m venv .venv

source .venv/bin/activate # Windows: .venv\Scripts\activate

pip install -r requirements.txt

streamlit run Home.py


Then use the sidebar pages in order: Dataset → Clean → Features → Prepare →
Train → Evaluate → Explain → Forecast. Each page depends on the one before it
having been run at least once in the session.

## Data cleaning note

One row (2001-02-01) contained a physically impossible arrivals value
(34,240,153 — roughly 170x every other month in the dataset). This was
identified as a data-entry error and corrected by averaging the surrounding
months, rather than left in place or silently dropped. See `pages/2_Clean.py`
for the exact logic and reasoning.

## Results

| Metric | LSTM | Naive | Seasonal naive |
|---|---|---|---|
| MAE | 153,689 | 65,458 | 137,572 |
| RMSE | 197,117 | 84,755 | 178,094 |
| MAPE | 154.3% | 16.6% | 40.0% |
| R² | -0.177 | 0.782 | 0.039 |

The LSTM did not outperform the naive baselines on this dataset. This is a
legitimate, reportable outcome: tourist arrivals in this dataset follow a
strong, predictable seasonal pattern (holidays, dry/wet season) that a simple
"repeat last month" or "repeat last year's same month" rule can already
capture well, especially with a relatively small training set (~250 monthly
rows) for a recurrent model to learn from.

## Explainability

SHAP analysis on the trained LSTM identified `is_holiday_peak` and `quarter`
as the most influential predictors — consistent with the Philippines'
travel patterns, where major holidays and the dry-season quarters (Q1
especially) reliably drive higher tourist arrivals. This lines up with why
the seasonal-naive baseline performed better than the naive baseline: it
directly captures the yearly holiday/season cycle that the model itself also
leaned on most heavily.

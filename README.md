# Earth System Trend Detective: Jordan and surroundings

NASA Space Apps Challenge 2026, challenge "Be an Earth System Trend Detective".

This project takes Earth science variables for a region, shows how they changed from 2000 to 2025, and answers four questions: **what** is changing, **where**, **how much**, and **is the change statistically significant?** The analysis behind the tool is a set of reproducible scripts that export JSON files for the web interface.

## What we found

Region: about 29.3-33.8 °N, 34.7-39.1 °E, split into 63 grid cells (0.5° x 0.625°). Period: 2000-2025.

| Variable | Whole region, change per decade | Significant? |
|---|---|---|
| Air temperature | +0.48 °C (95% CI 0.34 to 0.61), about 1.2 °C over the period | Yes, p < 0.001 |
| Specific humidity | +0.24 g/kg | Yes, p < 0.001 |
| Precipitation | +0.009 mm/day | No (p = 0.69) |
| Vegetation greenness (NDVI) | +0.0042 (about 3% of its average) | Yes, p < 0.001 |

**Opposite trends hidden by the average (precipitation):** the North-West zone became drier (-0.12 mm/day per decade, p = 0.022) while the South-West zone became wetter (+0.08 mm/day per decade, p = 0.008, from a very dry baseline). The region-wide average shows no trend and hides both.

Temperature rose in every zone (+0.40 to +0.55 °C per decade). The 2010 warm spike and 2011 cool dip do not drive the result: dropping any single year keeps the regional slope between 0.46 and 0.51 °C per decade.

NDVI is available only as one region-wide average, so it has no zone breakdown. We found no matching regional rainfall trend, and irrigation and land-use change are not in our data, so we make no claim about what drives the greening.

## Method

1. Monthly series for each variable and region.
2. Remove the seasonal cycle by subtracting each month's long-term average.
3. Average the anomalies by year (26 values).
4. Theil-Sen slope with a 95% confidence interval, and the Mann-Kendall significance test. OLS regression is included as a cross-check.
5. For the 63 individual grid cells we also apply a Benjamini-Hochberg false-discovery-rate correction.

## Extra: short-term vegetation forecast

An LSTM ensemble (5 seeds, 12-month window) forecasts the NDVI anomaly 1, 3 and 6 months ahead from past NDVI, temperature, precipitation and humidity. It is trained on 2000-2020 and tested on 2021-2025, and compared with three baselines (climatology, persistence, ridge regression).

* At **3 months** the LSTM has the lowest error (RMSE 0.0069 vs 0.0083 for persistence and 0.0087 for climatology).
* At **1 month** the simple baselines are better. At **6 months** the LSTM is barely better than climatology.
* Precipitation is the most consistently useful input. Importance values are noisy, and they describe associations, not causes.
* Forecasts for early 2026 are close to normal and are labelled with an evidence level.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# inputs:
#   data/clean/nasa_power_monthly_long.csv  (from src/clean_nasa_power.py)
#   data/raw/  MODIS MOD13Q1 statistics CSV + pixel-reliability QA CSV (AppEEARS export)
python src/run_pipeline.py              # everything (LSTM training takes roughly 5-15 minutes)
python src/run_pipeline.py --skip 07    # reuse the saved LSTM models
```

New NASA POWER data: `python src/run_pipeline.py --climate T2M.csv PRECTOTCORR.csv QV2M.csv`.
New NDVI data: put the new export in `data/raw/`; the newest file is used.

## Outputs for the interface (`outputs/export/`)

* `regions.json`: main file. Per region and variable: monthly and annual series, trend line, slope with 95% CI, p-value, plain-language variable definitions, how-to-read notes, limitations and three ready-made examples.
* `trend_data.json`: per-grid-cell trends and annual series for temperature, precipitation and humidity.
* `vegetation_forecast.json`: NDVI forecasts, skill against baselines, input importance and caveats.

## Repository layout

```
src/clean_nasa_power.py     clean and merge raw NASA POWER files
src/02_trends.py            per-cell trends (Mann-Kendall + Sen, FDR) and maps
src/03_export.py            climate JSON
src/06_prep_ndvi.py         monthly NDVI + climate table
src/07_lstm.py              LSTM + baselines, evaluation, importance
src/08_forecast_export.py   forecasts and vegetation JSON
src/09_regions.py           regional analysis, definitions, examples
src/run_pipeline.py         run everything
```

## Limitations

* 26 annual values per series; autocorrelation is not corrected.
* Climate variables are MERRA-2 model-based estimates, not station measurements.
* Zones are simple averages of grid cells; zone tests are not corrected for multiple testing.
* NDVI is one region-wide average dominated by desert pixels. Relative changes are unreliable where the baseline is near zero.
* Soil moisture is not included.
* The LSTM test set has 60 months; differences between models are indicative, not proven.

## Data and credits

* NASA POWER (MERRA-2 based): T2M, PRECTOTCORR, QV2M.
* MODIS Terra MOD13Q1 v061 NDVI (NASA LP DAAC), accessed through AppEEARS.

## Use of AI

Parts of the analysis design and code were developed with the help of an AI assistant (Claude, by Anthropic). The team ran the scripts, checked the outputs, and is responsible for the results. *(Edit this paragraph so it matches exactly what your team did.)*

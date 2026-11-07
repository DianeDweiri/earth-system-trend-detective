import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import json, glob
import numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import pymannkendall as mk
from scipy.stats import theilslopes
from tensorflow import keras

pre = json.load(open("outputs/models/preproc.json"))
L, TRAIN_END, VARS = pre["L"], pre["train_end"], pre["vars"]
clim = pd.DataFrame(pre["clim"]); clim.index = clim.index.astype(int)
mu, sd = np.array(pre["mu"]), np.array(pre["sd"])

df = pd.read_csv("data/clean/model_table.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
A = df[VARS].values - clim.loc[df.month, VARS].values
Z = (A - mu) / sd
F = np.hstack([Z, np.sin(2 * np.pi * df.month.values / 12)[:, None],
               np.cos(2 * np.pi * df.month.values / 12)[:, None]]).astype("float32")
x = F[-L:][None]                      # the last L months of data -> one input window
last = df.date.iloc[-1]

metrics = pd.read_csv("outputs/results/lstm_metrics.csv")
imp = pd.read_csv("outputs/results/lstm_importance.csv")
tp = pd.read_csv("outputs/results/lstm_test_predictions.csv")

# ---- forecasts beyond the last month of data ----
forecasts = []
for h in pre["horizons"]:
    files = sorted(glob.glob(f"outputs/models/lstm_h{h}_s*.keras"))
    if not files:
        continue
    z = np.array([keras.models.load_model(f, compile=False).predict(x, verbose=0).ravel()[0] for f in files])
    anom = z * sd[0] + mu[0]
    target = last + pd.DateOffset(months=h)
    normal = float(clim.loc[target.month, "ndvi"])
    m = metrics[(metrics.horizon_months == h) & (metrics.model == "LSTM")].iloc[0]
    worst = min(m.skill_vs_climatology, m.skill_vs_persistence)   # must beat BOTH simple baselines
    am = float(anom.mean())
    agree = float(np.mean(np.sign(anom) == np.sign(am)))
    # a call is "clear" only if the forecast anomaly exceeds the model's typical test error AND the 5 seeds agree
    clear = bool(abs(am) >= float(m.rmse_ndvi) and agree >= 0.8)
    direction = ("above normal" if am > 0 else "below normal") if clear else "near normal"
    forecasts.append(dict(
        horizon_months=h, target_month=target.strftime("%Y-%m"),
        ndvi_normal=round(normal, 4), ndvi_forecast=round(normal + float(anom.mean()), 4),
        anomaly=round(float(anom.mean()), 4), anomaly_spread=round(float(anom.std()), 4),
        direction=direction, signal_clear=clear,
        seed_agreement=round(agree, 2),
        evidence_level="moderate" if worst >= 0.10 else ("weak" if worst > 0 else "none"),
        test_rmse=float(m.rmse_ndvi), skill_vs_persistence=float(m.skill_vs_persistence),
        skill_vs_climatology=float(m.skill_vs_climatology)))

# ---- regional NDVI trend (same method as the climate trends) ----
t = df.copy()
t["a"] = t.ndvi - t.groupby("month").ndvi.transform("mean")
ann = t.groupby("year").a.mean()
trend = dict(sen_slope_per_decade=round(float(theilslopes(ann.values, ann.index.values)[0] * 10), 5),
             p=round(float(mk.original_test(ann.values).p), 5),
             relative_pct_per_decade=round(float(theilslopes(ann.values, ann.index.values)[0] * 10 / df.ndvi.mean() * 100), 2))

test_preds = {int(h): dict(months=g.date.str[:7].tolist(), actual=g.ndvi_actual.round(4).tolist(),
                           lstm=g.ndvi_lstm.round(4).tolist(), persistence=g.ndvi_persistence.round(4).tolist())
              for h, g in tp.groupby("horizon_months")}

payload = {
    "meta": {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "schema_version": 2,
        "target": "regional mean NDVI (MODIS MOD13Q1), monthly",
        "drivers": ["temperature", "precipitation", "humidity (NASA POWER / MERRA-2)"],
        "data_through": last.strftime("%Y-%m"),
        "train_period": f"2000-{TRAIN_END}", "test_period": f"{TRAIN_END + 1}-{int(df.year.max())}",
        "model": "LSTM ensemble (5 seeds), 12-month input window; baselines: climatology, persistence, ridge",
        "caveats": [
            "Single regional average, dominated by desert pixels: NDVI is low and changes are small.",
            "Skill is shown only where the LSTM beats both baselines on the 2021-2025 test; see evidence_level.",
            "'direction' compares with the normal NDVI of that month (not with last month). 'near normal' means the forecast anomaly is smaller than the model's typical error, or the ensemble members disagree.",
            "Input importance and correlations describe associations, not causes (irrigation, land use not modelled).",
            "Climate inputs are MERRA-2 model-based data; soil moisture is not included.",
        ],
    },
    "ndvi_trend": trend,
    "forecasts": forecasts,
    "skill": json.loads(metrics.to_json(orient="records")),
    "importance": imp.to_dict("records"),
    "test_predictions": test_preds,
}
out = Path("outputs/export"); out.mkdir(parents=True, exist_ok=True)
p = out / "vegetation_forecast.json"
json.dump(payload, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
          default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("saved", p, f"({p.stat().st_size / 1024:.0f} KB)")
print("NDVI trend:", trend)
print(pd.DataFrame(forecasts)[["horizon_months", "target_month", "ndvi_normal", "ndvi_forecast", "anomaly", "direction", "seed_agreement", "evidence_level"]].to_string(index=False))

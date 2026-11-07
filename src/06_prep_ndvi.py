import pandas as pd, numpy as np
from scipy.stats import theilslopes

from pathlib import Path

def newest(pattern, exclude=None):
    """Pick the most recently modified matching file in data/raw (so a fresh export is used automatically)."""
    f = [p for p in Path("data/raw").glob(pattern) if not (exclude and exclude in p.name)]
    if not f:
        raise SystemExit(f"No file matching {pattern} in data/raw")
    return max(f, key=lambda p: p.stat().st_mtime)

stats_file = newest("MOD13Q1*Statistics*.csv", exclude="QA")
qa_file = newest("*pixel-reliability-Statistics-QA.csv")
print("NDVI file:", stats_file.name, "| QA file:", qa_file.name)

s = pd.read_csv(stats_file, parse_dates=["Date"])
nd = s[s.Dataset.str.endswith("_NDVI")][["Date", "Mean"]].rename(columns={"Mean": "ndvi"})

q = pd.read_csv(qa_file).fillna(0)
q["Date"] = pd.to_datetime(q["Date"], utc=True).dt.tz_localize(None).dt.normalize()
q["good"] = q["0"] / q[["0", "1", "2", "3"]].sum(axis=1)

d = nd.merge(q[["Date", "good"]], on="Date", how="left")
d["date"] = d.Date.dt.to_period("M").dt.to_timestamp()
nm = d.groupby("date").agg(ndvi=("ndvi", "mean"), good_frac=("good", "mean")).reset_index()

c = pd.read_csv("data/clean/nasa_power_monthly_long.csv", parse_dates=["date"])
cm = c.groupby("date")[["temp_c", "precip_mm_day", "humidity_g_kg"]].mean().reset_index()

t = nm.merge(cm, on="date", how="inner").sort_values("date").reset_index(drop=True)
t["year"], t["month"] = t.date.dt.year, t.date.dt.month
t.to_csv("data/clean/model_table.csv", index=False)
print("rows:", len(t), "|", t.date.min().date(), "->", t.date.max().date(), "| NaN:", int(t.isna().sum().sum()))

# quick check (exploration only: anomalies use all years)
for col in ["ndvi", "temp_c", "precip_mm_day", "humidity_g_kg"]:
    t[col + "_a"] = t[col] - t.groupby("month")[col].transform("mean")
ann = t.groupby("year").ndvi_a.mean()
print("NDVI Sen slope per decade:", round(theilslopes(ann.values, ann.index.values)[0] * 10, 4))
for v in ["temp_c", "precip_mm_day", "humidity_g_kg"]:
    print(v, "corr with NDVI at lag 0..3:", [round(t.ndvi_a.corr(t[v + "_a"].shift(k)), 2) for k in range(4)])

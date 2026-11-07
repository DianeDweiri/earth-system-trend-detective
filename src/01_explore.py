import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

df = pd.read_csv("data/clean/nasa_power_monthly_long.csv", parse_dates=["date"])
out = Path("outputs/figures"); out.mkdir(parents=True, exist_ok=True)
VARS = {"temp_c": "Temperature (°C)",
        "precip_mm_day": "Precipitation (mm/day)",
        "humidity_g_kg": "Humidity (g/kg)"}

# ---------- 0. sanity checks ----------
print("shape:", df.shape, "| years:", df.year.min(), "-", df.year.max(),
      "| points:", df.point_id.nunique(), "| imputed rows:", int(df.imputed.sum()))
print(df[list(VARS)].describe().round(2))

# ---------- 1. regional annual mean + quick linear trend ----------
annual = df.groupby("year")[list(VARS)].mean()
fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
for ax, (v, label) in zip(axes, VARS.items()):
    s = stats.linregress(annual.index, annual[v])
    ax.plot(annual.index, annual[v], marker="o")
    ax.plot(annual.index, s.intercept + s.slope * annual.index, "--")
    ax.set_ylabel(label)
    ax.set_title(f"{label}: {s.slope*10:+.3f} per decade (p={s.pvalue:.3f})")
    print(f"{v}: slope/decade = {s.slope*10:+.3f}, p = {s.pvalue:.4f}")
plt.tight_layout(); plt.savefig(out / "01_regional_annual.png", dpi=150); plt.close()

# ---------- 2. seasonal cycle (this is why we need anomalies) ----------
clim = df.groupby("month")[list(VARS)].mean()
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (v, label) in zip(axes, VARS.items()):
    ax.plot(clim.index, clim[v], marker="o"); ax.set_title(label); ax.set_xlabel("month")
plt.tight_layout(); plt.savefig(out / "02_seasonal_cycle.png", dpi=150); plt.close()

# ---------- 3. quick per-point temperature slope map (preliminary) ----------
pt = df.groupby(["point_id", "lat", "lon", "year"]).temp_c.mean().reset_index()
rows = []
for (pid, lat, lon), g in pt.groupby(["point_id", "lat", "lon"]):
    s = stats.linregress(g.year, g.temp_c)
    rows.append((pid, lat, lon, s.slope * 10, s.pvalue))
slopes = pd.DataFrame(rows, columns=["point_id", "lat", "lon", "slope_per_decade", "p"])
print("\nTemp slope per decade across points:")
print(slopes.slope_per_decade.describe().round(3))
print("points with p<0.05:", int((slopes.p < 0.05).sum()), "of", len(slopes))

fig, ax = plt.subplots(figsize=(7, 7))
sc = ax.scatter(slopes.lon, slopes.lat, c=slopes.slope_per_decade, cmap="coolwarm", s=450, edgecolor="k")
sig = slopes[slopes.p < 0.05]
ax.scatter(sig.lon, sig.lat, marker="x", c="k", s=60, label="p < 0.05")
plt.colorbar(sc, label="°C per decade"); ax.legend()
ax.set_title("Preliminary temperature trend per grid point")
plt.savefig(out / "03_temp_slope_map.png", dpi=150); plt.close()

# ---------- 4. check the imputed grid cell ----------
imp = df[df.imputed].point_id.unique()
print("\nimputed point(s):", imp)
slopes.to_csv("outputs/results/prelim_temp_slopes.csv", index=False)
print("\nSaved figures to outputs/figures/")

import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymannkendall as mk
from scipy.stats import false_discovery_control
from pathlib import Path

df = pd.read_csv("data/clean/nasa_power_monthly_long.csv", parse_dates=["date"])
fig_dir = Path("outputs/figures"); fig_dir.mkdir(parents=True, exist_ok=True)
res_dir = Path("outputs/results"); res_dir.mkdir(parents=True, exist_ok=True)
VARS = {"temp_c": ("Temperature", "°C/decade"),
        "precip_mm_day": ("Precipitation", "mm/day per decade"),
        "humidity_g_kg": ("Humidity", "g/kg per decade")}

# ---------- 1. anomalies: remove each point's monthly climatology ----------
for v in VARS:
    clim = df.groupby(["point_id", "month"])[v].transform("mean")
    df[v + "_anom"] = df[v] - clim

# ---------- 2. check the 2010/2011 temperature jump ----------
print("=== Regional monthly temp anomaly, 2009-2012 ===")
chk = df[df.year.between(2009, 2012)].groupby(["year", "month"]).temp_c_anom.mean().unstack(0).round(2)
print(chk)
pa = df.groupby(["point_id", "year"]).temp_c_anom.mean().unstack()
print("\n=== Annual temp anomaly across the 63 points ===")
for y in (2009, 2010, 2011, 2012):
    print(y, pa[y].describe()[["min", "mean", "max"]].round(2).to_dict())

# ---------- 3. Mann-Kendall + Sen's slope per point ----------
summary = []
for v, (name, unit) in VARS.items():
    ann = df.groupby(["point_id", "lat", "lon", "year"])[v + "_anom"].mean().reset_index()
    rows = []
    for (pid, lat, lon), g in ann.groupby(["point_id", "lat", "lon"]):
        r = mk.original_test(g.sort_values("year")[v + "_anom"].values)
        rows.append((pid, lat, lon, r.slope * 10, r.p, r.trend))
    t = pd.DataFrame(rows, columns=["point_id", "lat", "lon", "sen_slope_per_decade", "p", "trend"])
    t["p_fdr"] = false_discovery_control(t.p.values, method="bh")
    t["significant"] = t.p_fdr < 0.05
    t.to_csv(res_dir / f"trends_{v}.csv", index=False)
    print(f"\n[{name}] significant after FDR: {int(t.significant.sum())}/{len(t)} | "
          f"slope range {t.sen_slope_per_decade.min():.3f} to {t.sen_slope_per_decade.max():.3f} {unit}")

    # map
    m = max(abs(t.sen_slope_per_decade).max(), 1e-9)
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(t.lon, t.lat, c=t.sen_slope_per_decade, cmap="coolwarm", vmin=-m, vmax=m,
                    s=450, edgecolor="k")
    s = t[t.significant]
    ax.scatter(s.lon, s.lat, marker=".", c="k", s=40, label="significant (FDR 5%)")
    plt.colorbar(sc, label=unit); ax.legend(loc="upper left")
    ax.set_title(f"{name} trend (Sen's slope)")
    plt.savefig(fig_dir / f"04_map_{v}.png", dpi=150); plt.close()

    # ---------- 4. regional trend + sensitivity ----------
    reg = df.groupby("year")[v + "_anom"].mean()
    full = mk.original_test(reg.values)
    loo = [mk.original_test(reg.drop(y).values).slope * 10 for y in reg.index]
    drop = reg.drop([2010, 2011]) if v == "temp_c" else reg
    summary.append(dict(variable=v, sen_slope_per_decade=full.slope * 10, p=full.p, trend=full.trend,
                        loo_min=min(loo), loo_max=max(loo),
                        slope_without_2010_2011=mk.original_test(drop.values).slope * 10,
                        points_significant=int(t.significant.sum())))

S = pd.DataFrame(summary)
S.to_csv(res_dir / "regional_summary.csv", index=False)
print("\n=== Regional summary ===")
print(S.round(4).to_string(index=False))
print("\nSaved: outputs/results/*.csv and outputs/figures/04_map_*.png")

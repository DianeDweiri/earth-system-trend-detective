import pandas as pd, json
from pathlib import Path
from datetime import datetime, timezone

res = Path("outputs/results"); out = Path("outputs/export"); out.mkdir(parents=True, exist_ok=True)
df = pd.read_csv("data/clean/nasa_power_monthly_long.csv")
VARS = ["temp_c", "precip_mm_day", "humidity_g_kg"]
trends = {v: pd.read_csv(res / f"trends_{v}.csv").set_index("point_id") for v in VARS}
ann = df.groupby(["point_id", "year"])[VARS].mean().round(3)

points = []
for pid, g in df.groupby("point_id"):
    a = ann.loc[pid]
    points.append({
        "id": pid, "lat": float(g.lat.iloc[0]), "lon": float(g.lon.iloc[0]),
        "trend": {v: {"slope_per_decade": round(float(trends[v].loc[pid, "sen_slope_per_decade"]), 4),
                      "p_fdr": round(float(trends[v].loc[pid, "p_fdr"]), 5),
                      "significant": bool(trends[v].loc[pid, "significant"])} for v in VARS},
        "annual": {"years": a.index.tolist(), **{v: a[v].tolist() for v in VARS}},
    })

reg = df.groupby("year")[VARS].mean().round(3)
summ = pd.read_csv(res / "regional_summary.csv").set_index("variable")
payload = {
    "meta": {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "NASA POWER (MERRA-2 based), monthly",
        "period": [int(df.year.min()), int(df.year.max())],
        "grid_points": int(df.point_id.nunique()),
        "units": {"temp_c": "°C", "precip_mm_day": "mm/day", "humidity_g_kg": "g/kg"},
        "method": "Mann-Kendall + Sen's slope on annual mean anomalies; Benjamini-Hochberg FDR 5%",
        "imputed_points": sorted(df[df.imputed].point_id.unique().tolist()),
        "schema_version": 1,
    },
    "regional": {
        "years": reg.index.tolist(),
        **{v: reg[v].tolist() for v in VARS},
        "trend": summ[["sen_slope_per_decade", "p", "trend", "points_significant"]].round(5).to_dict("index"),
    },
    "points": points,
}
p = out / "trend_data.json"
json.dump(payload, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
          default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("saved", p, f"({p.stat().st_size/1024:.0f} KB)")

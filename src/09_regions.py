"""Regional trend analysis for the 'Earth System Trend Detective' tool.

For every variable and region: monthly series, annual (deseasonalised) series, rate of change (Theil-Sen slope),
95% confidence interval, Mann-Kendall significance, plus plain-language definitions and ready-made examples.
Output: outputs/export/regions.json
"""
import json
import numpy as np, pandas as pd
import pymannkendall as mk
from pathlib import Path
from datetime import datetime, timezone
from scipy.stats import theilslopes, linregress

clim = pd.read_csv("data/clean/nasa_power_monthly_long.csv", parse_dates=["date"])
ndvi = pd.read_csv("data/clean/model_table.csv", parse_dates=["date"])[["date", "ndvi"]]

VARIABLES = {
    "temp_c": dict(name="Air temperature", unit="°C", source="NASA POWER (MERRA-2 model)",
                   definition="Air temperature about 2 metres above the ground. Rising values mean warming."),
    "precip_mm_day": dict(name="Precipitation", unit="mm/day", source="NASA POWER (MERRA-2 model, bias-corrected)",
                          definition="Average rain and snow, in millimetres of water per day."),
    "humidity_g_kg": dict(name="Specific humidity", unit="g/kg", source="NASA POWER (MERRA-2 model)",
                          definition="Grams of water vapour in each kilogram of air at 2 metres. Warmer air can hold more water vapour."),
    "ndvi": dict(name="Vegetation greenness (NDVI)", unit="index", source="MODIS Terra MOD13Q1 (250 m, 16-day composites, averaged monthly)",
                 definition="How green the land looks from space. Near 0 means bare ground or desert; higher values mean more or healthier plants."),
}
CLIMATE = ["temp_c", "precip_mm_day", "humidity_g_kg"]


def zone(lat, lon):
    ns = "North" if lat >= 32.0 else ("Central" if lat >= 31.0 else "South")
    return f"{ns}-{'West' if lon <= 36.25 else 'East'}"


pts = clim[["point_id", "lat", "lon"]].drop_duplicates().copy()
pts["zone"] = [zone(a, b) for a, b in zip(pts.lat, pts.lon)]
clim = clim.merge(pts[["point_id", "zone"]], on="point_id")


def analyze(m, v):
    """m: DataFrame[date, value] with one row per month."""
    m = m.sort_values("date").copy()
    m["year"], m["month"] = m.date.dt.year, m.date.dt.month
    m["anom"] = m.value - m.groupby("month").value.transform("mean")      # remove the seasonal cycle
    ann = m.groupby("year").anom.mean()
    x, y = ann.index.values.astype(float), ann.values
    base = float(m.value.mean())
    sen, icpt, lo, hi = theilslopes(y, x, alpha=0.95)
    r = mk.original_test(y)
    ols = linregress(x, y)
    sig = bool(r.p < 0.05)
    direction = ("increasing" if sen > 0 else "decreasing") if sig else "no significant trend"
    return {
        "monthly": {"dates": m.date.dt.strftime("%Y-%m").tolist(), "values": m.value.round(4).tolist()},
        "annual": {"years": ann.index.tolist(), "level": (base + y).round(4).tolist(),
                   "trend_line": (base + icpt + sen * x).round(4).tolist()},
        "trend": {
            "slope_per_decade": round(float(sen * 10), 5),
            "ci95_per_decade": [round(float(lo * 10), 5), round(float(hi * 10), 5)],
            "total_change_over_period": round(float(sen * (x.max() - x.min())), 4),
            "period": [int(x.min()), int(x.max())],
            "mann_kendall_p": round(float(r.p), 5), "significant": sig, "direction": direction,
            "ci_excludes_zero": bool(lo * 10 > 0 or hi * 10 < 0),
            "ols_slope_per_decade": round(float(ols.slope * 10), 5), "ols_p": round(float(ols.pvalue), 5),
            "baseline_mean": round(base, 4),
            "relative_pct_per_decade": None if v == "temp_c" else round(float(sen * 10 / base * 100), 2),
            "relative_unreliable": bool(v == "precip_mm_day" and base < 0.25),
            "n_years": int(len(x)),
        },
    }


def slug(s):
    return s.lower().replace(" ", "-")


regions = []
specs = [("Whole region", clim, pts)] + [(z, clim[clim.zone == z], pts[pts.zone == z]) for z in sorted(pts.zone.unique())]
for name, g, p in specs:
    entry = {"id": slug(name), "name": name, "n_points": int(len(p)),
             "bbox": {"lat_min": float(p.lat.min() - 0.25), "lat_max": float(p.lat.max() + 0.25),
                      "lon_min": float(p.lon.min() - 0.3125), "lon_max": float(p.lon.max() + 0.3125)},
             "variables": {}}
    for v in CLIMATE:
        entry["variables"][v] = analyze(g.groupby("date")[v].mean().reset_index().rename(columns={v: "value"}), v)
    if name == "Whole region":                       # NDVI is one region-wide average only
        entry["variables"]["ndvi"] = analyze(ndvi.rename(columns={"ndvi": "value"}), "ndvi")
    regions.append(entry)

summary = []
for r in regions:
    for v, blk in r["variables"].items():
        t = blk["trend"]
        summary.append({"region_id": r["id"], "region": r["name"], "variable": v,
                        **{k: t[k] for k in ("slope_per_decade", "ci95_per_decade", "mann_kendall_p", "significant",
                                              "direction", "relative_pct_per_decade", "relative_unreliable")}})


def T(region_name, v):
    return next(r for r in regions if r["name"] == region_name)["variables"][v]["trend"]


def pfmt(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


# ---------- ready-made examples (text is generated from the computed numbers) ----------
tw = T("Whole region", "temp_c")
nw = T("Whole region", "ndvi")
pr = [s for s in summary if s["variable"] == "precip_mm_day" and s["region"] != "Whole region" and s["significant"]]
neg = min(pr, key=lambda s: s["slope_per_decade"])
pos = max(pr, key=lambda s: s["slope_per_decade"]) if any(s["slope_per_decade"] > 0 for s in pr) else None
pw = T("Whole region", "precip_mm_day")
ex_rain = (f"In the {neg['region']} zone rainfall fell by {abs(neg['slope_per_decade']):.2f} mm/day per decade ({pfmt(neg['mann_kendall_p'])})")
if pos:
    base_pos = T(pos["region"], "precip_mm_day")["baseline_mean"]
    ex_rain += f", while in the {pos['region']} zone it rose by {pos['slope_per_decade']:.2f} mm/day per decade ({pfmt(pos['mann_kendall_p'])})"
    if base_pos < 0.5:
        ex_rain += f" (that area is very dry, averaging only {base_pos:.2f} mm/day, so the absolute change is small)"
ex_rain += f". The whole-region average shows no significant trend ({pfmt(pw['mann_kendall_p'])}), hiding both."
examples = [
    dict(id="warming", title="Is the region getting warmer?", region="whole-region", variable="temp_c",
         takeaway=(f"Average air temperature rose about {tw['slope_per_decade']:.2f} °C per decade "
                   f"(95% CI {tw['ci95_per_decade'][0]:.2f} to {tw['ci95_per_decade'][1]:.2f}), roughly "
                   f"{tw['total_change_over_period']:.1f} °C between {tw['period'][0]} and {tw['period'][1]} ({pfmt(tw['mann_kendall_p'])}): significant.")),
    dict(id="opposite-rain", title="Same region, opposite rainfall trends", region=slug(neg["region"]), variable="precip_mm_day",
         compare_region=slug(pos["region"]) if pos else None, takeaway=ex_rain),
    dict(id="greening", title="Is the vegetation getting greener?", region="whole-region", variable="ndvi",
         takeaway=(f"The region-wide NDVI rose about {nw['slope_per_decade']:.4f} per decade ({nw['relative_pct_per_decade']:.1f}% of its average), "
                   f"{pfmt(nw['mann_kendall_p'])}. Most of the area is desert, so values are low. There is no matching regional rainfall trend "
                   f"({pfmt(pw['mann_kendall_p'])}), and irrigation or land-use change are not in our data, so we cannot say what drives it.")),
]

payload = {
    "meta": {
        "schema_version": 3, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "period": [int(clim.year.min()), int(clim.year.max())],
        "method": ("Monthly series -> subtract each month's long-term average (removes the seasonal cycle) -> annual mean of anomalies -> "
                   "Theil-Sen slope with 95% confidence interval + Mann-Kendall significance test (OLS regression as a cross-check)."),
        "how_to_read": [
            "Slope per decade: how fast the variable changes (Theil-Sen estimate).",
            "95% confidence interval: plausible range for the slope; if it excludes zero the trend is likely real.",
            "p < 0.05 (Mann-Kendall): the trend is 'statistically significant'. Not significant does not mean no change - a longer record may be needed.",
            "Seasonality is removed first, so summer-vs-winter swings do not hide the long-term trend.",
        ],
        "limitations": [
            "Only 26 annual values; autocorrelation is not corrected.",
            "Zones are simple averages of 0.5 x 0.625 degree grid cells (63 cells); zone tests are not corrected for multiple testing.",
            "NDVI is available only as one region-wide average (no zones). Relative changes are unreliable when the baseline is near zero.",
            "Climate variables are MERRA-2 model-based estimates, not station measurements.",
            "Trends describe associations, not causes.",
        ],
    },
    "variables": VARIABLES,
    "regions": regions,
    "summary": summary,
    "examples": examples,
}
out = Path("outputs/export"); out.mkdir(parents=True, exist_ok=True)
p = out / "regions.json"
json.dump(payload, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
          default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("saved", p, f"({p.stat().st_size / 1024:.0f} KB)")
print(pd.DataFrame(summary).assign(ci=lambda d: d.ci95_per_decade.map(lambda c: f"[{c[0]:+.3f},{c[1]:+.3f}]"))
      [["region", "variable", "slope_per_decade", "ci", "mann_kendall_p", "direction"]].to_string(index=False))
print("\nExamples:")
for e in examples:
    print("-", e["title"], "->", e["takeaway"])

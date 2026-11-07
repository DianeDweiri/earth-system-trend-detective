"""Run the whole analysis with one command.

  python src/run_pipeline.py                       # re-run everything on the data already in data/
  python src/run_pipeline.py --climate T.csv P.csv Q.csv   # also re-clean new raw NASA POWER files first
  python src/run_pipeline.py --skip 07             # skip a step (e.g. skip LSTM training, reuse saved models)

New NDVI data: drop the new MODIS export (and its pixel-reliability QA file) into data/raw/ - the newest file is used.
"""
import sys, time, argparse, subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--climate", nargs=3, metavar=("T2M_CSV", "PRECTOTCORR_CSV", "QV2M_CSV"),
                help="new raw NASA POWER files (one parameter each) to clean first")
ap.add_argument("--skip", nargs="*", default=[], help="step ids to skip: 02 03 06 07 08 09")
a = ap.parse_args()

if not Path("src").is_dir() or not Path("data").is_dir():
    sys.exit("Run this from the project root (the trend-detective folder).")

STEPS = [("02", "climate trends", ["src/02_trends.py"]),
         ("03", "climate JSON for the UI", ["src/03_export.py"]),
         ("06", "NDVI + climate table", ["src/06_prep_ndvi.py"]),
         ("07", "LSTM training + evaluation", ["src/07_lstm.py"]),
         ("08", "forecasts + vegetation JSON for the UI", ["src/08_forecast_export.py"]),
         ("09", "regions, definitions and examples JSON for the UI", ["src/09_regions.py"])]
if a.climate:
    STEPS.insert(0, ("clean", "clean new NASA POWER data", ["src/clean_nasa_power.py", *a.climate, "data/clean"]))

t0 = time.time()
for sid, name, cmd in STEPS:
    if sid in a.skip:
        print(f"\n--- [{sid}] {name}: skipped"); continue
    print(f"\n=== [{sid}] {name} ===", flush=True)
    t = time.time()
    r = subprocess.run([sys.executable, *cmd])
    if r.returncode != 0:
        sys.exit(f"\nStep {sid} failed (exit code {r.returncode}). Fix it, then re-run; use --skip for finished steps.")
    print(f"[{sid}] done in {time.time() - t:.0f}s", flush=True)

print(f"\nAll done in {(time.time() - t0) / 60:.1f} min. Files for the UI:")
for f in sorted(Path("outputs/export").glob("*.json")):
    print(f"  {f}  ({f.stat().st_size / 1024:.0f} KB)")

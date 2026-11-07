import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge
import tensorflow as tf
from tensorflow import keras

TRAIN_END, VAL_START = 2020, 2018      # test = 2021-2025 (never used for training/tuning)
L = 12                                 # months of history the LSTM sees
HORIZONS = [1, 3, 6]                   # forecast NDVI h months ahead
SEEDS = [0, 1, 2, 3, 4]                # ensemble of 5 LSTMs
VARS = ["ndvi", "temp_c", "precip_mm_day", "humidity_g_kg"]
NAMES = ["NDVI (past)", "Temperature", "Precipitation", "Humidity"]

for p in ("outputs/figures", "outputs/results", "outputs/models"):
    Path(p).mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/clean/model_table.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
assert df.date.diff().dropna().dt.days.between(28, 31).all(), "months are not contiguous"

# ---- anomalies: climatology + scaling from years <= TRAIN_END only (no leakage) ----
clim = df[df.year <= TRAIN_END].groupby("month")[VARS].mean()
A = df[VARS].values - clim.loc[df.month].values
fit = (df.year <= TRAIN_END).values
mu, sd = A[fit].mean(0), A[fit].std(0)
Z = (A - mu) / sd
F = np.hstack([Z, np.sin(2 * np.pi * df.month.values / 12)[:, None],
               np.cos(2 * np.pi * df.month.values / 12)[:, None]]).astype("float32")  # (T, 6)


def make(h):
    X, y, idx = [], [], []
    for t in range(L + h - 1, len(df)):          # window = months [t-h-L+1 .. t-h], target = month t
        s = t - h - L + 1
        X.append(F[s:s + L]); y.append(Z[t, 0]); idx.append(t)
    return np.array(X, "float32"), np.array(y, "float32"), np.array(idx)


def build():
    m = keras.Sequential([keras.layers.Input((L, F.shape[1])),
                          keras.layers.LSTM(16), keras.layers.Dropout(0.2),
                          keras.layers.Dense(8, activation="relu"), keras.layers.Dense(1)])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m


def fit_lstm(Xtr, ytr, Xva, yva, seed):
    keras.utils.set_random_seed(seed)
    m = build()
    es = keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True)
    hist = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=300, batch_size=16, callbacks=[es], verbose=0)
    best = max(int(np.argmin(hist.history["val_loss"])) + 1, 10)
    keras.utils.set_random_seed(seed)                      # retrain on train+val with the chosen #epochs
    m2 = build()
    m2.fit(np.vstack([Xtr, Xva]), np.concatenate([ytr, yva]), epochs=best, batch_size=16, verbose=0)
    return m2


rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
rows, preds_out, imp_rows = [], [], []
fig, axes = plt.subplots(len(HORIZONS), 1, figsize=(11, 3.2 * len(HORIZONS)), sharex=False)
rng = np.random.default_rng(0)

for ax, h in zip(np.atleast_1d(axes), HORIZONS):
    X, y, idx = make(h)
    yr = df.year.values[idx]
    tr, va, te = yr < VAL_START, (yr >= VAL_START) & (yr <= TRAIN_END), yr > TRAIN_END
    persist = Z[idx - h, 0]                               # "next = last known anomaly"
    flat = X.reshape(len(X), -1)

    # ridge baseline (alpha picked on validation)
    best_a = min([1, 10, 100, 1000], key=lambda a: rmse(Ridge(alpha=a).fit(flat[tr], y[tr]).predict(flat[va]), y[va]))
    ridge = Ridge(alpha=best_a).fit(flat[tr | va], y[tr | va])
    p_ridge = ridge.predict(flat[te])

    models = [fit_lstm(X[tr], y[tr], X[va], y[va], s) for s in SEEDS]
    pred = lambda XX: np.mean([m.predict(XX, verbose=0).ravel() for m in models], axis=0)
    p_lstm = pred(X[te])

    yt = y[te]
    cand = {"climatology (zero)": np.zeros(te.sum()), "persistence": persist[te], "ridge": p_ridge, "LSTM": p_lstm}
    r0 = rmse(cand["climatology (zero)"], yt); rp = rmse(cand["persistence"], yt)
    for name, p in cand.items():
        r = rmse(p, yt)
        acc = float(np.mean(np.sign(p) == np.sign(yt))) if name != "climatology (zero)" else np.nan
        rows.append(dict(horizon_months=h, model=name, rmse_ndvi=round(r * sd[0], 5),
                         mae_ndvi=round(float(np.mean(np.abs(p - yt))) * sd[0], 5),
                         skill_vs_climatology=round(1 - r / r0, 3), skill_vs_persistence=round(1 - r / rp, 3),
                         above_below_normal_accuracy=None if np.isnan(acc) else round(acc, 3)))

    # permutation importance on the test set (how much does the error grow if we scramble one input?)
    base = rmse(p_lstm, yt)
    for c, nm in enumerate(NAMES):
        inc = []
        for _ in range(10):
            Xp = X[te].copy(); Xp[:, :, c] = Xp[rng.permutation(len(Xp))][:, :, c]
            inc.append(rmse(pred(Xp), yt) - base)
        imp_rows.append(dict(horizon_months=h, variable=nm, rmse_increase_ndvi=round(float(np.mean(inc)) * sd[0], 5)))

    dates = df.date.values[idx][te]
    cl = clim.loc[df.month.values[idx][te], "ndvi"].values
    preds_out.append(pd.DataFrame({"date": dates, "horizon_months": h,
                                   "ndvi_actual": cl + yt * sd[0] + mu[0], "ndvi_lstm": cl + p_lstm * sd[0] + mu[0],
                                   "ndvi_persistence": cl + persist[te] * sd[0] + mu[0],
                                   "anom_actual": yt * sd[0], "anom_lstm": p_lstm * sd[0], "anom_persistence": persist[te] * sd[0]}))
    ax.plot(dates, yt * sd[0], "k-", lw=2, label="actual")
    ax.plot(dates, p_lstm * sd[0], label="LSTM"); ax.plot(dates, persist[te] * sd[0], alpha=.6, label="persistence")
    ax.axhline(0, color="gray", lw=.5); ax.set_ylabel("NDVI anomaly"); ax.set_title(f"{h}-month-ahead forecast, test 2021-2025"); ax.legend(ncol=3)
    for i, m in enumerate(models): m.save(f"outputs/models/lstm_h{h}_s{i}.keras")
    print(f"horizon {h}: done (ridge alpha {best_a}, train {tr.sum()}, val {va.sum()}, test {te.sum()})")

plt.tight_layout(); plt.savefig("outputs/figures/06_lstm_test.png", dpi=150); plt.close()

M = pd.DataFrame(rows); I = pd.DataFrame(imp_rows)
M.to_csv("outputs/results/lstm_metrics.csv", index=False)
I.to_csv("outputs/results/lstm_importance.csv", index=False)
pd.concat(preds_out).to_csv("outputs/results/lstm_test_predictions.csv", index=False)
json.dump({"L": L, "horizons": HORIZONS, "train_end": TRAIN_END, "vars": VARS,
           "clim": clim.round(6).to_dict(), "mu": mu.tolist(), "sd": sd.tolist()},
          open("outputs/models/preproc.json", "w"), indent=1)

pv = I.pivot(index="variable", columns="horizon_months", values="rmse_increase_ndvi").loc[NAMES]
pv.plot.bar(figsize=(8, 4)); plt.ylabel("RMSE increase when scrambled (NDVI)"); plt.title("What drives the LSTM forecast?")
plt.xticks(rotation=0); plt.tight_layout(); plt.savefig("outputs/figures/07_importance.png", dpi=150); plt.close()

print("\n=== TEST METRICS (RMSE in NDVI units, lower is better) ===")
print(M.to_string(index=False))
print("\n=== INPUT IMPORTANCE (bigger = the model relies on it more) ===")
print(pv.to_string())

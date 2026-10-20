"""
mes_forecast_and_figures.py
---------------------------
Reads the Gold serving tables produced by mes_pipeline_spark.py and
  1. trains day-ahead daily PEAK-demand forecasting models with Spark MLlib
     (seasonal-naive baseline, linear regression, gradient-boosted trees)
     using a strict time-based split (train Jan-Oct, test Nov-Dec 2025);
  2. evaluates the substation anomaly detector against the injected
     ground-truth events (event recall, interval precision);
  3. renders every figure used in Section 4 of the report.
"""
import os
import json
import numpy as np
import pandas as pd
import holidays
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression, GBTRegressor

BASE = os.path.dirname(os.path.abspath(__file__))
RAW, GOLD = os.path.join(BASE, "data", "raw"), os.path.join(BASE, "data", "lake", "gold")
FIG = os.path.join(BASE, "outputs", "figures")
os.makedirs(FIG, exist_ok=True)

# reference categorical palette (validated order) + text tokens
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": INK2,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "axes.titleweight": "bold",
    "axes.titlesize": 10.5, "axes.titlecolor": INK, "legend.frameon": False,
    "lines.linewidth": 2.0, "savefig.dpi": 200, "figure.dpi": 100})
RESULTS = {}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def slot_axis(ax):
    ax.set_xticks(range(0, 49, 6))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 3)])
    ax.set_xlim(0, 47)
    ax.set_xlabel("Local time of day")


# ===================================================================== forecasting
def forecasting(spark):
    daily = spark.read.parquet(f"{GOLD}/daily_peak").toPandas()
    daily["date"] = pd.to_datetime(daily["date"])
    meters = pd.read_csv(os.path.join(RAW, "meters.csv"))
    share = meters.region.value_counts(normalize=True)
    wx = pd.read_csv(os.path.join(RAW, "weather_hourly.csv"))
    wx["date"] = pd.to_datetime(wx.ts_utc.str[:10])
    wd = wx.groupby(["date", "region"]).temperature_c.mean().unstack()
    temp = (wd * share.reindex(wd.columns)).sum(axis=1).rename("temp_c")
    hols = {c: holidays.country_holidays(c, years=[2024, 2025]) for c in ["UK", "DE", "NL", "ES"]}
    cshare = meters.country.value_counts(normalize=True)

    df = daily.merge(temp, left_on="date", right_index=True)
    df["hdd"] = np.clip(15.5 - df.temp_c, 0, None)
    df["cdd"] = np.clip(df.temp_c - 22, 0, None)
    df["dow"] = df.date.dt.dayofweek
    df["holiday_share"] = [sum(cshare[c] for c in cshare.index if d in hols[c]) for d in df.date.dt.date]
    # known business calendar: industrial customers' contracted Christmas shutdown
    df["xmas_shutdown"] = (((df.date.dt.month == 12) & (df.date.dt.day >= 23))
                           | ((df.date.dt.month == 1) & (df.date.dt.day <= 2))).astype(int)
    df["doy_sin"] = np.sin(2 * np.pi * df.date.dt.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df.date.dt.dayofyear / 365.25)
    df = df.sort_values("date")
    df["lag1"] = df.peak_kw.shift(1)      # yesterday's peak: known at day-ahead decision time
    df["lag7"] = df.peak_kw.shift(7)      # same weekday last week
    df["holiday_share_lag7"] = df.holiday_share.shift(7)   # was the reference day itself a holiday?
    df["xmas_lag7"] = df.xmas_shutdown.shift(7)
    df["hdd_lag7"] = df.hdd.shift(7)
    df = df.dropna().reset_index(drop=True)
    df["is_test"] = df.date >= "2025-11-01"
    df["key"] = np.arange(len(df))

    sdf = spark.createDataFrame(df.drop(columns=["peak_interval_utc"]))
    feats = ["dow_vec", "hdd", "cdd", "holiday_share", "xmas_shutdown", "doy_sin", "doy_cos", "lag1", "lag7",
             "holiday_share_lag7", "xmas_lag7", "hdd_lag7"]
    enc = OneHotEncoder(inputCols=["dow"], outputCols=["dow_vec"], dropLast=True)
    asm = VectorAssembler(inputCols=feats, outputCol="features")
    train, test = sdf.filter(~F.col("is_test")), sdf.filter(F.col("is_test"))

    preds = df[df.is_test][["key", "date", "peak_kw", "lag7"]].rename(columns={"lag7": "Seasonal naive"}).copy()
    models = {"Linear regression": LinearRegression(labelCol="peak_kw", regParam=0.1, elasticNetParam=0.0),
              "Gradient-boosted trees": GBTRegressor(labelCol="peak_kw", maxIter=200, maxDepth=3,
                                                     stepSize=0.05, subsamplingRate=0.8, seed=7)}
    fitted = {}
    for name, est in models.items():
        pm = Pipeline(stages=[enc, asm, est]).fit(train)
        fitted[name] = pm
        p = pm.transform(test).select("key", "prediction").toPandas()
        preds = preds.merge(p.rename(columns={"prediction": name}), on="key")

    def metrics(y, yhat):
        e = y - yhat
        return dict(MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e ** 2))),
                    MAPE=float(np.mean(np.abs(e) / y) * 100),
                    R2=float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)))
    MODELS = ["Seasonal naive", "Linear regression", "Gradient-boosted trees"]
    regular = preds.date < "2025-12-23"            # normal operating regime
    out = {}
    for label, mask in [("all_test", preds.date.notna()), ("regular_days", regular),
                        ("holiday_shutdown", ~regular)]:
        mt = {m: metrics(preds.peak_kw[mask].values, preds[m][mask].values) for m in MODELS}
        base_mae = mt["Seasonal naive"]["MAE"]
        for m in mt:
            mt[m]["Skill_vs_naive_pct"] = (1 - mt[m]["MAE"] / base_mae) * 100
        mt["n_days"] = int(mask.sum())
        out[label] = mt
    RESULTS["forecast"] = out
    RESULTS["forecast_meta"] = dict(train_days=int((~df.is_test).sum()), test_days=int(df.is_test.sum()),
                                    peak_min=float(df.peak_kw.min()), peak_max=float(df.peak_kw.max()))
    # feature importance (GBT), dow one-hot folded into one bar
    gbt = fitted["Gradient-boosted trees"].stages[-1]
    imp = gbt.featureImportances.toArray()
    names = [f"dow_{i}" for i in range(6)] + feats[1:]
    fi = pd.Series(imp, index=names)
    fi = pd.concat([pd.Series({"day_of_week": fi[[n for n in names if n.startswith("dow_")]].sum()}),
                    fi[[n for n in names if not n.startswith("dow_")]]]).sort_values()
    RESULTS["feature_importance"] = fi.round(4).to_dict()

    # ---- Figure: forecast vs actual
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.plot(preds.date, preds.peak_kw, color=INK, lw=2.2, label="Actual daily peak")
    ax.plot(preds.date, preds["Seasonal naive"], color=C[3], lw=1.6, ls="--", label="Seasonal naive (t−7)")
    ax.plot(preds.date, preds["Linear regression"], color=C[0], lw=1.6, label="Linear regression")
    ax.plot(preds.date, preds["Gradient-boosted trees"], color=C[1], lw=1.8, label="Gradient-boosted trees")
    ax.set_ylabel("System peak demand (kW)")
    ax.set_title("Day-ahead daily peak forecast on the hold-out period (Nov–Dec 2025)", loc="left")
    ax.axvspan(pd.Timestamp("2025-12-23"), preds.date.max() + pd.Timedelta(hours=12), color="#f1f0ec", zorder=0)
    ax.text(pd.Timestamp("2025-12-27"), 1550, "Holiday\nshutdown", ha="center", va="center", fontsize=8, color=INK2,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))
    ax.legend(ncol=2, loc="lower left", fontsize=8.5)
    fig.autofmt_xdate()
    save(fig, "fig4_8_forecast.png")

    # ---- Figure: feature importance
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    labels = {"day_of_week": "Day of week", "hdd": "Heating degree-days", "cdd": "Cooling degree-days",
              "holiday_share": "Public-holiday share", "xmas_shutdown": "Christmas shutdown flag",
              "doy_sin": "Season (sin)", "doy_cos": "Season (cos)", "lag1": "Peak yesterday (t−1)",
              "lag7": "Peak same weekday last week (t−7)",
              "holiday_share_lag7": "Holiday share at t−7", "xmas_lag7": "Shutdown flag at t−7",
              "hdd_lag7": "Heating degree-days at t−7"}
    ax.barh([labels[i] for i in fi.index], fi.values, color=C[0], height=0.6)
    for y, v in enumerate(fi.values):
        ax.text(v + 0.005, y, f"{v:.2f}", va="center", fontsize=8.5, color=INK2)
    ax.set_xlabel("Relative importance (GBT)")
    ax.set_title("What drives the peak forecast", loc="left")
    ax.grid(axis="y", visible=False)
    save(fig, "fig4_9_feature_importance.png")

    # ---- Figure: temperature vs peak (motivates HDD feature) on full year
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    wk = df.dow < 5
    ax.scatter(df.temp_c[wk], df.peak_kw[wk], s=14, color=C[0], alpha=0.75, label="Weekday", edgecolor="white", linewidth=0.4)
    ax.scatter(df.temp_c[~wk], df.peak_kw[~wk], s=14, color=C[1], alpha=0.75, label="Weekend", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Load-weighted mean daily temperature (°C)")
    ax.set_ylabel("Daily system peak (kW)")
    ax.set_title("Daily peak versus temperature, 2024–2025", loc="left")
    ax.legend()
    save(fig, "fig4_7_peak_vs_temperature.png")
    return df


# ===================================================================== anomaly evaluation
def anomaly_eval(spark):
    sc = spark.read.parquet(f"{GOLD}/scada_scored").toPandas()
    sc["ts"] = pd.to_datetime(sc.ts)
    for c in ["alert", "rule_en50160", "stat_z", "stuck"]:
        sc[c] = sc[c].fillna(False).astype(bool)
    truth = pd.read_csv(os.path.join(RAW, "scada_events_truth.csv"))
    truth["start"] = pd.to_datetime(truth.start_ts_utc.str.replace("Z", ""))
    sc = sc.sort_values(["substation_id", "ts"])
    sc["true_anom"] = False
    det = {k: 0 for k in truth.event_type.unique()}
    tot = truth.event_type.value_counts().to_dict()
    for ev in truth.itertuples():
        end = ev.start + pd.Timedelta(minutes=10 * ev.n_intervals)
        m = (sc.substation_id == ev.substation_id) & (sc.ts >= ev.start) & (sc.ts < end + pd.Timedelta(minutes=10 * 11))
        mt = (sc.substation_id == ev.substation_id) & (sc.ts >= ev.start) & (sc.ts < end)
        sc.loc[mt, "true_anom"] = True
        if sc.loc[m, "alert"].any():
            det[ev.event_type] += 1
    # an alert is a true positive if it falls inside an event or the 2h after (stuck detector lag)
    sc["near_event"] = False
    for ev in truth.itertuples():
        end = ev.start + pd.Timedelta(minutes=10 * (ev.n_intervals + 12))
        mk = (sc.substation_id == ev.substation_id) & (sc.ts >= ev.start) & (sc.ts < end)
        sc.loc[mk, "near_event"] = True
    alerts = sc[sc.alert]
    # which detector caught each event type (does the statistical rule add unique value?)
    caught_by = {}
    for ev in truth.itertuples():
        end = ev.start + pd.Timedelta(minutes=10 * (ev.n_intervals + 12))
        w = sc[(sc.substation_id == ev.substation_id) & (sc.ts >= ev.start) & (sc.ts < end)]
        for r in ["rule_en50160", "stat_z", "stuck"]:
            caught_by.setdefault(ev.event_type, {}).setdefault(r, 0)
            caught_by[ev.event_type][r] += int(w[r].any())
    prec = float(alerts.near_event.mean())
    by_rule = {r: dict(alerts=int(sc[r].sum()), precision=float(sc.loc[sc[r], "near_event"].mean()))
               for r in ["rule_en50160", "stat_z", "stuck"]}
    RESULTS["anomaly"] = dict(events=tot, detected=det,
                              recall={k: det[k] / tot[k] for k in tot},
                              overall_recall=sum(det.values()) / sum(tot.values()),
                              alert_intervals=int(len(alerts)), interval_precision=prec,
                              by_rule=by_rule, caught_by=caught_by, total_events=int(len(truth)))
    # ---- Figure: example week at one substation with a sag and its detection
    # pick a substation-window containing both a sag/swell and a subtle in-band dip if possible
    ex = truth[truth.event_type == "subtle_dip"].iloc[0]
    s0 = ex.start - pd.Timedelta(days=1.5)
    win = sc[(sc.substation_id == ex.substation_id) & (sc.ts >= s0) & (sc.ts < s0 + pd.Timedelta(days=3))]
    fig, ax = plt.subplots(figsize=(8.6, 3.4))
    ax.axhspan(207, 253, color="#f1f0ec", zorder=0, label="EN 50160 band (230 V ±10%)")
    ax.plot(win.ts, win.voltage_v, color=C[0], lw=1.4, label="Voltage (10-min mean)")
    a = win[win.alert]
    ax.scatter(a.ts, a.voltage_v, color=C[7], s=36, zorder=5, edgecolor="white", linewidth=1.2, label="Alert raised")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"Substation {ex.substation_id}: voltage telemetry and detector alerts", loc="left")
    ax.legend(loc="lower left", fontsize=8.5, ncol=3)
    fig.autofmt_xdate()
    save(fig, "fig4_10_scada_anomaly.png")


# ===================================================================== descriptive figures
def descriptive(spark):
    seg = spark.read.parquet(f"{GOLD}/segment_profile").toPandas()
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.0), sharex=True)
    for ax, s in zip(axes, ["Residential", "Commercial", "Industrial"]):
        for dt, col in [("Weekday", C[0]), ("Weekend", C[1])]:
            d = seg[(seg.customer_segment == s) & (seg.day_type == dt)].sort_values("slot")
            ax.plot(d.slot, d.avg_kw, color=col, label=dt)
        ax.set_title(s, loc="left")
        slot_axis(ax); ax.set_xticks(range(0, 49, 12)); ax.set_xticklabels(["00", "06", "12", "18", "24"])
        ax.set_xlabel("Local hour")
    axes[0].set_ylabel("Mean demand per meter (kW)")
    axes[0].legend(fontsize=8.5)
    save(fig, "fig4_3_segment_profiles.png")

    hm = spark.read.parquet(f"{GOLD}/month_hour_heatmap").toPandas()
    piv = hm.pivot(index="lhour", columns="month", values="avg_kw").sort_index()
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    im = ax.imshow(piv.values, aspect="auto", cmap="Blues", origin="lower")
    ax.set_xticks(range(12)); ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_yticks(range(0, 24, 3)); ax.set_yticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])
    ax.set_ylabel("Local hour"); ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035); cb.set_label("Mean system net demand (kW)", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("System net demand by month and hour of day", loc="left")
    save(fig, "fig4_4_month_hour_heatmap.png")

    tp = spark.read.parquet(f"{GOLD}/tariff_profile").toPandas()
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    ax.axvspan(32, 38, color="#f1f0ec", zorder=0)
    ax.text(35, ax.get_ylim()[1] if False else 0, "")
    for t, col, lab in [("Std", C[0], "Standard tariff"), ("ToU", C[1], "Time-of-Use tariff")]:
        d = tp[tp.tariff == t].sort_values("slot")
        ax.plot(d.slot, d.avg_kw, color=col, label=lab)
    slot_axis(ax)
    ymax = ax.get_ylim()[1]
    ax.text(31.6, 1.12, "16:00–19:00 peak price band →", ha="right", va="center", fontsize=8, color=INK2)
    ax.set_ylabel("Mean demand per household (kW)")
    ax.set_title("Residential winter-weekday profile: Standard vs Time-of-Use", loc="left")
    ax.legend(loc="upper left")
    save(fig, "fig4_5_tariff_profile.png")
    RESULTS["tou_peak_reduction_pct"] = float(
        100 * (1 - tp[(tp.tariff == "ToU") & tp.slot.between(32, 37)].avg_kw.mean()
               / tp[(tp.tariff == "Std") & tp.slot.between(32, 37)].avg_kw.mean()))

    pv = spark.read.parquet(f"{GOLD}/pv_net_profile").toPandas()
    ev = spark.read.parquet(f"{GOLD}/ev_profile").toPandas()
    full = pd.MultiIndex.from_product([[False, True], range(48)], names=["smart_charging", "slot"])
    ev = (ev.set_index(["smart_charging", "slot"])[["avg_kw_per_charger"]]
            .reindex(full, fill_value=0.0).reset_index())
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
    ax = axes[0]
    for mth, col, lab in [(12, C[0], "December"), (6, C[1], "June")]:
        d = pv[pv.month == mth].sort_values("slot")
        ax.plot(d.slot, d.avg_net_kw, color=col, label=lab)
    ax.axhline(0, color=INK2, lw=0.8)
    slot_axis(ax); ax.set_xticks(range(0, 49, 12)); ax.set_xticklabels(["00", "06", "12", "18", "24"])
    ax.set_xlabel("Local hour")
    ax.set_ylabel("Net grid demand per PV home (kW)")
    ax.set_title("(a) PV households: net demand", loc="left")
    ax.legend(fontsize=8.5)
    ax = axes[1]
    for sm, col, lab in [(False, C[7], "Unmanaged charging"), (True, C[2], "Smart (managed) charging")]:
        d = ev[ev.smart_charging == sm].sort_values("slot")
        ax.plot(d.slot, d.avg_kw_per_charger, color=col, label=lab)
    ax.axvspan(32, 38, color="#f1f0ec", zorder=0)
    slot_axis(ax); ax.set_xticks(range(0, 49, 12)); ax.set_xticklabels(["00", "06", "12", "18", "24"])
    ax.set_xlabel("Local hour")
    ax.set_ylabel("Mean EV load per charger (kW)")
    ax.set_title("(b) EV charging load", loc="left")
    ax.legend(fontsize=8.5)
    save(fig, "fig4_6_pv_ev.png")
    d_un = ev[(~ev.smart_charging) & ev.slot.between(32, 37)].avg_kw_per_charger.mean()
    d_sm = ev[(ev.smart_charging) & ev.slot.between(32, 37)].avg_kw_per_charger.mean()
    RESULTS["ev_peak_window_kw"] = dict(unmanaged=float(d_un), managed=float(d_sm))
    pvj = pv[pv.month == 6]
    RESULTS["pv_june_min_net_kw"] = float(pvj.avg_net_kw.min())
    RESULTS["pv_june_evening_max_kw"] = float(pvj[pvj.slot.between(34, 42)].avg_net_kw.max())


def dq_figure(metrics):
    q = metrics["quarantine"]
    items = [("Exact duplicate deliveries", metrics["exact_duplicates"], "removed"),
             ("Superseded corrected re-sends", metrics["superseded_resends"], "removed"),
             ("Null measurement values", metrics["null_values"], "imputed"),
             ("Negative values", q.get("NEGATIVE_VALUE", 0), "quarantined"),
             ("Unparseable timestamps", q.get("TS_UNPARSEABLE", 0), "quarantined"),
             ("Out-of-range (firmware overflow)", q.get("OUT_OF_RANGE", 0), "quarantined"),
             ("DST-ambiguous local timestamps", q.get("DST_AMBIGUOUS", 0), "quarantined"),
             ("Statistical spike vs meter history", q.get("STAT_SPIKE", 0), "quarantined")]
    items = sorted(items, key=lambda x: x[1])
    colmap = {"removed": C[0], "quarantined": C[1], "imputed": C[2]}
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ys = range(len(items))
    ax.barh(ys, [i[1] for i in items], color=[colmap[i[2]] for i in items], height=0.62)
    ax.set_yticks(ys); ax.set_yticklabels([i[0] for i in items])
    for y, i in zip(ys, items):
        ax.text(i[1] * 1.01 + 50, y, f"{i[1]:,}", va="center", fontsize=8.5, color=INK2)
    for k, v in colmap.items():
        ax.bar(0, 0, color=v, label=k.capitalize())
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_xlabel("Records affected")
    ax.grid(axis="y", visible=False)
    ax.set_title("Data-quality outcomes in the Silver layer", loc="left")
    save(fig, "fig4_1_dq_outcomes.png")


def spark_log_figure():
    lines = open(os.path.join(BASE, "logs", "pipeline_run.log")).read().strip().splitlines()
    lines = [l.replace("\t", " ") for l in lines if l.startswith("[") or l.startswith("real")]
    import textwrap
    wrapped = []
    for l in lines:
        wrapped += textwrap.wrap(l, 108, subsequent_indent="           ", break_on_hyphens=False)
    wrapped = ["$ python mes_pipeline_spark.py"] + wrapped
    fig = plt.figure(figsize=(9.6, 0.21 * len(wrapped) + 0.4))
    fig.patch.set_facecolor("#1e1e1e")
    for i, l in enumerate(wrapped):
        col = "#9cdcfe" if l.startswith("$") else ("#b5cea8" if "[done]" in l else "#d4d4d4")
        fig.text(0.012, 1 - (i + 1) / (len(wrapped) + 1), l, family="DejaVu Sans Mono",
                 fontsize=7.6, color=col, va="center")
    fig.savefig(os.path.join(FIG, "fig4_2_spark_execution_log.png"), dpi=200, facecolor="#1e1e1e")
    plt.close(fig)


if __name__ == "__main__":
    spark = (SparkSession.builder.appName("MES-Forecast").master("local[2]")
             .config("spark.driver.memory", "4g").config("spark.sql.shuffle.partitions", "8")
             .config("spark.sql.session.timeZone", "UTC").config("spark.ui.showConsoleProgress", "false")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    metrics = json.load(open(os.path.join(BASE, "outputs", "pipeline_metrics.json")))
    dq_figure(metrics)
    spark_log_figure()
    descriptive(spark)
    forecasting(spark)
    anomaly_eval(spark)
    with open(os.path.join(BASE, "outputs", "analysis_results.json"), "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print(json.dumps(RESULTS, indent=2, default=float))
    spark.stop()

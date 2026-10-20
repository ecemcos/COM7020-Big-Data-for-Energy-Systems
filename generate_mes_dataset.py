"""
generate_mes_dataset.py
-----------------------
Generates a multi-table synthetic dataset for MetroEnergy Solutions (MES).

The meter-reading table follows the structure of the UK Power Networks
"SmartMeter Energy Consumption Data in London Households" (Low Carbon London)
open dataset: one row per meter per half-hour, a meter identifier, a tariff
flag (Std / ToU) and an ACORN socio-demographic group. It is extended with the
other MES sources named in the brief: solar PV, EV charge-point sessions
(OCPP-style), substation SCADA telemetry and regional weather.

Three meter vendors deliver data in different formats (the brief's
"inconsistent data formats across vendors"):
    Vendor A  ISO-8601 UTC timestamps, kWh
    Vendor B  local wall-clock timestamps without offset, kWh
    Vendor C  legacy, dd/MM/yyyy HH:mm local timestamps, Wh

Realistic data-quality defects are injected so the pipeline has real work:
duplicate deliveries, corrected re-sends, negative values, firmware overflow
spikes, null values, malformed timestamps, missing intervals and multi-hour
communication outages. DST-ambiguous local timestamps arise naturally from
Vendor B/C at the October clock change.

Output (CSV): data/raw/meters.csv, data/raw/readings/readings_YYYY-MM.csv (Jan 2024 - Dec 2025),
data/raw/ev_sessions.csv, data/raw/substation_scada.csv,
data/raw/weather_hourly.csv, data/raw/vendor_units.csv,
data/raw/scada_events_truth.csv (ground truth used only for evaluation).
"""
import os
import numpy as np
import pandas as pd
import holidays

SEED = 20251
rng = np.random.default_rng(SEED)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
os.makedirs(os.path.join(OUT, "readings"), exist_ok=True)

# ---------------------------------------------------------------- regions
REGIONS = pd.DataFrame([
    # region, country, tz, lat, mean temp, seasonal amplitude, cloudiness
    ("London", "UK", "Europe/London", 51.5, -0.1, 11.8, 6.8, 0.55),
    ("Greater Manchester", "UK", "Europe/London", 53.5, -2.2, 10.2, 6.2, 0.62),
    ("Edinburgh", "UK", "Europe/London", 55.9, -3.2, 9.1, 5.6, 0.64),
    ("Berlin-Brandenburg", "DE", "Europe/Berlin", 52.5, 13.4, 10.3, 9.4, 0.52),
    ("Noord-Holland", "NL", "Europe/Amsterdam", 52.4, 4.9, 10.6, 6.9, 0.57),
    ("Valencia", "ES", "Europe/Madrid", 39.5, -0.4, 18.3, 7.4, 0.25),
], columns=["region", "country", "tz", "lat", "lon", "t_mean", "t_amp", "cloud"])

HOL = {c: holidays.country_holidays(c, years=[2024, 2025])
       for c in ["UK", "DE", "NL", "ES"]}

IDX = pd.date_range("2024-01-01 00:00", "2025-12-31 23:30", freq="30min", tz="UTC")
N = len(IDX)                      # 35,088 half-hours (2 years, 2024 is a leap year)


# ---------------------------------------------------------------- weather
def make_weather():
    hourly = pd.date_range("2024-01-01", "2025-12-31 23:00", freq="h", tz="UTC")
    out = []
    for r in REGIONS.itertuples():
        loc = hourly.tz_convert(r.tz)
        doy = loc.dayofyear.values
        hr = loc.hour.values
        dayno = (loc.normalize() - loc.normalize()[0]).days.values
        # persistent day-to-day anomaly (AR(1) on daily values): cold snaps / warm spells
        days = int(dayno.max()) + 1
        anom = np.zeros(days)
        for d in range(1, days):
            anom[d] = 0.8 * anom[d - 1] + rng.normal(0, 2.2)
        daily_anom = anom[dayno]
        temp = (r.t_mean - r.t_amp * np.cos(2 * np.pi * (doy - 18) / 365.25)
                + 3.2 * np.sin(2 * np.pi * (hr - 9) / 24) + daily_anom
                + rng.normal(0, 0.6, len(hourly)))
        # simple clear-sky irradiance from solar elevation
        decl = 23.44 * np.sin(np.radians(360 / 365 * (doy - 81)))
        hour_angle = 15 * (hourly.hour.values + 0.5 + r.lon / 15 - 12)   # solar time from UTC
        lat = np.radians(r.lat)
        sin_el = (np.sin(lat) * np.sin(np.radians(decl))
                  + np.cos(lat) * np.cos(np.radians(decl)) * np.cos(np.radians(hour_angle)))
        clear = np.clip(1050 * sin_el, 0, None)
        day_cloud = rng.beta(2, 2 * (1 - r.cloud) / r.cloud + 0.5, days)
        ghi = clear * (1 - 0.75 * day_cloud[dayno])
        out.append(pd.DataFrame({
            "region": r.region, "ts_utc": hourly.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "temperature_c": temp.round(1), "ghi_wm2": ghi.round(0),
            "wind_ms": np.abs(rng.normal(4.5, 2.0, len(hourly))).round(1)}))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- meters
def make_meters():
    rows = []
    n_res, n_com, n_ind = 200, 32, 8
    acorn = ["Affluent", "Comfortable", "Adversity"]
    k = 0
    for seg, n in [("Residential", n_res), ("Commercial", n_com), ("Industrial", n_ind)]:
        for _ in range(n):
            k += 1
            reg = REGIONS.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
            if reg.country == "UK":
                vendor = rng.choice(["A", "B", "C"], p=[0.40, 0.25, 0.35])
            else:
                vendor = rng.choice(["A", "B", "C"], p=[0.40, 0.50, 0.10])
            has_solar = rng.random() < (0.30 if seg == "Residential" else 0.40)
            if reg.region == "Valencia":
                has_solar = rng.random() < 0.55
            rows.append(dict(
                meter_id=f"MAC{k:06d}", customer_segment=seg,
                tariff=("ToU" if (seg == "Residential" and rng.random() < 0.25) else "Std"),
                acorn_group=(rng.choice(acorn, p=[0.35, 0.30, 0.35]) if seg == "Residential" else "n/a"),
                region=reg.region, country=reg.country,
                substation_id=f"SS-{REGIONS.index[REGIONS.region == reg.region][0] * 2 + int(rng.integers(0, 2)) + 1:02d}",
                vendor=vendor,
                heat_pump=bool(seg == "Residential" and rng.random() < (0.12 if reg.country == "UK" else 0.30)),
                has_solar=bool(has_solar),
                pv_kwp=(round(float(rng.uniform(3, 6)), 1) if (has_solar and seg == "Residential")
                        else round(float(rng.uniform(30, 120)), 1) if has_solar else 0.0),
                has_ev=bool(seg == "Residential" and rng.random() < 0.25),
                smart_charging=False))
    m = pd.DataFrame(rows)
    ev = m.has_ev
    m.loc[ev, "smart_charging"] = rng.random(ev.sum()) < 0.5
    return m


# ---------------------------------------------------------------- loads
def local_calendar(tz, country):
    loc = IDX.tz_convert(tz)
    hod = loc.hour.values + loc.minute.values / 60
    dow = loc.dayofweek.values
    hol = np.array([d in HOL[country] for d in loc.date])
    return loc, hod, dow, hol


def gauss(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2)


def residential_load(m, hod, dow, hol, temp):
    weekend = (dow >= 5) | hol
    base = {"Affluent": 0.14, "Comfortable": 0.11, "Adversity": 0.09}[m.acorn_group]
    shape = (0.55
             + 1.1 * gauss(hod, 7.6, 0.9) * ~weekend + 1.0 * gauss(hod, 9.5, 1.5) * weekend
             + 0.8 * gauss(hod, 13, 2.5) * weekend
             + 2.2 * gauss(hod, 18.7, 1.6))
    if m.tariff == "ToU":                           # price response: shift out of 16-19
        peak = (hod >= 16) & (hod < 19)
        shape = np.where(peak, shape * 0.80, shape)
        shape = shape + 0.25 * gauss(hod, 22.5, 1.0)
    hdd = np.clip(15.5 - temp, 0, None)
    heat = (0.09 if m.heat_pump else 0.006) * hdd * (0.6 + 0.5 * gauss(hod, 7.5, 2) + 0.5 * gauss(hod, 18.5, 2.5))
    load = base * shape + heat
    return load * rng.lognormal(0, 0.28, N)


def commercial_load(m, hod, dow, hol, temp, region):
    open_ = (dow < 5) & ~hol & (hod >= 7.5) & (hod < 18.5)
    sat = (dow == 5) & ~hol & (hod >= 9) & (hod < 15)
    level = rng.uniform(4, 14)
    load = level * (0.28 + 0.72 * open_ + 0.35 * sat)
    cdd = np.clip(temp - 22, 0, None)
    hdd = np.clip(14 - temp, 0, None)
    load = load + level * (0.05 * cdd + 0.015 * hdd) * open_
    return load * rng.lognormal(0, 0.10, N)


def industrial_load(m, hod, dow, hol, loc):
    shift = (hod >= 6) & (hod < 22)
    weekday = (dow < 5) & ~hol
    level = rng.uniform(45, 90)
    load = level * np.where(weekday, np.where(shift, 1.0, 0.55), 0.35)
    xmas = ((loc.month == 12) & (loc.day >= 23)) | ((loc.month == 1) & (loc.day <= 2))
    load = np.where(xmas, level * 0.25, load)
    return load * rng.lognormal(0, 0.06, N)


def ev_sessions(m, loc, dow):
    """OCPP-style charging sessions; returns per-interval kWh and a session list."""
    kwh = np.zeros(N)
    sessions = []
    days = pd.Series(loc.normalize().unique())
    for d in days:
        wk = d.dayofweek < 5
        if rng.random() > (0.42 if wk else 0.28):
            continue
        energy = float(rng.uniform(6, 30))
        if m.smart_charging:
            start = d + pd.Timedelta(hours=24 + 0.5 + float(rng.uniform(0, 2.5)))
        else:
            start = d + pd.Timedelta(hours=float(np.clip(rng.normal(18.4, 1.3), 15, 23)))
        n_slots = int(np.ceil(energy / 3.6))      # 7.2 kW charger = 3.6 kWh per half-hour
        s_utc = start.tz_convert("UTC").floor("30min")
        i0 = IDX.searchsorted(s_utc)
        if i0 >= N:
            continue
        i1 = min(i0 + n_slots, N)
        per = energy / n_slots
        kwh[i0:i1] += per
        sessions.append(dict(
            session_id=f"EVS-{m.meter_id}-{len(sessions) + 1:04d}",
            charger_id=f"CP-{m.meter_id[3:]}", meter_id=m.meter_id,
            start_ts_utc=IDX[i0].strftime("%Y-%m-%dT%H:%M:%SZ"),
            stop_ts_utc=(IDX[i1 - 1] + pd.Timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            energy_kwh=round(per * (i1 - i0), 2), max_power_kw=7.2,
            smart_charging=bool(m.smart_charging)))
    return kwh, sessions


def pv_generation(m, wx_region):
    if not m.has_solar:
        return np.zeros(N)
    ghi_h = wx_region.set_index("ts_utc")["ghi_wm2"].values
    ghi = np.repeat(ghi_h, 2)[:N]
    return m.pv_kwp * ghi / 1000 * 0.78 * 0.5 * rng.lognormal(0, 0.05, N)


def fmt_ts(vendor, tz):
    if vendor == "A":
        return IDX.strftime("%Y-%m-%dT%H:%M:%SZ")
    loc = IDX.tz_convert(tz)
    if vendor == "B":
        return loc.strftime("%Y-%m-%d %H:%M:%S")
    return loc.strftime("%d/%m/%Y %H:%M")


# ---------------------------------------------------------------- main build
TOTAL = [0]


def main():
    import shutil
    shutil.rmtree(os.path.join(OUT, "readings"), ignore_errors=True)
    os.makedirs(os.path.join(OUT, "readings"))
    wx = make_weather()
    wx.to_csv(os.path.join(OUT, "weather_hourly.csv"), index=False)
    meters = make_meters()
    meters.to_csv(os.path.join(OUT, "meters.csv"), index=False)
    pd.DataFrame([("A", "kWh", 1.0), ("B", "kWh", 1.0), ("C", "Wh", 0.001)],
                 columns=["vendor", "unit", "to_kwh_factor"]).to_csv(
        os.path.join(OUT, "vendor_units.csv"), index=False)

    all_sessions = []
    sub_load = {}                                   # for coherent SCADA voltage
    files = {}                                      # one CSV per year-month, appended per meter
    rid = 0
    for m in meters.itertuples():
        reg = REGIONS[REGIONS.region == m.region].iloc[0]
        loc, hod, dow, hol = local_calendar(reg.tz, reg.country)
        w = wx[wx.region == m.region]
        temp = np.repeat(w["temperature_c"].values, 2)[:N]
        if m.customer_segment == "Residential":
            load = residential_load(m, hod, dow, hol, temp)
        elif m.customer_segment == "Commercial":
            load = commercial_load(m, hod, dow, hol, temp, m.region)
        else:
            load = industrial_load(m, hod, dow, hol, loc)
        ev = np.zeros(N)
        if m.has_ev:
            ev, sess = ev_sessions(m, loc, dow)
            all_sessions += sess
        pv = pv_generation(m, w)
        gross = load + ev
        self_use = np.minimum(gross, pv)
        imp = gross - self_use
        exp = pv - self_use
        sub_load.setdefault(m.substation_id, np.zeros(N))
        sub_load[m.substation_id] += imp - exp

        factor = 1000.0 if m.vendor == "C" else 1.0
        df = pd.DataFrame({
            "meter_id": m.meter_id, "vendor": m.vendor,
            "ts": fmt_ts(m.vendor, reg.tz),
            "import_value": (imp * factor).round(3 if factor == 1 else 0),
            "export_value": (exp * factor).round(3 if factor == 1 else 0),
            "unit": "Wh" if m.vendor == "C" else "kWh",
            "sent_at": (IDX + pd.Timedelta(minutes=35)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "_ym": IDX.strftime("%Y-%m")})
        # ---- missing intervals (random gaps) and comms outages (6-48 h)
        keep = rng.random(N) > 0.004
        for _ in range(int(rng.poisson(1.6))):
            s0 = int(rng.integers(0, N - 100)); keep[s0:s0 + int(rng.integers(12, 96))] = False
        raw = df[keep].reset_index(drop=True)
        n = len(raw)
        raw.insert(0, "reading_id", [f"R{i:09d}" for i in range(rid, rid + n)])
        rid += n
        # ---- value defects
        idx = rng.choice(n, int(n * 0.0005), replace=False); raw.loc[idx, "import_value"] *= -1
        idx = rng.choice(n, int(n * 0.0002), replace=False); raw.loc[idx, "import_value"] = 99999.0
        idx = rng.choice(n, int(n * 0.002), replace=False); raw.loc[idx, "import_value"] = np.nan
        idx = rng.choice(n, int(n * 0.0002), replace=False)
        raw.loc[idx, "ts"] = rng.choice(["", "NULL", "2025-13-41 25:61:00", "not_a_date"], len(idx))
        # ---- at-least-once duplicates (identical re-delivery)
        dup = raw.sample(frac=0.003, random_state=int(rng.integers(1e9)))
        # ---- corrected re-sends: same meter+interval, new id, revised value, later sent_at
        cor = raw.sample(frac=0.0008, random_state=int(rng.integers(1e9))).copy()
        cor["reading_id"] = ["C" + r[1:] for r in cor["reading_id"]]
        cor["import_value"] = (cor["import_value"] * rng.uniform(0.9, 1.1, len(cor))).round(3)
        cor["sent_at"] = (pd.to_datetime(cor["sent_at"]) + pd.Timedelta(days=2)).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        raw = pd.concat([raw, dup, cor], ignore_index=True)
        for ym, part in raw.groupby("_ym"):
            path = os.path.join(OUT, "readings", f"readings_{ym}.csv")
            part.drop(columns="_ym").to_csv(path, mode="a", header=path not in files, index=False)
            files[path] = True
        TOTAL[0] += len(raw)
    pd.DataFrame(all_sessions).to_csv(os.path.join(OUT, "ev_sessions.csv"), index=False)

    # ---------------------------------------------------------- substation SCADA (10-min)
    t10 = pd.date_range("2025-01-01", "2025-12-31 23:50", freq="10min", tz="UTC")
    rows, truth = [], []
    freq_area = {"UK": rng.normal(50, 0.045, len(t10)), "CE": rng.normal(50, 0.025, len(t10))}
    for ss, load in sorted(sub_load.items()):
        reg = meters[meters.substation_id == ss].region.iloc[0]
        area = "UK" if REGIONS.set_index("region").loc[reg, "country"] == "UK" else "CE"
        l10 = np.repeat(load[-17520:], 3)[:len(t10)]     # SCADA kept for 2025 only
        lmax = np.percentile(np.abs(l10), 99.5)
        v = 236.0 - 9.0 * np.clip(l10, 0, None) / lmax + 7.5 * np.clip(-l10, 0, None) / lmax
        v = v + rng.normal(0, 0.9, len(t10))
        load_pct = 100 * np.clip(l10, 0, None) / (lmax * 1.15)
        f = freq_area[area].copy()
        # inject labelled events
        for _ in range(int(rng.integers(10, 16))):
            kind = rng.choice(["sag", "swell", "stuck_sensor", "subtle_dip"], p=[0.35, 0.25, 0.20, 0.20])
            s = int(rng.integers(0, len(t10) - 20)); d = int(rng.integers(1, 7))
            if kind == "subtle_dip":       # in-band but abnormal: e.g. loose neutral / incipient fault
                d = int(rng.integers(3, 7)); v[s:s + d] = v[s:s + d] - rng.uniform(11, 16)
            elif kind == "sag":
                v[s:s + d] = rng.uniform(194, 205, d)
            elif kind == "swell":
                v[s:s + d] = rng.uniform(254, 262, d)
            else:
                d = 12; v[s:s + d] = round(float(v[s]), 1)
            truth.append(dict(substation_id=ss, event_type=kind,
                              start_ts_utc=t10[s].strftime("%Y-%m-%dT%H:%M:%SZ"),
                              n_intervals=d))
        rows.append(pd.DataFrame({
            "substation_id": ss, "region": reg,
            "ts_utc": t10.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "voltage_v": v.round(1), "frequency_hz": f.round(3),
            "transformer_load_pct": load_pct.round(1),
            "oil_temp_c": (35 + 0.35 * load_pct + rng.normal(0, 1.5, len(t10))).round(1)}))
    pd.concat(rows, ignore_index=True).to_csv(os.path.join(OUT, "substation_scada.csv"), index=False)
    pd.DataFrame(truth).to_csv(os.path.join(OUT, "scada_events_truth.csv"), index=False)

    print(f"meters={len(meters)} readings={TOTAL[0]:,} ev_sessions={len(all_sessions):,} "
          f"scada={len(t10) * len(sub_load):,} events={len(truth)}")


if __name__ == "__main__":
    main()

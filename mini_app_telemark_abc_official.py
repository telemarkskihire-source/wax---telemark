# app.py  — Telemark · Snow Temps + Wax (A/B/C)  — single-file

import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import time
from dateutil import tz

# ---------------------- BASIC PAGE ----------------------
st.set_page_config(page_title="Telemark · Snow Temps + Wax", page_icon="❄️", layout="wide")

# ---------------------- THEME & CSS ----------------------
PRIMARY = "#10bfcf"      # Telemark turquoise
BG = "#0f172a"           # slate-900
CARD = "#111827"
TEXT = "#e5e7eb"

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
  background: linear-gradient(180deg, {BG} 0%, #111827 100%);
}}
.block-container {{ padding-top: 1rem; }}
.hero {{ display:flex;gap:14px;align-items:center;margin-bottom:6px; }}
.hero h1 {{ margin:0;font-size:1.6rem;color:{TEXT};letter-spacing:0.2px; }}
.badge {{
  border:1px solid rgba(255,255,255,.15);
  color:{TEXT};opacity:0.9;
  padding:4px 10px;border-radius:999px;font-size:.78rem;
  background: rgba(255,255,255,.03);
}}
.card {{
  background:{CARD}; border:1px solid rgba(255,255,255,.08);
  border-radius:16px; padding:14px; box-shadow:0 8px 20px rgba(0,0,0,.25);
}}
.kpi {{ display:flex;flex-direction:column;gap:2px;
  background:rgba(16,191,207,.06); border:1px dashed rgba(16,191,207,.45);
  padding:10px 12px; border-radius:12px; }}
.kpi .label {{ font-size:.78rem; color:#93c5fd; }}
.kpi .value {{ font-size:1rem; font-weight:700; color:{TEXT}; }}
.tag {{ display:inline-block; margin-right:6px; margin-top:6px;
  padding:3px 8px; border-radius:999px; font-size:.72rem;
  border:1px solid rgba(255,255,255,.12); color:{TEXT}; opacity:.9; }}
.section-title {{ color:{TEXT}; opacity:.95; margin:8px 0 4px 2px; }}
</style>
""", unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
cA, cB = st.columns([1,3], vertical_alignment="center")
with cA:
    st.markdown(f"<div class='hero'><h1>Telemark · Snow Temps + Wax</h1></div>", unsafe_allow_html=True)
with cB:
    st.markdown("<span class='badge'>Mobile-first · Open-Meteo · Consigli sciolina</span>", unsafe_allow_html=True)

# ---------------------- CONTROLS ----------------------
st.markdown("<p class='section-title'>1) Scegli lo spot e scarica i dati</p>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    spot = st.selectbox("Località", ["Champoluc (Ramey)", "Gressoney (Stafal)", "Alagna (Pianalunga)", "Personalizzata"])
    if spot == "Champoluc (Ramey)":
        lat, lon = 45.831, 7.730
    elif spot == "Gressoney (Stafal)":
        lat, lon = 45.824, 7.827
    elif spot == "Alagna (Pianalunga)":
        lat, lon = 45.855, 7.941
    else:
        lat = st.number_input("Latitudine", value=45.831, format="%.6f")
        lon = st.number_input("Longitudine", value=7.730, format="%.6f")
with col2:
    tzname = st.selectbox("Timezone", ["Europe/Rome","UTC"], index=0)
    hours = st.slider("Ore previsione", 12, 168, 72, 12)

st.markdown("<p class='section-title'>2) Finestre orarie (oggi): A · B · C</p>", unsafe_allow_html=True)
b1,b2,b3 = st.columns(3)
with b1:
    st.markdown("**Blocco A**")
    A_start = st.time_input("Inizio A", value=time(9,0), key="A_s")
    A_end   = st.time_input("Fine A",   value=time(11,0), key="A_e")
with b2:
    st.markdown("**Blocco B**")
    B_start = st.time_input("Inizio B", value=time(11,0), key="B_s")
    B_end   = st.time_input("Fine B",   value=time(13,0), key="B_e")
with b3:
    st.markdown("**Blocco C**")
    C_start = st.time_input("Inizio C", value=time(13,0), key="C_s")
    C_end   = st.time_input("Fine C",   value=time(16,0), key="C_e")

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
go_col, up_col = st.columns([1,1])
with go_col:
    go = st.button("Scarica previsioni (Open-Meteo)", type="primary")
with up_col:
    upl = st.file_uploader("…oppure carica CSV compatibile", type=["csv"])

# ---------------------- DATA FETCH / PREP ----------------------
def fetch_open_meteo(lat, lon, timezone_str):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon, "timezone": timezone_str,
        "hourly": "temperature_2m,dew_point_2m,precipitation,rain,snowfall,cloudcover,windspeed_10m,is_day,weathercode",
        "forecast_days": 7
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def prp_type(df):
    snow_codes = {71,73,75,77,85,86}; rain_codes = {51,53,55,61,63,65,80,81,82}
    def f(row):
        prp = row.precipitation; rain = getattr(row,"rain",0.0); snow = getattr(row,"snowfall",0.0)
        if prp<=0 or pd.isna(prp): return "none"
        if rain>0 and snow>0: return "mixed"
        if snow>0 and rain==0: return "snow"
        if rain>0 and snow==0: return "rain"
        code = int(getattr(row,"weathercode",0)) if pd.notna(getattr(row,"weathercode",None)) else 0
        if code in snow_codes: return "snow"
        if code in rain_codes: return "rain"
        return "mixed"
    return df.apply(f, axis=1)

def build_df(js, hours, tzname):
    # 👉 keep times NAIVE to avoid tz-aware vs naive comparison errors
    h = js["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])     # naive
    now_naive = pd.Timestamp.now().floor("H")   # naive
    df = df[df["time"] >= now_naive].head(hours).reset_index(drop=True)

    out = pd.DataFrame()
    out["time"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    out["T2m"] = df["temperature_2m"].astype(float)
    out["cloud"] = (df["cloudcover"].astype(float)/100).clip(0,1)
    out["wind"] = (df["windspeed_10m"].astype(float)/3.6).round(3)
    out["sunup"] = df["is_day"].astype(int)
    out["prp_mmph"] = df["precipitation"].astype(float)
    extra = df[["precipitation","rain","snowfall","weathercode"]].copy()
    out["prp_type"] = prp_type(extra)
    out["td"] = df["dew_point_2m"].astype(float)
    return out

# ---------------------- SNOW TEMPERATURE MODEL ----------------------
def compute_snow_temperature(df, dt_hours=1.0):
    """Heuristic, robust for hourly forecast."""
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    required = ["T2m","cloud","wind","sunup","prp_mmph","prp_type"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    if "td" not in df.columns:
        df["td"] = float("nan")
    df = df.sort_values("time").reset_index(drop=True)

    rain = df["prp_type"].str.lower().isin(["rain","mixed"])
    snow = df["prp_type"].str.lower().eq("snow")
    sunup = df["sunup"].astype(int) == 1
    tw = (df["T2m"] + df["td"]) / 2.0
    wet = (rain | (df["T2m"] > 0) |
           (sunup & (df["cloud"] < 0.3) & (df["T2m"] >= -3)) |
           (snow & (df["T2m"] >= -1)) |
           (snow & tw.ge(-0.5).fillna(False)))

    T_surf = pd.Series(index=df.index, dtype=float)
    T_surf.loc[wet] = 0.0

    dry = ~wet
    clear = (1.0 - df["cloud"]).clip(0.0, 1.0)
    windc = df["wind"].clip(upper=6.0)
    drad = (1.5 + 3.0*clear - 0.3*windc).clip(0.5, 4.5)
    T_surf.loc[dry] = df["T2m"][dry] - drad[dry]

    sunny_cold = sunup & dry & df["T2m"].between(-10, 0, inclusive="both")
    T_surf.loc[sunny_cold] = pd.concat([
        (df["T2m"] + 0.5*(1.0 - df["cloud"]))[sunny_cold],
        pd.Series(-0.5, index=df.index)[sunny_cold]
    ], axis=1).min(axis=1)

    T_top5 = pd.Series(index=df.index, dtype=float)
    tau = pd.Series(6.0, index=df.index, dtype=float)
    tau.loc[rain | snow | (df["wind"] >= 6)] = 3.0
    tau.loc[(~sunup) & (df["wind"] < 2) & (df["cloud"] < 0.3)] = 8.0
    alpha = 1.0 - (2.718281828 ** (-dt_hours / tau))
    if len(df) > 0:
        T_top5.iloc[0] = min(df["T2m"].iloc[0], 0.0)
        for i in range(1, len(df)):
            T_top5.iloc[i] = T_top5.iloc[i-1] + alpha.iloc[i] * (T_surf.iloc[i] - T_top5.iloc[i-1])

    df["T_surf"] = T_surf
    df["T_top5"] = T_top5
    return df

# ---------------------- WAX SUGGESTIONS ----------------------
SWIX = [("PS5 Turquoise", -18,-10), ("PS6 Blue",-12,-6), ("PS7 Violet",-8,-2), ("PS8 Red",-4,4), ("PS10 Yellow",0,10)]
TOKO = [("Blue",-30,-9), ("Red",-12,-4), ("Yellow",-6,0)]
VOLA = [("MX-E Violet/Blue",-12,-4), ("MX-E Red",-5,0), ("MX-E Warm",-2,10)]
RODE = [("R20 Blue",-18,-8), ("R30 Violet",-10,-3), ("R40 Red",-5,0), ("R50 Yellow",-1,10)]

def pick(bands, t):
    for n,tmin,tmax in bands:
        if t>=tmin and t<=tmax:
            return n
    return bands[-1][0] if t>bands[-1][2] else bands[0][0]

def wax_cards(t_med, wet):
    cols = st.columns(4)
    cols[0].markdown(f"<div class='kpi'><div class='label'>Swix</div><div class='value'>{pick(SWIX,t_med)}</div></div>", unsafe_allow_html=True)
    cols[1].markdown(f"<div class='kpi'><div class='label'>Toko</div><div class='value'>{pick(TOKO,t_med)}</div></div>", unsafe_allow_html=True)
    cols[2].markdown(f"<div class='kpi'><div class='label'>Vola</div><div class='value'>{pick(VOLA,t_med)}</div></div>", unsafe_allow_html=True)
    cols[3].markdown(f"<div class='kpi'><div class='label'>Rode</div><div class='value'>{pick(RODE,t_med)}</div></div>", unsafe_allow_html=True)
    if t_med <= -10:
        st.markdown("<span class='tag'>Neve fredda · struttura fine</span>", unsafe_allow_html=True)
    if -3 <= t_med <= 1 and wet:
        st.markdown("<span class='tag'>Neve umida · top-coat liquido</span>", unsafe_allow_html=True)

def badge(win, tzname):
    tmin = float(win["T_surf"].min()); tmax = float(win["T_surf"].max())
    wet = bool(((win["prp_type"].isin(["rain","mixed"])) | (win["prp_mmph"]>0.5)).any())
    if wet or tmax > 0.5:
        color, title, desc = "#ef4444", "CRITICAL", "Possibile neve bagnata/pioggia · struttura grossa"
    elif tmax > -1.0:
        color, title, desc = "#f59e0b", "WATCH", "Vicino a 0°C · cere medio-morbide"
    else:
        color, title, desc = "#22c55e", "OK", "Neve fredda/asciutta · cere dure"
    st.markdown(f"""
    <div class='card' style='border-color:{color}'>
      <div style='font-weight:800;color:{color};margin-bottom:4px'>{title}</div>
      <div style='color:{TEXT};opacity:.95'>{desc}</div>
      <div style='font-size:12px;opacity:.7;margin-top:6px'>T_surf min {tmin:.1f}°C · max {tmax:.1f}°C</div>
    </div>""", unsafe_allow_html=True)
    return wet

# ---------------------- PLOTS ----------------------
def make_plots(res):
    fig1 = plt.figure()
    t = pd.to_datetime(res["time"])
    plt.plot(t, res["T2m"], label="T2m")
    plt.plot(t, res["T_surf"], label="T_surf")
    plt.plot(t, res["T_top5"], label="T_top5")
    plt.legend(); plt.title("Temperature vs tempo"); plt.xlabel("Ora"); plt.ylabel("°C")
    fig2 = plt.figure()
    plt.bar(t, res["prp_mmph"])
    plt.title("Precipitazione (mm/h)"); plt.xlabel("Ora"); plt.ylabel("mm/h")
    return fig1, fig2

# ---------------------- WINDOW SLICE ----------------------
def window_slice(res, tzname, s, e):
    t = pd.to_datetime(res["time"]).dt.tz_localize(tz.gettz(tzname), nonexistent='shift_forward', ambiguous='NaT')
    D = res.copy(); D["dt"] = t
    today = pd.Timestamp.now(tz=tz.gettz(tzname)).date()
    W = D[(D["dt"].dt.date==today) & (D["dt"].dt.time>=s) & (D["dt"].dt.time<=e)]
    return W if not W.empty else D.head(7)

# ---------------------- MAIN RUN ----------------------
def run_all(res):
    st.markdown("### Risultati")
    st.dataframe(res, use_container_width=True)
    f1,f2 = make_plots(res); st.pyplot(f1); st.pyplot(f2)
    st.download_button("Scarica CSV", data=res.to_csv(index=False), file_name="forecast_with_snowT.csv", mime="text/csv")

    st.markdown("### Consigli per blocchi A · B · C")
    for label,(s,e) in {"A":(A_start,A_end),"B":(B_start,B_end),"C":(C_start,C_end)}.items():
        st.markdown(f"#### Blocco {label}")
        W = window_slice(res, tzname, s, e)
        wet = badge(W, tzname)
        t_med = float(W["T_surf"].mean())
        wax_cards(t_med, wet)

try:
    if upl is not None:
        df_upl = pd.read_csv(upl)
        res = compute_snow_temperature(df_upl, dt_hours=1.0)
        st.success("CSV caricato.")
        run_all(res)
    elif go:
        js = fetch_open_meteo(lat, lon, tzname)
        df = build_df(js, hours, tzname)
        res = compute_snow_temperature(df, dt_hours=1.0)
        st.success("Previsioni scaricate.")
        run_all(res)
    else:
        st.info("Seleziona lo spot, definisci A/B/C e premi **Scarica previsioni** (oppure carica un CSV).")
except Exception as e:
    st.error(f"Errore: {e}")

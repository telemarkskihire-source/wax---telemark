# mini_app_telemark_pretty_autocomplete.py
# Telemark · Snow Temps + Wax · Mobile-first UI con ricerca località (autocomplete)
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import time
from dateutil import tz

# --------- MODEL ---------
def compute_snow_temperature(df, dt_hours=1.0):
    import math
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
    wet = (
        rain |
        (df["T2m"] > 0) |
        (sunup & (df["cloud"] < 0.3) & (df["T2m"] >= -3)) |
        (snow & (df["T2m"] >= -1)) |
        (snow & tw.ge(-0.5).fillna(False))
    )
    T_surf = pd.Series(index=df.index, dtype=float)
    T_surf.loc[wet] = 0.0
    dry = ~wet
    clear = (1.0 - df["cloud"]).clip(lower=0.0, upper=1.0)
    windc = df["wind"].clip(upper=6.0)
    drad = (1.5 + 3.0*clear - 0.3*windc).clip(lower=0.5, upper=4.5)
    T_surf.loc[dry] = df["T2m"][dry] - drad[dry]
    sunny_cold = sunup & dry & df["T2m"].between(-10, 0, inclusive="both")
    if sunny_cold.any():
        T_surf.loc[sunny_cold] = pd.concat([
            (df["T2m"] + 0.5*(1.0 - df["cloud"]))[sunny_cold],
            pd.Series(-0.5, index=df.index)[sunny_cold]
        ], axis=1).min(axis=1)
    T_top5 = pd.Series(index=df.index, dtype=float)
    tau = pd.Series(6.0, index=df.index, dtype=float)
    tau.loc[rain | snow | (df["wind"] >= 6)] = 3.0
    tau.loc[(~sunup) & (df["wind"] < 2) & (df["cloud"] < 0.3)] = 8.0
    alpha = 1.0 - (math.e ** (-dt_hours / tau))
    if len(df) > 0:
        T_top5.iloc[0] = min(df["T2m"].iloc[0], 0.0)
        for i in range(1, len(df)):
            T_top5.iloc[i] = T_top5.iloc[i-1] + alpha.iloc[i] * (T_surf.iloc[i] - T_top5.iloc[i-1])
    df["T_surf"] = T_surf
    df["T_top5"] = T_top5
    return df

# --------- THEME / CSS ---------
PRIMARY = "#10bfcf"; BG = "#0f172a"; CARD = "#111827"; TEXT = "#e5e7eb"
st.set_page_config(page_title="Telemark · Snow Temps + Wax", page_icon="❄️", layout="wide")
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] > .main {{ background: linear-gradient(180deg, {BG} 0%, #111827 100%); }}
.block-container {{ padding-top: 0.8rem; }}
.hero h1 {{ margin:0;font-size:1.55rem;color:{TEXT};letter-spacing:.2px; }}
.badge {{ border:1px solid rgba(255,255,255,.15); color:{TEXT};opacity:.9; padding:4px 10px;border-radius:999px;font-size:.78rem; background: rgba(255,255,255,.03); }}
.card {{ background:{CARD}; border:1px solid rgba(255,255,255,.08); border-radius:16px; padding:14px; box-shadow: 0 8px 20px rgba(0,0,0,.25); }}
.kpi {{ display:flex; flex-direction:column; gap:2px; background:rgba(16,191,207,.06); border:1px dashed rgba(16,191,207,.45); padding:10px 12px; border-radius:12px; }}
.kpi .label {{ font-size:.78rem; color:#93c5fd; }}
.kpi .value {{ font-size:1rem; font-weight:700; color:{TEXT}; }}
.tag {{ display:inline-block; margin-right:6px; margin-top:6px; padding:3px 8px; border-radius:999px; font-size:.72rem; border:1px solid rgba(255,255,255,.12); color:{TEXT}; opacity:.9; }}
.btn-primary button {{ width:100%; background:{PRIMARY} !important; color:#002b30 !important; border:none; font-weight:700; border-radius:12px; }}
.section-title {{ color:{TEXT}; opacity:.95; margin:8px 0 4px 2px; }}
hr {{ border-color: rgba(255,255,255,.1) }}
</style>
""", unsafe_allow_html=True)

# --------- HEADER ---------
cA, cB = st.columns([1,3], vertical_alignment="center")
with cA:
    st.markdown("<div class='hero'><h1>Telemark · Snow Temps + Wax</h1></div>", unsafe_allow_html=True)
with cB:
    st.markdown("<span class='badge'>Mobile-first · Open-Meteo · Consigli sciolina</span>", unsafe_allow_html=True)

# --------- AUTOCOMPLETE LOCALITÀ ---------
@st.cache_data(show_spinner=False)
def geocode_search(q: str, count: int = 8):
    if not q or len(q) < 2:
        return []
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": q, "count": count, "language": "it", "format": "json"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    js = r.json()
    results = []
    for it in js.get("results", [])[:count]:
        label = f"{it.get('name','')} ({it.get('country_code','')}) – {it.get('admin1','') or ''}".strip()
        results.append({
            "label": label,
            "name": it.get("name"),
            "lat": it.get("latitude"),
            "lon": it.get("longitude"),
            "cc": it.get("country_code"),
            "admin1": it.get("admin1")
        })
    return results

st.markdown("<p class='section-title'>1) Cerca località</p>", unsafe_allow_html=True)
col_search, col_tz, col_hours = st.columns([3,1,2])
with col_search:
    q = st.text_input("Digita la località (autocompletamento)", placeholder="es. Champoluc, Gressoney, Alagna, Aosta…")
    suggestions = geocode_search(q) if q and len(q) >= 2 else []
    labels = [x["label"] for x in suggestions] or []
    sel = st.selectbox("Suggerimenti", options=["— nessun risultato —"]+labels, index=0 if not labels else 1)
    if labels and sel in labels:
        chosen = suggestions[labels.index(sel)]
        lat, lon = float(chosen["lat"]), float(chosen["lon"])
        spot_name = chosen["label"]
    else:
        # preset fallback
        spot_name = "Champoluc (Ramey)"
        lat, lon = 45.831, 7.730
with col_tz:
    tzname = st.selectbox("Timezone", ["Europe/Rome","UTC"], index=0)
with col_hours:
    hours = st.slider("Ore previsione", 12, 168, 72, 12)

# --------- FINESTRE A/B/C ---------
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
go = st.button("Scarica previsioni & Calcola", type="primary")

# --------- HELPERS ---------
def fetch_open_meteo(lat, lon, timezone_str):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat,"longitude": lon,"timezone": timezone_str,
              "hourly": "temperature_2m,dew_point_2m,precipitation,rain,snowfall,cloudcover,windspeed_10m,is_day,weathercode",
              "forecast_days": 7}
    r = requests.get(url, params=params, timeout=30); r.raise_for_status()
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

def build_df(js, hours):
    # Tutto in NAIVE per evitare errori tz-aware vs naive
    h = js["hourly"]; df = pd.DataFrame(h); df["time"] = pd.to_datetime(df["time"])
    now_naive = pd.Timestamp.now().floor("H")
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

def window_slice(res, tzname, s, e):
    t = pd.to_datetime(res["time"]).dt.tz_localize(tz.gettz(tzname), nonexistent='shift_forward', ambiguous='NaT')
    D = res.copy(); D["dt"] = t
    today = pd.Timestamp.now(tz=tz.gettz(tzname)).date()
    win = D[(D["dt"].dt.date==today) & (D["dt"].dt.time>=s) & (D["dt"].dt.time<=e)]
    return win if not win.empty else D.head(7)

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

def wax_cards(t_med, wet):
    SWIX = [("PS5 Turquoise", -18,-10), ("PS6 Blue",-12,-6), ("PS7 Violet",-8,-2), ("PS8 Red",-4,4), ("PS10 Yellow",0,10)]
    TOKO = [("Blue",-30,-9), ("Red",-12,-4), ("Yellow",-6,0)]
    VOLA = [("MX-E Violet/Blue",-12,-4), ("MX-E Red",-5,0), ("MX-E Warm",-2,10)]
    RODE = [("R20 Blue",-18,-8), ("R30 Violet",-10,-3), ("R40 Red",-5,0), ("R50 Yellow",-1,10)]
    def pick(bands,t):
        for n,tmin,tmax in bands:
            if t>=tmin and t<=tmax: return n
        return bands[-1][0] if t>bands[-1][2] else bands[0][0]
    rows = st.columns(4)
    rows[0].markdown(f"<div class='kpi'><div class='label'>Swix</div><div class='value'>{pick(SWIX,t_med)}</div></div>", unsafe_allow_html=True)
    rows[1].markdown(f"<div class='kpi'><div class='label'>Toko</div><div class='value'>{pick(TOKO,t_med)}</div></div>", unsafe_allow_html=True)
    rows[2].markdown(f"<div class='kpi'><div class='label'>Vola</div><div class='value'>{pick(VOLA,t_med)}</div></div>", unsafe_allow_html=True)
    rows[3].markdown(f"<div class='kpi'><div class='label'>Rode</div><div class='value'>{pick(RODE,t_med)}</div></div>", unsafe_allow_html=True)
    if t_med <= -10:
        st.markdown("<span class='tag'>Neve fredda · struttura fine</span>", unsafe_allow_html=True)
    if -3 <= t_med <= 1 and wet:
        st.markdown("<span class='tag'>Neve umida · top-coat liquido</span>", unsafe_allow_html=True)

def badge(win):
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
      <div style='font-size:12px;opacity:.7;margin-top:6px'>
        T_surf min {tmin:.1f}°C / max {tmax:.1f}°C
      </div>
    </div>""", unsafe_allow_html=True)
    return wet

# --------- RUN ---------
def run(res):
    st.markdown("### Risultati")
    st.dataframe(res, use_container_width=True)
    fig1, fig2 = make_plots(res); st.pyplot(fig1); st.pyplot(fig2)
    st.download_button("Scarica CSV", data=res.to_csv(index=False), file_name="forecast_with_snowT.csv", mime="text/csv")
    st.markdown("### Consigli per blocchi A · B · C")
    def slice_(s,e): 
        t = pd.to_datetime(res["time"]).dt.tz_localize(tz.gettz(tzname), nonexistent='shift_forward', ambiguous='NaT')
        D = res.copy(); D["dt"] = t
        today = pd.Timestamp.now(tz=tz.gettz(tzname)).date()
        W = D[(D["dt"].dt.date==today) & (D["dt"].dt.time>=s) & (D["dt"].dt.time<=e)]
        return W if not W.empty else D.head(7)
    for label,(s,e) in {"A":(A_start,A_end),"B":(B_start,B_end),"C":(C_start,C_end)}.items():
        st.markdown(f"#### Blocco {label}")
        W = slice_(s,e)
        wet = badge(W)
        t_med = float(W["T_surf"].mean())
        wax_cards(t_med, wet)

if go:
    try:
        js = fetch_open_meteo(lat, lon, tzname)
        src = build_df(js, hours)
        res = compute_snow_temperature(src, dt_hours=1.0)
        st.success(f"Previsioni scaricate per {spot_name}.")
        run(res)
    except Exception as e:
        st.error(f"Errore: {e}")
else:
    st.info("Digita una località, scegli A/B/C e premi **Scarica previsioni & Calcola**.")

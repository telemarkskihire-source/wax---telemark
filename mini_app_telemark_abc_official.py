# Telemark · Snow Temps + Wax — mobile-first, autocomplete + geolocalizzazione
# Requisiti: streamlit, pandas, requests, python-dateutil, matplotlib, reportlab
# File locale necessario: core_snowtemp.py (già nel repo)

import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import time
from dateutil import tz
from streamlit.components.v1 import html

from core_snowtemp import compute_snow_temperature

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="Telemark · Snow Temps + Wax", page_icon="❄️", layout="wide")

PRIMARY = "#10bfcf"
BG = "#0f172a"
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
.badge {{ border:1px solid rgba(255,255,255,.15); color:{TEXT};opacity:0.9;
         padding:4px 10px;border-radius:999px;font-size:.78rem;
         background: rgba(255,255,255,.03); }}
.card {{ background:{CARD}; border:1px solid rgba(255,255,255,.08);
         border-radius:16px; padding:14px; box-shadow: 0 8px 20px rgba(0,0,0,.25); }}
.kpi {{ display:flex; flex-direction:column; gap:2px;
       background:rgba(16,191,207,.06); border:1px dashed rgba(16,191,207,.45);
       padding:10px 12px; border-radius:12px; }}
.kpi .label {{ font-size:.78rem; color:#93c5fd; }}
.kpi .value {{ font-size:1rem; font-weight:700; color:{TEXT}; }}
.tag {{ display:inline-block; margin-right:6px; margin-top:6px;
       padding:3px 8px; border-radius:999px; font-size:.72rem;
       border:1px solid rgba(255,255,255,.12); color:{TEXT}; opacity:.9; }}
.btn-primary button {{ width:100%; background:{PRIMARY} !important; color:#002b30 !important;
                      border:none; font-weight:700; border-radius:12px; }}
.section-title {{ color:{TEXT}; opacity:.95; margin:8px 0 4px 2px; }}
hr {{ border-color: rgba(255,255,255,.1) }}
.suggest-item {{ padding:8px 10px; border-bottom:1px solid rgba(255,255,255,.06);
                 cursor:pointer; }}
.suggest-item:hover {{ background: rgba(255,255,255,.05); }}
.suggest-wrap {{ max-height:280px; overflow:auto; border:1px solid rgba(255,255,255,.1);
                 border-radius:12px; background:{CARD}; }}
.suggest-title {{ color:{TEXT}; font-weight:600; }}
.suggest-sub {{ color:#9ca3af; font-size:.78rem; }}
</style>
""", unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
colA, colB = st.columns([1,3], vertical_alignment="center")
with colA:
    st.markdown(f"<div class='hero'><h1>Telemark · Snow Temps + Wax</h1></div>", unsafe_allow_html=True)
with colB:
    st.markdown(f"<span class='badge'>Autocomplete tipo Meteoblue · Geolocalizzazione · A/B/C</span>", unsafe_allow_html=True)

# ---------------------- AUTOCOMPLETE + GEO ----------------------
st.markdown("### 1) Cerca la località")

def geocode(query: str, count: int = 10, language: str = "it"):
    """Open-Meteo Geocoding API (https://geocoding-api.open-meteo.com)."""
    if not query or len(query.strip()) < 2:
        return []
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": query.strip(), "count": count, "language": language, "format": "json"}
    try:
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        js = r.json()
        return js.get("results", []) or []
    except Exception:
        return []

def suggest_ui(query_key="q_city"):
    q = st.text_input("Digita il nome (es. Champoluc, Zermatt…)", key=query_key, placeholder="Cerca città, località sci, valle…")
    picked = None
    lat = lon = None

    # Geolocalizzazione via browser
    colg1, colg2 = st.columns([1,3])
    with colg1:
        if st.button("📍 Usa la mia posizione", use_container_width=True):
            res = html("""
                <script>
                const send = (data) => {{
                  const msg = {{ isStreamlitMessage: true,
                                 type: "streamlit:setComponentValue",
                                 value: data }};
                  window.parent.postMessage(msg, "*");
                }};
                navigator.geolocation.getCurrentPosition(
                  (pos)=>{{ send({lat: pos.coords.latitude, lon: pos.coords.longitude}); }},
                  (err)=>{{ send({error: err.message}); }},
                  {{ enableHighAccuracy: true, timeout: 8000, maximumAge: 0 }}
                );
                </script>
                """, height=0)
            if isinstance(res, dict) and "lat" in res and "lon" in res:
                st.session_state["_sel_name"] = "Posizione attuale"
                st.session_state["_sel_lat"] = float(res["lat"])
                st.session_state["_sel_lon"] = float(res["lon"])
            elif isinstance(res, dict) and "error" in res:
                st.error(f"Geolocalizzazione non concessa: {res['error']}")

    # Suggerimenti dinamici: aggiorno mentre digiti
    results = geocode(q, count=12, language="it") if len(q) >= 2 else []

    # Mostra lista stile Meteoblue
    cont = st.container()
    with cont:
        if len(q) < 2:
            st.caption("Digita almeno **2** caratteri per vedere i suggerimenti.")
        elif not results:
            st.caption("Nessun risultato. Prova con il *comune* o la *valle* (es. “Ayas”, “Gressoney”).")
        else:
            st.markdown("<div class='suggest-wrap'>", unsafe_allow_html=True)
            for i, r in enumerate(results):
                name = r.get("name", "")
                cc = r.get("country_code", "")
                adm1 = r.get("admin1", "") or r.get("admin2", "") or ""
                lat_i = r.get("latitude", None)
                lon_i = r.get("longitude", None)

                # Pulsante “trasparente” per scegliere
                col1, col2 = st.columns([5,1])
                with col1:
                    if st.button(f"{name}", key=f"sug_{i}", help=f"{adm1} · {cc}", use_container_width=True):
                        st.session_state["_sel_name"] = name
                        st.session_state["_sel_lat"] = float(lat_i)
                        st.session_state["_sel_lon"] = float(lon_i)
                with col2:
                    st.caption(f"{adm1} · {cc}")

            st.markdown("</div>", unsafe_allow_html=True)

    # Ritorna selezione corrente se presente
    if all(k in st.session_state for k in ["_sel_name", "_sel_lat", "_sel_lon"]):
        picked = st.session_state["_sel_name"]
        lat = st.session_state["_sel_lat"]
        lon = st.session_state["_sel_lon"]

    return picked, lat, lon

picked_name, lat, lon = suggest_ui()

# ---------------------- PARAMETRI A/B/C ----------------------
st.markdown("### 2) Finestre orarie di oggi · A · B · C")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Blocco A**")
    A_start = st.time_input("Inizio A", value=time(9,0), key="A_s")
    A_end   = st.time_input("Fine A",   value=time(11,0), key="A_e")
with c2:
    st.markdown("**Blocco B**")
    B_start = st.time_input("Inizio B", value=time(11,0), key="B_s")
    B_end   = st.time_input("Fine B",   value=time(13,0), key="B_e")
with c3:
    st.markdown("**Blocco C**")
    C_start = st.time_input("Inizio C", value=time(13,0), key="C_s")
    C_end   = st.time_input("Fine C",   value=time(16,0), key="C_e")

hours = st.slider("Ore previsione", 12, 168, 72, 12)
st.divider()

# ---------------------- METEO + MODELLO ----------------------
def fetch_open_meteo(lat: float, lon: float, timezone_str: str):
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
    # Mantengo i tempi NAIVE per evitare problemi tz-aware
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
    W = D[(D["dt"].dt.date==today) & (D["dt"].dt.time>=s) & (D["dt"].dt.time<=e)]
    return W if not W.empty else D.head(7)

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
    wet = bool(((win["prp_type"].isin(["rain","mixed"])) | (win["prp_mmph"]>0.5)).any()))
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

def run_all(res, tzname):
    st.markdown("### 3) Risultati")
    st.dataframe(res, use_container_width=True)
    fig1, fig2 = make_plots(res); st.pyplot(fig1); st.pyplot(fig2)
    st.download_button("Scarica CSV", data=res.to_csv(index=False), file_name="forecast_with_snowT.csv", mime="text/csv")

    st.markdown("### 4) Consigli per A · B · C")
    def slice_(s,e):
        return window_slice(res, tzname, s, e)
    for label,(s,e) in {"A":(A_start,A_end),"B":(B_start,B_end),"C":(C_start,C_end)}.items():
        st.markdown(f"#### Blocco {label}")
        W = slice_(s,e)
        wet = badge(W)
        t_med = float(W["T_surf"].mean())
        wax_cards(t_med, wet)

# ---------------------- AZIONE ----------------------
col_go1, col_go2 = st.columns([2,3])
with col_go1:
    go = st.button("Scarica previsioni e calcola", type="primary", use_container_width=True)

if go:
    if lat is None or lon is None:
        st.warning("Seleziona una località (o usa la tua posizione) prima di procedere.")
    else:
        st.success(f"Località: **{picked_name}** — lat {lat:.4f}, lon {lon:.4f}")
        js = fetch_open_meteo(lat, lon, "Europe/Rome")
        src = build_df(js, hours, "Europe/Rome")
        res = compute_snow_temperature(src, dt_hours=1.0)
        run_all(res, "Europe/Rome")
else:
    st.info("Cerca e seleziona la località (oppure **📍 Usa la mia posizione**) e poi premi **Scarica previsioni e calcola**.")

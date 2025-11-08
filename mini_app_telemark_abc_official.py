# mini_app_telemark_pretty_geo.py
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import time, datetime
from dateutil import tz

# --- modello: deve esserci core_snowtemp.py nella stessa cartella ---
from core_snowtemp import compute_snow_temperature

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
  border-radius:16px; padding:14px;
  box-shadow: 0 8px 20px rgba(0,0,0,.25);
}}
.kpi {{
  display:flex; flex-direction:column; gap:2px;
  background:rgba(16,191,207,.06); border:1px dashed rgba(16,191,207,.45);
  padding:10px 12px; border-radius:12px;
}}
.kpi .label {{ font-size:.78rem; color:#93c5fd; }}
.kpi .value {{ font-size:1rem; font-weight:700; color:{TEXT}; }}
.tag {{
  display:inline-block; margin-right:6px; margin-top:6px;
  padding:3px 8px; border-radius:999px; font-size:.72rem;
  border:1px solid rgba(255,255,255,.12); color:{TEXT}; opacity:.9;
}}
.btn-primary button {{
  width:100%; background:{PRIMARY} !important; color:#002b30 !important; 
  border:none; font-weight:700; border-radius:12px;
}}
.section-title {{ color:{TEXT}; opacity:.95; margin:8px 0 4px 2px; }}
hr {{ border-color: rgba(255,255,255,.1) }}
</style>
""", unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
colA, colB = st.columns([1,3], vertical_alignment="center")
with colA:
    st.markdown(f"<div class='hero'><h1>Telemark · Snow Temps + Wax</h1></div>", unsafe_allow_html=True)
with colB:
    st.markdown(f"<span class='badge'>Mobile-first · Previsioni Open-Meteo · Consigli sciolina</span>", unsafe_allow_html=True)

# ---------------------- GEO SEARCH ----------------------
st.markdown("<p class='section-title'>1) Cerca una località</p>", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def geocode(query: str, count: int = 10, lang: str = "it"):
    if not query.strip():
        return pd.DataFrame()
    url = "https://geocoding-api.open-meteo.com/v1/search"
    r = requests.get(url, params={"name": query, "count": count, "language": lang, "format": "json"}, timeout=20)
    r.raise_for_status()
    js = r.json()
    rows = js.get("results", []) or []
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # compose a nice label
    def label(row):
        parts = [row.get("name","")]
        if row.get("admin1"): parts.append(row["admin1"])
        if row.get("country_code"): parts.append(row["country_code"])
        return " · ".join([p for p in parts if p])
    df["label"] = df.apply(label, axis=1)
    return df

c_geo = st.columns([3,2,2,2])
with c_geo[0]:
    q = st.text_input("Cerca località (es. Champoluc, Zermatt, Åre)", value="Champoluc")
with c_geo[1]:
    if st.button("Cerca", type="primary"):
        st.session_state["_geo_df"] = geocode(q)
with c_geo[2]:
    preset = st.selectbox("Preset rapidi", ["—", "Champoluc (Ramey)", "Gressoney (Stafal)", "Alagna (Pianalunga)"])
with c_geo[3]:
    hours = st.slider("Ore previsione", 12, 168, 72, 12)

geo_df = st.session_state.get("_geo_df", geocode(q))
if preset != "—":
    if preset == "Champoluc (Ramey)":
        sel = {"name":"Champoluc","latitude":45.831,"longitude":7.730,"timezone":"Europe/Rome","label":"Champoluc · AO · IT"}
    elif preset == "Gressoney (Stafal)":
        sel = {"name":"Stafal","latitude":45.824,"longitude":7.827,"timezone":"Europe/Rome","label":"Gressoney-La-Trinité (Stafal) · AO · IT"}
    else:
        sel = {"name":"Alagna Valsesia","latitude":45.855,"longitude":7.941,"timezone":"Europe/Rome","label":"Alagna (Pianalunga) · VC · IT"}
else:
    if geo_df.empty:
        st.warning("Nessun risultato. Prova con un altro nome (anche in inglese).")
        st.stop()
    labels = geo_df["label"].tolist()
    idx = st.selectbox("Scegli tra i risultati", list(range(len(labels))), format_func=lambda i: labels[i])
    row = geo_df.iloc[int(idx)]
    sel = {
        "name": row.get("name",""),
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "timezone": row.get("timezone","auto"),
        "label": row["label"],
    }

lat = float(sel["latitude"]); lon = float(sel["longitude"])
tzname_api = sel["timezone"] or "auto"   # user-friendly: lasciamo auto

# ---------------------- BLOCCHI A/B/C ----------------------
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
fetch_col = st.columns([2,3,3])
with fetch_col[0]:
    go = st.button("Scarica previsioni", type="primary")
with fetch_col[1]:
    upl = st.file_uploader("…oppure carica CSV", type=["csv"])
with fetch_col[2]:
    st.caption("CSV: time, T2m, cloud(0..1), wind, sunup(0/1), prp_mmph, prp_type, td(opz.)")

# ---------------------- HELPERS ----------------------
def fetch_open_meteo(lat, lon, timezone="auto"):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,dew_point_2m,precipitation,rain,snowfall,cloudcover,windspeed_10m,is_day,weathercode",
        "forecast_days": 7,
        "timezone": timezone,   # usare "auto" => orari già locali alla località
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

def build_df(js, hours):
    # Manteniamo tempi NAIVE; Open-Meteo con timezone=auto già restituisce orari locali
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

def window_slice(res: pd.DataFrame, s: time, e: time):
    # Slicing sul "giorno di oggi" basandoci sulle date del dataset (locale già applicato a stringa)
    D = res.copy()
    D["t"] = pd.to_datetime(D["time"])
    if D.empty:
        return D
    today_date = pd.Timestamp.now().date()
    day_rows = D[D["t"].dt.date == today_date]
    if day_rows.empty:
        # fallback: usa le prime ore disponibili
        day_rows = D.head(8)
    W = day_rows[(day_rows["t"].dt.time >= s) & (day_rows["t"].dt.time <= e)]
    return W if not W.empty else day_rows

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

def wax_cards(t_med: float, wet: bool):
    # fasce essenziali (non-fluoro) — compatte
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
    tips = []
    if t_med <= -10: tips.append("Neve fredda · struttura fine")
    if -3 <= t_med <= 1 and wet: tips.append("Neve umida · top-coat liquido")
    if tips:
        st.markdown("".join([f"<span class='tag'>{t}</span>" for t in tips]), unsafe_allow_html=True)

def badge(win: pd.DataFrame):
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

# ---------------------- MAIN FLOW ----------------------
def run_full(res: pd.DataFrame):
    st.markdown("### Risultati")
    st.dataframe(res, use_container_width=True)
    fig1, fig2 = make_plots(res); st.pyplot(fig1); st.pyplot(fig2)
    st.download_button("Scarica CSV", data=res.to_csv(index=False), file_name="forecast_with_snowT.csv", mime="text/csv")

    st.markdown("### Consigli per blocchi A · B · C")
    for label,(s,e) in {"A":(A_start,A_end),"B":(B_start,B_end),"C":(C_start,C_end)}.items():
        st.markdown(f"#### Blocco {label}")
        W = window_slice(res, s, e)
        wet = badge(W)
        t_med = float(W["T_surf"].mean())
        wax_cards(t_med, wet)

try:
    if upl is not None:
        df_in = pd.read_csv(upl)
        res = compute_snow_temperature(df_in, dt_hours=1.0)
        st.success(f"CSV caricato · {sel['label']}")
        run_full(res)
    elif go:
        js = fetch_open_meteo(lat, lon, tzname_api)  # tz auto -> orari locali
        src = build_df(js, hours)
        res = compute_snow_temperature(src, dt_hours=1.0)
        st.success(f"Previsioni scaricate · {sel['label']}")
        run_full(res)
    else:
        st.info("Cerca la località, definisci A/B/C e premi **Scarica previsioni** (oppure carica un CSV).")
except Exception as e:
    st.error(f"Errore: {e}")

import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import time
from dateutil import tz

# se hai già il file a parte, lasciamo l'import
from core_snowtemp import compute_snow_temperature

# opzionale: per geolocalizzazione browser
try:
    from streamlit_js_eval import get_geolocation
    GEO_OK = True
except Exception:
    GEO_OK = False

st.set_page_config(page_title="Telemark · Snow Temps + Wax", page_icon="❄️", layout="wide")

# ---------------------- THEME & CSS ----------------------
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
.badge {{
  border:1px solid rgba(255,255,255,.15); color:{TEXT};opacity:0.9;
  padding:4px 10px;border-radius:999px;font-size:.78rem;background: rgba(255,255,255,.03);
}}
.card {{
  background:{CARD}; border:1px solid rgba(255,255,255,.08);
  border-radius:16px; padding:14px; box-shadow: 0 8px 20px rgba(0,0,0,.25);
}}
.kpi {{ display:flex; flex-direction:column; gap:2px;
  background:rgba(16,191,207,.06); border:1px dashed rgba(16,191,207,.45);
  padding:10px 12px; border-radius:12px; }}
.kpi .label {{ font-size:.78rem; color:#93c5fd; }}
.kpi .value {{ font-size:1rem; font-weight:700; color:{TEXT}; }}
.tag {{
  display:inline-block; margin-right:6px; margin-top:6px;
  padding:3px 8px; border-radius:999px; font-size:.72rem;
  border:1px solid rgba(255,255,255,.12); color:{TEXT}; opacity:.9;
}}
hr {{ border-color: rgba(255,255,255,.1) }}
</style>
""", unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
colA, colB = st.columns([1,3], vertical_alignment="center")
with colA:
    st.markdown(f"<div class='hero'><h1>Telemark · Snow Temps + Wax</h1></div>", unsafe_allow_html=True)
with colB:
    st.markdown(f"<span class='badge'>Mobile-first · Open-Meteo · Consigli sciolina</span>", unsafe_allow_html=True)

# ---------------------- SEARCH (autocomplete + geoloc) ----------------------
st.markdown("#### 1) Trova la località (digita e scegli)")

q = st.text_input("Cerca località (es. Champoluc, Zermatt, Courmayeur…)", value="", placeholder="Scrivi almeno 2 caratteri…")

def geocode(query: str, count: int = 12, lang: str = "it"):
    """Autocomplete stile Meteoblue con Open-Meteo Geocoding API (parziale mentre digiti)."""
    if len(query.strip()) < 2:
        return []
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": query.strip(), "count": count, "language": lang}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    js = r.json()
    results = js.get("results") or []
    out = []
    for it in results:
        name = it.get("name","")
        cc = it.get("country_code","")
        admin = it.get("admin1","") or it.get("country","")
        lat = it.get("latitude"); lon = it.get("longitude")
        label = f"{name} · {admin} · {cc}  ({lat:.3f},{lon:.3f})"
        out.append({"label":label,"lat":lat,"lon":lon,"tz":it.get("timezone","UTC")})
    return out

# suggerimenti live
suggestions = geocode(q) if q else []
labels = [s["label"] for s in suggestions] or ["— nessun risultato —"]
sel = st.selectbox("Suggerimenti", labels, index=0)

if suggestions and sel in labels:
    pick = suggestions[labels.index(sel)]
    lat, lon = float(pick["lat"]), float(pick["lon"])
    tzname = pick["tz"] or "Europe/Rome"
else:
    # default fallback a Champoluc
    lat, lon, tzname = 45.831, 7.730, "Europe/Rome"

col_geo_a, col_geo_b = st.columns([1,2])
with col_geo_a:
    if GEO_OK and st.button("📍 Usa posizione attuale"):
        g = get_geolocation()
        if isinstance(g, dict) and ("coords" in g or ("lat" in g and "lon" in g)):
            # due formati possibili a seconda della versione del package
            if "coords" in g:
                lat = float(g["coords"]["latitude"])
                lon = float(g["coords"]["longitude"])
            else:
                lat = float(g["lat"]); lon = float(g["lon"])
            st.success(f"Posizione rilevata: {lat:.4f}, {lon:.4f}")
        else:
            st.warning("Geolocalizzazione non concessa dal browser.")
with col_geo_b:
    hours = st.slider("Ore previsione", 12, 168, 72, 12)

# ---------------------- BLOCCHI A / B / C ----------------------
st.markdown("#### 2) Finestre orarie (oggi) — A · B · C")
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

# ---------------------- HELPERS METEO ----------------------
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
    snow_codes = {71,73,75,77,85,86}
    rain_codes = {51,53,55,61,63,65,80,81,82}
    def f(row):
        prp = row.get("precipitation", 0.0)
        rain = row.get("rain", 0.0)
        snow = row.get("snowfall", 0.0)
        if prp <= 0 or pd.isna(prp): return "none"
        if (rain > 0) and (snow > 0): return "mixed"
        if (snow > 0) and (rain == 0): return "snow"
        if (rain > 0) and (snow == 0): return "rain"
        code = int(row.get("weathercode", 0)) if pd.notna(row.get("weathercode", None)) else 0
        if code in snow_codes: return "snow"
        if code in rain_codes: return "rain"
        return "mixed"
    return df.apply(f, axis=1)

def build_df(js, hours, tzname):
    # Manteniamo timestamp NAIVE per evitare errori tz-aware/naive
    h = js["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])  # naive
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
    # localizziamo solo per il filtro del giorno/ora
    tloc = pd.to_datetime(res["time"]).dt.tz_localize(tz.gettz(tzname), nonexistent='shift_forward', ambiguous='NaT')
    D = res.copy()
    D["dt"] = tloc
    today = pd.Timestamp.now(tz=tz.gettz(tzname)).date()
    W = D[(D["dt"].dt.date == today) & (D["dt"].dt.time >= s) & (D["dt"].dt.time <= e)]
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

# Wax bands (Swix, Toko, Vola, Rode)
SWIX = [("PS5 Turquoise", -18,-10), ("PS6 Blue",-12,-6), ("PS7 Violet",-8,-2), ("PS8 Red",-4,4), ("PS10 Yellow",0,10)]
TOKO = [("Blue",-30,-9), ("Red",-12,-4), ("Yellow",-6,0)]
VOLA = [("MX-E Violet/Blue",-12,-4), ("MX-E Red",-5,0), ("MX-E Warm",-2,10)]
RODE = [("R20 Blue",-18,-8), ("R30 Violet",-10,-3), ("R40 Red",-5,0), ("R50 Yellow",-1,10)]
def pick(bands,t):
    for n,tmin,tmax in bands:
        if t>=tmin and t<=tmax: return n
    return bands[-1][0] if t>bands[-1][2] else bands[0][0]

def badge(win):
    tmin = float(win["T_surf"].min()); tmax = float(win["T_surf"].max())
    wet = bool(((win["prp_type"].isin(["rain","mixed"])) | (win["prp_mmph"] > 0.5)).any())
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

def wax_cards(t_med, wet):
    cols = st.columns(4)
    cols[0].markdown(f"<div class='kpi'><div class='label'>Swix</div><div class='value'>{pick(SWIX,t_med)}</div></div>", unsafe_allow_html=True)
    cols[1].markdown(f"<div class='kpi'><div class='label'>Toko</div><div class='value'>{pick(TOKO,t_med)}</div></div>", unsafe_allow_html=True)
    cols[2].markdown(f"<div class='kpi'><div class='label'>Vola</div><div class='value'>{pick(VOLA,t_med)}</div></div>", unsafe_allow_html=True)
    cols[3].markdown(f"<div class='kpi'><div class='label'>Rode</div><div class='value'>{pick(RODE,t_med)}</div></div>", unsafe_allow_html=True)
    if t_med <= -10:
        st.markdown("<span class='tag'>Neve fredda · struttura fine</span>", unsafe_allow_html=True)
    if (-3 <= t_med <= 1) and wet:
        st.markdown("<span class='tag'>Neve umida · top-coat liquido</span>", unsafe_allow_html=True)

# ---------------------- RUN ----------------------
st.markdown("#### 3) Scarica e vedi risultati")
go = st.button("Scarica previsioni Open-Meteo")

try:
    if go:
        js = fetch_open_meteo(lat, lon, tzname)
        src = build_df(js, hours, tzname)
        res = compute_snow_temperature(src, dt_hours=1.0)

        st.success(f"OK · {sel if suggestions else 'Località selezionata'} — tz: {tzname}")
        st.dataframe(res, use_container_width=True)
        f1, f2 = make_plots(res); st.pyplot(f1); st.pyplot(f2)
        st.download_button("Scarica CSV", data=res.to_csv(index=False), file_name="forecast_with_snowT.csv", mime="text/csv")

        st.markdown("### Consigli per blocchi A · B · C")
        for label,(s,e) in {"A":(A_start,A_end),"B":(B_start,B_end),"C":(C_start,C_end)}.items():
            st.markdown(f"#### Blocco {label}")
            W = window_slice(res, tzname, s, e)
            wet = badge(W)
            t_med = float(W["T_surf"].mean())
            wax_cards(t_med, wet)
    else:
        st.info("Digita la località (autocomplete), opzionalmente usa la geolocalizzazione, definisci A/B/C e premi **Scarica previsioni**.")
except Exception as e:
    st.error(f"Errore: {e}")

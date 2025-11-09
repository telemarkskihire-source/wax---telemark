# mini_app_telemark.py
import streamlit as st
import pandas as pd
import requests
import base64
import io
import math
from datetime import time
from dateutil import tz
import matplotlib.pyplot as plt

# ---------------------- PAGE & THEME ----------------------
PRIMARY = "#10bfcf"; BG = "#0f172a"; CARD = "#111827"; TEXT = "#e5e7eb"
st.set_page_config(page_title="Telemark · Pro Wax & Tune", page_icon="❄️", layout="wide")
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
  background: linear-gradient(180deg, {BG} 0%, #111827 100%);
}}
.block-container {{ padding-top: 0.8rem; }}
h1,h2,h3,h4,h5,p,span,div {{ color:{TEXT}; }}
.badge {{ border:1px solid rgba(255,255,255,.15); padding:6px 10px; border-radius:999px; font-size:.78rem; opacity:.85; }}
.card {{ background:{CARD}; border:1px solid rgba(255,255,255,.08); border-radius:16px; padding:14px; box-shadow:0 8px 20px rgba(0,0,0,.25); }}
.brand {{ display:flex; align-items:center; gap:8px; padding:8px 10px; border-radius:12px; background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08); }}
.brand img {{ height:22px; }}
.kpi {{ display:flex; gap:8px; align-items:center; background:rgba(16,191,207,.06); border:1px dashed rgba(16,191,207,.45); padding:10px 12px; border-radius:12px; }}
.kpi .lab {{ font-size:.78rem; color:#93c5fd; }}
.kpi .val {{ font-size:1rem; font-weight:800; color:{TEXT}; }}
.suggbtn > div > button {{
  width:100%; text-align:left; border-radius:10px; border:1px solid rgba(255,255,255,.12);
  background:rgba(255,255,255,.03); padding:8px 10px;
}}
.btn-primary button {{ width:100%; background:{PRIMARY} !important; color:#002b30 !important; border:none; font-weight:700; border-radius:12px; }}
hr {{ border-color: rgba(255,255,255,.1) }}
</style>
""", unsafe_allow_html=True)

st.markdown("### Telemark · Pro Wax & Tune")
st.markdown("<span class='badge'>Ricerca stile Meteoblue · Geolocalizzazione · Blocchi A/B/C · Sciolina + Struttura + Angoli</span>", unsafe_allow_html=True)

# ---------------------- WEATHER HELPERS ----------------------
def fetch_open_meteo(lat, lon, timezone_str):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon, "timezone": timezone_str,
        "hourly": "temperature_2m,dew_point_2m,precipitation,rain,snowfall,cloudcover,windspeed_10m,is_day,weathercode",
        "forecast_days": 7
    }
    r = requests.get(url, params=params, timeout=30); r.raise_for_status()
    return r.json()

def _prp_type(df):
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
    # Tenere tutto NAIVE per evitare errori tz-aware/naive
    h = js["hourly"]; df = pd.DataFrame(h); df["time"] = pd.to_datetime(df["time"])
    now_naive = pd.Timestamp.now().floor("H")
    df = df[df["time"] >= now_naive].head(hours).reset_index(drop=True)
    out = pd.DataFrame()
    out["time"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    out["T2m"] = df["temperature_2m"].astype(float)
    out["cloud"] = (df["cloudcover"].astype(float)/100).clip(0,1)
    out["wind"] = (df["windspeed_10m"].astype(float)/3.6).round(3)  # m/s
    out["sunup"] = df["is_day"].astype(int)
    out["prp_mmph"] = df["precipitation"].astype(float)
    extra = df[["precipitation","rain","snowfall","weathercode"]].copy()
    out["prp_type"] = _prp_type(extra)
    out["td"] = df["dew_point_2m"].astype(float)
    return out

def compute_snow_temperature(df, dt_hours=1.0):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    need = ["T2m","cloud","wind","sunup","prp_mmph","prp_type"]
    for c in need:
        if c not in df.columns: raise ValueError(f"Manca colonna: {c}")
    if "td" not in df.columns: df["td"] = float("nan")
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
    alpha = 1.0 - (math.e ** (-dt_hours / tau))
    if len(df) > 0:
        T_top5.iloc[0] = min(df["T2m"].iloc[0], 0.0)
        for i in range(1, len(df)):
            T_top5.iloc[i] = T_top5.iloc[i-1] + alpha.iloc[i]*(T_surf.iloc[i] - T_top5.iloc[i-1])

    df["T_surf"] = T_surf; df["T_top5"] = T_top5
    return df

def window_today_slice(res, tzname, s, e):
    t = pd.to_datetime(res["time"]).dt.tz_localize(tz.gettz(tzname), nonexistent='shift_forward', ambiguous='NaT')
    D = res.copy(); D["dt"] = t
    today = pd.Timestamp.now(tz=tz.gettz(tzname)).date()
    win = D[(D["dt"].dt.date==today) & (D["dt"].dt.time>=s) & (D["dt"].dt.time<=e)]
    return win if not win.empty else D.head(7)

# ---------------------- PLACE SEARCH (Meteoblue-like) ----------------------
# NOTE: Meteoblue fa ricerca live con una tendina aggiornata a ogni carattere.
# Replichiamo: ogni keypress ricerchiamo Nominatim e mostriamo una lista cliccabile.
def nominatim_search(q, limit=8):
    if not q or len(q) < 2: return []
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": q, "format":"json", "limit": limit, "addressdetails": 1},
            headers={"User-Agent": "telemark-pro-wax/1.0"},
            timeout=8
        )
        r.raise_for_status()
        js = r.json()
        out = []
        for it in js:
            name = it.get("display_name","")
            lat = float(it.get("lat", 0)); lon = float(it.get("lon", 0))
            out.append({"label": name, "lat": lat, "lon": lon})
        return out
    except Exception:
        return []

def ip_geolocate():
    try:
        r = requests.get("https://ipapi.co/json", timeout=8)
        if r.ok:
            j = r.json()
            return float(j.get("latitude",0)), float(j.get("longitude",0)), j.get("city","")
    except Exception:
        pass
    return None, None, ""

col_s, col_geo = st.columns([3,1])
with col_s:
    query = st.text_input("Cerca località (digita → compaiono i suggerimenti)", placeholder="Es. Champoluc, Cervinia, Sestriere…", key="q")
    suggestions = nominatim_search(query, limit=10)

    # Se ci sono suggerimenti: selezioniamo automaticamente il primo (come autocomplete)
    if suggestions:
        top = suggestions[0]
        # Mostra la tendina in stile elenco Meteoblue
        st.markdown("**Suggerimenti**")
        for i, s in enumerate(suggestions[:6]):
            colbtn = st.container()
            with colbtn:
                if st.button(s["label"], key=f"s_{i}"):
                    st.session_state["lat"] = s["lat"]; st.session_state["lon"] = s["lon"]
                    st.session_state["place_label"] = s["label"]
                    st.session_state["q"] = s["label"]
        # Aggiorna lat/lon automaticamente al primo risultato per dare sensazione “live”
        if "lat" not in st.session_state or st.session_state.get("q","") != top["label"]:
            st.session_state["lat"] = top["lat"]; st.session_state["lon"] = top["lon"]
            st.session_state["place_label"] = top["label"]
with col_geo:
    if st.button("📍 Usa la mia posizione"):
        la, lo, city = ip_geolocate()
        if la is not None:
            st.session_state["lat"] = la; st.session_state["lon"] = lo
            st.session_state["place_label"] = city or "La tua posizione"
            st.success("Posizione impostata.")
        else:
            st.error("Geolocalizzazione non disponibile.")

lat = st.session_state.get("lat", 45.831)
lon = st.session_state.get("lon", 7.730)
place_label = st.session_state.get("place_label", "Champoluc (Ramey)")

col_lat, col_lon, col_tz, col_h = st.columns([1,1,1.2,1.2])
with col_lat: lat = st.number_input("Lat", value=float(lat), format="%.6f", key="lat_num")
with col_lon: lon = st.number_input("Lon", value=float(lon), format="%.6f", key="lon_num")
with col_tz: tzname = st.selectbox("Timezone", ["Europe/Rome","UTC"], index=0)
with col_h: hours = st.slider("Ore previsione", 12, 168, 72, 12)

# ---------------------- STRUCTURE DRAWINGS ----------------------
# immagini semplici ma “da manuale macchinario”: linee nette, spacing coerente
def draw_structure(kind="linear", density=10):
    fig, ax = plt.subplots(figsize=(3.2, 1.4), dpi=200)
    ax.set_facecolor("#222"); ax.set_xlim(0,100); ax.set_ylim(0,40)
    ax.axis("off")

    if kind == "linear":  # righe dritte longitudinali
        step = 100//density
        for x in range(5, 100, step):
            ax.plot([x,x],[4,36], linewidth=1.6, alpha=0.95, solid_capstyle='round')
    elif kind == "waves":  # onde/convessa
        import numpy as np
        x = np.linspace(0,100,400)
        for offset in [6,12,18,24,30]:
            y = 20 + 10*np.sin((x/100)*2*math.pi + offset/10)
            ax.plot(x,y, linewidth=1.6, alpha=0.95)
    elif kind == "diagonal":  # 45°
        step = 100//density
        for k in range(-40, 100, step):
            ax.plot([k, k+40], [0,40], linewidth=1.6, alpha=0.95)
    elif kind == "fishbone":  # a lisca di pesce
        step = 100//(density//2 if density>=6 else 6)
        for x in range(10, 100, step):
            ax.plot([x,x],[5,35], linewidth=1.2, alpha=0.9)
            ax.plot([x-6,x+6],[22,35], linewidth=1.2, alpha=0.9)
            ax.plot([x-6,x+6],[18,5], linewidth=1.2, alpha=0.9)
    elif kind == "side-drain":  # scarico laterale
        step = 12
        for x in range(6, 100, step):
            ax.plot([x,x],[10,30], linewidth=1.6, alpha=0.95)
        ax.plot([0,100],[10,10], linewidth=2.0, alpha=0.9)
        ax.plot([0,100],[30,30], linewidth=2.0, alpha=0.9)
    buf = io.BytesIO(); fig.tight_layout(pad=0); plt.savefig(buf, format="png", transparent=True); plt.close(fig)
    return buf.getvalue()

STRUCTURE_IMAGES = {
    "Lineare (freddo/secco)": draw_structure("linear", density=14),
    "Onde/Convessa (universale freddo→medio)": draw_structure("waves", density=10),
    "Diagonale (universale)": draw_structure("diagonal", density=10),
    "Lisca/Spina (umido)": draw_structure("fishbone", density=12),
    "Scarico laterale (molto umido)": draw_structure("side-drain", density=10),
}

def structure_choice(t_surf):
    if t_surf <= -10: return "Lineare (freddo/secco)"
    if t_surf <= -3:  return "Onde/Convessa (universale freddo→medio)"
    if t_surf <= 0.5: return "Diagonale (universale)"
    return "Scarico laterale (molto umido)"

# ---------------------- WAX BANDS & LOGOS ----------------------
SWIX = [("PS5 Turquoise", -18,-10), ("PS6 Blue", -12,-6), ("PS7 Violet", -8,-2), ("PS8 Red", -4,4), ("PS10 Yellow", 0,10)]
TOKO = [("Blue", -30,-9), ("Red", -12,-4), ("Yellow", -6,0)]
VOLA = [("MX-E Blue/Violet", -25,-4), ("MX-E Red", -5,0), ("MX-E Warm", -2,10)]
RODE = [("R20 Blue", -18,-8), ("R30 Violet", -10,-3), ("R40 Red", -5,0), ("R50 Yellow", -1,10)]
HOLM = [("UltraMix Blue", -20,-8), ("BetaMix Red", -14,-4), ("AlphaMix Yellow", -4,0)]
MAPL = [("Universal Cold", -12,-6), ("Universal Medium", -7,-2), ("Universal Soft", -5,0)]
SKIGO= [("Blue", -12,-6), ("Red", -5,1)]
START= [("SG Blue", -12,-6), ("SG Purple", -8,-2), ("SG Red", -3,7)]

BRANDS = [
    ("Swix", SWIX, "#ef4444"),
    ("Toko", TOKO, "#f59e0b"),
    ("Vola", VOLA, "#3b82f6"),
    ("Rode", RODE, "#22c55e"),
    ("Holmenkol", HOLM, "#8b5cf6"),
    ("Maplus", MAPL, "#06b6d4"),
    ("Skigo", SKIGO, "#9ca3af"),
    ("Start", START, "#f43f5e"),
]

def svg_logo(text, color):
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='200' height='36'><rect width='200' height='36' rx='6' fill='{color}'/><text x='12' y='24' font-size='16' font-weight='700' fill='white'>{text}</text></svg>"
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode()

def band_pick(bands, t):
    for n,tmin,tmax in bands:
        if t>=tmin and t<=tmax: return n
    return bands[-1][0] if t>bands[-1][2] else bands[0][0]

# ---------------------- LAMINE (side stile 88°) ----------------------
def tune_for(t_surf, discipline):
    # side angle (88.5, 88, 87.5, 87); base edge in degrees (0.5..1.0)
    if t_surf <= -10:
        structure = "Lineare (fine)"; base = 0.5; side_map = {"SL":88.5,"GS":88.0,"SG":87.5,"DH":87.5}
    elif t_surf <= -3:
        structure = "Onde/Convessa (media)"; base = 0.7; side_map = {"SL":88.0,"GS":88.0,"SG":87.5,"DH":87.0}
    else:
        structure = "Scarico laterale / Lisca (umido)"; base = 0.8 if t_surf<=0.5 else 1.0; side_map = {"SL":88.0,"GS":87.5,"SG":87.0,"DH":87.0}
    return structure, side_map.get(discipline,88.0), base

# ---------------------- CONTROLS A/B/C ----------------------
st.markdown("#### Finestre A · B · C (oggi)")
c1,c2,c3 = st.columns(3)
with c1:
    A_start = st.time_input("Inizio A", value=time(9,0), key="A_s"); A_end = st.time_input("Fine A", value=time(11,0), key="A_e")
with c2:
    B_start = st.time_input("Inizio B", value=time(11,0), key="B_s"); B_end = st.time_input("Fine B", value=time(13,0), key="B_e")
with c3:
    C_start = st.time_input("Inizio C", value=time(13,0), key="C_s"); C_end = st.time_input("Fine C", value=time(16,0), key="C_e")

# ---------------------- RUN ----------------------
col_run1, col_run2 = st.columns([1,2])
with col_run1:
    go = st.button("Scarica previsioni per la località", type="primary")
with col_run2:
    upl = st.file_uploader("…oppure carica CSV (time,T2m,cloud,wind,sunup,prp_mmph,prp_type,td)", type=["csv"])

def plot_series(res):
    t = pd.to_datetime(res["time"])
    fig1 = plt.figure()
    plt.plot(t,res["T2m"],label="T2m")
    plt.plot(t,res["T_surf"],label="T_surf")
    plt.plot(t,res["T_top5"],label="T_top5")
    plt.legend(); plt.title("Temperature"); plt.xlabel("Ora"); plt.ylabel("°C")
    fig2 = plt.figure()
    plt.bar(t,res["prp_mmph"])
    plt.title("Precipitazione (mm/h)"); plt.xlabel("Ora"); plt.ylabel("mm/h")
    return fig1, fig2

def badge(win):
    tmin = float(win["T_surf"].min()); tmax = float(win["T_surf"].max())
    wet = bool(((win["prp_type"].isin(["rain","mixed"])) | (win["prp_mmph"]>0.5)).any())
    if wet or tmax > 0.5:
        color, title, desc = "#ef4444","CRITICAL","Possibile neve bagnata/pioggia · struttura grossa"
    elif tmax > -1.0:
        color, title, desc = "#f59e0b","WATCH","Vicino a 0°C · cere medio-morbide"
    else:
        color, title, desc = "#22c55e","OK","Neve fredda/asciutta · cere dure"
    st.markdown(f"""
    <div class='card' style='border-color:{color}'>
      <div style='font-weight:800;color:{color};margin-bottom:4px'>{title}</div>
      <div style='opacity:.95'>{desc}</div>
      <div style='font-size:12px;opacity:.7;margin-top:6px'>T_surf min {tmin:.1f}°C / max {tmax:.1f}°C</div>
    </div>""", unsafe_allow_html=True)
    return wet

def wax_and_tune_block(label, win, tzname):
    wet = badge(win)
    t_med = float(win["T_surf"].mean())
    st.markdown(f"**T_surf medio {label}: {t_med:.1f} °C**")

    # Struttura consigliata + preview immagine
    sname = structure_choice(t_med)
    img = STRUCTURE_IMAGES[sname]
    st.markdown(f"**Struttura consigliata:** {sname}")
    st.image(img, use_column_width=False)

    # Tuning discipline
    disc = st.multiselect(f"Discipline (Blocco {label})", ["SL","GS","SG","DH"], default=["SL","GS"], key=f"disc_{label}")
    if disc:
        rows = []
        for d in disc:
            sdesc, side, base = tune_for(t_med, d)
            rows.append([d, sdesc, f"{side:.1f}°", f"{base:.1f}°"])
        st.table(pd.DataFrame(rows, columns=["Disciplina","Struttura soluzione","Lamina SIDE (°)","Lamina BASE (°)"]))

    # Carte sciolina per 8 brand
    st.markdown("**Sciolina consigliata (non-fluoro):**")
    cols = st.columns(4)
    for i,(brand,bands,color) in enumerate(BRANDS):
        rec = band_pick(bands, t_med)
        logo = svg_logo(brand.upper(), color)
        cols[i%4].markdown(
            f"<div class='brand'><img src='{logo}'/><div><div style='font-size:.8rem;opacity:.85'>{brand}</div><div style='font-weight:800'>{rec}</div></div></div>",
            unsafe_allow_html=True
        )

def run_all(src, label):
    res = compute_snow_temperature(src, dt_hours=1.0)
    st.success(f"Previsioni per **{label}** pronte.")
    st.dataframe(res, use_container_width=True)
    f1,f2 = plot_series(res); st.pyplot(f1); st.pyplot(f2)
    st.download_button("Scarica CSV risultato", data=res.to_csv(index=False), file_name="forecast_with_snowT.csv", mime="text/csv")

    for L,(s,e) in {"A":(A_start,A_end), "B":(B_start,B_end), "C":(C_start,C_end)}.items():
        st.markdown(f"### Blocco {L}")
        W = window_today_slice(res, tzname, s, e)
        wax_and_tune_block(L, W, tzname)

# Run from CSV or API
if upl is not None:
    try:
        df_u = pd.read_csv(upl)
        run_all(df_u, place_label)
    except Exception as e:
        st.error(f"CSV non valido: {e}")

if go:
    try:
        js = fetch_open_meteo(lat, lon, tzname)
        src = build_df(js, hours, tzname)
        run_all(src, place_label)
    except Exception as e:
        st.error(f"Errore: {e}")

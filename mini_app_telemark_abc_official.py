# telemark_pro_app.py
# Telemark · Pro Wax & Tune — mobile-first app
# Autocomplete istantaneo, geolocalizzazione, blocchi A/B/C, sciolina multi-brand,
# struttura soletta (SVG), angoli lamine (SIDE 88°… & BASE).

import streamlit as st
import pandas as pd
import requests
import base64
import matplotlib.pyplot as plt
from datetime import time
from dateutil import tz

# =========================
# THEME
# =========================
PRIMARY = "#10bfcf"; BG = "#0f172a"; CARD = "#111827"; TEXT = "#e5e7eb"
st.set_page_config(page_title="Telemark · Pro Wax & Tune", page_icon="❄️", layout="wide")
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
  background: linear-gradient(180deg, {BG} 0%, #111827 100%);
}}
.block-container {{ padding-top: 12px; }}
h1,h2,h3,h4,h5,p,span,div {{ color:{TEXT}; }}
.badge {{
  border:1px solid rgba(255,255,255,.15);
  padding:6px 10px; border-radius:999px; font-size:.78rem; opacity:.85;
}}
.card {{
  background:{CARD}; border:1px solid rgba(255,255,255,.08);
  border-radius:16px; padding:14px; box-shadow: 0 8px 20px rgba(0,0,0,.25);
}}
.brand {{
  display:flex; align-items:center; gap:10px; padding:10px 12px;
  border-radius:12px; background:rgba(255,255,255,.03);
  border:1px solid rgba(255,255,255,.08);
}}
.brand img {{ height:22px; }}
.kpi {{
  display:flex; gap:8px; align-items:center;
  background:rgba(16,191,207,.06); border:1px dashed rgba(16,191,207,.45);
  padding:10px 12px; border-radius:12px;
}}
.kpi .lab {{ font-size:.78rem; color:#93c5fd; }}
.kpi .val {{ font-size:1rem; font-weight:800; color:{TEXT}; }}
.sugg-btn button {{
  width:100%; text-align:left; border-radius:10px !important;
  background:rgba(255,255,255,.04) !important; border:1px solid rgba(255,255,255,.12) !important;
}}
hr {{ border-color: rgba(255,255,255,.08) }}
</style>
""", unsafe_allow_html=True)

st.markdown("### Telemark · Pro Wax & Tune")
st.markdown("<span class='badge'>Autocomplete istantaneo · Geolocalizzazione · Blocchi A/B/C · Sciolina + Struttura + Lamine (SIDE)</span>", unsafe_allow_html=True)

# =========================
# MODELLO: temperatura neve (embedded)
# =========================
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
    clear = (1.0 - df["cloud"]).clip(0.0, 1.0)
    windc = df["wind"].clip(upper=6.0)
    drad = (1.5 + 3.0*clear - 0.3*windc).clip(lower=0.5, upper=4.5)
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
            T_top5.iloc[i] = T_top5.iloc[i-1] + alpha.iloc[i] * (T_surf.iloc[i] - T_top5.iloc[i-1])

    df["T_surf"] = T_surf
    df["T_top5"] = T_top5
    return df

# =========================
# RICERCA LOCALITÀ — stile “meteoblue-like”
# (Aggiorna ad ogni carattere; mostra lista sotto input; applica auto il best match)
# =========================
@st.cache_data(show_spinner=False)
def geocode(query: str, limit: int = 8):
    if not query or len(query.strip()) < 1:
        return []
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format":"json", "limit": limit, "addressdetails": 1},
            headers={"User-Agent": "telemark-pro-wax/1.0"},
            timeout=8
        )
        r.raise_for_status()
        items = r.json()
        out = []
        for it in items:
            out.append({
                "label": it.get("display_name",""),
                "lat": float(it.get("lat",0)),
                "lon": float(it.get("lon",0))
            })
        return out
    except Exception:
        return []

def ip_geolocate():
    try:
        r = requests.get("https://ipapi.co/json", timeout=6)
        if r.ok:
            j = r.json()
            return float(j.get("latitude",0)), float(j.get("longitude",0)), j.get("city","")
    except Exception:
        pass
    return None, None, ""

# Stato iniziale
if "chosen" not in st.session_state:
    st.session_state.chosen = {"label":"Champoluc (Ramey)","lat":45.831,"lon":7.730}
if "query" not in st.session_state:
    st.session_state.query = st.session_state.chosen["label"]

# Input “alla meteoblue”: un solo campo + lista live sotto
st.markdown("**Cerca località**")
q = st.text_input(
    label="Digita una località",
    value=st.session_state.query,
    placeholder="Es. Champoluc, Cervinia, Sestriere…",
    label_visibility="collapsed",
    key="q_input"
)

# Suggerimenti live
suggs = geocode(q, limit=8) if q else []
auto_applied = False
if suggs:
    # Applica automaticamente il best match alla prima battuta (senza click/Invio)
    best = suggs[0]
    if st.session_state.chosen["label"] != best["label"]:
        st.session_state.chosen = best
        st.session_state.query = best["label"]
        auto_applied = True

    with st.container():
        st.caption("Suggerimenti")
        # griglia di bottoni tipo menu a tendina
        for i, s in enumerate(suggs[:8]):
            if st.button(s["label"], key=f"sugg_{i}", help="Usa questa località", type="secondary"):
                st.session_state.chosen = s
                st.session_state.query = s["label"]

geo_col1, geo_col2 = st.columns([1,1])
with geo_col1:
    if st.button("📍 Usa la mia posizione"):
        lat, lon, city = ip_geolocate()
        if lat is not None:
            st.session_state.chosen = {"label":city or "La tua posizione", "lat":lat, "lon":lon}
            st.session_state.query = st.session_state.chosen["label"]
            st.success("Posizione impostata.")
        else:
            st.error("Geolocalizzazione non disponibile.")

with geo_col2:
    tzname = st.selectbox("Timezone", ["Europe/Rome","UTC"], index=0)

# Lettura selezione corrente (sempre aggiornata)
label = st.session_state.chosen["label"]
lat = st.session_state.chosen["lat"]
lon = st.session_state.chosen["lon"]

# Parametri orizzonte + blocchi A/B/C
cols_top = st.columns(2)
with cols_top[0]:
    hours = st.slider("Ore di previsione", 12, 168, 72, 12)
with cols_top[1]:
    st.write("")  # spazio

st.markdown("#### Finestre A · B · C (oggi)")
c1,c2,c3 = st.columns(3)
with c1:
    A_start = st.time_input("Inizio A", value=time(9,0), key="A_s")
    A_end   = st.time_input("Fine A",   value=time(11,0), key="A_e")
with c2:
    B_start = st.time_input("Inizio B", value=time(11,0), key="B_s")
    B_end   = st.time_input("Fine B",   value=time(13,0), key="B_e")
with c3:
    C_start = st.time_input("Inizio C", value=time(13,0), key="C_s")
    C_end   = st.time_input("Fine C",   value=time(16,0), key="C_e")

# =========================
# Meteo: fetch + build dataframe
# =========================
def fetch_open_meteo(lat, lon, timezone_str):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon, "timezone": timezone_str,
        "hourly": "temperature_2m,dew_point_2m,precipitation,rain,snowfall,cloudcover,windspeed_10m,is_day,weathercode",
        "forecast_days": 7
    }
    r = requests.get(url, params=params, timeout=20)
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
    h = js["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])        # naive
    now_naive = pd.Timestamp.now().floor("H")      # naive
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

# =========================
# WAX (multi-brand) + TUNING
# =========================
# Temperature bands (no-fluoro) — semplificate per uso operativo
SWIX = [("PS5 Turquoise", -18,-10), ("PS6 Blue",-12,-6), ("PS7 Violet",-8,-2), ("PS8 Red",-4,4), ("PS10 Yellow",0,10)]
TOKO = [("Blue",-30,-9), ("Red",-12,-4), ("Yellow",-6,0)]
VOLA = [("MX-E Violet/Blue",-12,-4), ("MX-E Red",-5,0), ("MX-E Warm",-2,10)]
RODE = [("R20 Blue",-18,-8), ("R30 Violet",-10,-3), ("R40 Red",-5,0), ("R50 Yellow",-1,10)]
HOLM = [("UltraMix Blue",-20,-8), ("BetaMix Red",-14,-4), ("AlphaMix Yellow",-4,5)]
MAPL = [("Universal Cold",-12,-6), ("Universal Medium",-7,-2), ("Universal Warm",-3,6)]
START = [("SG Blue",-12,-6), ("SG Purple",-8,-2), ("SG Red",-3,7)]
SKIGO = [("Paraffin Blue",-12,-6), ("Paraffin Violet",-8,-2), ("Paraffin Red",-3,2)]

def pick(bands, t):
    for n,tmin,tmax in bands:
        if t>=tmin and t<=tmax: return n
    return bands[-1][0] if t>bands[-1][2] else bands[0][0]

def svg_logo(text, color):
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='180' height='34'><rect width='180' height='34' rx='6' fill='{color}'/><text x='12' y='22' font-size='15' font-weight='700' fill='white'>{text}</text></svg>"
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")

LOGOS = {
    "Swix": svg_logo("SWIX","#ef4444"),
    "Toko": svg_logo("TOKO","#f59e0b"),
    "Vola": svg_logo("VOLA","#3b82f6"),
    "Rode": svg_logo("RODE","#22c55e"),
    "Holmenkol": svg_logo("HOLMENKOL","#60a5fa"),
    "Maplus": svg_logo("MAPLUS","#f97316"),
    "Start": svg_logo("START","#f43f5e"),
    "Skigo": svg_logo("SKIGO","#8b5cf6"),
}

def tuning_for(t_surf, discipline):
    # SIDE (88°, 87.5°, 87° …), BASE 0.5°–1.0°, struttura sintetica
    if t_surf <= -10:
        structure = "Fine (lineare/pitone)"
        base = 0.5
        side_map = {"SL": 88.5, "GS": 88.0, "SG": 87.5, "DH": 87.5}
    elif t_surf <= -3:
        structure = "Media (universale)"
        base = 0.7
        side_map = {"SL": 88.0, "GS": 88.0, "SG": 87.5, "DH": 87.0}
    else:
        structure = "Medio-Grossa (wave/chevron)"
        base = 0.8 if t_surf <= 0.5 else 1.0
        side_map = {"SL": 88.0, "GS": 87.5, "SG": 87.0, "DH": 87.0}
    return structure, side_map.get(discipline, 88.0), base

# =========================
# STRUTTURE — anteprima SVG chiara (tipo catalogo)
# =========================
def structure_svg(kind: str):
    # genera SVG 560x120 con pattern puliti
    W, H = 560, 120
    bg = "#0b1224"; grid = "rgba(255,255,255,0.06)"; stroke = "#a1a1aa"
    lines = []

    if kind == "Fine lineare":
        # 10 linee verticali sottili
        for x in range(20, W, 40):
            lines.append(f"<line x1='{x}' y1='12' x2='{x}' y2='{H-12}' stroke='{stroke}' stroke-width='2'/>")

    elif kind == "Universale":
        # lineare + leggere onde
        for x in range(20, W, 50):
            lines.append(f"<line x1='{x}' y1='14' x2='{x}' y2='{H-14}' stroke='{stroke}' stroke-width='3' opacity='0.8'/>")
        for x in range(0, W, 28):
            lines.append(f"<path d='M{x},{H/2} q14,-10 28,0 q14,10 28,0' fill='none' stroke='{stroke}' stroke-width='2' opacity='0.5'/>")

    elif kind == "Wave (onda)":
        for x in range(0, W, 24):
            lines.append(f"<path d='M{x},60 q12,-16 24,0 q12,16 24,0' fill='none' stroke='{stroke}' stroke-width='3'/>")

    elif kind == "Chevron (lisca)":
        # V ripetuti
        for x in range(16, W, 32):
            for y in range(20, H, 30):
                lines.append(f"<path d='M{x},{y} l12,12 l12,-12' fill='none' stroke='{stroke}' stroke-width='3'/>")

    elif kind == "Scarico laterale":
        # canale laterale + linee che puntano al bordo
        lines.append(f"<rect x='{W-40}' y='12' width='16' height='{H-24}' fill='{stroke}' opacity='0.6'/>")
        for y in range(16, H-12, 12):
            lines.append(f"<line x1='40' y1='{y}' x2='{W-40}' y2='{y+4}' stroke='{stroke}' stroke-width='2' opacity='0.8'/>")

    else:
        for x in range(20, W, 40):
            lines.append(f"<line x1='{x}' y1='12' x2='{x}' y2='{H-12}' stroke='{stroke}' stroke-width='2'/>")

    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}'>
      <rect width='{W}' height='{H}' rx='12' fill='{bg}' />
      <g opacity='0.35'>
        <line x1='0' y1='{H/2}' x2='{W}' y2='{H/2}' stroke='{grid}' />
      </g>
      {''.join(lines)}
    </svg>"""
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")

def show_structure(kind: str):
    st.markdown(f"**Struttura consigliata:** {kind}")
    st.image(structure_svg(kind))

# =========================
# UI — azioni
# =========================
col_go, col_upload = st.columns([1,2])
with col_go:
    go = st.button(f"Scarica previsioni per: {label}", type="primary")
with col_upload:
    upl = st.file_uploader("…oppure carica CSV (time,T2m,cloud,wind,sunup,prp_mmph,prp_type,td)", type=["csv"])

# =========================
# GRAFICI / BADGE
# =========================
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
      <div style='opacity:.95'>{desc}</div>
      <div style='font-size:12px;opacity:.7;margin-top:6px'>
        T_surf min {tmin:.1f}°C / max {tmax:.1f}°C
      </div>
    </div>""", unsafe_allow_html=True)
    return wet

def brand_cards(t_med):
    brands = [
        ("Swix", SWIX), ("Toko", TOKO), ("Vola", VOLA), ("Rode", RODE),
        ("Holmenkol", HOLM), ("Maplus", MAPL), ("Start", START), ("Skigo", SKIGO),
    ]
    # due righe da 4
    for row in (brands[:4], brands[4:]):
        cols = st.columns(4)
        for i,(name,bands_) in enumerate(row):
            rec = pick(bands_, t_med)
            cols[i].markdown(
                f"<div class='brand'><img src='{LOGOS[name]}'/><div><div style='font-size:.8rem;opacity:.85'>{name}</div><div style='font-weight:800'>{rec}</div></div></div>",
                unsafe_allow_html=True
            )

def run_all(src_df):
    res = compute_snow_temperature(src_df, dt_hours=1.0)

    st.success(f"Previsioni per **{label}** pronte.")
    st.dataframe(res, use_container_width=True)
    f1,f2 = make_plots(res); st.pyplot(f1); st.pyplot(f2)
    st.download_button("Scarica CSV risultato", data=res.to_csv(index=False), file_name="forecast_with_snowT.csv", mime="text/csv")
    st.markdown("---")

    # Blocchi A/B/C
    blocks = {"A": (A_start,A_end), "B": (B_start,B_end), "C": (C_start,C_end)}
    for L,(s,e) in blocks.items():
        st.markdown(f"### Blocco {L}")
        W = window_slice(res, tzname, s, e)
        wet = badge(W)
        t_med = float(W["T_surf"].mean())
        st.markdown(f"**T_surf medio {L}: {t_med:.1f} °C**")

        # Sciolina (tutte le marche)
        brand_cards(t_med)

        # Struttura consigliata + disegno
        # Mappa semplice dalla T neve
        if t_med <= -10:
            kind = "Fine lineare"
        elif t_med <= -3:
            kind = "Universale"
        else:
            # se molto vicino a 0, mostra onda/scarico
            kind = "Wave (onda)" if t_med <= 0.8 else "Scarico laterale"
        show_structure(kind)

        # Lamine per disciplina
        disc = st.multiselect(f"Discipline (Blocco {L})", ["SL","GS","SG","DH"], default=["SL","GS"], key=f"disc_{L}")
        rows = []
        for d in disc:
            structure, side, base = tuning_for(t_med, d)
            rows.append([d, structure, f"{side:.1f}°", f"{base:.1f}°"])
        if rows:
            st.table(pd.DataFrame(rows, columns=["Disciplina","Struttura suggerita","Lamina SIDE (°)","Lamina BASE (°)"]))

# =========================
# FLOW
# =========================
if upl is not None:
    try:
        df_u = pd.read_csv(upl)
        run_all(df_u)
    except Exception as e:
        st.error(f"CSV non valido: {e}")

if go:
    try:
        js = fetch_open_meteo(lat, lon, tzname)
        df_src = build_df(js, hours, tzname)
        run_all(df_src)
    except Exception as e:
        st.error(f"Errore: {e}")

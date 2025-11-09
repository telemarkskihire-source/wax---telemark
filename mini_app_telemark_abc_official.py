# telemark_pro_app.py
# Telemark · Pro Wax & Tune — mobile-first
# Ricerca località tipo Meteoblue (typeahead), geolocalizzazione,
# blocchi A/B/C, sciolina multi-brand, struttura soletta (SVG),
# angoli lamine in formato side (88°, 87.5°, 87°), design moderno.

import streamlit as st
import pandas as pd
import requests
import base64
import math
import matplotlib.pyplot as plt
from datetime import time
from dateutil import tz

# -------------------- UI THEME --------------------
PRIMARY = "#10bfcf"; BG = "#0f172a"; CARD = "#111827"; TEXT = "#e5e7eb"
st.set_page_config(page_title="Telemark · Pro Wax & Tune", page_icon="❄️", layout="wide")
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
  background: linear-gradient(180deg, {BG} 0%, #111827 100%);
}}
.block-container {{ padding-top: 1rem; }}
h1,h2,h3,h4,h5,p,span,div {{ color:{TEXT}; }}
.badge {{
  border:1px solid rgba(255,255,255,.15);
  padding:6px 10px; border-radius:999px;
  font-size:.78rem; opacity:.85;
}}
.card {{
  background:{CARD}; border:1px solid rgba(255,255,255,.08);
  border-radius:16px; padding:14px;
  box-shadow:0 8px 20px rgba(0,0,0,.25);
}}
.brand {{
  display:flex; align-items:center; gap:8px;
  padding:8px 10px; border-radius:12px;
  background:rgba(255,255,255,.03);
  border:1px solid rgba(255,255,255,.08);
}}
.brand img {{ height:22px; }}
.btn-primary button {{
  width:100%; background:{PRIMARY} !important; color:#002b30 !important;
  border:none; font-weight:700; border-radius:12px;
}}
.suggestion {{
  padding:8px 10px; margin:4px 0; border-radius:10px;
  background:rgba(255,255,255,.04);
  border:1px solid rgba(255,255,255,.10);
  cursor:pointer; font-size:.92rem;
}}
.suggestion:hover {{ border-color:{PRIMARY}; background:rgba(16,191,207,.08); }}
.hr {{ border-top:1px solid rgba(255,255,255,.12); margin:8px 0 12px 0; }}
.svgwrap {{
  background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08);
  border-radius:14px; padding:8px; display:flex; justify-content:center;
}}
</style>
""", unsafe_allow_html=True)

st.markdown("## Telemark · Pro Wax & Tune")
st.markdown("<span class='badge'>Typeahead Meteoblue · Geolocalizzazione · Blocchi A/B/C · Sciolina · Struttura · Lamine</span>", unsafe_allow_html=True)

# -------------------- MODELLI --------------------
def compute_snow_temperature(df, dt_hours=1.0):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    req = ["T2m","cloud","wind","sunup","prp_mmph","prp_type"]
    for c in req:
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
    clear = (1.0 - df["cloud"]).clip(0,1)
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
            T_top5.iloc[i] = T_top5.iloc[i-1] + alpha.iloc[i] * (T_surf.iloc[i] - T_top5.iloc[i-1])

    df["T_surf"] = T_surf
    df["T_top5"] = T_top5
    return df

# -------------------- RICERCA LOCALITÀ (tipo Meteoblue) --------------------
@st.cache_data(show_spinner=False)
def _nominatim(query: str, limit: int = 8, lang: str = "it"):
    if not query or len(query) < 2:
        return []
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": query,
                "format": "json",
                "limit": limit,
                "addressdetails": 1,
                "accept-language": lang,
            },
            headers={"User-Agent": "telemark-pro-wax/1.0"},
            timeout=8
        )
        r.raise_for_status()
        js = r.json()
        out = []
        for it in js:
            label = it.get("display_name", "")
            lat = float(it.get("lat", 0))
            lon = float(it.get("lon", 0))
            out.append({"label": label, "lat": lat, "lon": lon})
        return out
    except Exception:
        return []

def geo_ip():
    try:
        r = requests.get("https://ipapi.co/json", timeout=6)
        if r.ok:
            j = r.json()
            return float(j.get("latitude",0)), float(j.get("longitude",0)), j.get("city","")
    except Exception:
        pass
    return None, None, ""

# Barra di ricerca: aggiorna ad OGNI battuta (Streamlit rerun automatico)
col_search, col_geo = st.columns([3,1])
with col_search:
    q = st.text_input("Cerca località", placeholder="Es. Champoluc, Cervinia, Sestriere…", value=st.session_state.get("q",""))
    sugg = _nominatim(q, limit=10, lang="it")
    # Dropdown "visivo" come Meteoblue: lista cliccabile che si aggiorna live
    if q.strip():
        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        for i, s in enumerate(sugg):
            if st.button(s["label"], key=f"sugg_{i}", use_container_width=True):
                st.session_state["q"] = s["label"]
                st.session_state["lat"] = s["lat"]
                st.session_state["lon"] = s["lon"]
                st.session_state["place_label"] = s["label"]
                st.rerun()
with col_geo:
    if st.button("📍 Usa la mia posizione", use_container_width=True):
        lat0, lon0, city = geo_ip()
        if lat0 is not None:
            st.session_state["q"] = city or "La mia posizione"
            st.session_state["lat"] = lat0
            st.session_state["lon"] = lon0
            st.session_state["place_label"] = st.session_state["q"]
            st.success("Posizione impostata")
        else:
            st.error("Geolocalizzazione non disponibile")

lat = st.session_state.get("lat", 45.831)
lon = st.session_state.get("lon", 7.730)
place_label = st.session_state.get("place_label", "Champoluc (Ramey)")

# Parametri base
c0, c1, c2 = st.columns([1.2,1,1])
with c0: tzname = st.selectbox("Timezone", ["Europe/Rome","UTC"], index=0)
with c1: hours = st.slider("Ore previsione", 12, 168, 72, 12)
with c2:
    # Mostra coordinate attuali (modificabili)
    lat = st.number_input("Lat", value=float(lat), format="%.6f")
    lon = st.number_input("Lon", value=float(lon), format="%.6f")
    # salva subito
    st.session_state["lat"], st.session_state["lon"] = lat, lon

# Finestre A/B/C
st.markdown("### Finestre A · B · C (oggi)")
b1,b2,b3 = st.columns(3)
with b1:
    A_start = st.time_input("Inizio A", value=time(9,0), key="A_s")
    A_end   = st.time_input("Fine A",   value=time(11,0), key="A_e")
with b2:
    B_start = st.time_input("Inizio B", value=time(11,0), key="B_s")
    B_end   = st.time_input("Fine B",   value=time(13,0), key="B_e")
with b3:
    C_start = st.time_input("Inizio C", value=time(13,0), key="C_s")
    C_end   = st.time_input("Fine C",   value=time(16,0), key="C_e")

# -------------------- FETCH METEO --------------------
def fetch_open_meteo(lat, lon, timezone_str):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon, "timezone": timezone_str,
        "hourly": "temperature_2m,dew_point_2m,precipitation,rain,snowfall,cloudcover,windspeed_10m,is_day,weathercode",
        "forecast_days": 7
    }
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

def build_df(js, hours, tzname):
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

def slice_window(res, tzname, s, e):
    t = pd.to_datetime(res["time"]).dt.tz_localize(tz.gettz(tzname), nonexistent='shift_forward', ambiguous='NaT')
    D = res.copy(); D["dt"] = t
    today = pd.Timestamp.now(tz=tz.gettz(tzname)).date()
    W = D[(D["dt"].dt.date==today) & (D["dt"].dt.time>=s) & (D["dt"].dt.time<=e)]
    return W if not W.empty else D.head(7)

# -------------------- WAX BANDS (no-fluoro) --------------------
SWIX = [("PS5 Turquoise",-18,-10),("PS6 Blue",-12,-6),("PS7 Violet",-8,-2),("PS8 Red",-4,4),("PS10 Yellow",0,10)]
TOKO = [("Blue",-30,-9),("Red",-12,-4),("Yellow",-6,0)]
VOLA = [("MX-E Violet/Blue",-12,-4),("MX-E Red",-5,0),("MX-E Warm",-2,10)]
RODE = [("R20 Blue",-18,-8),("R30 Violet",-10,-3),("R40 Red",-5,0),("R50 Yellow",-1,10)]
HOLM = [("UltraMix Blue",-20,-8),("BetaMix Red",-14,-4),("AlphaMix Yellow",-4,4)]
MAPL = [("Universal Cold",-12,-6),("Universal Medium",-7,-2),("Universal Warm",-3,6)]
START= [("SG Blue",-12,-6),("SG Purple",-8,-2),("SG Red",-3,7)]
SKIGO= [("Paraffin Blue",-12,-6),("Paraffin Violet",-8,-2),("Paraffin Red",-3,2)]

BRANDS = [
    ("Swix", SWIX, "#ef4444"),
    ("Toko", TOKO, "#f59e0b"),
    ("Vola", VOLA, "#3b82f6"),
    ("Rode", RODE, "#22c55e"),
    ("Holmenkol", HOLM, "#06b6d4"),
    ("Maplus", MAPL, "#a855f7"),
    ("Start", START, "#fb7185"),
    ("Skigo", SKIGO, "#94a3b8"),
]

def pick(bands, t):
    for n,tmin,tmax in bands:
        if t >= tmin and t <= tmax:
            return n
    return bands[-1][0] if t > bands[-1][2] else bands[0][0]

def logo_badge(text, color):
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='110' height='28'><rect width='110' height='28' rx='6' fill='{color}'/><text x='10' y='19' font-size='14' font-weight='700' fill='white'>{text}</text></svg>"
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")

# -------------------- STRUTTURE SOLETTA (SVG pulite) --------------------
def svg_linear(density="fine"):
    # fine / medium / coarse
    spacing = {"fine":6, "medium":10, "coarse":14}[density]
    lines = "\n".join([f"<line x1='{x}' y1='5' x2='{x}' y2='95' stroke='white' stroke-width='1' opacity='0.8'/>" for x in range(10, 190, spacing)])
    return f"<svg viewBox='0 0 200 100' xmlns='http://www.w3.org/2000/svg'><rect width='200' height='100' rx='10' fill='#1f2937'/>{lines}</svg>"

def svg_chevron():
    # lisca/chevron
    path = []
    for y in range(10,95,10):
        path.append(f"<polyline points='10,{y} 100,{y+8} 190,{y}' fill='none' stroke='white' stroke-width='1' opacity='0.85'/>")
        path.append(f"<polyline points='10,{y+5} 100,{y-3} 190,{y+5}' fill='none' stroke='white' stroke-width='1' opacity='0.5'/>")
    return f"<svg viewBox='0 0 200 100' xmlns='http://www.w3.org/2000/svg'><rect width='200' height='100' rx='10' fill='#1f2937'/> {''.join(path)}</svg>"

def svg_wave():
    # onda
    parts=[]
    for phase in range(0,10):
        d = " ".join([f"L {x} {50 + 15*math.sin((x/18)+phase)}" for x in range(10,191,10)])
        parts.append(f"<path d='M 10 50 {d}' stroke='white' stroke-width='1' fill='none' opacity='0.8'/>")
    return f"<svg viewBox='0 0 200 100' xmlns='http://www.w3.org/2000/svg'><rect width='200' height='100' rx='10' fill='#1f2937'/> {''.join(parts)}</svg>"

def svg_lateral_drain():
    # scarico laterale
    lines = "\n".join([f"<line x1='{x}' y1='5' x2='{x}' y2='95' stroke='white' stroke-width='1' opacity='0.7'/>" for x in range(20, 181, 12)])
    gutters = "<rect x='6' y='5' width='8' height='90' fill='#0ea5e9' opacity='0.85'/><rect x='186' y='5' width='8' height='90' fill='#0ea5e9' opacity='0.85'/>"
    return f"<svg viewBox='0 0 200 100' xmlns='http://www.w3.org/2000/svg'><rect width='200' height='100' rx='10' fill='#1f2937'/> {gutters} {lines}</svg>"

def svg_data_uri(svg):
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")

def tune_for(t_surf, discipline):
    # Angolo SIDE (88°, 87.5°, 87°); BASE 0.5°-1.0°
    if t_surf <= -10:
        structure = ("Lineare fine", svg_linear("fine"))
        base = 0.5
        side_map = {"SL": 88.5, "GS": 88.0, "SG": 87.5, "DH": 87.5}
    elif t_surf <= -3:
        structure = ("Lineare media", svg_linear("medium"))
        base = 0.7
        side_map = {"SL": 88.0, "GS": 88.0, "SG": 87.5, "DH": 87.0}
    else:
        # neve vicino/oltre 0°C -> più drenaggio
        structure = ("Onda + scarico laterale", svg_wave())
        base = 0.8 if t_surf <= 0.5 else 1.0
        side_map = {"SL": 88.0, "GS": 87.5, "SG": 87.0, "DH": 87.0}
    side = side_map.get(discipline, 88.0)
    return structure, side, base

def badge(win):
    tmin = float(win["T_surf"].min()); tmax = float(win["T_surf"].max())
    wet = bool(((win["prp_type"].isin(["rain","mixed"])) | (win["prp_mmph"]>0.5)).any())
    if wet or tmax > 0.5:
        color, title, desc = "#ef4444", "CRITICAL", "Possibile neve bagnata/pioggia · struttura grossa/drain"
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
    cols = st.columns(4)
    for i,(name, bands, color) in enumerate(BRANDS[:4]):
        rec = pick(bands, t_med)
        cols[i].markdown(
            f"<div class='brand'><img src='{logo_badge(name.upper(), color)}'/>"
            f"<div><div style='font-size:.8rem;opacity:.85'>{name}</div>"
            f"<div style='font-weight:800'>{rec}</div></div></div>", unsafe_allow_html=True
        )
    cols2 = st.columns(4)
    for i,(name, bands, color) in enumerate(BRANDS[4:]):
        rec = pick(bands, t_med)
        cols2[i].markdown(
            f"<div class='brand'><img src='{logo_badge(name.upper(), color)}'/>"
            f"<div><div style='font-size:.8rem;opacity:.85'>{name}</div>"
            f"<div style='font-weight:800'>{rec}</div></div></div>", unsafe_allow_html=True
        )

def structure_gallery(t_surf):
    # Mostra 4 esempi chiari stile catalogo macchine
    A = ("Lineare fine", svg_linear("fine"))
    B = ("Lineare media", svg_linear("medium"))
    C = ("Chevron/Lisca", svg_chevron())
    D = ("Onda + Scarico laterale", svg_lateral_drain())
    st.markdown("**Anteprima struttura soletta (esempi)**")
    g1, g2, g3, g4 = st.columns(4)
    for col, (lab, svg) in zip([g1,g2,g3,g4],[A,B,C,D]):
        col.markdown(f"<div class='svgwrap'><img src='{svg_data_uri(svg)}'/></div>", unsafe_allow_html=True)
        col.caption(lab)

def plots(res):
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

# -------------------- RUN --------------------
cRun, cCsv = st.columns([1,2])
with cRun:
    run_btn = st.button("Scarica previsioni per la località", type="primary")
with cCsv:
    upl = st.file_uploader("…oppure carica CSV (time,T2m,cloud,wind,sunup,prp_mmph,prp_type,td)", type=["csv"])

if upl is not None:
    try:
        src = pd.read_csv(upl)
        res = compute_snow_temperature(src, dt_hours=1.0)
        st.success(f"CSV caricato per **{place_label}**")
        st.dataframe(res, use_container_width=True)
        f1,f2 = plots(res); st.pyplot(f1); st.pyplot(f2)
        st.download_button("Scarica CSV risultato", data=res.to_csv(index=False), file_name="forecast_with_snowT.csv", mime="text/csv")
    except Exception as e:
        st.error(f"CSV non valido: {e}")

elif run_btn:
    try:
        js = fetch_open_meteo(lat, lon, tzname)
        src = build_df(js, hours, tzname)
        res = compute_snow_temperature(src, dt_hours=1.0)
        st.success(f"Previsioni scaricate per **{place_label}**")
        st.dataframe(res, use_container_width=True)
        f1,f2 = plots(res); st.pyplot(f1); st.pyplot(f2)
        st.download_button("Scarica CSV risultato", data=res.to_csv(index=False), file_name="forecast_with_snowT.csv", mime="text/csv")

        # Blocchi A/B/C
        for label, (s,e) in {"A":(A_start,A_end), "B":(B_start,B_end), "C":(C_start,C_end)}.items():
            st.markdown(f"### Blocco {label}")
            W = slice_window(res, tzname, s, e)
            wet = badge(W)
            t_med = float(W["T_surf"].mean())
            st.markdown(f"**T_surf medio {label}: {t_med:.1f} °C**")
            brand_cards(t_med)

            # Tuning per discipline
            disc = st.multiselect(f"Discipline (Blocco {label})", ["SL","GS","SG","DH"], default=["SL","GS"], key=f"disc_{label}")
            rows = []
            thumbs = []
            for d in disc:
                (str_label, str_svg), side, base = tune_for(t_med, d)
                rows.append([d, str_label, f"{side:.1f}°", f"{base:.1f}°"])
            if rows:
                df_tune = pd.DataFrame(rows, columns=["Disciplina","Struttura consigliata","Lamina SIDE (°)","Lamina BASE (°)"])
                st.table(df_tune)
                structure_gallery(t_med)

    except Exception as e:
        st.error(f"Errore: {e}")
else:
    st.info("Digita una località (suggerimenti istantanei), oppure usa la tua posizione. Poi premi **Scarica previsioni**.")

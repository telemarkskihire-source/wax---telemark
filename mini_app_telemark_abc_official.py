# telemark_pro_wax_app.py
import streamlit as st
import pandas as pd
import requests
import base64
import matplotlib.pyplot as plt
from datetime import time
from dateutil import tz

# ------------------------------ CONFIG & THEME --------------------------------
st.set_page_config(page_title="Telemark · Pro Wax & Tune", page_icon="❄️", layout="wide")
PRIMARY = "#10bfcf"; BG = "#0f172a"; CARD = "#111827"; TEXT = "#e5e7eb"
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
  background: linear-gradient(180deg,{BG} 0%,#111827 100%);
}}
.block-container {{ padding-top: 1rem; }}
h1,h2,h3,h4,h5,p,span,div {{ color:{TEXT}; }}
.badge {{ border:1px solid rgba(255,255,255,.15); padding:6px 10px; border-radius:999px; font-size:.78rem; opacity:.85 }}
.card  {{ background:{CARD}; border:1px solid rgba(255,255,255,.08); border-radius:16px; padding:14px; box-shadow:0 8px 20px rgba(0,0,0,.25) }}
.brand {{ display:flex; align-items:center; gap:8px; padding:8px 10px; border-radius:12px; background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08) }}
.brand img {{ height:22px }}
.kpi   {{ display:flex; gap:8px; align-items:center; background:rgba(16,191,207,.06); border:1px dashed rgba(16,191,207,.45); padding:10px 12px; border-radius:12px }}
.small {{ font-size:.82rem; opacity:.85 }}
.btn-primary button {{ width:100%; background:{PRIMARY} !important; color:#002b30 !important; border:none; font-weight:700; border-radius:12px }}
</style>
""", unsafe_allow_html=True)

st.markdown("### Telemark · Pro Wax & Tune")
st.markdown("<span class='badge'>Ricerca tipo Meteoblue · Geolocalizzazione · Blocchi A/B/C · Sciolina + Struttura + Lamine</span>", unsafe_allow_html=True)

# ------------------------------ UTIL --------------------------------
def svg_data_uri(text, color):
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='200' height='36'><rect width='200' height='36' rx='6' fill='{color}'/><text x='12' y='24' font-size='16' font-weight='700' fill='white'>{text}</text></svg>"
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")

def svg_structure(kind:str) -> str:
    """Piccolo disegno della struttura: restituisce data URI SVG."""
    if kind.startswith("Fine"):
        pattern = "<path d='M0 6 H200 M0 18 H200 M0 30 H200' stroke='#8fb3ff' stroke-width='1'/>"
    elif "Media-Grossa" in kind or "Grossa" in kind:
        pattern = "<path d='M0 6 H200 M0 18 H200 M0 30 H200' stroke='#8fb3ff' stroke-width='2'/>" \
                  "<path d='M0 0 L200 36 M-20 0 L180 36' stroke='#65f0c6' stroke-width='1' opacity='.5'/>"
    else:  # Media
        pattern = "<path d='M0 6 H200 M0 18 H200 M0 30 H200' stroke='#8fb3ff' stroke-width='1.5'/>"
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='200' height='36' viewBox='0 0 200 36'><rect width='200' height='36' rx='8' fill='#0b1224'/> {pattern}</svg>"
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")

# ------------------------------ WEATHER MODEL --------------------------------
def compute_snow_temperature(df, dt_hours=1.0):
    import math
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    req = ["T2m","cloud","wind","sunup","prp_mmph","prp_type"]
    for c in req:
        if c not in df.columns: raise ValueError(f"Missing required column: {c}")
    if "td" not in df.columns: df["td"] = float("nan")
    df = df.sort_values("time").reset_index(drop=True)

    rain = df["prp_type"].str.lower().isin(["rain","mixed"])
    snow = df["prp_type"].str.lower().eq("snow")
    sun  = df["sunup"].astype(int) == 1
    tw   = (df["T2m"] + df["td"]) / 2.0
    wet  = (rain | (df["T2m"]>0) | (sun & (df["cloud"]<0.3) & (df["T2m"]>=-3)) | (snow & (df["T2m"]>=-1)) | (snow & tw.ge(-0.5).fillna(False)))

    T_surf = pd.Series(index=df.index, dtype=float); T_surf.loc[wet] = 0.0
    dry = ~wet
    clear = (1.0 - df["cloud"]).clip(0,1)
    windc = df["wind"].clip(upper=6.0)
    drad  = (1.5 + 3.0*clear - 0.3*windc).clip(0.5,4.5)
    T_surf.loc[dry] = df["T2m"][dry] - drad[dry]

    sunny_cold = sun & dry & df["T2m"].between(-10, 0, inclusive="both")
    T_surf.loc[sunny_cold] = pd.concat([
        (df["T2m"] + 0.5*(1.0-df["cloud"]))[sunny_cold],
        pd.Series(-0.5, index=df.index)[sunny_cold]
    ], axis=1).min(axis=1)

    T_top5 = pd.Series(index=df.index, dtype=float)
    tau = pd.Series(6.0, index=df.index, dtype=float)
    tau.loc[rain | snow | (df["wind"]>=6)] = 3.0
    tau.loc[(~sun) & (df["wind"]<2) & (df["cloud"]<0.3)] = 8.0
    alpha = 1.0 - (math.e ** (-dt_hours/tau))
    if len(df)>0:
        T_top5.iloc[0] = min(df["T2m"].iloc[0], 0.0)
        for i in range(1,len(df)):
            T_top5.iloc[i] = T_top5.iloc[i-1] + alpha.iloc[i]*(T_surf.iloc[i]-T_top5.iloc[i-1])

    df["T_surf"] = T_surf; df["T_top5"] = T_top5
    return df

# ------------------------------ GEO SEARCH (Meteoblue-style) ------------------
@st.cache_data(show_spinner=False, ttl=3600)
def geocode_autocomplete(q: str, limit: int = 10):
    """Nominatim, risultati ordinati; usata ad ogni battuta (no ENTER)."""
    if not q or len(q) < 2:
        return []
    r = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": q, "format": "json", "limit": limit, "addressdetails": 1},
        headers={"User-Agent": "telemark-pro-wax/1.0"},
        timeout=10,
    )
    r.raise_for_status()
    out = []
    for it in r.json():
        name = it.get("display_name","")
        lat  = float(it.get("lat",0)); lon = float(it.get("lon",0))
        out.append({"label": name, "lat": lat, "lon": lon})
    return out

def ip_geolocate():
    try:
        r = requests.get("https://ipapi.co/json", timeout=6)
        if r.ok:
            j = r.json()
            return float(j.get("latitude",0)), float(j.get("longitude",0)), j.get("city","")
    except Exception:
        pass
    return None, None, ""

# Barra di ricerca “live”
cL, cR = st.columns([3,1])
with cL:
    query = st.text_input(
        "Cerca località",
        placeholder="Es. Champoluc, Cervinia, Sestriere…",
        key="query",
        help="Digita: i risultati compaiono sotto automaticamente (tipo Meteoblue)."
    )
    suggestions = geocode_autocomplete(query, limit=12) if len(query) >= 2 else []
    labels = [s["label"] for s in suggestions]
    choice = st.selectbox("Suggerimenti (scegli)", labels if labels else ["—"], index=0, key="pick", disabled=(len(labels)==0))
with cR:
    if st.button("📍 Usa la mia posizione"):
        lat0, lon0, city = ip_geolocate()
        if lat0 is not None:
            st.session_state["sel_lat"] = lat0; st.session_state["sel_lon"] = lon0; st.session_state["sel_label"] = city or "La tua posizione"
            st.success("Posizione impostata.")
        else:
            st.error("Geolocalizzazione non disponibile.")

# Se l’utente ha scelto dalla tendina, salviamo la selezione
if labels and choice in labels:
    i = labels.index(choice)
    st.session_state["sel_lat"] = suggestions[i]["lat"]
    st.session_state["sel_lon"] = suggestions[i]["lon"]
    st.session_state["sel_label"] = suggestions[i]["label"]

lat = st.session_state.get("sel_lat", 45.831)
lon = st.session_state.get("sel_lon", 7.730)
label = st.session_state.get("sel_label", "Champoluc (Ramey)")

c1,c2,c3,c4 = st.columns([1,1,1.5,1.5])
with c1: lat = st.number_input("Lat", value=float(lat), format="%.6f")
with c2: lon = st.number_input("Lon", value=float(lon), format="%.6f")
with c3: tzname = st.selectbox("Timezone", ["Europe/Rome","UTC"], index=0)
with c4: hours  = st.slider("Ore previsione", 12, 168, 72, 12)

# ------------------------------ A/B/C windows --------------------------------
st.markdown("#### Finestre A · B · C (oggi)")
w1,w2,w3 = st.columns(3)
with w1:
    A_start = st.time_input("Inizio A", value=time(9,0), key="A_s")
    A_end   = st.time_input("Fine A",   value=time(11,0), key="A_e")
with w2:
    B_start = st.time_input("Inizio B", value=time(11,0), key="B_s")
    B_end   = st.time_input("Fine B",   value=time(13,0), key="B_e")
with w3:
    C_start = st.time_input("Inizio C", value=time(13,0), key="C_s")
    C_end   = st.time_input("Fine C",   value=time(16,0), key="C_e")

# ------------------------------ Open-Meteo -----------------------------------
def fetch_open_meteo(lat, lon, timezone_str):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon, "timezone": timezone_str,
        "hourly": "temperature_2m,dew_point_2m,precipitation,rain,snowfall,cloudcover,windspeed_10m,is_day,weathercode",
        "forecast_days": 7
    }
    r = requests.get(url, params=params, timeout=25); r.raise_for_status()
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
    h = js["hourly"]; df = pd.DataFrame(h); df["time"] = pd.to_datetime(df["time"])
    now_naive = pd.Timestamp.now().floor("H")
    df = df[df["time"] >= now_naive].head(hours).reset_index(drop=True)
    out = pd.DataFrame()
    out["time"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    out["T2m"] = df["temperature_2m"].astype(float)
    out["cloud"] = (df["cloudcover"].astype(float)/100).clip(0,1)
    out["wind"]  = (df["windspeed_10m"].astype(float)/3.6).round(3)
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

# ------------------------------ WAX CATALOG (many brands) --------------------
# (range semplificati non-fluoro)
BANDS = {
    "Swix":      [("PS5 Turquoise",-18,-10),("PS6 Blue",-12,-6),("PS7 Violet",-8,-2),("PS8 Red",-4,4),("PS10 Yellow",0,10)],
    "Toko":      [("Blue",-30,-9),("Red",-12,-4),("Yellow",-6,0)],
    "Vola":      [("MX-E Blue/Violet",-25,-4),("MX-E Red",-5,0),("MX-E Warm",-2,10)],
    "Rode":      [("R20 Blue",-18,-8),("R30 Violet",-10,-3),("R40 Red",-5,0),("R50 Yellow",-1,10)],
    "Holmenkol": [("UltraMix Blue",-20,-8),("BetaMix Red",-14,-4),("AlphaMix Yellow",-4,2)],
    "Maplus":    [("Universal Cold",-12,-6),("Universal Med",-7,-2),("Universal Soft",-5,2)],
    "Skigo":     [("Blue",-12,-6),("Violet",-8,-2),("Red",-4,2)],
    "Start":     [("SG Blue",-12,-6),("SG Purple",-8,-2),("SG Red",-3,7)],
}

BRAND_LOGO = {
    "Swix": svg_data_uri("SWIX","#ef4444"),
    "Toko": svg_data_uri("TOKO","#f59e0b"),
    "Vola": svg_data_uri("VOLA","#3b82f6"),
    "Rode": svg_data_uri("RODE","#22c55e"),
    "Holmenkol": svg_data_uri("HOLMENKOL","#64748b"),
    "Maplus": svg_data_uri("MAPLUS","#06b6d4"),
    "Skigo": svg_data_uri("SKIGO","#a855f7"),
    "Start": svg_data_uri("START","#f97316"),
}

def pick(bands, t):
    for n,tmin,tmax in bands:
        if t>=tmin and t<=tmax: return n
    return bands[-1][0] if t>bands[-1][2] else bands[0][0]

def tune_for(t_surf, discipline):
    # side in gradi (88°, 87.5°, …) / base in gradi
    if t_surf <= -10:
        structure = "Fine (lineare)"
        base = 0.5
        side = {"SL": 88.5, "GS": 88.0, "SG": 87.5, "DH": 87.5}.get(discipline, 88.0)
    elif t_surf <= -3:
        structure = "Media (universale)"
        base = 0.7
        side = {"SL": 88.0, "GS": 88.0, "SG": 87.5, "DH": 87.0}.get(discipline, 88.0)
    else:
        structure = "Media-Grossa (lisca/diamante)"
        base = 0.8 if t_surf <= 0.5 else 1.0
        side = {"SL": 88.0, "GS": 87.5, "SG": 87.0, "DH": 87.0}.get(discipline, 88.0)
    return structure, side, base

def wax_cards(t):
    cols = st.columns(4)
    brands = list(BANDS.keys())
    for i, b in enumerate(brands[:4]):
        rec = pick(BANDS[b], t)
        cols[i].markdown(
            f"<div class='brand'><img src='{BRAND_LOGO[b]}'/><div>"
            f"<div class='small'>{b}</div><div style='font-weight:800'>{rec}</div></div></div>",
            unsafe_allow_html=True
        )
    cols2 = st.columns(4)
    for i, b in enumerate(brands[4:8]):
        rec = pick(BANDS[b], t)
        cols2[i].markdown(
            f"<div class='brand'><img src='{BRAND_LOGO[b]}'/><div>"
            f"<div class='small'>{b}</div><div style='font-weight:800'>{rec}</div></div></div>",
            unsafe_allow_html=True
        )

def badge(win):
    tmin = float(win["T_surf"].min()); tmax = float(win["T_surf"].max())
    wet  = bool(((win["prp_type"].isin(["rain","mixed"])) | (win["prp_mmph"]>0.5)).any())
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
      <div class='small' style='margin-top:6px'>T_surf min {tmin:.1f}°C / max {tmax:.1f}°C</div>
    </div>""", unsafe_allow_html=True)
    return wet

def plots(res):
    t = pd.to_datetime(res["time"])
    fig1 = plt.figure(); plt.plot(t,res["T2m"],label="T2m"); plt.plot(t,res["T_surf"],label="T_surf"); plt.plot(t,res["T_top5"],label="T_top5"); plt.legend(); plt.title("Temperature vs tempo"); plt.xlabel("Ora"); plt.ylabel("°C")
    fig2 = plt.figure(); plt.bar(t,res["prp_mmph"]); plt.title("Precipitazione (mm/h)"); plt.xlabel("Ora"); plt.ylabel("mm/h")
    return fig1,fig2

# ------------------------------ RUN ------------------------------------------
left, right = st.columns([1,2])
with left:
    go = st.button("Scarica previsioni per la località", type="primary")
with right:
    upl = st.file_uploader("…oppure carica CSV (time,T2m,cloud,wind,sunup,prp_mmph,prp_type,td)", type=["csv"])

def run_all(src_df, place_label):
    res = compute_snow_temperature(src_df, dt_hours=1.0)
    st.success(f"Previsioni per **{place_label}** pronte.")
    st.dataframe(res, use_container_width=True)
    f1,f2 = plots(res); st.pyplot(f1); st.pyplot(f2)
    st.download_button("Scarica CSV risultato", data=res.to_csv(index=False), file_name="forecast_with_snowT.csv", mime="text/csv")

    for L,(s,e) in {"A":(A_start,A_end),"B":(B_start,B_end),"C":(C_start,C_end)}.items():
        st.markdown(f"### Blocco {L}")
        W = window_slice(res, tzname, s, e)
        wet = badge(W)
        t_med = float(W["T_surf"].mean())
        st.markdown(f"**T_surf medio {L}: {t_med:.1f} °C**")
        wax_cards(t_med)

        # Struttura + lamine per disciplina
        disc = st.multiselect(f"Discipline (Blocco {L})", ["SL","GS","SG","DH"], default=["SL","GS"], key=f"disc_{L}")
        rows=[]
        for d in disc:
            structure, side, base = tune_for(t_med, d)
            rows.append([d, structure, f"{side:.1f}°", f"{base:.1f}°",
                         f"![struct]({svg_structure(structure)})"])
        if rows:
            df_tune = pd.DataFrame(rows, columns=["Disciplina","Struttura consigliata","Lamina SIDE (°)","Lamina BASE (°)","Disegno struttura"])
            st.table(df_tune)

# CSV upload
if upl is not None:
    try:
        df_u = pd.read_csv(upl)
        run_all(df_u, label)
    except Exception as e:
        st.error(f"CSV non valido: {e}")

# Online fetch
if go:
    try:
        js = fetch_open_meteo(lat, lon, tzname)
        src = build_df(js, hours, tzname)
        run_all(src, label)
    except Exception as e:
        st.error(f"Errore: {e}")

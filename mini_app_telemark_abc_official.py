# Telemark · Pro Wax & Tune — mobile-first
# Streamlit app completa con ricerca "tipo meteoblue", geolocalizzazione,
# finestre A/B/C, scioline multi-marca, strutture soletta (SVG), angoli lamine.

import streamlit as st
import pandas as pd
import requests
import base64
import math
import matplotlib.pyplot as plt
from datetime import time
from dateutil import tz

# ------------------ PAGE / THEME ------------------
st.set_page_config(page_title="Telemark · Pro Wax & Tune", page_icon="❄️", layout="wide")
PRIMARY = "#10bfcf"; BG="#0f172a"; CARD="#111827"; TEXT="#e5e7eb"
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] > .main {{ background: linear-gradient(180deg,{BG} 0%,#111827 100%);}}
.block-container {{ padding-top: 0.8rem; }}
h1,h2,h3,h4,h5,p,span,div {{ color:{TEXT}; }}
.badge {{ border:1px solid rgba(255,255,255,.18); padding:6px 10px; border-radius:999px; font-size:.78rem; opacity:.9; }}
.card {{ background:{CARD}; border:1px solid rgba(255,255,255,.08); border-radius:16px; padding:14px;
        box-shadow: 0 8px 20px rgba(0,0,0,.25); }}
.selectable {{ cursor:pointer; }}
.brand {{ display:flex; gap:8px; align-items:center; padding:8px 10px; border-radius:12px;
         background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08);}}
.brand img {{ height:22px; }}
.btn-primary button {{ width:100%; background:{PRIMARY} !important; color:#062b2f !important; font-weight:700; border:none; border-radius:12px; }}
</style>
""", unsafe_allow_html=True)

st.markdown("### Telemark · Pro Wax & Tune")
st.markdown("<span class='badge'>Ricerca live · Geolocalizzazione · Blocchi A/B/C · Sciolina + Struttura + Angoli</span>", unsafe_allow_html=True)

# ------------------ UTILS ------------------
@st.cache_data(show_spinner=False)
def _nominatim(query: str, limit: int = 10):
    """Autocomplete Nominatim (OpenStreetMap)."""
    if not query or len(query) < 2:
        return []
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": limit, "addressdetails": 1},
            headers={"User-Agent": "telemark-wax-app/1.0"},
            timeout=8
        )
        r.raise_for_status()
        out = []
        for it in r.json():
            label = it.get("display_name","")
            lat = float(it.get("lat",0))
            lon = float(it.get("lon",0))
            cc  = (it.get("address",{}) or {}).get("country_code","")
            out.append({"label": label, "lat": lat, "lon": lon, "cc": cc})
        return out
    except Exception:
        return []

def _ip_geolocate():
    try:
        r = requests.get("https://ipapi.co/json", timeout=6)
        if r.ok:
            j = r.json()
            return float(j.get("latitude",0)), float(j.get("longitude",0)), j.get("city","")
    except Exception:
        pass
    return None, None, ""

def svg_text_logo(text, color):
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='160' height='36'><rect rx='6' width='160' height='36' fill='{color}'/><text x='10' y='24' font-size='16' font-weight='700' fill='#fff'>{text}</text></svg>"
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()

def structure_svg(style: str):
    """
    Ritorna un SVG (data URI) pulito e realistico per:
    - 'fine'      = lineare fine/verticale
    - 'onda'      = corona/onda universale
    - 'diagonale' = diagonale/chevron (scarico laterale)
    """
    if style == "fine":
        pattern = "".join([f"<rect x='{10+i*10}' y='8' width='3' height='60' rx='1' fill='#aab2bb'/>" for i in range(12)])
    elif style == "onda":
        # archi simmetrici
        arcs = []
        for k in range(8):
            y = 12 + k*7
            arcs.append(f"<path d='M12 {y} Q 80 {y-6} 148 {y} ' stroke='#aab2bb' stroke-width='3' fill='none'/>")
        pattern = "".join(arcs)
    else:  # diagonale / chevron leggero
        lines = []
        for i in range(-10, 16):
            x1 = i*12 + 12
            lines.append(f"<line x1='{x1}' y1='8' x2='{x1+40}' y2='70' stroke='#aab2bb' stroke-width='3'/>")
        pattern = "".join(lines)
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='180' height='80'><rect width='180' height='80' rx='10' fill='#0b1220' stroke='#2a3448'/>" \
          f"{pattern}</svg>"
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()

# ------------------ MODEL ------------------
def compute_snow_temperature(df: pd.DataFrame, dt_hours: float = 1.0) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    for c in ["T2m","cloud","wind","sunup","prp_mmph","prp_type"]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    if "td" not in df.columns:
        df["td"] = float("nan")

    df = df.sort_values("time").reset_index(drop=True)
    rain = df["prp_type"].str.lower().isin(["rain","mixed"])
    snow = df["prp_type"].str.lower().eq("snow")
    sun  = df["sunup"].astype(int) == 1
    tw   = (df["T2m"] + df["td"]) / 2.0

    wet = (rain | (df["T2m"] > 0) | (sun & (df["cloud"] < 0.3) & (df["T2m"] >= -3))
           | (snow & (df["T2m"] >= -1)) | (snow & tw.ge(-0.5).fillna(False)))

    T_surf = pd.Series(index=df.index, dtype=float); T_surf.loc[wet] = 0.0
    dry = ~wet
    clear = (1.0 - df["cloud"]).clip(0,1)
    windc = df["wind"].clip(upper=6.0)
    drad = (1.5 + 3.0*clear - 0.3*windc).clip(0.5,4.5)
    T_surf.loc[dry] = df["T2m"][dry] - drad[dry]

    sunny_cold = sun & dry & df["T2m"].between(-10,0)
    T_surf.loc[sunny_cold] = pd.concat([
        (df["T2m"] + 0.5*(1.0 - df["cloud"]))[sunny_cold],
        pd.Series(-0.5, index=df.index)[sunny_cold]
    ], axis=1).min(axis=1)

    T_top5 = pd.Series(index=df.index, dtype=float)
    tau = pd.Series(6.0, index=df.index, dtype=float)
    tau.loc[rain | snow | (df["wind"] >= 6)] = 3.0
    tau.loc[(~sun) & (df["wind"] < 2) & (df["cloud"] < 0.3)] = 8.0
    alpha = 1.0 - (math.e ** (-dt_hours / tau))

    if len(df) > 0:
        T_top5.iloc[0] = min(df["T2m"].iloc[0], 0.0)
        for i in range(1, len(df)):
            T_top5.iloc[i] = T_top5.iloc[i-1] + alpha.iloc[i]*(T_surf.iloc[i]-T_top5.iloc[i-1])

    df["T_surf"] = T_surf
    df["T_top5"] = T_top5
    return df

# ------------------ WEATHER FETCH ------------------
def prp_type_from_cols(df):
    snow_codes = {71,73,75,77,85,86}; rain_codes = {51,53,55,61,63,65,80,81,82}
    def f(row):
        prp = row.precipitation
        rain = getattr(row,"rain",0.0)
        snow = getattr(row,"snowfall",0.0)
        if prp<=0 or pd.isna(prp): return "none"
        if rain>0 and snow>0: return "mixed"
        if snow>0 and rain==0: return "snow"
        if rain>0 and snow==0: return "rain"
        code = int(getattr(row,"weathercode",0)) if pd.notna(getattr(row,"weathercode",None)) else 0
        if code in snow_codes: return "snow"
        if code in rain_codes: return "rain"
        return "mixed"
    return df.apply(f, axis=1)

@st.cache_data(show_spinner=False)
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

def build_df(js, hours):
    h = js["hourly"]; df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])           # naive
    now = pd.Timestamp.now().floor("H")
    df = df[df["time"] >= now].head(hours).reset_index(drop=True)

    out = pd.DataFrame()
    out["time"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    out["T2m"] = df["temperature_2m"].astype(float)
    out["cloud"] = (df["cloudcover"].astype(float)/100).clip(0,1)
    out["wind"] = (df["windspeed_10m"].astype(float)/3.6).round(3)
    out["sunup"] = df["is_day"].astype(int)
    out["prp_mmph"] = df["precipitation"].astype(float)
    extra = df[["precipitation","rain","snowfall","weathercode"]].copy()
    out["prp_type"] = prp_type_from_cols(extra)
    out["td"] = df["dew_point_2m"].astype(float)
    return out

# ------------------ WAX & TUNING ------------------
# banding non-fluoro, sintetico ma credibile
SWIX  = [("PS5 Turquoise",-18,-10),("PS6 Blue",-12,-6),("PS7 Violet",-8,-2),("PS8 Red",-4,4),("PS10 Yellow",0,10)]
TOKO  = [("Blue",-30,-9),("Red",-12,-4),("Yellow",-6,0)]
VOLA  = [("MX-E Violet/Blue",-12,-4),("MX-E Red",-5,0),("MX-E Warm",-2,10)]
RODE  = [("R20 Blue",-18,-8),("R30 Violet",-10,-3),("R40 Red",-5,0),("R50 Yellow",-1,10)]
HOLM  = [("UltraMix Blue",-18,-10),("UltraMix Red",-8,-2),("UltraMix Yellow",-3,5)]
BMAP  = [("BP1 Cold",-18,-10),("BP2 Med",-10,-2),("BP3 Warm",-3,6)]
STAR  = [("M21 Blue",-18,-8),("M23 Violet",-10,-3),("M25 Red",-5,1)]
DOM   = [("HyperZoom Cold",-20,-8),("HyperZoom Mid",-9,-2),("HyperZoom Warm",-3,6)]

BRANDS = [
    ("Swix"      ,"#ef4444",SWIX),
    ("Toko"      ,"#f59e0b",TOKO),
    ("Vola"      ,"#3b82f6",VOLA),
    ("Rode"      ,"#22c55e",RODE),
    ("Holmenkol" ,"#2563eb",HOLM),
    ("Briko-Maplus","#f97316",BMAP),
    ("STAR"      ,"#e11d48",STAR),
    ("Dominator" ,"#a855f7",DOM),
]

def pick_band(bands, t):
    for name,tmin,tmax in bands:
        if t>=tmin and t<=tmax:
            return name
    return bands[0][0] if t < bands[0][1] else bands[-1][0]

def tune_for(t_surf, discipline):
    # SIDE in gradi (88°, 87.5°…), BASE in gradi
    if t_surf <= -10:
        structure = ("fine","Struttura fine/lineare")
        base = 0.5
        side = {"SL":88.5,"GS":88.0,"SG":87.5,"DH":87.5}
    elif t_surf <= -3:
        structure = ("onda","Universale a onda/corona")
        base = 0.7
        side = {"SL":88.0,"GS":88.0,"SG":87.5,"DH":87.0}
    else:
        structure = ("diagonale","Diagonale/chevron (scarico)")
        base = 0.8 if t_surf <= 0.5 else 1.0
        side = {"SL":88.0,"GS":87.5,"SG":87.0,"DH":87.0}
    return structure, side.get(discipline,88.0), base

# ------------------ SEARCH UI (tipo meteoblue) ------------------
# Evitiamo di scrivere su st.session_state con la stessa chiave del widget:
# usiamo 'q' per input, e 'place_choice' per selectbox. Selezionare un item aggiorna i campi scelti.

st.markdown("#### 1) Cerca località")
col_q, col_geo = st.columns([3,1])
with col_q:
    q = st.text_input("Digita e scegli (aggiorna live)", key="q", placeholder="Es. Champoluc, Cervinia, Sestriere…")
    sugg = _nominatim(q, limit=12)
    labels = [s["label"] for s in sugg]
    place_choice = st.selectbox("Suggerimenti", options=["—"]+labels, index=0, label_visibility="collapsed")
with col_geo:
    if st.button("📍 Usa la mia posizione"):
        lat, lon, city = _ip_geolocate()
        if lat is not None:
            st.session_state["chosen_lat"] = lat
            st.session_state["chosen_lon"] = lon
            st.session_state["chosen_label"] = city or "La tua posizione"
            st.success("Posizione impostata")
        else:
            st.error("Geolocalizzazione non disponibile")

# Applica la scelta se selezionato qualcosa
if place_choice != "—":
    idx = labels.index(place_choice)
    st.session_state["chosen_lat"] = sugg[idx]["lat"]
    st.session_state["chosen_lon"] = sugg[idx]["lon"]
    st.session_state["chosen_label"] = sugg[idx]["label"]

lat  = st.session_state.get("chosen_lat", 45.831)
lon  = st.session_state.get("chosen_lon", 7.730)
name = st.session_state.get("chosen_label", "Champoluc (Ramey)")

c1,c2,c3,c4 = st.columns([1,1,1.2,1.2])
with c1: lat = st.number_input("Lat", value=float(lat), format="%.6f", key="lat_num")
with c2: lon = st.number_input("Lon", value=float(lon), format="%.6f", key="lon_num")
with c3: tzname = st.selectbox("Timezone", ["Europe/Rome","UTC"], index=0)
with c4: hours  = st.slider("Ore previsione", 12, 168, 72, 12)

# ------------------ A/B/C windows ------------------
st.markdown("#### 2) Finestre A · B · C (oggi)")
w1,w2,w3 = st.columns(3)
with w1:
    A_start = st.time_input("Inizio A", time(9,0), key="A_s")
    A_end   = st.time_input("Fine A",   time(11,0), key="A_e")
with w2:
    B_start = st.time_input("Inizio B", time(11,0), key="B_s")
    B_end   = st.time_input("Fine B",   time(13,0), key="B_e")
with w3:
    C_start = st.time_input("Inizio C", time(13,0), key="C_s")
    C_end   = st.time_input("Fine C",   time(16,0), key="C_e")

def window_slice(res, tzname, s, e):
    t = pd.to_datetime(res["time"]).dt.tz_localize(tz.gettz(tzname), nonexistent='shift_forward', ambiguous='NaT')
    D = res.copy(); D["dt"] = t
    today = pd.Timestamp.now(tz=tz.gettz(tzname)).date()
    W = D[(D["dt"].dt.date==today) & (D["dt"].dt.time>=s) & (D["dt"].dt.time<=e)]
    return W if not W.empty else D.head(7)

def plots(res):
    t = pd.to_datetime(res["time"])
    fig1 = plt.figure()
    plt.plot(t, res["T2m"], label="T2m")
    plt.plot(t, res["T_surf"], label="T_surf")
    plt.plot(t, res["T_top5"], label="T_top5")
    plt.legend(); plt.title("Temperature"); plt.xlabel("Ora"); plt.ylabel("°C")
    fig2 = plt.figure()
    plt.bar(t, res["prp_mmph"])
    plt.title("Precipitazione (mm/h)"); plt.xlabel("Ora"); plt.ylabel("mm/h")
    return fig1, fig2

# ------------------ RUN ------------------
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
col_run, col_csv = st.columns([1,2])
with col_run:
    go = st.button("Scarica previsioni per la località", type="primary")
with col_csv:
    upl = st.file_uploader("…oppure carica CSV (time,T2m,cloud,wind,sunup,prp_mmph,prp_type,td)", type=["csv"])

def badge(win):
    tmin = float(win["T_surf"].min()); tmax = float(win["T_surf"].max())
    wet  = bool(((win["prp_type"].isin(["rain","mixed"])) | (win["prp_mmph"]>0.5)).any())
    if wet or tmax > 0.5:
        color,title,desc = "#ef4444","CRITICAL","Possibile neve bagnata/pioggia → struttura grossa"
    elif tmax > -1.0:
        color,title,desc = "#f59e0b","WATCH","Vicino a 0°C → cere medie"
    else:
        color,title,desc = "#22c55e","OK","Neve fredda/asciutta → cere dure"
    st.markdown(f"""
    <div class='card' style='border-color:{color}'>
      <div style='font-weight:800;color:{color};margin-bottom:4px'>{title}</div>
      <div style='opacity:.95'>{desc}</div>
      <div style='font-size:12px;opacity:.7;margin-top:6px'>
        T_surf min {tmin:.1f}°C · max {tmax:.1f}°C
      </div>
    </div>
    """, unsafe_allow_html=True)
    return wet

def wax_cards(t_med):
    cols = st.columns(4)
    for i,(brand,color,bands) in enumerate(BRANDS[:4]):
        rec = pick_band(bands, t_med)
        cols[i].markdown(f"<div class='brand'><img src='{svg_text_logo(brand.upper(),color)}'/>"
                         f"<div><div style='font-size:.8rem;opacity:.85'>{brand}</div>"
                         f"<div style='font-weight:800'>{rec}</div></div></div>", unsafe_allow_html=True)
    cols2 = st.columns(4)
    for i,(brand,color,bands) in enumerate(BRANDS[4:]):
        rec = pick_band(bands, t_med)
        cols2[i].markdown(f"<div class='brand'><img src='{svg_text_logo(brand.upper(),color)}'/>"
                          f"<div><div style='font-size:.8rem;opacity:.85'>{brand}</div>"
                          f"<div style='font-weight:800'>{rec}</div></div></div>", unsafe_allow_html=True)

def structure_card(kind_key, kind_label):
    img = structure_svg(kind_key)
    st.markdown(f"**Struttura consigliata:** {kind_label}")
    st.image(img, use_column_width=False)

def run_all(src_df, place_label):
    res = compute_snow_temperature(src_df, dt_hours=1.0)
    st.success(f"Dati pronti per **{place_label}**")
    st.dataframe(res, use_container_width=True)
    f1,f2 = plots(res); st.pyplot(f1); st.pyplot(f2)
    st.download_button("Scarica CSV risultato", res.to_csv(index=False), "forecast_with_snowT.csv", "text/csv")

    for L,(s,e) in {"A":(A_start,A_end),"B":(B_start,B_end),"C":(C_start,C_end)}.items():
        st.markdown(f"### Blocco {L}")
        W = window_slice(res, tzname, s, e)
        wet = badge(W)
        t_med = float(W["T_surf"].mean())
        st.markdown(f"**T_surf medio {L}: {t_med:.1f} °C**")

        # Wax multi-marca
        wax_cards(t_med)

        # Tuning (struttura + angoli)
        colS, colT = st.columns([1,1])
        with colS:
            # Immagine struttura (realistica in SVG)
            if t_med <= -10:
                structure_card("fine", "Fine / Lineare")
            elif t_med <= -3:
                structure_card("onda", "Universale a onda/corona")
            else:
                structure_card("diagonale", "Diagonale / Chevron (scarico)")
        with colT:
            disc = st.multiselect(f"Discipline (Blocco {L})", ["SL","GS","SG","DH"], default=["SL","GS"], key=f"d_{L}")
            rows=[]
            for d in disc:
                (key,label), side, base = tune_for(t_med, d)
                rows.append([d, label, f"{side:.1f}°", f"{base:.1f}°"])
            if rows:
                st.table(pd.DataFrame(rows, columns=["Disciplina","Struttura soletta","Lamina SIDE (°)","Lamina BASE (°)"]))

# CSV o fetch
if upl is not None:
    try:
        df_u = pd.read_csv(upl)
        run_all(df_u, name)
    except Exception as e:
        st.error(f"CSV non valido: {e}")

if go:
    try:
        js  = fetch_open_meteo(lat, lon, tzname)
        src = build_df(js, hours)
        run_all(src, name)
    except Exception as e:
        st.error(f"Errore: {e}")

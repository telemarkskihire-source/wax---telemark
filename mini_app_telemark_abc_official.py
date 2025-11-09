# telemark_app.py  —  Telemark · Pro Wax & Tune (live search, A/B/C, strutture, lamine)

import streamlit as st
import pandas as pd
import requests
import math
import io
import base64
from datetime import time
from dateutil import tz
import matplotlib.pyplot as plt

# ------------------------ UI / THEME ------------------------
PRIMARY = "#10bfcf"; BG = "#0f172a"; CARD = "#111827"; TEXT = "#e5e7eb"
st.set_page_config(page_title="Telemark · Pro Wax & Tune", page_icon="❄️", layout="wide")
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
  background: linear-gradient(180deg, {BG} 0%, #111827 100%);
}}
.block-container {{ padding-top: .8rem; }}
h1,h2,h3,h4,h5,p,span,div {{ color:{TEXT}; }}
.card {{
  background:{CARD}; border:1px solid rgba(255,255,255,.08);
  border-radius:16px; padding:14px; box-shadow:0 8px 20px rgba(0,0,0,.25);
}}
.kpi {{
  display:flex; gap:8px; align-items:center;
  background:rgba(16,191,207,.06); border:1px dashed rgba(16,191,207,.45);
  padding:10px 12px; border-radius:12px;
}}
.brand {{ display:flex; gap:8px; align-items:center; }}
.brand img {{ height:22px; border-radius:4px }}
.badge {{
  border:1px solid rgba(255,255,255,.15);
  padding:6px 10px; border-radius:999px; font-size:.78rem; opacity:.85;
}}
.btn-primary button {{
  width:100%; background:{PRIMARY} !important; color:#002b30 !important;
  border:none; font-weight:700; border-radius:12px;
}}
</style>
""", unsafe_allow_html=True)

st.markdown("### Telemark · Pro Wax & Tune")
st.markdown("<span class='badge'>Ricerca località tipo Meteoblue · Geolocalizzazione · Blocchi A/B/C · Sciolina + Struttura + Lamine</span>", unsafe_allow_html=True)

# ------------------------ DATA / MODELS ------------------------
def compute_snow_temperature(df: pd.DataFrame, dt_hours=1.0) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    req = ["T2m","cloud","wind","sunup","prp_mmph","prp_type"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

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

# ------------------------ LIVE SEARCH (style Meteoblue) ------------------------
# Idea: 1 campo unico. Ogni battuta → chiamata Nominatim → lista sotto il campo aggiornabile.
# Selezione immediata (no Enter) tramite selectbox sincronizzato con i risultati.
def nominatim(query: str, limit=12):
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
            name = it.get("display_name","")
            out.append({
                "label": name,
                "lat": float(it.get("lat", 0)),
                "lon": float(it.get("lon", 0)),
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

# Stato iniziale pulito
if "place_label" not in st.session_state:
    st.session_state.place_label = "Champoluc (Ramey)"
    st.session_state.place_lat = 45.831
    st.session_state.place_lon = 7.730

cA, cB = st.columns([3,1])
with cA:
    query = st.text_input("Cerca località (scrivi e scegli)", value="", placeholder="Champoluc, Cervinia, Sestriere…", key="q", label_visibility="visible")
    results = nominatim(query, limit=12)
    labels = [r["label"] for r in results] if results else []
    # selectbox che si aggiorna ad ogni battuta (identico flusso a meteoblue)
    selected = st.selectbox("Suggerimenti", options=labels if labels else ["—"], index=0, key="q_select", label_visibility="collapsed")
    if results and selected and selected != "—":
        pick = next((r for r in results if r["label"] == selected), None)
        if pick:
            st.session_state.place_label = pick["label"]
            st.session_state.place_lat = pick["lat"]
            st.session_state.place_lon = pick["lon"]

with cB:
    if st.button("📍 Usa posizione"):
        lat, lon, city = ip_geolocate()
        if lat is not None:
            st.session_state.place_label = city or "La tua posizione"
            st.session_state.place_lat = lat
            st.session_state.place_lon = lon
            st.success("Posizione impostata.")
        else:
            st.error("Geolocalizzazione non disponibile.")

lat = st.session_state.place_lat
lon = st.session_state.place_lon
label = st.session_state.place_label

c1, c2, c3, c4 = st.columns([1,1,1.2,1.2])
with c1: st.number_input("Lat", value=lat, key="lat_num", format="%.6f")
with c2: st.number_input("Lon", value=lon, key="lon_num", format="%.6f")
with c3: tzname = st.selectbox("Timezone", ["Europe/Rome", "UTC"], index=0)
with c4: hours = st.slider("Ore previsione", 12, 168, 72, 12)

# ------------------------ FINSTRE A/B/C ------------------------
st.markdown("#### Finestre orarie (oggi) — A · B · C")
d1,d2,d3 = st.columns(3)
with d1:
    A_start = st.time_input("Inizio A", value=time(9,0));   A_end = st.time_input("Fine A", value=time(11,0))
with d2:
    B_start = st.time_input("Inizio B", value=time(11,0));  B_end = st.time_input("Fine B", value=time(13,0))
with d3:
    C_start = st.time_input("Inizio C", value=time(13,0));  C_end = st.time_input("Fine C", value=time(16,0))

# ------------------------ METEO → DATAFRAME ------------------------
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
    h = js["hourly"]; df = pd.DataFrame(h); df["time"] = pd.to_datetime(df["time"])  # naive
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
    out["prp_type"] = prp_type(extra)
    out["td"] = df["dew_point_2m"].astype(float)
    return out

def window_slice(res, tzname, s, e):
    t = pd.to_datetime(res["time"]).dt.tz_localize(tz.gettz(tzname), nonexistent='shift_forward', ambiguous='NaT')
    D = res.copy(); D["dt"] = t
    today = pd.Timestamp.now(tz=tz.gettz(tzname)).date()
    win = D[(D["dt"].dt.date==today) & (D["dt"].dt.time>=s) & (D["dt"].dt.time<=e)]
    return win if not win.empty else D.head(7)

# ------------------------ WAX & TUNING ------------------------
# Più marchi (non-fluoro): Swix, Toko, Vola, Rode, Holmenkol, Maplus, Start, Skigo
BRANDS = {
    "Swix":      [("PS5 Turquoise",-18,-10),("PS6 Blue",-12,-6),("PS7 Violet",-8,-2),("PS8 Red",-4,4),("PS10 Yellow",0,10)],
    "Toko":      [("Blue",-30,-9),("Red",-12,-4),("Yellow",-6,0)],
    "Vola":      [("MX-E Blue/Violet",-20,-6),("MX-E Red",-5,0),("MX-E Warm",-2,10)],
    "Rode":      [("R20 Blue",-18,-8),("R30 Violet",-10,-3),("R40 Red",-5,0),("R50 Yellow",-1,10)],
    "Holmenkol": [("UltraMix Blue",-20,-8),("BetaMix Red",-14,-4),("AlphaMix Yellow",-4,0)],
    "Maplus":    [("Universal Cold",-12,-6),("Universal Med.",-7,-2),("Universal Soft",-5,0)],
    "Start":     [("SG Blue",-12,-6),("SG Purple",-8,-2),("SG Red",-3,7)],
    "Skigo":     [("Blue",-12,-6),("Violet",-8,-2),("Red",-5,1)],
}
def pick_wax(bands, t):
    for name,tmin,tmax in bands:
        if t>=tmin and t<=tmax: return name
    return bands[-1][0] if t>bands[-1][2] else bands[0][0]

def tune_for(t_surf, discipline):
    # side = 88/87.5/87 in gradi (richiesto); base 0.5–1.0°
    if t_surf <= -10:
        structure = "Lineare fine (freddo)"
        base = 0.5
        side_map = {"SL":88.5, "GS":88.0, "SG":87.5, "DH":87.5}
    elif t_surf <= -3:
        structure = "Universale media"
        base = 0.7
        side_map = {"SL":88.0, "GS":88.0, "SG":87.5, "DH":87.0}
    else:
        structure = "Diagonale/onda (umido)"
        base = 0.8 if t_surf<=0.5 else 1.0
        side_map = {"SL":88.0, "GS":87.5, "SG":87.0, "DH":87.0}
    return structure, side_map.get(discipline,88.0), base

# ------------------------ STRUTTURE — disegni puliti stile manuale ------------------------
def structure_image(kind: str, width=540, height=160, lines=28) -> bytes:
    """
    kind: 'lineare', 'onda', 'diagonale'
    ritorna PNG bytes
    """
    fig = plt.figure(figsize=(width/120, height/120), dpi=120)
    ax = plt.gca()
    ax.set_facecolor("#0e141f")
    ax.axis("off")

    if kind == "lineare":
        # linee verticali sottili, equidistanti
        for i in range(lines):
            x = (i+1)/(lines+1)
            ax.plot([x,x],[0.07,0.93], linewidth=1.6, solid_capstyle="round")
    elif kind == "onda":
        import numpy as np
        xs = np.linspace(0.05,0.95,lines)
        y0 = 0.5; amp = 0.35
        t = np.linspace(0, 1, 220)
        for x in xs:
            y = y0 + amp*(1 - (abs(t-0.5)/0.5))  # archi regolari
            ax.plot(t, y, linewidth=1.8)
    else:  # diagonale scarico laterale
        step = 0.03
        x = -0.2
        while x < 1.2:
            ax.plot([x, x+0.6], [0.05, 0.95], linewidth=1.8)
            x += step

    buf = io.BytesIO()
    plt.tight_layout(pad=0.2)
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    return buf.getvalue()

def img_tag(png_bytes, w=260):
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"<img src='data:image/png;base64,{b64}' width='{w}'/>"

# ------------------------ RUN / OUTPUT ------------------------
cR, cU = st.columns([1,2])
with cR:
    go = st.button("Scarica previsioni per la località", type="primary")
with cU:
    upl = st.file_uploader("…oppure carica CSV (time,T2m,cloud,wind,sunup,prp_mmph,prp_type,td)", type=["csv"])

def badge(win: pd.DataFrame):
    tmin = float(win["T_surf"].min()); tmax = float(win["T_surf"].max())
    wet = bool(((win["prp_type"].isin(["rain","mixed"])) | (win["prp_mmph"]>0.5)).any())
    if wet or tmax > 0.5:
        color, title, desc = "#ef4444", "CRITICAL", "Possibile bagnato/pioggia · struttura grossa"
    elif tmax > -1.0:
        color, title, desc = "#f59e0b", "WATCH", "Vicino a 0°C · cere medio-morbide"
    else:
        color, title, desc = "#22c55e", "OK", "Freddo/asciutto · cere dure"
    st.markdown(
        f"<div class='card' style='border-color:{color}'>"
        f"<div style='font-weight:800;color:{color};margin-bottom:4px'>{title}</div>"
        f"<div style='opacity:.95'>{desc}</div>"
        f"<div style='font-size:12px;opacity:.7;margin-top:6px'>T_surf min {tmin:.1f}°C / max {tmax:.1f}°C</div>"
        f"</div>", unsafe_allow_html=True)
    return wet

def plot_series(res: pd.DataFrame):
    t = pd.to_datetime(res["time"])
    fig1 = plt.figure(); plt.plot(t,res["T2m"],label="T2m"); plt.plot(t,res["T_surf"],label="T_surf"); plt.plot(t,res["T_top5"],label="T_top5")
    plt.legend(); plt.title("Temperature"); plt.xlabel("Ora"); plt.ylabel("°C"); st.pyplot(fig1)
    fig2 = plt.figure(); plt.bar(t,res["prp_mmph"]); plt.title("Precipitazione (mm/h)"); plt.xlabel("Ora"); plt.ylabel("mm/h"); st.pyplot(fig2)

def wax_row(t_med):
    cols = st.columns(4)
    brands_list = list(BRANDS.items())[:4]
    for i,(brand,bands) in enumerate(brands_list):
        rec = pick_wax(bands, t_med)
        cols[i].markdown(f"<div class='brand'><img src='https://dummyimage.com/88x22/3b3b3b/ffffff.png&text={brand}'/><div><div style='font-size:.8rem;opacity:.8'>{brand}</div><div style='font-weight:800'>{rec}</div></div></div>", unsafe_allow_html=True)
    cols2 = st.columns(4)
    for i,(brand,bands) in enumerate(list(BRANDS.items())[4:8]):
        rec = pick_wax(bands, t_med)
        cols2[i].markdown(f"<div class='brand'><img src='https://dummyimage.com/88x22/3b3b3b/ffffff.png&text={brand}'/><div><div style='font-size:.8rem;opacity:.8'>{brand}</div><div style='font-weight:800'>{rec}</div></div></div>", unsafe_allow_html=True)

def structure_gallery(structure_name):
    if "lineare" in structure_name.lower():
        img = structure_image("lineare")
    elif "onda" in structure_name.lower():
        img = structure_image("onda")
    else:
        img = structure_image("diagonale")
    st.markdown(img_tag(img, w=420), unsafe_allow_html=True)

def run_all(src_df: pd.DataFrame, label: str):
    res = compute_snow_temperature(src_df, dt_hours=1.0)
    st.success(f"Previsioni per **{label}** pronte.")
    st.dataframe(res, use_container_width=True)
    plot_series(res)
    st.download_button("Scarica CSV risultato", data=res.to_csv(index=False), file_name="forecast_with_snowT.csv", mime="text/csv")

    blocks = {"A":(A_start,A_end), "B":(B_start,B_end), "C":(C_start,C_end)}
    for L,(s,e) in blocks.items():
        st.markdown(f"### Blocco {L}")
        W = window_slice(res, tzname, s, e)
        wet = badge(W)
        t_med = float(W["T_surf"].mean())
        st.markdown(f"**T_surf medio {L}: {t_med:.1f} °C**")

        # Wax per tutti i brand
        wax_row(t_med)

        # Tuning (lamine & struttura) per discipline
        disc = st.multiselect(f"Discipline (Blocco {L})", ["SL","GS","SG","DH"], default=["SL","GS"], key=f"d_{L}")
        rows = []
        for d in disc:
            structure, side, base = tune_for(t_med, d)
            rows.append([d, structure, f"{side:.1f}°", f"{base:.1f}°"])
        if rows:
            df_tune = pd.DataFrame(rows, columns=["Disciplina","Struttura consigliata","Lamina SIDE (°)","Lamina BASE (°)"])
            st.table(df_tune)
            # Anteprima struttura “alla Wintersteiger”
            structure_gallery(rows[0][1])

# Input
if upl is not None:
    try:
        user_df = pd.read_csv(upl)
        run_all(user_df, label)
    except Exception as e:
        st.error(f"CSV non valido: {e}")

if go:
    try:
        js = fetch_open_meteo(st.session_state.lat_num, st.session_state.lon_num, tzname)
        src = build_df(js, hours, tzname)
        run_all(src, label)
    except Exception as e:
        st.error(f"Errore: {e}")

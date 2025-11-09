# app.py — Telemark · Pro Wax & Tune (mobile-first, meteoblue-like search)

import streamlit as st
import pandas as pd
import requests
import base64
import io
import matplotlib.pyplot as plt
from datetime import time
from dateutil import tz

# ========== PAGE & THEME ==========
PRIMARY = "#10bfcf"; BG = "#0f172a"; CARD = "#111827"; TEXT = "#e5e7eb"
st.set_page_config(page_title="Telemark · Pro Wax & Tune", page_icon="❄️", layout="wide")
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
  background: linear-gradient(180deg, {BG} 0%, #111827 100%);
}}
.block-container {{ padding-top: 0.8rem; }}
h1,h2,h3,h4,h5,p,span,div {{ color:{TEXT}; }}
.badge {{
  border:1px solid rgba(255,255,255,.15);
  padding:6px 10px; border-radius:999px; font-size:.78rem; opacity:.9;
}}
.card {{
  background:{CARD}; border:1px solid rgba(255,255,255,.08);
  border-radius:16px; padding:14px; box-shadow: 0 8px 20px rgba(0,0,0,.25);
}}
.brand {{
  display:flex; align-items:center; gap:8px; padding:10px 12px; border-radius:12px;
  background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08);
}}
.brand img {{ height:22px; }}
.kpi {{
  display:flex; flex-direction:column; gap:2px; border-radius:12px;
  background:rgba(16,191,207,.06); border:1px dashed rgba(16,191,207,.45); padding:10px 12px;
}}
.kpi .lab {{ font-size:.78rem; color:#93c5fd; }}
.kpi .val {{ font-size:1rem; font-weight:800; color:{TEXT}; }}
.hr {{ height:1px; background:rgba(255,255,255,.12); margin:8px 0; }}
.sugg-item {{
  padding:8px 10px; border-radius:10px; border:1px solid rgba(255,255,255,.12);
  margin-top:6px; background:rgba(255,255,255,.03); cursor:pointer;
}}
.sugg-item:hover {{ background:rgba(255,255,255,.06); }}
input, select, textarea {{ border-radius:12px !important; }}
</style>
""", unsafe_allow_html=True)

st.markdown("### Telemark · Pro Wax & Tune")
st.markdown("<span class='badge'>Ricerca stile Meteoblue · Geolocalizzazione · Blocchi A/B/C · Wax + Struttura + Lamine</span>", unsafe_allow_html=True)

# ========== MODEL: snow temperature ==========
def compute_snow_temperature(df, dt_hours=1.0):
    """Heuristic surface & top-5cm snow temperature model."""
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

# ========== LOCATION SEARCH (meteoblue-like) ==========
# Debounced, live suggestions on each keystroke; suggestions under input; click to select.
def geocode_suggest(q, limit=8):
    if not q or len(q.strip()) < 2:
        return []
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": q, "format": "json", "limit": limit, "addressdetails": 1},
            headers={"User-Agent": "telemark-pro-wax/1.0"},
            timeout=8
        )
        r.raise_for_status()
        js = r.json()
        out = []
        for it in js:
            out.append({
                "label": it.get("display_name",""),
                "lat": float(it.get("lat", 0)),
                "lon": float(it.get("lon", 0))
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

# Live input
col_search, col_geo = st.columns([3,1])
with col_search:
    q = st.text_input(
        "Cerca località (digita e scegli)",
        value=st.session_state.get("q",""),
        placeholder="Es. Champoluc, Cervinia, Sestriere…",
        key="q_input"
    )
    # update suggestions at each rerun/keystroke
    sugg = geocode_suggest(st.session_state.get("q_input",""), limit=10)
    if sugg:
        st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
        st.caption("Suggerimenti")
        for i, s in enumerate(sugg):
            if st.button(s["label"], key=f"sugg_{i}"):
                st.session_state["chosen_lat"] = s["lat"]
                st.session_state["chosen_lon"] = s["lon"]
                st.session_state["chosen_label"] = s["label"]
                st.session_state["q"] = s["label"]
                st.session_state["q_input"] = s["label"]
                st.experimental_rerun()

with col_geo:
    if st.button("📍 Usa la mia posizione"):
        lat0, lon0, city = ip_geolocate()
        if lat0 is not None:
            st.session_state["chosen_lat"] = lat0
            st.session_state["chosen_lon"] = lon0
            st.session_state["chosen_label"] = city or "La tua posizione"
            st.session_state["q"] = city or "La tua posizione"
            st.session_state["q_input"] = city or "La tua posizione"
            st.success("Posizione impostata.")
        else:
            st.error("Geolocalizzazione non disponibile.")

# Current selection (defaults Champoluc)
lat = st.session_state.get("chosen_lat", 45.831)
lon = st.session_state.get("chosen_lon", 7.730)
label = st.session_state.get("chosen_label", "Champoluc (Ramey)")

c0, c1, c2, c3 = st.columns([2,1,1,1.2])
with c0: st.text_input("Località selezionata", value=label, key="label_show")
with c1: lat = st.number_input("Lat", value=float(lat), format="%.6f")
with c2: lon = st.number_input("Lon", value=float(lon), format="%.6f")
with c3: tzname = st.selectbox("Timezone", ["Europe/Rome","UTC"], index=0)
hours = st.slider("Ore previsione", 12, 168, 72, 12)

# A/B/C blocks (oggi)
st.markdown("#### Finestre A · B · C (oggi)")
cA, cB, cC = st.columns(3)
with cA:
    A_start = st.time_input("Inizio A", value=time(9,0), key="A_s")
    A_end   = st.time_input("Fine A",   value=time(11,0), key="A_e")
with cB:
    B_start = st.time_input("Inizio B", value=time(11,0), key="B_s")
    B_end   = st.time_input("Fine B",   value=time(13,0), key="B_e")
with cC:
    C_start = st.time_input("Inizio C", value=time(13,0), key="C_s")
    C_end   = st.time_input("Fine C",   value=time(16,0), key="C_e")

# ========== DATA FETCH ==========
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

def build_df(js, hours, tzname):
    # Keep all timestamps naive to avoid tz-aware comparison errors
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
    t = pd.to_datetime(res["time"]).dt.tz_localize(tz.gettz(tzname), nonexistent='shift_forward', ambiguous='NaT')
    D = res.copy(); D["dt"] = t
    today = pd.Timestamp.now(tz=tz.gettz(tzname)).date()
    W = D[(D["dt"].dt.date==today) & (D["dt"].dt.time>=s) & (D["dt"].dt.time<=e)]
    return W if not W.empty else D.head(7)

# ========== WAX BANDS (8 brands) ==========
SWIX  = [("PS5 Turq",-18,-10), ("PS6 Blue",-12,-6), ("PS7 Violet",-8,-2), ("PS8 Red",-4,4), ("PS10 Yellow",0,10)]
TOKO  = [("Blue",-30,-9),("Red",-12,-4),("Yellow",-6,0)]
VOLA  = [("MX-E Violet/Blue",-12,-4),("MX-E Red",-5,0),("MX-E Warm",-2,10)]
RODE  = [("R20 Blue",-18,-8),("R30 Violet",-10,-3),("R40 Red",-5,0),("R50 Yellow",-1,10)]
HOLM  = [("UltraMix Blue",-20,-8),("BetaMix Red",-14,-4),("AlphaMix Yellow",-4,5)]
MAPL  = [("Univ Cold",-12,-6),("Univ Medium",-7,-2),("Univ Warm",-3,6)]
START = [("SG Blue",-12,-6),("SG Purple",-8,-2),("SG Red",-3,7)]
SKIGO = [("Blue",-12,-6),("Violet",-8,-2),("Red",-3,2)]
BRANDS = [("Swix",SWIX),("Toko",TOKO),("Vola",VOLA),("Rode",RODE),("Holmenkol",HOLM),("Maplus",MAPL),("Start",START),("Skigo",SKIGO)]

def pick_band(bands, t):
    for name,tmin,tmax in bands:
        if t>=tmin and t<=tmax:
            return name
    return bands[-1][0] if t>bands[-1][2] else bands[0][0]

def svg_logo(text, color):
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='200' height='36'><rect width='200' height='36' rx='6' fill='{color}'/><text x='12' y='24' font-size='16' font-weight='700' fill='white'>{text}</text></svg>"
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")

LOGOS = {
    "Swix": svg_logo("SWIX","#ef4444"),
    "Toko": svg_logo("TOKO","#f59e0b"),
    "Vola": svg_logo("VOLA","#3b82f6"),
    "Rode": svg_logo("RODE","#22c55e"),
    "Holmenkol": svg_logo("HOLMENKOL","#06b6d4"),
    "Maplus": svg_logo("MAPLUS","#ea580c"),
    "Start": svg_logo("START","#d946ef"),
    "Skigo": svg_logo("SKIGO","#84cc16"),
}

# ========== STRUCTURE & EDGES ==========
def tune_for(t_surf, discipline):
    """Return (structure_name, side_deg (e.g., 88.0), base_deg)."""
    if t_surf <= -10:
        structure = "Fine lineare"
        base = 0.5
        side = {"SL": 88.5, "GS": 88.0, "SG": 87.5, "DH": 87.5}.get(discipline, 88.0)
    elif t_surf <= -3:
        structure = "Media universale"
        base = 0.7
        side = {"SL": 88.0, "GS": 88.0, "SG": 87.5, "DH": 87.0}.get(discipline, 88.0)
    else:
        structure = "Media-grossa (onda o scarico)"
        base = 0.8 if t_surf <= 0.5 else 1.0
        side = {"SL": 88.0, "GS": 87.5, "SG": 87.0, "DH": 87.0}.get(discipline, 87.5)
    return structure, side, base

def draw_structure(kind="lineare"):
    """Return PNG bytes with a schematic of base structure."""
    fig = plt.figure(figsize=(4,1.0), dpi=200)
    ax = plt.gca()
    ax.set_axis_off()
    ax.set_xlim(0,100); ax.set_ylim(-10,10)

    import numpy as np
    x = np.linspace(0,100,500)

    if kind == "lineare":
        for y in [-6,-2,2,6]:
            ax.plot([0,100],[y,y], linewidth=1)
    elif kind == "onda":
        for y0 in [-6,-2,2,6]:
            ax.plot(x, y0+2*np.sin(0.25*x), linewidth=1)
    elif kind == "chevron":
        for phase in [0,10,20,30]:
            y = ( (x+phase)%10 ) - 5
            ax.plot(x, y/2, linewidth=1)
    elif kind == "cross":
        for y in [-6,-2,2,6]:
            ax.plot([0,100],[y,y], linewidth=1)
        for xi in range(0,100,10):
            ax.plot([xi,xi+10],[ -8,8], linewidth=0.8)
    elif kind == "scarico laterale":
        # grooves denser on one side
        for i, xi in enumerate(range(0,100,6)):
            ax.plot([xi,xi],[ -8 + (i%4), 8 ], linewidth=1)
    else:
        for y in [-6,-2,2,6]:
            ax.plot([0,100],[y,y], linewidth=1)

    buf = io.BytesIO()
    fig.tight_layout(pad=0)
    plt.savefig(buf, format="png", transparent=True, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def structure_kind_from_temp(t_surf):
    if t_surf <= -10:
        return "lineare"
    elif t_surf <= -3:
        return "lineare"
    else:
        return "onda"  # o scarico laterale in caso di bagnato

# ========== UI Actions ==========
col_run1, col_run2 = st.columns([1,2])
with col_run1:
    fetch_btn = st.button("Scarica previsioni", type="primary")
with col_run2:
    upl = st.file_uploader("…oppure carica CSV (time,T2m,cloud,wind,sunup,prp_mmph,prp_type,td)", type=["csv"])

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

def badge(win):
    tmin = float(win["T_surf"].min()); tmax = float(win["T_surf"].max())
    wet = bool(((win["prp_type"].isin(["rain","mixed"])) | (win["prp_mmph"]>0.5)).any())
    if wet or tmax > 0.5:
        color, title, desc = "#ef4444", "CRITICAL", "Possibile neve bagnata/pioggia · struttura grossa/scarico"
    elif tmax > -1.0:
        color, title, desc = "#f59e0b", "WATCH", "Vicino 0°C · cere medio-morbide"
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

def wax_cards(t_med):
    rows = st.columns(4)
    rows2 = st.columns(4)
    all_cols = list(rows) + list(rows2)
    for i,(brand,bands) in enumerate(BRANDS):
        rec = pick_band(bands, t_med)
        logo = LOGOS.get(brand)
        all_cols[i].markdown(
            f"<div class='brand'><img src='{logo}'/><div><div style='font-size:.8rem;opacity:.85'>{brand}</div><div style='font-weight:800'>{rec}</div></div></div>",
            unsafe_allow_html=True
        )

def run_all(src, spot_label):
    res = compute_snow_temperature(src, dt_hours=1.0)
    st.success(f"Previsioni per **{spot_label}** pronte.")
    st.dataframe(res, use_container_width=True)
    f1, f2 = plots(res); st.pyplot(f1); st.pyplot(f2)
    st.download_button("Scarica CSV risultato", data=res.to_csv(index=False), file_name="forecast_with_snowT.csv", mime="text/csv")

    for L,(s,e) in {"A":(A_start,A_end),"B":(B_start,B_end),"C":(C_start,C_end)}.items():
        st.markdown(f"### Blocco {L}")
        W = window_slice(res, tzname, s, e)
        wet = badge(W)
        t_med = float(W["T_surf"].mean())
        st.markdown(f"**T_surf medio {L}: {t_med:.1f} °C**")

        # Wax
        wax_cards(t_med)

        # Structure preview + edges
        kind = structure_kind_from_temp(t_med)
        if wet and t_med > -2:
            kind = "scarico laterale"
        st.caption(f"Struttura suggerita: **{kind}**")
        img = draw_structure(kind)
        st.image(img, caption=f"Anteprima struttura: {kind}", use_column_width=True)

        # Discipline tuning
        disc = st.multiselect(f"Discipline (Blocco {L})", ["SL","GS","SG","DH"], default=["SL","GS"], key=f"d_{L}")
        rows = []
        for d in disc:
            structure, side, base = tune_for(t_med, d)
            rows.append([d, structure, f"{side:.1f}°", f"{base:.1f}°"])
        if rows:
            df_tune = pd.DataFrame(rows, columns=["Disciplina","Struttura soletta","Lamina SIDE (°)","Lamina BASE (°)"])
            st.table(df_tune)

# SOURCE: CSV or Open-Meteo
if upl is not None:
    try:
        df_u = pd.read_csv(upl)
        run_all(df_u, st.session_state.get("q_input", label))
    except Exception as e:
        st.error(f"CSV non valido: {e}")
elif fetch_btn:
    try:
        js = fetch_open_meteo(lat, lon, tzname)
        df_src = build_df(js, hours, tzname)
        run_all(df_src, st.session_state.get("q_input", label))
    except Exception as e:
        st.error(f"Errore: {e}")
else:
    st.info("Cerca una località (si aggiorna mentre scrivi) → seleziona dalla lista → **Scarica previsioni**.")

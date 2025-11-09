import streamlit as st
import pandas as pd
import requests, base64, math
import matplotlib.pyplot as plt
from datetime import time
from dateutil import tz

# ========== MODELLO NEVE ==========
def compute_snow_temperature(df, dt_hours=1.0):
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
    T_surf.loc[dry] = df.loc[dry, "T2m"] - drad[dry]
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

# ========== TEMA ==========
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
  padding:6px 10px; border-radius:999px; font-size:.78rem; opacity:.85;
}}
.card {{
  background:{CARD}; border:1px solid rgba(255,255,255,.08);
  border-radius:16px; padding:14px; box-shadow:0 8px 20px rgba(0,0,0,.25);
}}
.brand {{ display:flex; gap:10px; align-items:center; padding:10px 12px;
  border-radius:12px; background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08); }}
.brand img {{ height:22px; }}
hr {{ border-color: rgba(255,255,255,.1) }}
.sugg-btn button {{ width:100%; text-align:left; }}
</style>
""", unsafe_allow_html=True)

st.markdown("### Telemark · Pro Wax & Tune")
st.markdown("<span class='badge'>Ricerca tipo Meteoblue · Geolocalizzazione · Blocchi A/B/C · Sciolina + Struttura + Lamine</span>", unsafe_allow_html=True)

# ========== RICERCA LOCALITÀ (stile Meteoblue: auto-suggest continuo) ==========
@st.cache_data(show_spinner=False)
def geo_search(q: str, limit: int = 8):
    """Usa l'API Geocoding di Open-Meteo (supporta fuzzy search e parziali)."""
    if not q or len(q) < 2:
        return []
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": q, "count": limit, "language": "it", "format": "json"},
            timeout=8
        )
        r.raise_for_status()
        js = r.json()
        out = []
        for it in js.get("results", []):
            label = f"{it.get('name','')} ({it.get('country_code','')})"
            if it.get("admin1"): label = f"{it['name']}, {it['admin1']} ({it.get('country_code','')})"
            out.append({"label": label, "lat": it["latitude"], "lon": it["longitude"]})
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

# Barra di ricerca che aggiorna i suggerimenti ad ogni battuta
c1, c2 = st.columns([3,1])
with c1:
    q = st.text_input("Cerca località", placeholder="Digita: Champoluc, Cervinia, Sestriere…", key="q", help="Suggerimenti dinamici, seleziona dalla lista.")
    suggestions = geo_search(q, limit=10)
    if suggestions:
        st.caption("Suggerimenti")
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(s["label"], key=f"sugg_{i}", use_container_width=True):
                    st.session_state["lat"] = s["lat"]
                    st.session_state["lon"] = s["lon"]
                    st.session_state["place_label"] = s["label"]

with c2:
    if st.button("📍 Geolocalizza", use_container_width=True):
        la, lo, city = ip_geolocate()
        if la is not None:
            st.session_state["lat"] = la; st.session_state["lon"] = lo
            st.session_state["place_label"] = city or "La tua posizione"
            st.success("Posizione impostata")
        else:
            st.error("Geolocalizzazione non disponibile")

lat = st.session_state.get("lat", 45.831)
lon = st.session_state.get("lon", 7.730)
place_label = st.session_state.get("place_label", "Champoluc (AO)")

c3, c4, c5 = st.columns([1,1,2])
with c3: lat = st.number_input("Lat", value=float(lat), format="%.6f", key="lat_num")
with c4: lon = st.number_input("Lon", value=float(lon), format="%.6f", key="lon_num")
with c5: tzname = st.selectbox("Timezone", ["Europe/Rome","UTC"], index=0)

hours = st.slider("Ore previsione", 12, 168, 72, 12)

# ========== FINSTRE A/B/C ==========
st.markdown("#### Finestre A · B · C (oggi)")
a1,a2,a3 = st.columns(3)
with a1:
    A_start = st.time_input("Inizio A", value=time(9,0), key="A_s")
    A_end   = st.time_input("Fine A",   value=time(11,0), key="A_e")
with a2:
    B_start = st.time_input("Inizio B", value=time(11,0), key="B_s")
    B_end   = st.time_input("Fine B",   value=time(13,0), key="B_e")
with a3:
    C_start = st.time_input("Inizio C", value=time(13,0), key="C_s")
    C_end   = st.time_input("Fine C",   value=time(16,0), key="C_e")

# ========== FETCH PREVISIONI ==========
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

def prp_type_from(js_hourly):
    df = pd.DataFrame(js_hourly)
    snow_codes = {71,73,75,77,85,86}; rain_codes = {51,53,55,61,63,65,80,81,82}
    def f(row):
        prp = row["precipitation"]; rain = row.get("rain",0.0); snow = row.get("snowfall",0.0)
        if prp<=0 or pd.isna(prp): return "none"
        if rain>0 and snow>0: return "mixed"
        if snow>0 and rain==0: return "snow"
        if rain>0 and snow==0: return "rain"
        code = int(row.get("weathercode",0)) if pd.notna(row.get("weathercode",None)) else 0
        if code in snow_codes: return "snow"
        if code in rain_codes: return "rain"
        return "mixed"
    return df.apply(f, axis=1)

def build_df(js, hours, tzname):
    h = js["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])          # naive per evitare tz-mix
    now_naive = pd.Timestamp.now().floor("H")
    df = df[df["time"] >= now_naive].head(hours).reset_index(drop=True)

    out = pd.DataFrame()
    out["time"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    out["T2m"] = df["temperature_2m"].astype(float)
    out["cloud"] = (df["cloudcover"].astype(float)/100).clip(0,1)
    out["wind"] = (df["windspeed_10m"].astype(float)/3.6).round(3)
    out["sunup"] = df["is_day"].astype(int)
    out["prp_mmph"] = df["precipitation"].astype(float)
    out["prp_type"] = prp_type_from(h)
    out["td"] = df["dew_point_2m"].astype(float)
    return out

# ========== SCIOLINA: 8 MARCHI ==========
SWIX = [("PS5 Turquoise", -18,-10), ("PS6 Blue",-12,-6), ("PS7 Violet",-8,-2), ("PS8 Red",-4,4), ("PS10 Yellow",0,10)]
TOKO = [("Blue",-30,-9), ("Red",-12,-4), ("Yellow",-6,0)]
VOLA = [("MX-E Blue/Violet",-12,-4), ("MX-E Red",-5,0), ("MX-E Warm",-2,10)]
RODE = [("R20 Blue",-18,-8), ("R30 Violet",-10,-3), ("R40 Red",-5,0), ("R50 Yellow",-1,10)]
HOLM = [("UltraMix Blue",-20,-8), ("BetaMix Red",-14,-4), ("AlphaMix Yellow",-4,5)]
MAPL = [("Universal Cold",-12,-6), ("Universal Medium",-7,-2), ("Universal Warm",-3,6)]
STAR = [("SG Blue",-12,-6), ("SG Purple",-8,-2), ("SG Red",-3,7)]
SKIG = [("Paraffin Blue",-12,-6), ("Paraffin Violet",-8,-2), ("Paraffin Red",-3,2)]

def pick(bands, t):
    for name,tmin,tmax in bands:
        if t >= tmin and t <= tmax:
            return name
    return bands[-1][0] if t > bands[-1][2] else bands[0][0]

def brand_svg(text, color):
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='160' height='30'><rect width='160' height='30' rx='6' fill='{color}'/><text x='10' y='20' font-size='14' font-weight='700' fill='white'>{text}</text></svg>"
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()

BRANDS = [
    ("Swix", SWIX, "#ef4444"),
    ("Toko", TOKO, "#f59e0b"),
    ("Vola", VOLA, "#3b82f6"),
    ("Rode", RODE, "#22c55e"),
    ("Holmenkol", HOLM, "#06b6d4"),
    ("Maplus", MAPL, "#8b5cf6"),
    ("Start", STAR, "#f97316"),
    ("Skigo", SKIG, "#10b981"),
]

# ========== TUNING (struttura + lamine) ==========
def tune_for(t_surf, discipline):
    # SIDE espresso come 88°, 87.5°, 87°; BASE 0.5°-1.0°
    if t_surf <= -10:
        structure = "FINE lineare"
        base = 0.5
        side_map = {"SL": 88.5, "GS": 88.0, "SG": 87.5, "DH": 87.5}
    elif t_surf <= -3:
        structure = "MEDIA universale"
        base = 0.7
        side_map = {"SL": 88.0, "GS": 88.0, "SG": 87.5, "DH": 87.0}
    else:
        structure = "MEDIA-GROSSA (a onda/chevron)"
        base = 0.8 if t_surf <= 0.5 else 1.0
        side_map = {"SL": 88.0, "GS": 87.5, "SG": 87.0, "DH": 87.0}
    return structure, side_map.get(discipline, 88.0), base

def draw_structure(kind: str):
    """
    Rende una mini-anteprima chiara:
    - 'FINE lineare'    : righe molto fitte verticali
    - 'MEDIA universale': righe + leggere diagonali (universale)
    - 'MEDIA-GROSSA...' : pattern a onde/chevron (scarico laterale evidente)
    """
    fig = plt.figure()
    ax = plt.gca()
    ax.set_facecolor("white")
    ax.set_xlim(0,100); ax.set_ylim(0,60)
    ax.axis("off")
    if "FINE" in kind:
        # lineare fine
        for x in range(5, 100, 4):
            ax.plot([x,x],[5,55])
    elif "universale" in kind:
        # lineare + diagonali leggere
        for x in range(5, 100, 6):
            ax.plot([x,x],[5,55])
        for x in range(-40, 100, 12):
            ax.plot([x, x+50], [5, 55], alpha=0.5)
    else:
        # onde/chevron (scarico laterale)
        import numpy as np
        for y in range(8, 56, 10):
            xs = np.linspace(5,95,200)
            ys = y + 6*np.sin(xs/6)
            ax.plot(xs, ys, linewidth=1.8)
        # “canali” laterali marcati
        ax.plot([8,8],[5,55], linewidth=2.4)
        ax.plot([92,92],[5,55], linewidth=2.4)
    return fig

# ========== PULSANTI ==========
bL, bR = st.columns([1,2])
go = bL.button("Scarica previsioni per la località", use_container_width=True)
upl = bR.file_uploader("…oppure carica CSV (time,T2m,cloud,wind,sunup,prp_mmph,prp_type,td opz.)", type=["csv"])

# ========== PIPELINE ==========
def window_slice(res, tzname, s, e):
    t = pd.to_datetime(res["time"]).dt.tz_localize(tz.gettz(tzname), nonexistent='shift_forward', ambiguous='NaT')
    D = res.copy(); D["dt"] = t
    today = pd.Timestamp.now(tz=tz.gettz(tzname)).date()
    win = D[(D["dt"].dt.date==today) & (D["dt"].dt.time>=s) & (D["dt"].dt.time<=e)]
    return win if not win.empty else D.head(7)

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

def show_wax_cards(t_med):
    cols = st.columns(4)
    for i,(name,bands,color) in enumerate(BRANDS):
        rec = pick(bands, t_med)
        logo = brand_svg(name.upper(), color)
        cols[i%4].markdown(
            f"<div class='brand'><img src='{logo}'/>"
            f"<div><div style='font-size:.8rem;opacity:.8'>{name}</div>"
            f"<div style='font-weight:800'>{rec}</div></div></div>",
            unsafe_allow_html=True
        )

def show_block(label, W, tzname):
    st.markdown(f"### Blocco {label}")
    tmin = float(W["T_surf"].min()); tmax = float(W["T_surf"].max())
    wet = bool(((W["prp_type"].isin(["rain","mixed"])) | (W["prp_mmph"]>0.5)).any())
    color, status, desc = ("#22c55e","OK","Neve fredda/asciutta · cere dure")
    if wet or tmax > 0.5:
        color, status, desc = ("#ef4444","CRITICAL","Possibile neve bagnata/pioggia · struttura grossa")
    elif tmax > -1.0:
        color, status, desc = ("#f59e0b","WATCH","Vicino a 0°C · cere medio-morbide")

    st.markdown(f"""
    <div class='card' style='border-color:{color}'>
      <div style='font-weight:800;color:{color};margin-bottom:4px'>{status}</div>
      <div style='opacity:.95'>{desc}</div>
      <div style='font-size:12px;opacity:.7;margin-top:6px'>
        T_surf min {tmin:.1f}°C / max {tmax:.1f}°C
      </div>
    </div>
    """, unsafe_allow_html=True)

    t_med = float(W["T_surf"].mean())
    st.markdown(f"**T_surf medio {label}: {t_med:.1f}°C**")
    show_wax_cards(t_med)

    # Tuning discipline + anteprima struttura
    disc = st.multiselect(f"Discipline (Blocco {label})", ["SL","GS","SG","DH"], default=["SL","GS"], key=f"disc_{label}")
    if disc:
        rows = []
        # struttura consigliata in base a t_med
        structure, _, _ = tune_for(t_med, "GS")
        fig = draw_structure(structure)
        st.pyplot(fig)

        for d in disc:
            structure, side, base = tune_for(t_med, d)
            rows.append([d, structure, f"{side:.1f}°", f"{base:.1f}°"])
        st.table(pd.DataFrame(rows, columns=["Disciplina","Struttura soletta","Lamina SIDE (°)","Lamina BASE (°)"]))

def run_all(src_df):
    res = compute_snow_temperature(src_df, dt_hours=1.0)
    st.success(f"Previsioni pronte per **{place_label}**")
    st.dataframe(res, use_container_width=True)
    f1,f2 = plots(res); st.pyplot(f1); st.pyplot(f2)
    st.download_button("Scarica CSV risultato", data=res.to_csv(index=False), file_name="forecast_with_snowT.csv", mime="text/csv")

    W_A = window_slice(res, tzname, A_start, A_end)
    W_B = window_slice(res, tzname, B_start, B_end)
    W_C = window_slice(res, tzname, C_start, C_end)

    show_block("A", W_A, tzname)
    show_block("B", W_B, tzname)
    show_block("C", W_C, tzname)

# Sorgente dati: CSV o Fetch
if upl is not None:
    try:
        df_u = pd.read_csv(upl)
        run_all(df_u)
    except Exception as e:
        st.error(f"CSV non valido: {e}")

if go:
    try:
        js = fetch_open_meteo(lat, lon, tzname)
        src = build_df(js, hours, tzname)
        run_all(src)
    except Exception as e:
        st.error(f"Errore: {e}")

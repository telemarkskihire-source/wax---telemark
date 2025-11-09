# app.py — Telemark · Pro Wax & Tune (ricerca “stile Meteoblue”, blocchi A/B/C, wax + strutture + lamine)

import streamlit as st
import pandas as pd
import requests
import math
import base64
import io
import matplotlib.pyplot as plt
from datetime import time
from dateutil import tz

# =============== CONFIG & THEME =================================================
st.set_page_config(page_title="Telemark · Pro Wax & Tune", page_icon="❄️", layout="wide")

PRIMARY = "#10bfcf"; BG = "#0f172a"; CARD = "#111827"; TEXT = "#e5e7eb"
st.markdown(f"""
<style>
/* page bg */
[data-testid="stAppViewContainer"] > .main {{
  background: linear-gradient(180deg, {BG} 0%, #111827 100%);
}}
.block-container {{ padding-top: 0.8rem; }}
h1,h2,h3,h4,h5,h6,p,span,div {{ color:{TEXT}; }}
.badge {{
  border:1px solid rgba(255,255,255,.15); padding:6px 10px; border-radius:999px;
  font-size:.78rem; opacity:.85;
}}
.card {{
  background:{CARD}; border:1px solid rgba(255,255,255,.08); border-radius:16px; padding:14px;
  box-shadow: 0 8px 20px rgba(0,0,0,.25);
}}
.brand {{
  display:flex; align-items:center; gap:10px; padding:10px 12px; border-radius:12px;
  background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08);
}}
.brand img {{ height:22px; }}
.kpi {{
  display:flex; gap:8px; align-items:center; background:rgba(16,191,207,.06);
  border:1px dashed rgba(16,191,207,.45); padding:10px 12px; border-radius:12px;
}}
.kpi .lab {{ font-size:.78rem; color:#93c5fd; }}
.kpi .val {{ font-size:1rem; font-weight:800; color:{TEXT}; }}
.btn-primary button {{
  width:100%; background:{PRIMARY} !important; color:#002b30 !important; border:none; font-weight:700; border-radius:12px;
}}
.small {{ font-size:.8rem; opacity:.85; }}
hr {{ border-color: rgba(255,255,255,.1) }}
.sugg-dd {{ margin-top: -8px; }}
</style>
""", unsafe_allow_html=True)

st.markdown("## Telemark · Pro Wax & Tune")
st.markdown("<span class='badge'>Ricerca località tipo Meteoblue · Geolocalizzazione · Blocchi A/B/C · Sciolina + Strutture + Lamine</span>", unsafe_allow_html=True)

# =============== UTIL: bandierine e testo breve =================================
COUNTRY_FLAG = lambda cc: chr(127397 + ord(cc.upper()[0])) + chr(127397 + ord(cc.upper()[1])) if cc and len(cc)==2 else "🌍"

def short_label(item):
    # Compatta lo "display_name" di Nominatim in: FLAG City, Admin (CC)
    addr = item.get("address", {})
    city = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("hamlet") or item.get("name") or ""
    admin = addr.get("state") or addr.get("county") or ""
    cc = addr.get("country_code","").upper()
    flag = COUNTRY_FLAG(cc) if cc else "🌍"
    parts = [p for p in [city, admin] if p]
    base = (", ".join(parts))[:60]
    return f"{flag} {base} ({cc})"

# =============== RICERCA LOCALITÀ (stile Meteoblue-like) ========================
# Aggiorna suggerimenti ad OGNI battuta (Streamlit rerun automatico su text_input)
def search_places(q, limit=8):
    if not q or len(q) < 2:
        return []
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": q, "format":"json", "limit": limit, "addressdetails": 1},
            headers={"User-Agent": "telemark-pro-wax/1.0"},
            timeout=8
        )
        r.raise_for_status()
        items = r.json()
        out = []
        for it in items:
            try:
                lat = float(it.get("lat", 0)); lon = float(it.get("lon", 0))
            except:
                continue
            it["label_short"] = short_label(it)
            it["lat_f"] = lat; it["lon_f"] = lon
            out.append(it)
        return out
    except Exception:
        return []

# Campo di ricerca (nessun Enter richiesto: ogni carattere fa rerun → query)
col_search, col_geo = st.columns([3,1])
with col_search:
    q = st.text_input("Cerca località", placeholder="Es. Champoluc, Cervinia, Sestriere…", key="q")
    suggestions = search_places(q, limit=10) if q else []
    # Dropdown compatto sotto il campo (selectbox) con etichette brevi + bandiera
    labels = [it["label_short"] for it in suggestions]
    if labels:
        st.markdown("<div class='sugg-dd small'>", unsafe_allow_html=True)
        sel = st.selectbox("Suggerimenti", labels, index=0, label_visibility="collapsed", key="selbox")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        sel = None

with col_geo:
    if st.button("📍 Usa la mia posizione"):
        try:
            r = requests.get("https://ipapi.co/json", timeout=6)
            if r.ok:
                j = r.json()
                st.session_state["chosen_lat"] = float(j.get("latitude",45.83))
                st.session_state["chosen_lon"] = float(j.get("longitude",7.73))
                st.session_state["chosen_label"] = (j.get("city") or "La tua posizione")
                st.success("Posizione impostata.")
            else:
                st.error("Geolocalizzazione non disponibile.")
        except Exception:
            st.error("Geolocalizzazione non disponibile.")

# Applica selezione
if sel:
    idx = labels.index(sel)
    pick = suggestions[idx]
    st.session_state["chosen_lat"] = pick["lat_f"]
    st.session_state["chosen_lon"] = pick["lon_f"]
    st.session_state["chosen_label"] = pick["label_short"]

# Default se non scelto
lat = st.session_state.get("chosen_lat", 45.831)
lon = st.session_state.get("chosen_lon", 7.730)
label_loc = st.session_state.get("chosen_label", "🇮🇹 Champoluc, Aosta (IT)")

# Parametri base
col_tz, col_h = st.columns([1,1])
with col_tz:
    tzname = st.selectbox("Timezone", ["Europe/Rome","UTC"], index=0)
with col_h:
    hours = st.slider("Ore previsione", 12, 168, 72, 12)

# =============== FINESTRE A/B/C =================================================
st.markdown("### Finestre orarie (oggi) — A · B · C")
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

# =============== FETCH & PREP METEO ============================================
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
    df["time"] = pd.to_datetime(df["time"])   # naive
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

# =============== MODELLO NEVE (T_surf, T_top5) =================================
def compute_snow_temperature(df, dt_hours=1.0):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    req = ["T2m","cloud","wind","sunup","prp_mmph","prp_type"]
    for c in req:
        if c not in df.columns: raise ValueError(f"Missing column: {c}")
    if "td" not in df.columns: df["td"] = float("nan")
    df = df.sort_values("time").reset_index(drop=True)

    rain = df["prp_type"].str.lower().isin(["rain","mixed"])
    snow = df["prp_type"].str.lower().eq("snow")
    sunup = df["sunup"].astype(int) == 1
    tw = (df["T2m"] + df["td"]) / 2.0
    wet = ( rain |
            (df["T2m"] > 0) |
            (sunup & (df["cloud"] < 0.3) & (df["T2m"] >= -3)) |
            (snow & (df["T2m"] >= -1)) |
            (snow & tw.ge(-0.5).fillna(False)) )

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

# =============== WAX BRANDS =====================================================
# Bande temperatura (no-fluoro). Intervalli “industry-like”.
SWIX  = [("PS5 Turquoise", -18,-10), ("PS6 Blue",-12,-6), ("PS7 Violet",-8,-2), ("PS8 Red",-4,4), ("PS10 Yellow",0,10)]
TOKO  = [("Blue",-30,-9), ("Red",-12,-4), ("Yellow",-6,0)]
VOLA  = [("MX-E Violet/Blue",-12,-4), ("MX-E Red",-5,0), ("MX-E Warm",-2,10)]
RODE  = [("R20 Blue",-18,-8), ("R30 Violet",-10,-3), ("R40 Red",-5,0), ("R50 Yellow",-1,10)]
HOLM  = [("UltraMix Blue",-20,-8), ("BetaMix Red",-14,-4), ("AlphaMix Yellow",-4,5)]
MAPL  = [("Universal Cold",-12,-6), ("Universal Medium",-7,-2), ("Universal Warm",-3,6)]
START = [("SG Blue",-12,-6), ("SG Purple",-8,-2), ("SG Red",-3,7)]
SKIGO = [("Paraffin Blue",-12,-6), ("Paraffin Violet",-8,-2), ("Paraffin Red",-3,2)]
BRANDS = [("Swix",SWIX,"#ef4444"),("Toko",TOKO,"#f59e0b"),("Vola",VOLA,"#3b82f6"),("Rode",RODE,"#22c55e"),
          ("Holmenkol",HOLM,"#06b6d4"),("Maplus",MAPL,"#eab308"),("Start",START,"#f97316"),("Skigo",SKIGO,"#6366f1")]

def pick(bands,t):
    for n,tmin,tmax in bands:
        if t>=tmin and t<=tmax: return n
    return bands[-1][0] if t>bands[-1][2] else bands[0][0]

def logo_chip(text, color):
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='140' height='28'><rect width='140' height='28' rx='6' fill='{color}'/><text x='10' y='19' font-size='14' font-weight='700' fill='white'>{text}</text></svg>"
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")

# =============== LAMINE & STRUTTURE ============================================
def tune_for(t_surf, discipline):
    # side (°) nel formato 88, 87.5, 87 ... base 0.5–1.0 in base a neve
    if t_surf <= -10:
        structure = "Lineare fine (fredda)"
        base = 0.5
        side_map = {"SL": 88.5, "GS": 88.0, "SG": 87.5, "DH": 87.5}
    elif t_surf <= -3:
        structure = "Lineare media / universale"
        base = 0.7
        side_map = {"SL": 88.0, "GS": 88.0, "SG": 87.5, "DH": 87.0}
    else:
        structure = "Chevron/Wave o Scarico laterale (calda/umida)"
        base = 0.8 if t_surf <= 0.5 else 1.0
        side_map = {"SL": 88.0, "GS": 87.5, "SG": 87.0, "DH": 87.0}
    return structure, side_map.get(discipline, 88.0), base

def draw_structure(struct_type, width=600, height=120):
    """Disegna anteprima struttura in stile 'macchina' (linee nette, contrasto realistico)."""
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    ax = plt.gca()
    ax.set_facecolor("#1f2937")  # soletta scura
    ax.set_xlim(0, 100); ax.set_ylim(0, 20)
    ax.axis('off')

    # line style helper
    def line(x1,y1,x2,y2,lw=2, alpha=0.9):
        ax.plot([x1,x2],[y1,y2], linewidth=lw, alpha=alpha, solid_capstyle='round', color="#9ca3af")

    if "fine" in struct_type.lower():
        # numerose linee sottili verticali
        for x in range(2, 100, 2):
            line(x,1, x,19, lw=1.3, alpha=0.8)
    elif "universale" in struct_type.lower() or "media" in struct_type.lower():
        # linee verticali + qualche micro-chevron
        for x in range(3, 100, 3):
            line(x,1, x,19, lw=1.6, alpha=0.9)
        for x in range(5, 100, 10):
            line(x-1,4, x+1,6, lw=1.2, alpha=0.7)
            line(x-1,14, x+1,16, lw=1.2, alpha=0.7)
    elif "chevron" in struct_type.lower() or "wave" in struct_type.lower():
        # pattern a V / onda
        for x in range(0, 100, 6):
            line(x, 5, x+3,10, lw=2.2)
            line(x+3,10, x,15, lw=2.2)
    elif "scarico" in struct_type.lower():
        # canali laterali + linee centrali
        for y in [3,17]:
            line(0,y, 100,y, lw=4.5)
        for x in range(10, 95, 6):
            line(x,1, x,19, lw=1.8)
    else:
        # default: media
        for x in range(3, 100, 3):
            line(x,1, x,19, lw=1.6)

    buf = io.BytesIO()
    plt.tight_layout(pad=0)
    fig.savefig(buf, format="png", dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf

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
      <div class='small' style='margin-top:6px'>T_surf min {tmin:.1f}°C / max {tmax:.1f}°C</div>
    </div>""", unsafe_allow_html=True)
    return wet

def wax_cards(temp):
    cols = st.columns(4)
    for i,(name,bands,color) in enumerate(BRANDS[:4]):
        chip = logo_chip(name, color)
        rec = pick(bands, temp)
        cols[i].markdown(f"<div class='brand'><img src='{chip}'/><div><div class='small' style='opacity:.85'>{name}</div><div style='font-weight:800'>{rec}</div></div></div>", unsafe_allow_html=True)
    cols2 = st.columns(4)
    for i,(name,bands,color) in enumerate(BRANDS[4:]):
        chip = logo_chip(name, color)
        rec = pick(bands, temp)
        cols2[i].markdown(f"<div class='brand'><img src='{chip}'/><div><div class='small' style='opacity:.85'>{name}</div><div style='font-weight:800'>{rec}</div></div></div>", unsafe_allow_html=True)

# =============== RUN ============================================================
col_run1, col_run2 = st.columns([1,2])
with col_run1:
    go = st.button("Scarica previsioni per la località", type="primary")
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

def run_all(src_df):
    res = compute_snow_temperature(src_df, dt_hours=1.0)
    st.success(f"Dati per **{label_loc}** pronti.")
    st.dataframe(res, use_container_width=True)
    f1,f2 = plots(res); st.pyplot(f1); st.pyplot(f2)
    st.download_button("Scarica CSV risultato", data=res.to_csv(index=False), file_name="forecast_with_snowT.csv", mime="text/csv")

    for L,(s,e) in {"A":(A_start,A_end), "B":(B_start,B_end), "C":(C_start,C_end)}.items():
        st.markdown(f"### Blocco {L}")
        W = window_slice(res, tzname, s, e)
        wet = badge(W)
        t_med = float(W["T_surf"].mean())
        st.markdown(f"**T_surf medio {L}: {t_med:.1f} °C**")
        wax_cards(t_med)

        # Tuning consigli + anteprima struttura
        disc = st.multiselect(f"Discipline (Blocco {L})", ["SL","GS","SG","DH"], default=["SL","GS"], key=f"d_{L}")
        rows = []
        struct_label, _, _ = tune_for(t_med, "GS")
        img = draw_structure(struct_label)
        st.image(img, caption=f"Struttura consigliata: {struct_label}", use_column_width=True)

        for d in disc:
            structure, side, base = tune_for(t_med, d)
            rows.append([d, structure, f"{side:.1f}°", f"{base:.1f}°"])
        if rows:
            df_tune = pd.DataFrame(rows, columns=["Disciplina","Struttura","Lamina SIDE (°)","Lamina BASE (°)"])
            st.table(df_tune)

# Sorgente dati
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

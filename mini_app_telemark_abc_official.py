import streamlit as st
import pandas as pd
import requests
import base64
import math
import matplotlib.pyplot as plt
from datetime import time
from dateutil import tz

# =========================
# UI THEME
# =========================
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
  border-radius:16px; padding:14px; box-shadow: 0 8px 20px rgba(0,0,0,.25);
}}
.brand {{ display:flex; align-items:center; gap:8px; padding:8px 10px; border-radius:12px;
         background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08);}}
.brand img {{ height:22px; }}
.kpi {{ display:flex; gap:8px; align-items:center; background:rgba(16,191,207,.06);
       border:1px dashed rgba(16,191,207,.45); padding:10px 12px; border-radius:12px; }}
.kpi .lab {{ font-size:.78rem; color:#93c5fd; }}
.kpi .val {{ font-size:1rem; font-weight:800; color:{TEXT}; }}
.select-tight div[data-baseweb="select"] > div {{
  border-radius: 12px; border-color: rgba(255,255,255,.2);
}}
</style>
""", unsafe_allow_html=True)

st.markdown("### Telemark · Pro Wax & Tune")
st.markdown("<span class='badge'>Ricerca stile Meteoblue · Geolocalizzazione · A/B/C · Sciolina · Struttura · Lamine</span>", unsafe_allow_html=True)

# =========================
# GEOCODING (Meteoblue-like UX)
# =========================
def flag_emoji(cc):
    if not cc or len(cc) != 2: return "🏳️"
    base = 127397
    return chr(ord(cc[0].upper())+base) + chr(ord(cc[1].upper())+base)

@st.cache_data(show_spinner=False, ttl=300)
def geocode_autocomplete(q: str, limit: int = 8):
    """Nominatim autocomplete; return list of dicts with label,lat,lon,country_code."""
    if not q or len(q) < 2:
        return []
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": q, "format": "json", "limit": limit, "addressdetails": 1},
            headers={"User-Agent": "telemark-pro-wax/1.0"},
            timeout=10
        )
        r.raise_for_status()
        out = []
        for it in r.json():
            name = it.get("display_name","").split(",")
            short = ", ".join(name[:2]) if len(name) >= 2 else it.get("display_name","")
            cc = (it.get("address") or {}).get("country_code","")
            out.append({
                "label_full": it.get("display_name",""),
                "label": short,
                "lat": float(it.get("lat", 0)),
                "lon": float(it.get("lon", 0)),
                "cc": cc.upper()
            })
        # De-dup by label
        seen=set(); dedup=[]
        for a in out:
            key=(a["label"],a["cc"])
            if key in seen: continue
            seen.add(key); dedup.append(a)
        return dedup[:limit]
    except Exception:
        return []

def ip_geolocate():
    try:
        r = requests.get("https://ipapi.co/json", timeout=8)
        if r.ok:
            j = r.json()
            return float(j.get("latitude",0)), float(j.get("longitude",0)), j.get("city",""), (j.get("country_code","") or "").upper()
    except Exception:
        pass
    return None, None, "", ""

# --- Search bar like Meteoblue: text field + live dropdown (selectbox fed by suggestions) ---
col_search, col_geo = st.columns([3,1])
with col_search:
    q = st.text_input("Cerca località (digita – suggerimenti in tempo reale)", placeholder="Es. Champoluc, Cervinia, Sestriere…", key="q_live")
    sugg = geocode_autocomplete(q, limit=10) if len(q) >= 2 else []
    options = []
    if sugg:
        for s in sugg:
            options.append(f"{flag_emoji(s['cc'])}  {s['label']}")
        # Default to first match for immediate UX
        default_index = 0
        choice = st.selectbox("Suggerimenti", options, index=default_index, key="sugg_select",
                              help="Seleziona la località dalla tendina. (Aggiorna automaticamente ad ogni carattere)",
                              placeholder="Seleziona una località")
        # Map back
        sel = sugg[options.index(choice)] if options else None
        if sel:
            st.session_state["spot_lat"] = sel["lat"]
            st.session_state["spot_lon"] = sel["lon"]
            st.session_state["spot_label"] = sel["label"]
            st.session_state["spot_cc"] = sel["cc"]
    else:
        st.caption("Suggerimenti mostrati mentre digiti (>= 2 caratteri).")

with col_geo:
    if st.button("📍 Usa posizione"):
        lat, lon, city, cc = ip_geolocate()
        if lat is not None:
            st.session_state["spot_lat"] = lat
            st.session_state["spot_lon"] = lon
            st.session_state["spot_label"] = city or "La tua posizione"
            st.session_state["spot_cc"] = cc or "IT"
            st.success("Posizione impostata.")
        else:
            st.error("Geolocalizzazione non disponibile.")

lat = st.session_state.get("spot_lat", 45.831)
lon = st.session_state.get("spot_lon", 7.730)
label = st.session_state.get("spot_label", "Champoluc (Ramey)")
cc = st.session_state.get("spot_cc", "IT")

# =========================
# A/B/C WINDOWS + CONTROLS
# =========================
col_tz, col_hours = st.columns([2,1])
with col_tz:
    tzname = st.selectbox("Timezone", ["Europe/Rome","UTC"], index=0)
with col_hours:
    hours = st.slider("Ore previsione", 12, 168, 72, 12)

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
# METEO FETCH + MODEL
# =========================
def fetch_open_meteo(lat, lon, timezone_str):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,"longitude": lon,"timezone": timezone_str,
        "hourly": "temperature_2m,dew_point_2m,precipitation,rain,snowfall,cloudcover,windspeed_10m,is_day,weathercode",
        "forecast_days": 7
    }
    r = requests.get(url, params=params, timeout=30); r.raise_for_status(); return r.json()

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
    h = js["hourly"]; df = pd.DataFrame(h); df["time"] = pd.to_datetime(df["time"])  # naive
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
    out["prp_type"] = _prp_type(extra)
    out["td"] = df["dew_point_2m"].astype(float)
    return out

def compute_snow_temperature(df, dt_hours=1.0):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    req = ["T2m","cloud","wind","sunup","prp_mmph","prp_type"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    if "td" not in df.columns: df["td"] = float("nan")
    df = df.sort_values("time").reset_index(drop=True)
    rain = df["prp_type"].str.lower().isin(["rain","mixed"])
    snow = df["prp_type"].str.lower().eq("snow")
    sunup = df["sunup"].astype(int) == 1
    tw = (df["T2m"] + df["td"]) / 2.0
    wet = (rain | (df["T2m"] > 0) | (sunup & (df["cloud"] < 0.3) & (df["T2m"] >= -3)) | (snow & (df["T2m"] >= -1)) | (snow & tw.ge(-0.5).fillna(False)))
    T_surf = pd.Series(index=df.index, dtype=float); T_surf.loc[wet] = 0.0
    dry = ~wet
    clear = (1.0 - df["cloud"]).clip(lower=0.0, upper=1.0)
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
    df["T_surf"] = T_surf; df["T_top5"] = T_top5; return df

def window_slice(res, tzname, s, e):
    t = pd.to_datetime(res["time"]).dt.tz_localize(tz.gettz(tzname), nonexistent='shift_forward', ambiguous='NaT')
    D = res.copy(); D["dt"] = t
    today = pd.Timestamp.now(tz=tz.gettz(tzname)).date()
    win = D[(D["dt"].dt.date==today) & (D["dt"].dt.time>=s) & (D["dt"].dt.time<=e)]
    return win if not win.empty else D.head(7)

# =========================
# WAX BRANDS + TUNING
# =========================
SWIX  = [("PS5 Turquoise",-18,-10),("PS6 Blue",-12,-6),("PS7 Violet",-8,-2),("PS8 Red",-4,4),("PS10 Yellow",0,10)]
TOKO  = [("Blue",-30,-9),("Red",-12,-4),("Yellow",-6,0)]
VOLA  = [("MX-E Violet/Blue",-12,-4),("MX-E Red",-5,0),("MX-E Warm",-2,10)]
RODE  = [("R20 Blue",-18,-8),("R30 Violet",-10,-3),("R40 Red",-5,0),("R50 Yellow",-1,10)]
HOLM  = [("UltraMix Blue",-20,-8),("BetaMix Red",-14,-4),("AlphaMix Yellow",-4,4)]
MAPL  = [("Universal Cold",-12,-6),("Universal Medium",-7,-2),("Universal Warm",-3,6)]
START = [("SG Blue",-12,-6),("SG Purple",-8,-2),("SG Red",-3,7)]
SKIGO = [("Paraffin Blue",-12,-6),("Paraffin Violet",-8,-2),("Paraffin Red",-3,2)]

def pick(bands, t):
    for n,tmin,tmax in bands:
        if t>=tmin and t<=tmax: return n
    return bands[-1][0] if t>bands[-1][2] else bands[0][0]

def tune_for(t_surf, discipline):
    # SIDE (88°, 87.5°, 87°), BASE (0.5°..1.0°)
    if t_surf <= -10:
        structure = "Lineare fine / Croce fine"
        base = 0.5
        side_map = {"SL": 88.5, "GS": 88.0, "SG": 87.5, "DH": 87.5}
    elif t_surf <= -3:
        structure = "Lineare media / Chevron leggero"
        base = 0.7
        side_map = {"SL": 88.0, "GS": 88.0, "SG": 87.5, "DH": 87.0}
    else:
        structure = "Chevron / Onda / Scarico laterale"
        base = 0.8 if t_surf <= 0.5 else 1.0
        side_map = {"SL": 88.0, "GS": 87.5, "SG": 87.0, "DH": 87.0}
    return structure, side_map.get(discipline, 88.0), base

def svg_badge(text, color):
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='200' height='36'><rect width='200' height='36' rx='6' fill='{color}'/><text x='12' y='24' font-size='16' font-weight='700' fill='white'>{text}</text></svg>"
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")

brand_svgs = {
    "Swix": svg_badge("SWIX","#ef4444"),
    "Toko": svg_badge("TOKO","#f59e0b"),
    "Vola": svg_badge("VOLA","#3b82f6"),
    "Rode": svg_badge("RODE","#22c55e"),
    "Holmenkol": svg_badge("HOLMENKOL","#334155"),
    "Maplus": svg_badge("MAPLUS","#6b7280"),
    "Start": svg_badge("START","#f43f5e"),
    "Skigo": svg_badge("SKIGO","#16a34a"),
}

# =========================
# STRUCTURE DRAWINGS (SVG, stile “catalogo” Wintersteiger)
# =========================
def structure_svg(kind: str):
    """
    kind: 'linear_fine', 'linear_medium', 'cross', 'chevron', 'wave', 'lateral_drain'
    """
    W,H = 420,100
    fg = "#9ca3af"   # incisione
    bg = "#0b1222"   # soletta scura
    lines = []
    if kind == "linear_fine":
        step=8
        for x in range(10, W, step):
            lines.append(f"<line x1='{x}' y1='0' x2='{x}' y2='{H}' stroke='{fg}' stroke-width='1'/>")
    elif kind == "linear_medium":
        step=14
        for x in range(10, W, step):
            lines.append(f"<line x1='{x}' y1='0' x2='{x}' y2='{H}' stroke='{fg}' stroke-width='2'/>")
    elif kind == "cross":
        # verticale
        for x in range(12, W, 14):
            lines.append(f"<line x1='{x}' y1='0' x2='{x}' y2='{H}' stroke='{fg}' stroke-width='1.5'/>")
        # orizzontale
        for y in range(8, H, 12):
            lines.append(f"<line x1='0' y1='{y}' x2='{W}' y2='{y}' stroke='{fg}' stroke-width='1'/>")
    elif kind == "chevron":
        # V ripetuti
        step=20
        for x in range(0, W, step):
            for y in range(10, H, 20):
                lines.append(f"<path d='M{x},{y} l10,-10 l10,10' fill='none' stroke='{fg}' stroke-width='2'/>")
    elif kind == "wave":
        # sinusoidi parallele
        amp=8; step=12
        for y in range(10, H, step):
            path = "M0,{y} ".format(y=y)
            x=0
            while x < W:
                path += f"q 10,-{amp} 20,0 q 10,{amp} 20,0 "
                x += 40
            lines.append(f"<path d='{path}' fill='none' stroke='{fg}' stroke-width='1.6'/>")
    elif kind == "lateral_drain":
        # canali laterali + lineare centrale
        # canali grossi lato
        lines.append(f"<rect x='0' y='0' width='20' height='{H}' fill='#1f2937'/>")
        lines.append(f"<rect x='{W-20}' y='0' width='20' height='{H}' fill='#1f2937'/>")
        for x in range(35, W-35, 14):
            lines.append(f"<line x1='{x}' y1='0' x2='{x}' y2='{H}' stroke='{fg}' stroke-width='1.3'/>")
        # scarico inclinato verso i lati
        for y in range(10, H, 22):
            lines.append(f"<path d='M20,{y} L80,{y+8}' stroke='{fg}' stroke-width='1'/>")
            lines.append(f"<path d='M{W-80},{y} L{W-20},{y+8}' stroke='{fg}' stroke-width='1'/>")
    else:
        return ""
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}'><rect width='{W}' height='{H}' fill='{bg}'/>{''.join(lines)}</svg>"
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")

def show_structures(reco_label: str):
    st.caption("Strutture consigliate (visualizzazione stilizzata)")
    kinds_order = {
        "cold": ["linear_fine","cross"],
        "mid":  ["linear_medium","chevron"],
        "warm": ["chevron","wave","lateral_drain"]
    }
    if "fine" in reco_label or "Croce" in reco_label:
        kinds = kinds_order["cold"]
    elif "media" in reco_label.lower():
        kinds = kinds_order["mid"]
    else:
        kinds = kinds_order["warm"]
    cols = st.columns(len(kinds))
    names = {
        "linear_fine":"Lineare fine",
        "linear_medium":"Lineare media",
        "cross":"Croce",
        "chevron":"Chevron",
        "wave":"Onda",
        "lateral_drain":"Scarico laterale"
    }
    for i,k in enumerate(kinds):
        cols[i].markdown(f"**{names[k]}**")
        cols[i].markdown(f"<img src='{structure_svg(k)}'/>", unsafe_allow_html=True)

# =========================
# ACTIONS
# =========================
col_run1, col_run2 = st.columns([1,2])
with col_run1:
    go = st.button(f"Scarica previsioni per {flag_emoji(cc)} {label}", type="primary")
with col_run2:
    upl = st.file_uploader("…oppure carica CSV (time,T2m,cloud,wind,sunup,prp_mmph,prp_type,td)", type=["csv"])

def make_plots(res):
    fig1 = plt.figure()
    t = pd.to_datetime(res["time"])
    plt.plot(t,res["T2m"],label="T2m")
    plt.plot(t,res["T_surf"],label="T_surf")
    plt.plot(t,res["T_top5"],label="T_top5")
    plt.legend(); plt.title("Temperature vs tempo"); plt.xlabel("Ora"); plt.ylabel("°C")
    fig2 = plt.figure()
    plt.bar(t,res["prp_mmph"])
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

def wax_row(title, t):
    bands_map = {
        "Swix": SWIX, "Toko": TOKO, "Vola": VOLA, "Rode": RODE,
        "Holmenkol": HOLM, "Maplus": MAPL, "Start": START, "Skigo": SKIGO
    }
    cols = st.columns(4)
    br = list(bands_map.keys())
    for i,b in enumerate(br[:4]):
        rec = pick(bands_map[b], t)
        cols[i].markdown(
            f"<div class='brand'><img src='{brand_svgs[b]}'/><div>"
            f"<div style='font-size:.8rem;opacity:.85'>{b}</div>"
            f"<div style='font-weight:800'>{rec}</div></div></div>", unsafe_allow_html=True)
    cols2 = st.columns(4)
    for i,b in enumerate(br[4:]):
        rec = pick(bands_map[b], t)
        cols2[i].markdown(
            f"<div class='brand'><img src='{brand_svgs[b]}'/><div>"
            f"<div style='font-size:.8rem;opacity:.85'>{b}</div>"
            f"<div style='font-weight:800'>{rec}</div></div></div>", unsafe_allow_html=True)

def run_pipeline(src_df, spot_label):
    res = compute_snow_temperature(src_df, dt_hours=1.0)
    st.success(f"Dati pronti per **{spot_label}**")
    st.dataframe(res, use_container_width=True)
    f1,f2 = make_plots(res); st.pyplot(f1); st.pyplot(f2)
    st.download_button("Scarica CSV risultato",
                       data=res.to_csv(index=False),
                       file_name="forecast_with_snowT.csv",
                       mime="text/csv")

    # Blocchi
    for L,(s,e) in {"A":(A_start,A_end),"B":(B_start,B_end),"C":(C_start,C_end)}.items():
        st.markdown(f"### Blocco {L}")
        W = window_slice(res, tzname, s, e)
        wet = badge(W)
        t_med = float(W["T_surf"].mean())
        st.markdown(f"**T_surf medio {L}: {t_med:.1f} °C**")

        # Scioline (8 marchi)
        wax_row(f"Wax {L}", t_med)

        # Tuning
        disc = st.multiselect(f"Discipline (Blocco {L})", ["SL","GS","SG","DH"], default=["SL","GS"], key=f"disc_{L}")
        rows=[]
        for d in disc:
            structure, side, base = tune_for(t_med, d)
            rows.append([d, structure, f"{side:.1f}°", f"{base:.1f}°"])
        if rows:
            st.table(pd.DataFrame(rows, columns=["Disciplina","Struttura consigliata","Lamina SIDE (°)","Lamina BASE (°)"]))
            # Anteprima struttura in stile catalogo
            # Selezioniamo famiglia in base a struttura
            fam = "cold" if "fine" in rows[0][1].lower() or "croce" in rows[0][1].lower() else ("mid" if "media" in rows[0][1].lower() else "warm")
            show_structures(rows[0][1])

# =========================
# MAIN FLOW
# =========================
if upl is not None:
    try:
        df_u = pd.read_csv(upl)
        run_pipeline(df_u, f"{label} ({flag_emoji(cc)})")
    except Exception as e:
        st.error(f"CSV non valido: {e}")

if go:
    try:
        js = fetch_open_meteo(lat, lon, tzname)
        df_src = build_df(js, hours, tzname)
        run_pipeline(df_src, f"{label} ({flag_emoji(cc)})")
    except Exception as e:
        st.error(f"Errore: {e}")

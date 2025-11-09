# telemark_pro_app.py
import streamlit as st
import pandas as pd
import requests, base64, math
from datetime import time
from dateutil import tz
import matplotlib.pyplot as plt
from io import BytesIO

# ------------- PAGE SETUP & THEME -------------
PRIMARY = "#10bfcf"; BG = "#0f172a"; CARD = "#111827"; TEXT = "#e5e7eb"
st.set_page_config(page_title="Telemark · Pro Wax & Tune", page_icon="❄️", layout="wide")
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
  background: linear-gradient(180deg, {BG} 0%, #111827 100%);
}}
.block-container {{ padding-top: 0.6rem; }}
h1,h2,h3,h4,h5,p,span,div {{ color:{TEXT}; }}
.badge {{ border:1px solid rgba(255,255,255,.15); padding:6px 10px; border-radius:999px; font-size:.78rem; opacity:.85; }}
.card {{ background:{CARD}; border:1px solid rgba(255,255,255,.08); border-radius:16px; padding:14px; box-shadow: 0 8px 20px rgba(0,0,0,.25);}}
.brand {{ display:flex; align-items:center; gap:8px; padding:8px 10px; border-radius:12px; background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08);}}
.brand img {{ height:22px; }}
.kpi {{ display:flex; gap:8px; align-items:center; background:rgba(16,191,207,.06); border:1px dashed rgba(16,191,207,.45); padding:10px 12px; border-radius:12px; }}
.kpi .lab {{ font-size:.78rem; color:#93c5fd; }}
.kpi .val {{ font-size:1rem; font-weight:800; color:{TEXT}; }}
.select-like {{
  background:{CARD}; border:1px solid rgba(255,255,255,.12); border-radius:12px; padding:8px;
}}
.sugg-item {{
  padding:8px; border-radius:10px; cursor:pointer; border:1px solid transparent;
}}
.sugg-item:hover {{ background:rgba(255,255,255,.06); border-color:rgba(255,255,255,.12); }}
.flag {{ width:18px; height:18px; border-radius:3px; display:inline-block; margin-right:8px; vertical-align:-3px; }}
</style>
""", unsafe_allow_html=True)

st.markdown("### Telemark · Pro Wax & Tune")
st.markdown("<span class='badge'>Ricerca rapida stile Meteoblue · Geolocalizzazione · Blocchi A/B/C · Sciolina, Struttura, Lamine</span>", unsafe_allow_html=True)

# ------------- UTIL: FLAGS & LOGOS -------------
def cc_to_flag(cc: str) -> str:
    """Convert country code to emoji flag (fallback colored dot)."""
    if not cc or len(cc) != 2:
        return "🏳️"
    base = 127397
    return chr(ord(cc[0].upper()) + base) + chr(ord(cc[1].upper()) + base)

def svg_badge(text, color):
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='200' height='36'><rect width='200' height='36' rx='6' fill='{color}'/><text x='12' y='24' font-size='16' font-weight='700' fill='white'>{text}</text></svg>"
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")

BRAND_SVGS = {
    "Swix": svg_badge("SWIX", "#ef4444"),
    "Toko": svg_badge("TOKO", "#f59e0b"),
    "Vola": svg_badge("VOLA", "#3b82f6"),
    "Rode": svg_badge("RODE", "#22c55e"),
    "Holmenkol": svg_badge("HOLMENKOL", "#64748b"),
    "Maplus": svg_badge("MAPLUS", "#ff6b00"),
    "Start": svg_badge("START", "#a855f7"),
    "Skigo": svg_badge("SKIGO", "#14b8a6"),
}

# ------------- WEATHER PIPELINE -------------
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

def build_df(js, hours):
    # keep times naive + simple selection from "now"
    h = js["hourly"]; df = pd.DataFrame(h); df["time"] = pd.to_datetime(df["time"])
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
    wet = (rain | (df["T2m"] > 0) | (sunup & (df["cloud"] < 0.3) & (df["T2m"] >= -3)) |
           (snow & (df["T2m"] >= -1)) | (snow & tw.ge(-0.5).fillna(False)))

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

    df["T_surf"] = T_surf; df["T_top5"] = T_top5
    return df

# ------------- LIVE SEARCH (Meteoblue-like) -------------
# We simulate "realtime" suggestions with a gentle auto-refresh and fetch on every keystroke.
st_autorefresh = st.experimental_rerun  # alias if needed later
st_autorefresh_token = st.experimental_data_editor if False else None  # placeholder (no-op)

# Small, non-intrusive periodic refresh to catch keystrokes without ENTER
st.experimental_set_query_params()  # noop to avoid lint
st_autorefresh_id = st.autorefresh(interval=700, key="auto_refresh", limit=0)

def geocode_autocomplete(q: str, limit=8):
    if not q or len(q) < 2:
        return []
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": q, "format":"json", "limit": limit, "addressdetails": 1, "accept-language":"it,en"},
            headers={"User-Agent": "telemark-pro-wax/1.0"},
            timeout=10
        )
        r.raise_for_status()
        out = []
        for item in r.json():
            lat = float(item.get("lat", 0)); lon = float(item.get("lon", 0))
            addr = item.get("address", {})
            city = addr.get("city") or addr.get("town") or addr.get("village") or ""
            country = addr.get("country", "")
            cc = addr.get("country_code", "")
            label = f"{city or item.get('display_name','')}"
            out.append({
                "label": label if city else item.get("display_name",""),
                "lat": lat, "lon": lon,
                "country": country, "cc": cc.upper()
            })
        return out
    except Exception:
        return []

def ip_geolocate():
    try:
        r = requests.get("https://ipapi.co/json", timeout=8)
        if r.ok:
            j = r.json()
            return float(j.get("latitude",0)), float(j.get("longitude",0)), j.get("city",""), j.get("country_code","")
    except Exception:
        pass
    return None, None, "", ""

col_search, col_geo = st.columns([3,1])
with col_search:
    q = st.text_input("Cerca località", placeholder="Digita per cercare (es. Champoluc, Cervinia, Sestriere…)", label_visibility="visible", key="q_text")
    sugg = geocode_autocomplete(q, limit=8)

    selected = None
    if sugg:
        st.markdown("<div class='select-like'>", unsafe_allow_html=True)
        opts = []
        for i, s in enumerate(sugg):
            flag = cc_to_flag(s["cc"])
            opts.append(f"{flag}  {s['label']}")
        choice = st.selectbox("Suggerimenti", options=["— scegli —"]+opts, index=0, label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)
        if choice != "— scegli —":
            idx = opts.index(choice)
            selected = sugg[idx]

with col_geo:
    if st.button("📍 Usa la mia posizione"):
        lat, lon, city, cc = ip_geolocate()
        if lat is not None:
            st.session_state["chosen"] = {"lat": lat, "lon": lon, "label": city or "La tua posizione", "cc": cc}
            st.success("Posizione impostata.")
        else:
            st.error("Geolocalizzazione non disponibile.")

if selected:
    st.session_state["chosen"] = {"lat": selected["lat"], "lon": selected["lon"], "label": selected["label"], "cc": selected["cc"]}

# Defaults if none chosen yet
if "chosen" not in st.session_state:
    st.session_state["chosen"] = {"lat": 45.831, "lon": 7.730, "label": "Champoluc (Ramey)", "cc":"IT"}

spot = st.session_state["chosen"]
tzname = st.selectbox("Timezone", ["Europe/Rome","UTC"], index=0)
hours = st.slider("Ore previsione", 12, 168, 72, 12)

# ------------- TIME WINDOWS A/B/C -------------
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

# ------------- FETCH / UPLOAD -------------
col_run1, col_run2 = st.columns([1,2])
with col_run1:
    go = st.button(f"Scarica previsioni per {cc_to_flag(spot['cc'])} {spot['label']}", type="primary")
with col_run2:
    upl = st.file_uploader("…oppure carica CSV (time,T2m,cloud,wind,sunup,prp_mmph,prp_type,td)", type=["csv"])

# ------------- STRUCTURE DRAWINGS -------------
def structure_image(structure: str) -> bytes:
    """
    Draw clean, technical-style structure previews:
    - 'fine_linear' : solchi sottili e ravvicinati
    - 'medium_linear': solchi medi, spaziatura maggiore
    - 'cross' : incroci (X) regolari
    - 'chevron' : lisca/onda direzionale
    """
    fig = plt.figure(figsize=(3.2, 1.2), dpi=180)
    ax = plt.gca()
    ax.set_facecolor("#222831")
    plt.xlim(0, 100); plt.ylim(0, 30)
    plt.axis("off")

    if structure == "fine_linear":
        for x in range(5, 100, 4):
            plt.plot([x, x], [2, 28])
    elif structure == "medium_linear":
        for x in range(6, 100, 7):
            plt.plot([x, x], [2, 28])
    elif structure == "cross":
        # diagonal lines one direction
        for x in range(-30, 120, 8):
            plt.plot([x, x+40], [2, 28])
        # and the other direction
        for x in range(0, 150, 8):
            plt.plot([x, x-40], [2, 28])
    elif structure == "chevron":
        # angled repeated V pattern
        step = 10
        for x0 in range(0, 100, step):
            plt.plot([x0, x0+step/2], [2, 15])
            plt.plot([x0+step/2, x0+step], [15, 2])
            plt.plot([x0, x0+step/2], [28, 15])
            plt.plot([x0+step/2, x0+step], [15, 28])

    buf = BytesIO()
    plt.tight_layout(pad=0.2)
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def best_structure_for_temp(t_surf: float) -> str:
    # Very cold -> fine linear; mid -> medium linear; near/warm -> chevron; if wet peak -> cross (drainage multi-dir)
    if t_surf <= -10:
        return "fine_linear"
    if t_surf <= -3:
        return "medium_linear"
    return "chevron"

# ------------- WAX BANDS -------------
SWIX  = [("PS5 Turquoise",-18,-10), ("PS6 Blue",-12,-6), ("PS7 Violet",-8,-2), ("PS8 Red",-4,4), ("PS10 Yellow",0,10)]
TOKO  = [("Blue",-30,-9), ("Red",-12,-4), ("Yellow",-6,0)]
VOLA  = [("MX-E Violet/Blue",-12,-4), ("MX-E Red",-5,0), ("MX-E Warm",-2,10)]
RODE  = [("R20 Blue",-18,-8), ("R30 Violet",-10,-3), ("R40 Red",-5,0), ("R50 Yellow",-1,10)]
HOLM  = [("Ultra/Alpha Mix Blue",-20,-8), ("BetaMix Red",-14,-4), ("AlphaMix Yellow",-4,5)]
MAPL  = [("Universal Cold",-12,-6), ("Universal Medium",-7,-2), ("Universal Warm",-3,6)]
START = [("SG Blue",-12,-6), ("SG Purple",-8,-2), ("SG Red",-3,7)]
SKIGO = [("Paraffin Blue",-12,-6), ("Paraffin Violet",-8,-2), ("Paraffin Red",-3,2)]

def pick_band(bands, t):
    for name, tmin, tmax in bands:
        if t >= tmin and t <= tmax:
            return name
    return bands[-1][0] if t > bands[-1][2] else bands[0][0]

def wax_cards(t):
    rows = [
        ("Swix", SWIX), ("Toko", TOKO), ("Vola", VOLA), ("Rode", RODE),
        ("Holmenkol", HOLM), ("Maplus", MAPL), ("Start", START), ("Skigo", SKIGO),
    ]
    c1, c2, c3, c4 = st.columns(4)
    cols = [c1, c2, c3, c4]
    for i, (brand, spec) in enumerate(rows):
        col = cols[i%4]
        rec = pick_band(spec, t)
        col.markdown(
            f"<div class='brand'><img src='{BRAND_SVGS[brand]}'/>"
            f"<div><div style='font-size:.8rem;opacity:.85'>{brand}</div>"
            f"<div style='font-weight:800'>{rec}</div></div></div>",
            unsafe_allow_html=True
        )

# ------------- EDGE ANGLES (SIDE + BASE) -------------
def tune_for(t_surf, discipline):
    # side angle in degrees (88°, 87.5°, 87°), base (0.5..1.0°)
    if t_surf <= -10:
        structure = "Linear fine"
        base = 0.5
        side_map = {"SL": 88.5, "GS": 88.0, "SG": 87.5, "DH": 87.5}
    elif t_surf <= -3:
        structure = "Linear media"
        base = 0.7
        side_map = {"SL": 88.0, "GS": 88.0, "SG": 87.5, "DH": 87.0}
    else:
        structure = "Chevron / onda"
        base = 0.8 if t_surf <= 0.5 else 1.0
        side_map = {"SL": 88.0, "GS": 87.5, "SG": 87.0, "DH": 87.0}
    return structure, side_map.get(discipline, 88.0), base

# ------------- PLOTS -------------
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

def window_slice(res, tzname, s, e):
    times = pd.to_datetime(res["time"]).dt.tz_localize(tz.gettz(tzname), nonexistent='shift_forward', ambiguous='NaT')
    D = res.copy(); D["dt"] = times
    today = pd.Timestamp.now(tz=tz.gettz(tzname)).date()
    W = D[(D["dt"].dt.date==today) & (D["dt"].dt.time>=s) & (D["dt"].dt.time<=e)]
    return W if not W.empty else D.head(7)

def badge(win):
    tmin = float(win["T_surf"].min()); tmax = float(win["T_surf"].max())
    wet = bool(((win["prp_type"].isin(["rain","mixed"])) | (win["prp_mmph"]>0.5)).any())
    if wet or tmax > 0.5:
        color, title, desc = "#ef4444", "CRITICAL", "Possibile neve bagnata/pioggia · struttura grossa"
    elif tmax > -1.0:
        color, title, desc = "#f59e0b", "WATCH", "Neve vicino a 0°C · cere medio-morbide"
    else:
        color, title, desc = "#22c55e", "OK", "Neve fredda/asciutta · cere dure"
    st.markdown(
        f"<div class='card' style='border-color:{color}'>"
        f"<div style='font-weight:800;color:{color};margin-bottom:4px'>{title}</div>"
        f"<div style='opacity:.95'>{desc}</div>"
        f"<div style='font-size:12px;opacity:.7;margin-top:6px'>T_surf min {tmin:.1f}°C / max {tmax:.1f}°C</div>"
        f"</div>", unsafe_allow_html=True
    )
    return wet

# ------------- RUN -------------
def run_all(src):
    res = compute_snow_temperature(src, dt_hours=1.0)
    st.success(f"Dati pronti per {cc_to_flag(spot['cc'])} {spot['label']}")
    st.dataframe(res, use_container_width=True)
    f1,f2 = plots(res); st.pyplot(f1); st.pyplot(f2)
    st.download_button("Scarica CSV risultato", data=res.to_csv(index=False), file_name="forecast_with_snowT.csv", mime="text/csv")

    for L,(s,e) in {"A":(A_start,A_end), "B":(B_start,B_end), "C":(C_start,C_end)}.items():
        st.markdown(f"### Blocco {L}")
        W = window_slice(res, tzname, s, e)
        wet = badge(W)
        t_med = float(W["T_surf"].mean())
        st.markdown(f"**T_surf medio {L}: {t_med:.1f} °C**")

        # Wax
        wax_cards(t_med)

        # Structure preview (technical style)
        structure_key = best_structure_for_temp(t_med)
        img_bytes = structure_image(structure_key)
        st.image(img_bytes, caption=f"Struttura consigliata: {structure_key.replace('_',' ').title()}", use_column_width=False)

        # Tuning table
        disc = st.multiselect(f"Discipline (Blocco {L})", ["SL","GS","SG","DH"], default=["SL","GS"], key=f"disc_{L}")
        rows = []
        for d in disc:
            sname, side, base = tune_for(t_med, d)
            rows.append([d, sname, f"{side:.1f}°", f"{base:.1f}°"])
        if rows:
            df_tune = pd.DataFrame(rows, columns=["Disciplina","Struttura","Lamina SIDE (°)","Lamina BASE (°)"])
            st.table(df_tune)

# INPUT SOURCES
if upl is not None:
    try:
        df_u = pd.read_csv(upl)
        run_all(df_u)
    except Exception as e:
        st.error(f"CSV non valido: {e}")

if go and upl is None:
    try:
        js = fetch_open_meteo(spot["lat"], spot["lon"], tzname)
        df_src = build_df(js, hours)
        run_all(df_src)
    except Exception as e:
        st.error(f"Errore: {e}")

# telemark_pro_app.py
import streamlit as st
import pandas as pd
import requests
import base64
import io
import math
import matplotlib.pyplot as plt
from datetime import time
from dateutil import tz

# -------------------- UI / THEME --------------------
PRIMARY = "#10bfcf"; BG = "#0f172a"; CARD = "#111827"; TEXT = "#e5e7eb"
st.set_page_config(page_title="Telemark · Pro Wax & Tune", page_icon="❄️", layout="wide")
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] > .main {{ background: linear-gradient(180deg, {BG} 0%, #111827 100%); }}
.block-container {{ padding-top: 0.75rem; }}
h1,h2,h3,h4,h5,p,span,div {{ color:{TEXT}; }}
.badge {{ border:1px solid rgba(255,255,255,.15); padding:6px 10px; border-radius:999px; font-size:.78rem; opacity:.85; }}
.card {{ background:{CARD}; border:1px solid rgba(255,255,255,.08); border-radius:16px; padding:14px; box-shadow:0 8px 20px rgba(0,0,0,.25); }}
.brand {{ display:flex; align-items:center; gap:8px; padding:8px 10px; border-radius:12px; background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08); }}
.brand img {{ height:22px; }}
.kpibox {{ display:flex; gap:10px; flex-wrap:wrap; }}
.kpi {{ background:rgba(16,191,207,.08); border:1px dashed rgba(16,191,207,.45); padding:8px 10px; border-radius:10px; min-width:150px; }}
.kpi .lab {{ font-size:.78rem; color:#93c5fd; }}
.kpi .val {{ font-weight:800; }}
.select-row {{ display:flex; gap:8px; align-items:center; }}
.flag {{ font-size:18px; }}
hr {{ border-color: rgba(255,255,255,.12) }}
</style>
""", unsafe_allow_html=True)

st.markdown("### Telemark · Pro Wax & Tune")
st.markdown("<span class='badge'>Ricerca località in tempo reale · Blocchi A/B/C · Sciolina + Strutture · Lamine (side 88°/87.5°)</span>", unsafe_allow_html=True)

# -------------------- MODEL --------------------
def compute_snow_temperature(df, dt_hours=1.0):
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
            T_top5.iloc[i] = T_top5.iloc[i-1] + alpha.iloc[i]*(T_surf.iloc[i] - T_top5.iloc[i-1])

    df["T_surf"] = T_surf
    df["T_top5"] = T_top5
    return df

# -------------------- SEARCH (Meteoblue-like) --------------------
# Nota: Streamlit rerunna ad ogni tasto in text_input (se NON è dentro un form).
# Così aggiorniamo i suggerimenti “live”, senza Enter.

COUNTRY_FLAGS = {}  # cache iso2 -> flag
def flag_emoji(iso2):
    if not iso2 or len(iso2) != 2: return ""
    iso2 = iso2.upper()
    if iso2 in COUNTRY_FLAGS: return COUNTRY_FLAGS[iso2]
    base = 127397
    f = chr(ord(iso2[0]) + base) + chr(ord(iso2[1]) + base)
    COUNTRY_FLAGS[iso2] = f
    return f

@st.cache_data(show_spinner=False, ttl=300)
def geocode_nominatim(query: str, limit: int = 8):
    if not query or len(query.strip()) < 1:
        return []
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format":"json", "limit": limit, "addressdetails":1, "accept-language":"it"},
            headers={"User-Agent":"telemark-pro/1.0"},
            timeout=8
        )
        r.raise_for_status()
        out = []
        for it in r.json():
            name = it.get("display_name","")
            lat = float(it.get("lat", 0)); lon = float(it.get("lon", 0))
            addr = it.get("address", {})
            country = addr.get("country", "")
            iso2 = addr.get("country_code", "")
            flag = flag_emoji(iso2)
            label = f"{name} {(' ' + flag) if flag else ''}"
            out.append({"label": label, "short": f"{addr.get('town') or addr.get('city') or addr.get('village') or ''} {flag}".strip(), "lat": lat, "lon": lon})
        return out
    except Exception:
        return []

left, right = st.columns([3,1])
with left:
    st.write("**Cerca località**")
    q = st.text_input("Digita per cercare (aggiorna live)", placeholder="Champoluc, Cervinia, Sestriere…", label_visibility="collapsed", key="search_q")
    suggestions = geocode_nominatim(q, limit=12) if q else []
    # tendina interattiva semplice: selectbox sempre visibile, aggiorna mentre scrivi
    labels = [s["label"] for s in suggestions] if suggestions else []
    choice = st.selectbox("Suggerimenti", labels, index=0 if labels else None, placeholder="—", label_visibility="collapsed")
with right:
    if st.button("📍 Usa la mia posizione"):
        try:
            r = requests.get("https://ipapi.co/json", timeout=6)
            if r.ok:
                j = r.json()
                st.session_state["lat"] = float(j.get("latitude", 45.831))
                st.session_state["lon"] = float(j.get("longitude", 7.730))
                st.session_state["label"] = j.get("city","La tua posizione")
                st.success("Posizione impostata")
        except Exception:
            st.error("Geolocalizzazione non disponibile")

# applica scelta
if choice and suggestions:
    sel = suggestions[labels.index(choice)]
    st.session_state["lat"] = sel["lat"]
    st.session_state["lon"] = sel["lon"]
    st.session_state["label"] = sel["short"] or sel["label"]

lat = st.session_state.get("lat", 45.831)
lon = st.session_state.get("lon", 7.730)
label = st.session_state.get("label", "Champoluc (Ramey)")

# -------------------- CONTROLS --------------------
c1, c2, c3 = st.columns([1.2,1.2,2])
with c1:
    tzname = st.selectbox("Timezone", ["Europe/Rome","UTC"], index=0)
with c2:
    hours = st.slider("Ore previsione", 12, 168, 72, 12)
with c3:
    upl = st.file_uploader("…oppure carica CSV (time,T2m,cloud,wind,sunup,prp_mmph,prp_type,td)", type=["csv"])

st.markdown("**Finestre A · B · C (oggi)**")
b1,b2,b3 = st.columns(3)
with b1:
    A_start = st.time_input("Inizio A", time(9,0), key="A_s")
    A_end   = st.time_input("Fine A",   time(11,0), key="A_e")
with b2:
    B_start = st.time_input("Inizio B", time(11,0), key="B_s")
    B_end   = st.time_input("Fine B",   time(13,0), key="B_e")
with b3:
    C_start = st.time_input("Inizio C", time(13,0), key="C_s")
    C_end   = st.time_input("Fine C",   time(16,0), key="C_e")

# -------------------- DATA FETCH --------------------
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

# -------------------- WAX BRANDS (+ logos) --------------------
def svg_logo(text, color):
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='200' height='36'><rect width='200' height='36' rx='6' fill='{color}'/><text x='12' y='24' font-size='16' font-weight='700' fill='white'>{text}</text></svg>"
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")

LOGOS = {
    "Swix": svg_logo("SWIX","#ef4444"),
    "Toko": svg_logo("TOKO","#f59e0b"),
    "Vola": svg_logo("VOLA","#3b82f6"),
    "Rode": svg_logo("RODE","#22c55e"),
    "Holmenkol": svg_logo("HOLMENKOL","#0ea5e9"),
    "Maplus": svg_logo("MAPLUS","#f43f5e"),
    "Start": svg_logo("START","#f97316"),
    "Skigo": svg_logo("SKIGO","#6366f1"),
}

SWIX = [("PS5 Turquoise", -18,-10), ("PS6 Blue",-12,-6), ("PS7 Violet",-8,-2), ("PS8 Red",-4,4), ("PS10 Yellow",0,10)]
TOKO = [("Blue",-30,-9), ("Red",-12,-4), ("Yellow",-6,0)]
VOLA = [("MX-E Violet/Blue",-12,-4), ("MX-E Red",-5,0), ("MX-E Warm",-2,10)]
RODE = [("R20 Blue",-18,-8), ("R30 Violet",-10,-3), ("R40 Red",-5,0), ("R50 Yellow",-1,10)]
HOLM = [("Ultra/Alpha Blue",-20,-8), ("BetaMix Red",-14,-4), ("AlphaMix Yellow",-4,5)]
MAPL = [("Universal Cold",-12,-6), ("Universal Medium",-7,-2), ("Universal Warm",-3,6)]
START = [("SG Blue",-12,-6), ("SG Purple",-8,-2), ("SG Red",-3,7)]
SKIGO = [("Paraffin Blue",-12,-6), ("Paraffin Violet",-8,-2), ("Paraffin Red",-3,2)]

def pick_band(bands, t):
    for name, tmin, tmax in bands:
        if t >= tmin and t <= tmax:
            return name
    return bands[-1][0] if t > bands[-1][2] else bands[0][0]

def wax_cards(t_med):
    brands = [
        ("Swix", SWIX), ("Toko", TOKO), ("Vola", VOLA), ("Rode", RODE),
        ("Holmenkol", HOLM), ("Maplus", MAPL), ("Start", START), ("Skigo", SKIGO),
    ]
    rows = [st.columns(4), st.columns(4)]
    for i, (brand, bands) in enumerate(brands):
        rec = pick_band(bands, t_med)
        row = rows[0] if i < 4 else rows[1]
        j = i if i < 4 else i-4
        row[j].markdown(
            f"<div class='brand'><img src='{LOGOS[brand]}'/>"
            f"<div><div style='font-size:.8rem;opacity:.85'>{brand}</div>"
            f"<div style='font-weight:800'>{rec}</div></div></div>",
            unsafe_allow_html=True
        )

# -------------------- STRUCTURE DRAWINGS (Wintersteiger-like) --------------------
def draw_structure(pattern: str, width=600, height=140):
    """
    Render clean, high-contrast structure previews:
    - 'LinearFine'
    - 'LinearMedium'
    - 'Cross'
    - 'Chevron'
    - 'Wave'
    - 'SideDrain'
    """
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    ax.set_facecolor("#0b1222")
    ax.set_xlim(0, 100); ax.set_ylim(0, 20)
    ax.axis("off")

    # base
    ax.add_patch(plt.Rectangle((0,0), 100, 20, color="#0b1222"))

    def line(x1,y1,x2,y2,lw=1.8, alpha=0.95):
        ax.plot([x1,x2],[y1,y2], linewidth=lw, alpha=alpha, color="#9fb3ff")

    if pattern == "LinearFine":
        for x in range(3, 100, 3):
            line(x, 1, x, 19, lw=1.2)
    elif pattern == "LinearMedium":
        for x in range(5, 100, 5):
            line(x, 1, x, 19, lw=2.1)
    elif pattern == "Cross":
        # vertical sparse + diagonal
        for x in range(8, 100, 8):
            line(x, 1, x, 19, lw=1.6)
        for x in range(-40, 140, 6):
            line(x, 0, x+30, 20, lw=1.2, alpha=0.7)
        for x in range(-40, 140, 10):
            line(x, 20, x+30, 0, lw=1.2, alpha=0.7)
    elif pattern == "Chevron":
        # V repeating
        step = 8
        for x in range(0, 100, step):
            line(x, 1, x+step/2, 10, lw=1.8)
            line(x+step/2, 10, x+step, 1, lw=1.8)
            line(x, 19, x+step/2, 10, lw=1.8)
            line(x+step/2, 10, x+step, 19, lw=1.8)
    elif pattern == "Wave":
        # sine-like ribs
        import numpy as np
        xs = np.linspace(0, 100, 400)
        for phase in [0, 2, 4]:
            ys = 10 + 6*np.sin(0.25*xs + phase)
            ax.plot(xs, ys, color="#9fb3ff", linewidth=1.8, alpha=0.95)
    elif pattern == "SideDrain":
        # smooth center + heavy side grooves
        for x in range(10, 90, 10):
            line(x, 1, x, 19, lw=1.2, alpha=0.5)
        for x in [3, 6, 94, 97]:
            line(x, 1, x, 19, lw=3.0, alpha=1.0)

    buf = io.BytesIO()
    fig.tight_layout(pad=0)
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def tuning_for(t_surf, discipline):
    # side angle in degrees; base in degrees
    if t_surf <= -10:
        structure = "LinearFine"
        base = 0.5
        side_map = {"SL": 88.5, "GS": 88.0, "SG": 87.5, "DH": 87.5}
    elif t_surf <= -3:
        structure = "LinearMedium"
        base = 0.7
        side_map = {"SL": 88.0, "GS": 88.0, "SG": 87.5, "DH": 87.0}
    else:
        # vicino/ sopra 0: drenaggi e strutture più marcate
        structure = "SideDrain"
        base = 0.8 if t_surf <= 0.5 else 1.0
        side_map = {"SL": 88.0, "GS": 87.5, "SG": 87.0, "DH": 87.0}
    return structure, side_map.get(discipline, 88.0), base

def badge(win):
    tmin = float(win["T_surf"].min()); tmax = float(win["T_surf"].max())
    wet = bool(((win["prp_type"].isin(["rain","mixed"])) | (win["prp_mmph"]>0.5)).any())
    if wet or tmax > 0.5:
        color, title, desc = "#ef4444", "CRITICAL", "Possibile neve bagnata/pioggia · struttura grossa"
    elif tmax > -1.0:
        color, title, desc = "#f59e0b", "WATCH", "Vicino a 0°C · cere medio-morbide"
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

def plots(res):
    t = pd.to_datetime(res["time"])
    fig1 = plt.figure()
    plt.plot(t, res["T2m"], label="T2m")
    plt.plot(t, res["T_surf"], label="T_surf")
    plt.plot(t, res["T_top5"], label="T_top5")
    plt.legend(); plt.title("Temperature vs tempo"); plt.xlabel("Ora"); plt.ylabel("°C")
    fig2 = plt.figure()
    plt.bar(t, res["prp_mmph"])
    plt.title("Precipitazione (mm/h)"); plt.xlabel("Ora"); plt.ylabel("mm/h")
    return fig1, fig2

# -------------------- RUN --------------------
go = st.button("Scarica previsioni per la località", type="primary")

if upl is not None:
    try:
        src = pd.read_csv(upl)
        res = compute_snow_temperature(src, dt_hours=1.0)
        st.success("CSV caricato")
        st.dataframe(res, use_container_width=True)
        f1, f2 = plots(res); st.pyplot(f1); st.pyplot(f2)
    except Exception as e:
        st.error(f"CSV non valido: {e}")

if go and upl is None:
    try:
        js = fetch_open_meteo(lat, lon, tzname)
        src = build_df(js, hours, tzname)
        res = compute_snow_temperature(src, dt_hours=1.0)
        st.success(f"Previsioni per **{label}** pronte")
        st.dataframe(res, use_container_width=True)
        f1, f2 = plots(res); st.pyplot(f1); st.pyplot(f2)
    except Exception as e:
        st.error(f"Errore: {e}")
        res = None
else:
    res = None if upl is None else compute_snow_temperature(pd.read_csv(upl), dt_hours=1.0)

# Blocchi A/B/C + consigli
if res is not None:
    st.markdown("### Consigli per blocchi A · B · C")
    for L, (s, e) in {"A": (A_start, A_end), "B": (B_start, B_end), "C": (C_start, C_end)}.items():
        st.markdown(f"#### Blocco {L}")
        W = window_slice(res, tzname, s, e)
        wet = badge(W)
        t_med = float(W["T_surf"].mean())
        st.markdown(f"**T_surf medio {L}: {t_med:.1f} °C**")
        wax_cards(t_med)

        # struttura “alla Wintersteiger” + lamine
        disc = st.multiselect(f"Discipline (Blocco {L})", ["SL","GS","SG","DH"], default=["SL","GS"], key=f"disc_{L}")
        rows = []
        shown_patterns = set()
        for d in disc:
            pattern, side, base = tuning_for(t_med, d)
            rows.append([d, pattern, f"{side:.1f}°", f"{base:.1f}°"])
            shown_patterns.add(pattern)
        if rows:
            st.table(pd.DataFrame(rows, columns=["Disciplina","Struttura","Lamina SIDE (°)","Lamina BASE (°)"]))
        # anteprime strutture richieste
        if shown_patterns:
            cols = st.columns(min(3, len(shown_patterns)))
            for i, p in enumerate(sorted(list(shown_patterns))):
                img = draw_structure(p)
                cols[i % len(cols)].image(img, caption=p, use_column_width=True)

    st.download_button("Scarica CSV risultato", data=res.to_csv(index=False), file_name="forecast_with_snowT.csv", mime="text/csv")

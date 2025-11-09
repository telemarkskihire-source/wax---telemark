# streamlit_app.py  — Telemark · Pro Wax & Tune (autocomplete + A/B/C + wax + structure + edges)
import streamlit as st
import pandas as pd
import requests
import base64
import io
import math
import matplotlib.pyplot as plt
from datetime import time
from dateutil import tz

# -------------------- PAGE & THEME --------------------
PRIMARY = "#10bfcf"; BG = "#0f172a"; CARD = "#111827"; TEXT = "#e5e7eb"
st.set_page_config(page_title="Telemark · Pro Wax & Tune", page_icon="❄️", layout="wide")
st.markdown(f"""
<style>
:root {{ --primary:{PRIMARY}; --bg:{BG}; --card:{CARD}; --text:{TEXT}; }}
[data-testid="stAppViewContainer"] > .main {{ background: linear-gradient(180deg, var(--bg) 0%, #111827 100%); }}
.block-container {{ padding-top: 0.8rem; }}
h1,h2,h3,h4,h5,p,span,div {{ color: var(--text); }}
.badge {{ border:1px solid rgba(255,255,255,.15); padding:6px 10px; border-radius:999px; font-size:.78rem; opacity:.85; }}
.card {{ background:var(--card); border:1px solid rgba(255,255,255,.08); border-radius:16px; padding:14px; box-shadow: 0 8px 20px rgba(0,0,0,.25); }}
.brand {{ display:flex; align-items:center; gap:10px; padding:10px 12px; border-radius:12px;
         background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08); }}
.brand img {{ height:22px; }}
.kpi {{ display:flex; gap:8px; align-items:center; background:rgba(16,191,207,.06);
       border:1px dashed rgba(16,191,207,.45); padding:10px 12px; border-radius:12px; }}
.kpi .lab {{ font-size:.78rem; color:#93c5fd; }}
.kpi .val {{ font-size:1rem; font-weight:800; color:var(--text); }}
.input-slim input {{ border-radius:12px !important; }}
.btn-primary button {{ width:100%; background:var(--primary) !important; color:#002b30 !important; border:none; font-weight:700; border-radius:12px; }}
ul.suggest {{ list-style:none; margin:.25rem 0 0 0; padding:0; border:1px solid rgba(255,255,255,.12); border-radius:12px; overflow:hidden;
              background:#0b1222; }}
ul.suggest li {{ padding:8px 12px; border-bottom:1px solid rgba(255,255,255,.08); cursor:pointer; }}
ul.suggest li:last-child {{ border-bottom:none; }}
ul.suggest li:hover {{ background:rgba(255,255,255,.06); }}
</style>
""", unsafe_allow_html=True)

st.markdown("### Telemark · Pro Wax & Tune")
st.markdown("<span class='badge'>Ricerca tipo meteoblue · Geolocalizzazione · Blocchi A/B/C · Sciolina + Struttura + Lamine</span>", unsafe_allow_html=True)

# -------------------- MODEL (snow temperature) --------------------
def compute_snow_temperature(df, dt_hours=1.0):
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
            T_top5.iloc[i] = T_top5.iloc[i-1] + alpha.iloc[i]*(T_surf.iloc[i] - T_top5.iloc[i-1])

    df["T_surf"] = T_surf
    df["T_top5"] = T_top5
    return df

# -------------------- LOCATION SEARCH (meteoblue-style UX) --------------------
# We simulate the “as-you-type” dropdown suggestions without extra libs:
# - text_input updates value on each keystroke -> Streamlit reruns -> we call Nominatim
# - we render a UL dropdown; each item is a clickable button-like element.

def geocode_autocomplete(q, limit=8):
    if not q or len(q.strip()) < 2:
        return []
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": q, "format":"json", "limit": limit, "addressdetails": 1},
            headers={"User-Agent": "telemark-pro-wax/1.0"},
            timeout=10
        )
        r.raise_for_status()
        js = r.json()
        out = []
        for it in js:
            out.append({
                "label": it.get("display_name",""),
                "lat": float(it.get("lat",0)),
                "lon": float(it.get("lon",0)),
            })
        return out
    except Exception:
        return []

def ip_geolocate():
    try:
        r = requests.get("https://ipapi.co/json", timeout=8)
        if r.ok:
            j = r.json()
            return float(j.get("latitude",0)), float(j.get("longitude",0)), j.get("city","")
    except Exception:
        pass
    return None, None, ""

# Search bar + live suggestions (no lat/lon fields)
col_search, col_geo = st.columns([3,1])
with col_search:
    q = st.text_input("Cerca località", placeholder="Es. Champoluc, Cervinia, Sestriere…", key="q", label_visibility="visible")
    sugg = geocode_autocomplete(q, limit=10)
    chosen = None
    if sugg:
        # Render a dropdown list; each item sets session state when clicked
        st.markdown("<ul class='suggest'>", unsafe_allow_html=True)
        for i, s in enumerate(sugg):
            if st.button(s["label"], key=f"s_{i}"):
                st.session_state["loc_label"] = s["label"]
                st.session_state["loc_lat"] = s["lat"]
                st.session_state["loc_lon"] = s["lon"]
                chosen = s
        st.markdown("</ul>", unsafe_allow_html=True)

with col_geo:
    if st.button("📍 Geolocalizza"):
        lat, lon, city = ip_geolocate()
        if lat is not None:
            st.session_state["loc_label"] = city or "La tua posizione"
            st.session_state["loc_lat"] = lat
            st.session_state["loc_lon"] = lon
            st.success("Posizione impostata.")
        else:
            st.error("Geolocalizzazione non disponibile.")

loc_label = st.session_state.get("loc_label", "Champoluc (Ramey)")
loc_lat   = st.session_state.get("loc_lat", 45.831)
loc_lon   = st.session_state.get("loc_lon", 7.730)

col_tz, col_hours = st.columns([1.2, 1.2])
with col_tz:
    tzname = st.selectbox("Timezone", ["Europe/Rome","UTC"], index=0)
with col_hours:
    hours = st.slider("Ore previsione", 12, 168, 72, 12)

# -------------------- A/B/C WINDOWS --------------------
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

# -------------------- FETCH & BUILD --------------------
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
    df["time"] = pd.to_datetime(df["time"])  # keep naive to avoid tz comparisons
    now_naive = pd.Timestamp.now().floor("H")
    df = df[df["time"] >= now_naive].head(hours).reset_index(drop=True)

    out = pd.DataFrame()
    out["time"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    out["T2m"] = df["temperature_2m"].astype(float)
    out["cloud"] = (df["cloudcover"].astype(float)/100).clip(0,1)
    out["wind"] = (df["windspeed_10m"].astype(float)/3.6).round(3)   # km/h -> m/s
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

# -------------------- WAX BANDS (multi-brand) --------------------
BANDS = {
    "Swix":       [("PS5 Turquoise",-18,-10),("PS6 Blue",-12,-6),("PS7 Violet",-8,-2),("PS8 Red",-4,4),("PS10 Yellow",0,10)],
    "Toko":       [("Blue",-30,-9),("Red",-12,-4),("Yellow",-6,0)],
    "Vola":       [("MX-E Violet/Blue",-12,-4),("MX-E Red",-5,0),("MX-E Warm",-2,10)],
    "Rode":       [("R20 Blue",-18,-8),("R30 Violet",-10,-3),("R40 Red",-5,0),("R50 Yellow",-1,10)],
    "Holmenkol":  [("UltraMix Blue",-20,-8),("BetaMix Red",-14,-4),("AlphaMix Yellow",-4,5)],
    "Maplus":     [("Universal Cold",-12,-6),("Universal Medium",-7,-2),("Universal Warm",-3,6)],
    "Start":      [("SG Blue",-12,-6),("SG Purple",-8,-2),("SG Red",-3,7)],
    "Skigo":      [("Paraffin Blue",-12,-6),("Paraffin Violet",-8,-2),("Paraffin Red",-3,2)],
}
def pick_band(bands, t):
    for name, tmin, tmax in bands:
        if t >= tmin and t <= tmax:
            return name
    return bands[-1][0] if t > bands[-1][2] else bands[0][0]

def wax_cards(t_med):
    # brand “logos” as colored SVG pills (placeholders)
    def svg_data_uri(text, color):
        svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='200' height='36'><rect width='200' height='36' rx='6' fill='{color}'/><text x='12' y='24' font-size='16' font-weight='700' fill='white'>{text}</text></svg>"
        return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    COLORS = {
        "Swix":"#ef4444","Toko":"#f59e0b","Vola":"#3b82f6","Rode":"#22c55e",
        "Holmenkol":"#06b6d4","Maplus":"#f43f5e","Start":"#8b5cf6","Skigo":"#a3e635",
    }
    brands = list(BANDS.keys())
    rows = [st.columns(4), st.columns(4)]
    for idx, brand in enumerate(brands):
        rec = pick_band(BANDS[brand], t_med)
        pill = svg_data_uri(brand.upper(), COLORS[brand])
        col = rows[0] if idx < 4 else rows[1]
        col[idx % 4].markdown(
            f"<div class='brand'><img src='{pill}'/><div>"
            f"<div style='font-size:.8rem;opacity:.85'>{brand}</div>"
            f"<div style='font-weight:800'>{rec}</div></div></div>", unsafe_allow_html=True
        )

# -------------------- STRUCTURE + EDGES --------------------
def draw_structure(kind="linear", width=420, height=90):
    """Return PNG bytes of a schematic structure (Wintersteiger-like)."""
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    ax = plt.gca()
    ax.set_facecolor("#222831")
    ax.set_xlim(0, 100); ax.set_ylim(0, 20)
    ax.axis("off")

    # base strip
    ax.add_patch(plt.Rectangle((0,0), 100, 20, color="#2b3440"))

    def line(x1,y1,x2,y2,lw=1.5,alpha=0.85):
        ax.plot([x1,x2],[y1,y2], linewidth=lw, color="#cfd8e3", alpha=alpha)

    if kind == "linear":
        # parallel fine lines
        for x in range(0, 101, 4):
            line(x, 1, x, 19, lw=1.2, alpha=0.9)
    elif kind == "cross":
        # cross hatch
        for d in range(-40, 140, 6):
            line(d, 0, d+40, 20, lw=1.1)
        for d in range(0, 180, 6):
            line(d, 0, d-40, 20, lw=1.1)
    elif kind == "wave":
        import numpy as np
        xs = np.linspace(0, 100, 400)
        for ph in (0, 6, 12):
            ys = 10 + 6*np.sin((xs/7.0)+ph)
            ax.plot(xs, ys, color="#cfd8e3", linewidth=1.6, alpha=0.9)
    elif kind == "fishbone":
        # chevron / lisca di pesce
        for x in range(0, 101, 8):
            line(x, 0, x+12, 10, lw=1.3)
            line(x, 20, x+12, 10, lw=1.3)
    else:
        for x in range(0, 101, 5):
            line(x, 1, x, 19, lw=1.2)
    buf = io.BytesIO()
    plt.tight_layout(pad=0.1)
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def tune_for(t_surf, discipline):
    # side in degrees like 88.0, 87.5... ; base in degrees 0.5..1.0
    if t_surf <= -10:
        structure = "Fine lineare"
        base = 0.5
        side_map = {"SL": 88.5, "GS": 88.0, "SG": 87.5, "DH": 87.5}
        kind = "linear"
    elif t_surf <= -3:
        structure = "Media universale"
        base = 0.7
        side_map = {"SL": 88.0, "GS": 88.0, "SG": 87.5, "DH": 87.0}
        kind = "cross"
    else:
        structure = "Media-grossa (onda/lisca)"
        base = 0.8 if t_surf <= 0.5 else 1.0
        side_map = {"SL": 88.0, "GS": 87.5, "SG": 87.0, "DH": 87.0}
        kind = "wave"
    return structure, side_map.get(discipline, 88.0), base, kind

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

# -------------------- RUN PIPELINE --------------------
col_run1, col_run2 = st.columns([1,2])
with col_run1:
    go = st.button("Scarica previsioni", type="primary")
with col_run2:
    upl = st.file_uploader("…oppure carica CSV (time,T2m,cloud,wind,sunup,prp_mmph,prp_type,td)", type=["csv"])

def make_plots(res):
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

def run_all(src_df):
    res = compute_snow_temperature(src_df, dt_hours=1.0)
    st.success(f"Dati pronti per **{loc_label}**")
    st.dataframe(res, use_container_width=True)
    f1,f2 = make_plots(res); st.pyplot(f1); st.pyplot(f2)

    st.download_button("Scarica CSV risultato", data=res.to_csv(index=False),
                       file_name="forecast_with_snowT.csv", mime="text/csv")

    # Blocks A/B/C
    for L,(s,e) in {"A":(A_start,A_end), "B":(B_start,B_end), "C":(C_start,C_end)}.items():
        st.markdown(f"### Blocco {L}")
        W = window_slice(res, tzname, s, e)
        _ = badge(W)
        t_med = float(W["T_surf"].mean())
        st.markdown(f"**T_surf medio {L}: {t_med:.1f} °C**")
        wax_cards(t_med)

        # Tuning (discipline)
        disc = st.multiselect(f"Discipline (Blocco {L})", ["SL","GS","SG","DH"],
                              default=["SL","GS"], key=f"d_{L}")
        rows = []
        st_cols = st.columns(2)
        with st_cols[0]:
            # table of angles + structure text
            for d in disc:
                structure, side, base, kind = tune_for(t_med, d)
                rows.append([d, structure, f"{side:.1f}°", f"{base:.1f}°"])
            if rows:
                st.table(pd.DataFrame(rows, columns=["Disciplina","Struttura consigliata",
                                                     "Lamina SIDE (°)","Lamina BASE (°)"]))
        with st_cols[1]:
            # preview image for typical structure in this block
            # choose most “aggressiva” se caldo/umido
            k = "wave" if t_med > -3 else ("cross" if t_med > -10 else "linear")
            img = draw_structure(k)
            st.image(img, caption=f"Anteprima struttura: {k}", use_container_width=True)

# Source: CSV or fetch
if upl is not None:
    try:
        df_u = pd.read_csv(upl)
        run_all(df_u)
    except Exception as e:
        st.error(f"CSV non valido: {e}")

if go:
    try:
        js = fetch_open_meteo(st.session_state.get("loc_lat", loc_lat),
                              st.session_state.get("loc_lon", loc_lon),
                              tzname)
        df_src = build_df(js, hours, tzname)
        run_all(df_src)
    except Exception as e:
        st.error(f"Errore: {e}")

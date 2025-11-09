import streamlit as st
import pandas as pd
import requests
import math
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import time
from dateutil import tz
from streamlit_searchbox import st_searchbox

# -------------------- CONFIG & THEME --------------------
st.set_page_config(page_title="Telemark · Pro Wax & Tune", page_icon="❄️", layout="wide")

PRIMARY = "#10bfcf"; BG = "#0f172a"; CARD = "#111827"; TEXT = "#e5e7eb"
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
  background: linear-gradient(180deg, {BG} 0%, #111827 100%);
}}
.block-container {{ padding-top: 1rem; }}
h1,h2,h3,h4,h5,p,span,div {{ color:{TEXT}; }}
.badge {{
  border:1px solid rgba(255,255,255,.15); padding:6px 10px;
  border-radius:999px; font-size:.78rem; opacity:.85;
}}
.card {{
  background:{CARD}; border:1px solid rgba(255,255,255,.08);
  border-radius:16px; padding:14px; box-shadow: 0 8px 20px rgba(0,0,0,.25);
}}
.kpi {{
  display:flex; gap:8px; align-items:center;
  background:rgba(16,191,207,.06); border:1px dashed rgba(16,191,207,.45);
  padding:10px 12px; border-radius:12px;
}}
.kpi .lab {{ font-size:.78rem; color:#93c5fd; }}
.kpi .val {{ font-size:1rem; font-weight:800; color:{TEXT}; }}
.brand {{
  display:flex; align-items:center; gap:8px; padding:8px 10px;
  border-radius:12px; background:rgba(255,255,255,.03);
  border:1px solid rgba(255,255,255,.08);
}}
.brand img {{ height:22px; }}
hr {{ border-color: rgba(255,255,255,.1) }}
</style>
""", unsafe_allow_html=True)

st.markdown("### Telemark · Pro Wax & Tune")
st.markdown("<span class='badge'>Ricerca tipo Meteoblue · Geolocalizzazione · Blocchi A/B/C · Sciolina + Strutture + Lamine</span>", unsafe_allow_html=True)

# -------------------- UTILS --------------------
def country_flag(cc: str) -> str:
    """Convert ISO country code to emoji flag."""
    if not cc or len(cc) != 2: return ""
    base = 127397
    return chr(ord(cc[0].upper()) + base) + chr(ord(cc[1].upper()) + base)

@st.cache_data(show_spinner=False)
def nominatim_search(q: str, limit: int = 10):
    """Live search to Nominatim (like Meteoblue suggestions)."""
    if not q or len(q) < 2:
        return []
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": q, "format":"json", "limit": limit, "addressdetails": 1}
    headers = {"User-Agent": "telemark-pro-wax/1.0"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        out = []
        for item in r.json():
            addr = item.get("address", {})
            city = addr.get("city") or addr.get("town") or addr.get("village") or ""
            state = addr.get("state") or ""
            country = addr.get("country") or ""
            cc = addr.get("country_code", "")
            label = ", ".join([s for s in [city, state, country] if s])
            if not label:
                label = item.get("display_name", "")[:80]
            flag = country_flag(cc)
            out.append({
                "id": f'{item.get("osm_type","")}_{item.get("osm_id","")}',
                "name": f"{flag} {label}".strip(),
                "lat": float(item.get("lat", 0.0)),
                "lon": float(item.get("lon", 0.0)),
                "raw": item
            })
        return out
    except Exception:
        return []

def searchbox_fn(searchterm: str):
    """Adapter for streamlit-searchbox: returns list of strings or dicts."""
    res = nominatim_search(searchterm, limit=10)
    # streamlit-searchbox supports list of dicts; each dict must have 'name' or 'title'
    return res

def ip_geolocate():
    try:
        r = requests.get("https://ipapi.co/json", timeout=8)
        if r.ok:
            j = r.json()
            return float(j.get("latitude",0)), float(j.get("longitude",0)), j.get("city","")
    except Exception:
        pass
    return None, None, ""

# -------------------- FORECAST PIPELINE --------------------
def prp_type_from_codes(df):
    snow_codes = {71,73,75,77,85,86}
    rain_codes = {51,53,55,61,63,65,80,81,82}
    def f(row):
        prp = row.get("precipitation", 0.0)
        rain = row.get("rain", 0.0)
        snow = row.get("snowfall", 0.0)
        if prp <= 0 or pd.isna(prp): return "none"
        if rain>0 and snow>0: return "mixed"
        if snow>0 and rain==0: return "snow"
        if rain>0 and snow==0: return "rain"
        code = int(row.get("weathercode", 0)) if pd.notna(row.get("weathercode", None)) else 0
        if code in snow_codes: return "snow"
        if code in rain_codes: return "rain"
        return "mixed"
    return df.apply(f, axis=1)

def fetch_open_meteo(lat, lon, timezone_str):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon, "timezone": timezone_str,
        "hourly": ",".join([
            "temperature_2m","dew_point_2m","precipitation","rain","snowfall",
            "cloudcover","windspeed_10m","is_day","weathercode"
        ]),
        "forecast_days": 7
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def build_df(js, hours, tzname):
    h = js["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])  # keep naive to avoid tz mixing
    now = pd.Timestamp.now().floor("H")
    df = df[df["time"] >= now].head(hours).reset_index(drop=True)
    out = pd.DataFrame()
    out["time"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    out["T2m"] = df["temperature_2m"].astype(float)
    out["cloud"] = (df["cloudcover"].astype(float) / 100).clip(0,1)
    out["wind"]  = (df["windspeed_10m"].astype(float) / 3.6).round(3)  # km/h -> m/s
    out["sunup"] = df["is_day"].astype(int)
    out["prp_mmph"] = df["precipitation"].astype(float)
    extra = df[["precipitation","rain","snowfall","weathercode"]].copy()
    out["prp_type"] = prp_type_from_codes(extra.to_dict("records"))
    out["td"] = df["dew_point_2m"].astype(float)
    return out

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

# -------------------- STRUCTURE RENDERING --------------------
def draw_structure(structure: str, fineness: str) -> BytesIO:
    """
    Render realistic base structures:
    - structure: 'lineare', 'croce', 'chevron', 'onda'
    - fineness: 'fine' | 'media' | 'grossa'  (spaziatura)
    """
    spacing = {"fine": 6, "media": 10, "grossa": 16}[fineness]
    lw = {"fine": 1.2, "media": 1.8, "grossa": 2.4}[fineness]

    fig = plt.figure(figsize=(4, 2), dpi=200)
    ax = plt.gca()
    ax.set_facecolor("#d9d9d9")
    ax.set_xlim(0, 200); ax.set_ylim(0, 100)
    ax.axis("off")

    def draw_linear():
        for x in range(0, 210, spacing):
            ax.plot([x, x], [0, 100], linewidth=lw, color="#555555")

    def draw_cross():
        # vertical
        for x in range(0, 210, spacing):
            ax.plot([x, x], [0, 100], linewidth=lw*0.9, color="#565656")
        # 45°
        for k in range(-100, 300, spacing):
            ax.plot([k, k+100], [0, 100], linewidth=lw*0.9, color="#4c4c4c")

    def draw_chevron():
        # V a spina di pesce
        for x in range(-100, 300, spacing*2):
            ax.plot([x, x+80], [0, 80], linewidth=lw, color="#505050")
            ax.plot([x+80, x+160], [80, 0], linewidth=lw, color="#505050")

    def draw_wave():
        import numpy as np
        for x0 in range(0, 200, spacing):
            xs = np.linspace(0, 200, 400)
            ys = 10 * np.sin(0.08*xs + x0/10.0) + 50
            ax.plot(xs, ys, linewidth=lw, color="#545454")

    if structure == "lineare":
        draw_linear()
    elif structure == "croce":
        draw_cross()
    elif structure == "chevron":
        draw_chevron()
    elif structure == "onda":
        draw_wave()
    else:
        draw_linear()

    buf = BytesIO()
    plt.tight_layout(pad=0.2)
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def structure_for_temp(t_surf: float):
    """
    Scelta struttura in base alla T neve:
    <= -10: lineare fine
    -10 .. -3: lineare/croce media
    > -3: chevron o onda grossa
    """
    if t_surf <= -10:
        return ("lineare", "fine")
    elif t_surf <= -3:
        return ("croce", "media")
    else:
        return ("chevron", "grossa")

# -------------------- WAX BANDS (8 brands) --------------------
SWIX = [("PS5 Turquoise",-18,-10),("PS6 Blue",-12,-6),("PS7 Violet",-8,-2),("PS8 Red",-4,4),("PS10 Yellow",0,10)]
TOKO = [("Blue",-30,-9),("Red",-12,-4),("Yellow",-6,0)]
VOLA = [("MX-E Violet/Blue",-12,-4),("MX-E Red",-5,0),("MX-E Warm",-2,10)]
RODE = [("R20 Blue",-18,-8),("R30 Violet",-10,-3),("R40 Red",-5,0),("R50 Yellow",-1,10)]
HOLM = [("Ultra/Alpha Blue",-20,-8),("Beta/Red",-14,-4),("Alpha Yellow",-4,5)]
MAPL = [("Universal Cold",-12,-6),("Universal Medium",-7,-2),("Universal Soft",-5,1)]
START= [("SG Blue",-12,-6),("SG Purple",-8,-2),("SG Red",-3,7)]
SKIGO= [("Blue",-12,-6),("Violet",-8,-2),("Red",-3,2)]

BRANDS = [
    ("Swix", SWIX),
    ("Toko", TOKO),
    ("Vola", VOLA),
    ("Rode", RODE),
    ("Holmenkol", HOLM),
    ("Maplus", MAPL),
    ("Start", START),
    ("Skigo", SKIGO),
]

def pick_band(bands, t):
    for name, tmin, tmax in bands:
        if t >= tmin and t <= tmax:
            return name
    # fallback
    return bands[0][0] if t < bands[0][1] else bands[-1][0]

# -------------------- LAMINE (SIDE/BASE) --------------------
def tune_for(t_surf, discipline):
    # SIDE angle (88°, 87.5°, 87°), BASE angle (0.5°..1.0°)
    if t_surf <= -10:
        structure = "Fine lineare"
        base = 0.5
        side_map = {"SL": 88.5, "GS": 88.0, "SG": 87.5, "DH": 87.5}
    elif t_surf <= -3:
        structure = "Media (lineare/croce)"
        base = 0.7
        side_map = {"SL": 88.0, "GS": 88.0, "SG": 87.5, "DH": 87.0}
    else:
        structure = "Grossa (chevron/onda)"
        base = 0.8 if t_surf <= 0.5 else 1.0
        side_map = {"SL": 88.0, "GS": 87.5, "SG": 87.0, "DH": 87.0}
    return structure, side_map.get(discipline, 88.0), base

# -------------------- UI: LOCATION (LIKE METEOBLUE) --------------------
colL, colG = st.columns([3,1])
with colL:
    st.markdown("**Cerca località**")
    selected = st_searchbox(
        search_function=searchbox_fn,
        key="locbox",
        label=None,
        default=None,
        placeholder="Digita per cercare (es. Champoluc, Cervinia, Sestriere…)",
        clear_on_submit=False,              # la tendina si aggiorna in tempo reale, nessun Enter richiesto
        debounce=150,                       # reattivo come meteoblue
        rerun=True,                         # forza rerun ad ogni carattere
    )
with colG:
    if st.button("📍 Usa la mia posizione"):
        lat, lon, city = ip_geolocate()
        if lat is not None:
            st.session_state["sel_lat"] = lat
            st.session_state["sel_lon"] = lon
            st.session_state["sel_label"] = city or "La tua posizione"
            st.success("Posizione impostata.")
        else:
            st.error("Geolocalizzazione non disponibile.")

# result of searchbox
if selected and isinstance(selected, dict):
    st.session_state["sel_lat"] = selected["lat"]
    st.session_state["sel_lon"] = selected["lon"]
    st.session_state["sel_label"] = selected["name"]

# Defaults if none selected yet
lat = st.session_state.get("sel_lat", 45.831)
lon = st.session_state.get("sel_lon", 7.730)
spot_label = st.session_state.get("sel_label", "Champoluc (Ramey)")

colTZ, colH = st.columns([1,1])
with colTZ:
    tzname = st.selectbox("Timezone", ["Europe/Rome","UTC"], index=0)
with colH:
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

go = st.button("Scarica previsioni", type="primary")

# -------------------- MAIN FLOW --------------------
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
        color, title, desc = "#ef4444", "CRITICAL", "Neve bagnata/pioggia · struttura grossa"
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

def plot_series(res):
    fig1 = plt.figure()
    t = pd.to_datetime(res["time"])
    plt.plot(t, res["T2m"], label="T2m")
    plt.plot(t, res["T_surf"], label="T_surf")
    plt.plot(t, res["T_top5"], label="T_top5")
    plt.legend(); plt.title("Temperature vs tempo"); plt.xlabel("Ora"); plt.ylabel("°C")
    return fig1

if go:
    try:
        js = fetch_open_meteo(lat, lon, tzname)
        src = build_df(js, hours, tzname)
        res = compute_snow_temperature(src, dt_hours=1.0)

        st.success(f"Dati per **{spot_label}** pronti.")
        st.dataframe(res, use_container_width=True)
        st.pyplot(plot_series(res))
        st.download_button("Scarica CSV", data=res.to_csv(index=False), file_name="forecast_with_snowT.csv", mime="text/csv")

        blocks = {"A":(A_start,A_end),"B":(B_start,B_end),"C":(C_start,C_end)}
        for L,(s,e) in blocks.items():
            st.markdown(f"### Blocco {L}")
            W = window_slice(res, tzname, s, e)
            wet = badge(W)
            t_med = float(W["T_surf"].mean())
            st.markdown(f"**T neve media: {t_med:.1f} °C**")

            # Wax brands (8)
            cols = st.columns(4)
            for i,(brand,bands) in enumerate(BRANDS[:4]):
                cols[i].markdown(
                    f"<div class='brand'><div><div style='font-size:.8rem;opacity:.85'>{brand}</div><div style='font-weight:800'>{pick_band(bands,t_med)}</div></div></div>",
                    unsafe_allow_html=True
                )
            cols2 = st.columns(4)
            for i,(brand,bands) in enumerate(BRANDS[4:]):
                cols2[i].markdown(
                    f"<div class='brand'><div><div style='font-size:.8rem;opacity:.85'>{brand}</div><div style='font-weight:800'>{pick_band(bands,t_med)}</div></div></div>",
                    unsafe_allow_html=True
                )

            # Structure preview (stile Wintersteiger semplificato)
            struct, finez = structure_for_temp(t_med)
            img = draw_structure(struct, finez)
            st.markdown(f"*Struttura consigliata:* **{struct} · {finez}**")
            st.image(img, caption=f"Pattern: {struct} · {finez}", use_column_width=False)

            # Edge tuning
            disc = st.multiselect(f"Discipline per lamine (Blocco {L})", ["SL","GS","SG","DH"], default=["SL","GS"], key=f"disc_{L}")
            rows = []
            for d in disc:
                struc_txt, side, base = tune_for(t_med, d)
                rows.append([d, struc_txt, f"{side:.1f}°", f"{base:.1f}°"])
            if rows:
                st.table(pd.DataFrame(rows, columns=["Disciplina","Struttura","Lamina SIDE (°)","Lamina BASE (°)"]))

    except Exception as e:
        st.error(f"Errore: {e}")
else:
    st.info("Cerca una località (la tendina si aggiorna mentre digiti) e premi **Scarica previsioni**.")

import streamlit as st
import pandas as pd
import requests, base64, os, io
import matplotlib.pyplot as plt
from datetime import time
from dateutil import tz
from streamlit_searchbox import st_searchbox  # AUTOCOMPLETE LIVE

# ----------------- CONFIG & THEME -----------------
st.set_page_config(page_title="Telemark · Pro Wax & Tune", page_icon="❄️", layout="wide")
PRIMARY = "#10bfcf"; BG = "#0f172a"; CARD = "#111827"; TEXT = "#e5e7eb"

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
  background: linear-gradient(180deg, {BG} 0%, #111827 100%);
}}
.block-container {{ padding-top: 0.8rem; }}
h1,h2,h3,h4,h5,p,span,div {{ color:{TEXT}; }}
.badge {{ border:1px solid rgba(255,255,255,.15); padding:6px 10px; border-radius:999px; font-size:.78rem; opacity:.85; }}
.card {{ background:{CARD}; border:1px solid rgba(255,255,255,.10); border-radius:16px; padding:14px; box-shadow: 0 8px 20px rgba(0,0,0,.25);}}
.kpi {{ display:flex; gap:8px; align-items:center; background:rgba(16,191,207,.06); border:1px dashed rgba(16,191,207,.45); padding:10px 12px; border-radius:12px; }}
.kpi .lab {{ font-size:.78rem; color:#93c5fd; }}
.kpi .val {{ font-size:1rem; font-weight:800; color:{TEXT}; }}
.brand {{ display:flex; align-items:center; gap:10px; padding:10px 12px; border-radius:12px; background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08);}}
.brand img {{ height:22px; }}
hr {{ border-color: rgba(255,255,255,.12) }}
</style>
""", unsafe_allow_html=True)

st.markdown("### Telemark · Pro Wax & Tune")
st.markdown("<span class='badge'>Ricerca tipo Meteoblue · Geolocalizzazione · Blocchi A/B/C · Sciolina · Struttura · Lamine</span>", unsafe_allow_html=True)

# ----------------- SNOW MODEL -----------------
def compute_snow_temperature(df, dt_hours=1.0):
    import math
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    req = ["T2m","cloud","wind","sunup","prp_mmph","prp_type"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Manca colonna {c}")
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

# ----------------- PLACE SEARCH (Meteoblue-like UX) -----------------
# Callback called on each keypress by st_searchbox -> returns list of (label, id)
def _nominatim(query: str):
    if not query or len(query) < 2:
        return []
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 10, "addressdetails": 1},
            headers={"User-Agent": "telemark-pro-wax/1.0"},
            timeout=8
        )
        r.raise_for_status()
        js = r.json()
        opts = []
        for i, item in enumerate(js):
            label = item.get("display_name","")
            lat = float(item.get("lat", 0.0)); lon = float(item.get("lon", 0.0))
            # Bundle coords into a compact id string to retrieve later
            opts.append((label, f"{lat:.6f},{lon:.6f}"))
        return opts
    except Exception:
        return []

def geolocate_ip():
    try:
        r = requests.get("https://ipapi.co/json", timeout=6)
        if r.ok:
            j = r.json()
            return (float(j.get("latitude",0)), float(j.get("longitude",0)), j.get("city",""))
    except Exception:
        pass
    return (None, None, "")

left, right = st.columns([3,1])
with left:
    place = st_searchbox(
        search_function=_nominatim,
        placeholder="Cerca località (digita e scegli) — tipo Meteoblue",
        key="place_search",
        default="Champoluc, Aosta Valley, Italy",
        highlight=True
    )
with right:
    if st.button("📍 Usa la mia posizione"):
        lat_ip, lon_ip, city = geolocate_ip()
        if lat_ip is not None:
            st.session_state["chosen_lat"] = lat_ip
            st.session_state["chosen_lon"] = lon_ip
            st.session_state["chosen_label"] = city or "La tua posizione"
            st.success("Posizione impostata.")
        else:
            st.error("Geolocalizzazione non disponibile.")

# Decode selected place
if place and isinstance(place, str) and "," in place and place.count(",") == 1:
    # It might already be "lat,lon" if user selected from dropdown
    try:
        lat, lon = [float(x) for x in place.split(",")]
        label = st.session_state.get("chosen_label", "Località")
    except:
        lat = 45.831; lon = 7.730; label = "Champoluc (Ramey)"
else:
    # If 'place' is label (first run), we try to geocode once
    lat = st.session_state.get("chosen_lat", 45.831)
    lon = st.session_state.get("chosen_lon", 7.730)
    label = st.session_state.get("chosen_label", "Champoluc (Ramey)")

col_lat, col_lon, col_tz, col_h = st.columns([1,1,1.3,1.3])
with col_lat: lat = st.number_input("Lat", value=float(lat), format="%.6f")
with col_lon: lon = st.number_input("Lon", value=float(lon), format="%.6f")
with col_tz: tzname = st.selectbox("Timezone", ["Europe/Rome","UTC"], index=0)
with col_h: hours = st.slider("Ore previsione", 12, 168, 72, 12)

# ----------------- A/B/C WINDOWS -----------------
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

# ----------------- WEATHER FETCH -----------------
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
    h = js["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])   # naive
    now_naive = pd.Timestamp.now().floor("H")
    df = df[df["time"] >= now_naive].head(hours).reset_index(drop=True)

    out = pd.DataFrame()
    out["time"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    out["T2m"] = df["temperature_2m"].astype(float)
    out["cloud"] = (df["cloudcover"].astype(float)/100).clip(0,1)
    out["wind"] = (df["windspeed_10m"].astype(float)/3.6).round(3)  # km/h -> m/s
    out["sunup"] = df["is_day"].astype(int)
    out["prp_mmph"] = df["precipitation"].astype(float)
    extra = df[["precipitation","rain","snowfall","weathercode"]].copy()
    out["prp_type"] = prp_type(extra)
    out["td"] = df["dew_point_2m"].astype(float)
    return out

# ----------------- BRAND WAX BANDS -----------------
BRANDS = {
    "Swix": [("PS5 Turquoise", -18,-10), ("PS6 Blue",-12,-6), ("PS7 Violet",-8,-2), ("PS8 Red",-4,4), ("PS10 Yellow",0,10)],
    "Toko": [("Blue",-30,-9), ("Red",-12,-4), ("Yellow",-6,0)],
    "Vola": [("MX-E Cold",-12,-4), ("MX-E Red",-5,0), ("MX-E Warm",-2,10)],
    "Rode": [("R20 Blue",-18,-8), ("R30 Violet",-10,-3), ("R40 Red",-5,0), ("R50 Yellow",-1,10)],
    "Holmenkol": [("Ultra Mix Blue",-20,-8), ("BetaMix Red",-14,-4), ("AlphaMix Yellow",-4,5)],
    "Maplus": [("Universal Cold",-12,-6), ("Universal Medium",-7,-2), ("Universal Warm",-3,6)],
    "Start": [("SG Blue",-12,-6), ("SG Purple",-8,-2), ("SG Red",-3,7)],
    "Skigo": [("Blue",-12,-6), ("Violet",-8,-2), ("Red",-3,2)],
}
def pick_wax(bands, t):
    for name, tmin, tmax in bands:
        if t >= tmin and t <= tmax:
            return name
    return bands[-1][0] if t > bands[-1][2] else bands[0][0]

# ----------------- STRUCTURE DRAWINGS -----------------
def draw_structure(kind="linear_fine"):
    """Return PNG bytes for a more 'technical' structure preview."""
    fig = plt.figure(figsize=(3.6, 0.9), dpi=200)  # wide, shallow strip like a base
    ax = plt.gca()
    ax.set_facecolor("#ececec")
    ax.set_xlim(0,100); ax.set_ylim(0,20)
    ax.axis("off")

    def lines(step, amp=4, phase=0, angled=False):
        import numpy as np
        x = np.arange(0, 101, 0.2)
        if not angled:
            for xi in range(0, 101, step):
                ax.plot([xi, xi], [0, 20], lw=1.2, color="#555")
        else:
            for xi in range(-40, 140, step):
                ax.plot([xi, xi+40], [0, 20], lw=1.2, color="#555")

    def chevron(step=10, opening=10):
        for x0 in range(0,100,step):
            ax.plot([x0, x0+opening], [0, 10], color="#555", lw=1.2)
            ax.plot([x0+opening, x0], [10, 20], color="#555", lw=1.2)

    def wave(step=6, amp=2.5):
        import numpy as np
        xs = np.linspace(0,100,400)
        for k in range(0,20,6):
            ys = 10 + amp*np.sin(2*np.pi*(xs/step) + (k*0.4))
            ax.plot(xs, ys, color="#555", lw=1.2)

    def lateral_drain(step=10):
        # dense parallel + gutters near edges
        lines(step=6)
        ax.add_patch(plt.Rectangle((0,0), 100, 2.5, color="#777"))
        ax.add_patch(plt.Rectangle((0,17.5), 100, 2.5, color="#777"))

    if kind == "linear_fine":
        lines(step=8)
    elif kind == "linear_medium":
        lines(step=12)
    elif kind == "linear_coarse":
        lines(step=18)
    elif kind == "chevron":
        chevron(step=12, opening=10)
    elif kind == "wave":
        wave(step=7, amp=3)
    elif kind == "lateral_drain":
        lateral_drain(step=10)

    buf = io.BytesIO()
    plt.tight_layout(pad=0.2)
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

STRUCTURE_LABELS = {
    "linear_fine": "Lineare fine (freddo)",
    "linear_medium": "Lineare media (universale)",
    "linear_coarse": "Lineare grossa (caldo/umido)",
    "chevron": "Chevron (scarico multi-direz.)",
    "wave": "Onda (neve bagnata)",
    "lateral_drain": "Scarico laterale (bagnata)"
}

# ----------------- EDGE TUNING -----------------
def tuning_for_temp(t_surf, discipline):
    # SIDE angle in degrees (88°, 87.5°, 87°); BASE in degrees (0.5–1.0)
    if t_surf <= -10:
        structure = "linear_fine"; base = 0.5; sides = {"SL": 88.5, "GS": 88.0, "SG": 87.5, "DH": 87.5}
    elif t_surf <= -3:
        structure = "linear_medium"; base = 0.7; sides = {"SL": 88.0, "GS": 88.0, "SG": 87.5, "DH": 87.0}
    else:
        # vicino/oltre 0: più scarico
        structure = "linear_coarse"; base = 0.8 if t_surf <= 0.5 else 1.0; sides = {"SL": 88.0, "GS": 87.5, "SG": 87.0, "DH": 87.0}
    # versioni “bagnato” speciali
    if t_surf > -1.0:
        structure = "wave"
    if t_surf > 0.5:
        structure = "lateral_drain"
    return structure, sides.get(discipline, 88.0), base

# ----------------- LOGOS -----------------
def brand_logo_src(name: str):
    # If you put logos in ./logos/{name}.png or .svg they will show; else colored label
    for ext in (".png", ".svg", ".jpg"):
        p = f"logos/{name.lower()}{ext}"
        if os.path.exists(p):
            if ext == ".svg":
                with open(p, "rb") as f:
                    enc = base64.b64encode(f.read()).decode("utf-8")
                return f"data:image/svg+xml;base64,{enc}"
            else:
                with open(p, "rb") as f:
                    enc = base64.b64encode(f.read()).decode("utf-8")
                return f"data:image/png;base64,{enc}"
    # fallback badge
    color = {"Swix":"#ef4444","Toko":"#f59e0b","Vola":"#3b82f6","Rode":"#22c55e"}.get(name, "#64748b")
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='120' height='28'><rect width='120' height='28' rx='6' fill='{color}'/><text x='10' y='19' font-size='14' font-weight='700' fill='white'>{name.upper()}</text></svg>"
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")

# ----------------- ACTIONS -----------------
colA, colB = st.columns([1,2])
with colA:
    go = st.button("Scarica previsioni per località", type="primary")
with colB:
    upl = st.file_uploader("…oppure carica CSV (time,T2m,cloud,wind,sunup,prp_mmph,prp_type,td)", type=["csv"])

def make_plots(res_df):
    fig1 = plt.figure()
    t = pd.to_datetime(res_df["time"])
    plt.plot(t, res_df["T2m"], label="T2m")
    plt.plot(t, res_df["T_surf"], label="T_surf")
    plt.plot(t, res_df["T_top5"], label="T_top5")
    plt.legend(); plt.title("Temperature vs tempo"); plt.xlabel("Ora"); plt.ylabel("°C")
    fig2 = plt.figure()
    plt.bar(t, res_df["prp_mmph"])
    plt.title("Precipitazione (mm/h)"); plt.xlabel("Ora"); plt.ylabel("mm/h")
    return fig1, fig2

def window_slice(res, tzname, s, e):
    t = pd.to_datetime(res["time"]).dt.tz_localize(tz.gettz(tzname), nonexistent='shift_forward', ambiguous='NaT')
    D = res.copy(); D["dt"] = t
    today = pd.Timestamp.now(tz=tz.gettz(tzname)).date()
    w = D[(D["dt"].dt.date == today) & (D["dt"].dt.time >= s) & (D["dt"].dt.time <= e)]
    return w if not w.empty else D.head(7)

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
      <div style='font-size:12px;opacity:.7;margin-top:6px'>T_surf min {tmin:.1f}°C / max {tmax:.1f}°C</div>
    </div>""", unsafe_allow_html=True)
    return wet

def show_wax_cards(t_med):
    # render 8 brands
    cols = st.columns(4)
    items = list(BRANDS.items())
    for i, (brand, bands) in enumerate(items[:4]):
        rec = pick_wax(bands, t_med)
        cols[i].markdown(f"<div class='brand'><img src='{brand_logo_src(brand)}'/><div><div style='font-size:.8rem;opacity:.85'>{brand}</div><div style='font-weight:800'>{rec}</div></div></div>", unsafe_allow_html=True)
    cols2 = st.columns(4)
    for i, (brand, bands) in enumerate(items[4:8]):
        rec = pick_wax(bands, t_med)
        cols2[i].markdown(f"<div class='brand'><img src='{brand_logo_src(brand)}'/><div><div style='font-size:.8rem;opacity:.85'>{brand}</div><div style='font-weight:800'>{rec}</div></div></div>", unsafe_allow_html=True)

def show_structure_card(code_key, title, t_med):
    # pick target structure based on temp (for GS as reference)
    structure_key, side, base = tuning_for_temp(t_med, "GS")
    # allow manual override per blocco
    structure_key = st.selectbox(f"{title} · struttura soletta", list(STRUCTURE_LABELS.keys()), index=list(STRUCTURE_LABELS.keys()).index(structure_key), key=f"str_{code_key}")
    img = draw_structure(structure_key)
    st.image(img, caption=STRUCTURE_LABELS[structure_key], use_column_width=True)

    # discipline tuning table
    disc = st.multiselect(f"{title} · Discipline", ["SL","GS","SG","DH"], default=["SL","GS"], key=f"disc_{code_key}")
    rows = []
    for d in disc:
        skey, side_ang, base_ang = tuning_for_temp(t_med, d)
        rows.append([d, STRUCTURE_LABELS[skey], f"{side_ang:.1f}°", f"{base_ang:.1f}°"])
    if rows:
        st.table(pd.DataFrame(rows, columns=["Disciplina","Struttura consigliata","Lamina SIDE (°)","Lamina BASE (°)"]))

def run_all(src_df, label):
    res = compute_snow_temperature(src_df, dt_hours=1.0)
    st.success(f"Previsioni per **{label}** pronte.")
    st.dataframe(res, use_container_width=True)
    fig1, fig2 = make_plots(res); st.pyplot(fig1); st.pyplot(fig2)
    st.download_button("Scarica CSV risultato", data=res.to_csv(index=False), file_name="forecast_with_snowT.csv", mime="text/csv")

    for code, (s,e) in {"A":(A_start,A_end), "B":(B_start,B_end), "C":(C_start,C_end)}.items():
        st.markdown(f"### Blocco {code}")
        W = window_slice(res, tzname, s, e)
        wet = badge(W)
        t_med = float(W["T_surf"].mean())
        st.markdown(f"**T_surf medio blocco {code}: {t_med:.1f} °C**")
        show_wax_cards(t_med)
        show_structure_card(code, f"Blocco {code}", t_med)

# ----------------- FLOW -----------------
if upl is not None:
    try:
        df_upl = pd.read_csv(upl)
        run_all(df_upl, label)
    except Exception as e:
        st.error(f"CSV non valido: {e}")

if 'df_src' not in st.session_state and (upl is None):
    st.info("Scegli la località dal box sopra (menu a comparsa mentre digiti) oppure usa la tua posizione. Poi premi **Scarica previsioni**.")

if st.button("↻ Aggiorna ricerca (se la lista non compare subito)"):
    st.experimental_rerun()

if go:
    try:
        js = fetch_open_meteo(lat, lon, tzname)
        df_src = build_df(js, hours, tzname)
        run_all(df_src, label)
    except Exception as e:
        st.error(f"Errore: {e}")

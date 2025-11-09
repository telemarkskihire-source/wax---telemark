# telemark_pro_wax_app.py
# Telemark · Pro Wax & Tune — ricerca località, A/B/C, wax multi-brand, strutture stile Wintersteiger, angoli lamine

import streamlit as st
import pandas as pd
import requests
import base64
import io
import math
import matplotlib.pyplot as plt
from datetime import time
from dateutil import tz

# =============== CONFIG & THEME ===============
PRIMARY = "#10bfcf"; BG = "#0f172a"; CARD = "#111827"; TEXT = "#e5e7eb"
st.set_page_config(page_title="Telemark · Pro Wax & Tune", page_icon="❄️", layout="wide")
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] > .main {{ background: linear-gradient(180deg, {BG} 0%, #111827 100%); }}
.block-container {{ padding-top: 0.9rem; }}
h1,h2,h3,h4,h5,p,span,div {{ color:{TEXT}; }}
.small {{ font-size:.8rem; opacity:.8; }}
.card {{ background:{CARD}; border:1px solid rgba(255,255,255,.08); border-radius:16px; padding:14px; box-shadow: 0 8px 20px rgba(0,0,0,.25); }}
.badge {{ border:1px solid rgba(255,255,255,.12); padding:6px 10px; border-radius:999px; font-size:.78rem; opacity:.85; }}
.brand {{ display:flex; align-items:center; gap:8px; padding:8px 10px; border-radius:12px; background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08); }}
.brand img {{ height:22px; }}
.kpi {{ display:flex; flex-direction:column; gap:2px; background:rgba(16,191,207,.06); border:1px dashed rgba(16,191,207,.45); padding:10px 12px; border-radius:12px; }}
.kpi .lab {{ font-size:.78rem; color:#93c5fd; }}
.kpi .val {{ font-size:1rem; font-weight:800; color:{TEXT}; }}
.select-suggestion .stSelectbox div[data-baseweb="select"] {{ border-radius:12px; }}
hr {{ border-color: rgba(255,255,255,.1) }}
</style>
""", unsafe_allow_html=True)

st.markdown("### Telemark · Pro Wax & Tune")
st.markdown("<span class='badge'>Ricerca rapida · Finestre A/B/C · Sciolina multi-brand · Strutture stile Wintersteiger · Angoli lamine</span>", unsafe_allow_html=True)

# =============== HELPER: FLAG EMOJI ===============
def flag_emoji(cc: str) -> str:
    if not cc: return "🏳️"
    cc = cc.upper()
    return "".join(chr(127397 + ord(c)) for c in cc) if len(cc) == 2 else "🏳️"

# =============== LOCATION SEARCH (autocomplete stile Meteoblue) ===============
@st.cache_data(show_spinner=False)
def geocode_autocomplete(q: str, limit: int = 10):
    if not q or len(q) < 2:
        return []
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": q, "format": "json", "limit": limit, "addressdetails": 1},
            headers={"User-Agent": "telemark-pro-wax/1.0"},
            timeout=10,
        )
        r.raise_for_status()
        out = []
        for it in r.json():
            name = it.get("display_name", "")
            lat = float(it.get("lat", 0))
            lon = float(it.get("lon", 0))
            cc = (it.get("address") or {}).get("country_code", "")
            out.append(
                {
                    "label": f"{flag_emoji(cc)}  {name}",
                    "lat": lat,
                    "lon": lon,
                    "cc": cc,
                }
            )
        return out
    except Exception:
        return []

# BOX di ricerca tipo Meteoblue (enter “virtuale”: text_input causa rerun ad ogni carattere → aggiorna la tendina)
colA, colB = st.columns([3,1])
with colA:
    q = st.text_input("Cerca località", placeholder="Es. Champoluc, Cervinia, Sestriere…", key="search_q")
    suggestions = geocode_autocomplete(q, limit=12) if q else []
    nice_labels = [s["label"] for s in suggestions] or ["— digita per cercare —"]
    # La select è la “tendina” come su Meteoblue; si aggiorna ad ogni carattere (rerun automatico di Streamlit)
    choice = st.selectbox("Suggerimenti", nice_labels, index=0, key="search_sel",
                          help="Seleziona una località dalla lista; la lista si aggiorna mentre scrivi.",
                          label_visibility="collapsed")
with colB:
    if st.button("📍 Usa posizione IP"):
        try:
            r = requests.get("https://ipapi.co/json", timeout=8)
            if r.ok:
                j = r.json()
                st.session_state["lat"] = float(j.get("latitude", 45.831))
                st.session_state["lon"] = float(j.get("longitude", 7.730))
                st.session_state["place_label"] = f"{flag_emoji(j.get('country_code', ''))}  {j.get('city','Posizione IP')}"
                st.success("Posizione impostata.")
        except Exception:
            st.error("Geolocalizzazione non disponibile ora.")

# Se è stato scelto un suggerimento valido → aggiorna lat/lon
if suggestions and choice in [s["label"] for s in suggestions]:
    sel = suggestions[[s["label"] for s in suggestions].index(choice)]
    st.session_state["lat"] = sel["lat"]
    st.session_state["lon"] = sel["lon"]
    st.session_state["place_label"] = sel["label"]

lat = float(st.session_state.get("lat", 45.831))
lon = float(st.session_state.get("lon", 7.730))
place_label = st.session_state.get("place_label", "🇮🇹  Champoluc, Aosta Valley, Italy")

tzname = st.selectbox("Timezone", ["Europe/Rome", "UTC"], index=0)
hours = st.slider("Ore previsione", 12, 168, 72, 12)

# =============== FINESTRE A/B/C ===============
st.markdown("#### Finestre A · B · C (oggi)")
c1, c2, c3 = st.columns(3)
with c1:
    A_start = st.time_input("Inizio A", value=time(9, 0), key="A_s")
    A_end   = st.time_input("Fine A",   value=time(11, 0), key="A_e")
with c2:
    B_start = st.time_input("Inizio B", value=time(11, 0), key="B_s")
    B_end   = st.time_input("Fine B",   value=time(13, 0), key="B_e")
with c3:
    C_start = st.time_input("Inizio C", value=time(13, 0), key="C_s")
    C_end   = st.time_input("Fine C",   value=time(16, 0), key="C_e")

# =============== WEATHER FETCH & PIPELINE ===============
def fetch_open_meteo(lat, lon, timezone_str):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": timezone_str,
        "hourly": "temperature_2m,dew_point_2m,precipitation,rain,snowfall,cloudcover,windspeed_10m,is_day,weathercode",
        "forecast_days": 7,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def prp_type(df):
    snow_codes = {71, 73, 75, 77, 85, 86}
    rain_codes = {51, 53, 55, 61, 63, 65, 80, 81, 82}
    def f(row):
        prp = row.precipitation
        rain = getattr(row, "rain", 0.0)
        snow = getattr(row, "snowfall", 0.0)
        if prp <= 0 or pd.isna(prp):   return "none"
        if rain > 0 and snow > 0:      return "mixed"
        if snow > 0 and rain == 0:     return "snow"
        if rain > 0 and snow == 0:     return "rain"
        code = int(getattr(row, "weathercode", 0)) if pd.notna(getattr(row, "weathercode", None)) else 0
        if code in snow_codes: return "snow"
        if code in rain_codes: return "rain"
        return "mixed"
    return df.apply(f, axis=1)

def build_df(js, hours, tzname):
    h = js["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])
    now_naive = pd.Timestamp.now().floor("H")
    df = df[df["time"] >= now_naive].head(hours).reset_index(drop=True)
    out = pd.DataFrame()
    out["time"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    out["T2m"] = df["temperature_2m"].astype(float)
    out["cloud"] = (df["cloudcover"].astype(float) / 100).clip(0, 1)
    out["wind"] = (df["windspeed_10m"].astype(float) / 3.6).round(3)
    out["sunup"] = df["is_day"].astype(int)
    out["prp_mmph"] = df["precipitation"].astype(float)
    extra = df[["precipitation", "rain", "snowfall", "weathercode"]].copy()
    out["prp_type"] = prp_type(extra)
    out["td"] = df["dew_point_2m"].astype(float)
    return out

def compute_snow_temperature(df, dt_hours=1.0):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    for c in ["T2m", "cloud", "wind", "sunup", "prp_mmph", "prp_type"]:
        if c not in df.columns:
            raise ValueError(f"Missing {c}")
    if "td" not in df.columns:
        df["td"] = float("nan")
    df = df.sort_values("time").reset_index(drop=True)

    rain = df["prp_type"].str.lower().isin(["rain", "mixed"])
    snow = df["prp_type"].str.lower().eq("snow")
    sunup = df["sunup"].astype(int) == 1
    tw = (df["T2m"] + df["td"]) / 2.0

    wet = (
        rain
        | (df["T2m"] > 0)
        | (sunup & (df["cloud"] < 0.3) & (df["T2m"] >= -3))
        | (snow & (df["T2m"] >= -1))
        | (snow & tw.ge(-0.5).fillna(False))
    )

    T_surf = pd.Series(index=df.index, dtype=float)
    T_surf.loc[wet] = 0.0
    dry = ~wet

    clear = (1.0 - df["cloud"]).clip(lower=0.0, upper=1.0)
    windc = df["wind"].clip(upper=6.0)
    drad = (1.5 + 3.0 * clear - 0.3 * windc).clip(lower=0.5, upper=4.5)
    T_surf.loc[dry] = df["T2m"][dry] - drad[dry]

    sunny_cold = sunup & dry & df["T2m"].between(-10, 0, inclusive="both")
    T_surf.loc[sunny_cold] = pd.concat(
        [(df["T2m"] + 0.5 * (1.0 - df["cloud"]))[sunny_cold], pd.Series(-0.5, index=df.index)[sunny_cold]],
        axis=1,
    ).min(axis=1)

    T_top5 = pd.Series(index=df.index, dtype=float)
    tau = pd.Series(6.0, index=df.index, dtype=float)
    tau.loc[rain | snow | (df["wind"] >= 6)] = 3.0
    tau.loc[(~sunup) & (df["wind"] < 2) & (df["cloud"] < 0.3)] = 8.0
    alpha = 1.0 - (math.e ** (-dt_hours / tau))

    if len(df) > 0:
        T_top5.iloc[0] = min(df["T2m"].iloc[0], 0.0)
        for i in range(1, len(df)):
            T_top5.iloc[i] = T_top5.iloc[i - 1] + alpha.iloc[i] * (T_surf.iloc[i] - T_top5.iloc[i - 1])

    df["T_surf"] = T_surf
    df["T_top5"] = T_top5
    return df

# =============== WAX BANDS (NO-FLUORO) ===============
# marchi: Swix, Toko, Vola, Rode, Holmenkol, Maplus, Star, HWK, Rex
WAX_BANDS = {
    "Swix":      [("PS5 Turquoise", -18, -10), ("PS6 Blue", -12, -6), ("PS7 Violet", -8, -2), ("PS8 Red", -4,  4), ("PS10 Yellow", 0, 10)],
    "Toko":      [("Blue", -30, -9), ("Red", -12, -4), ("Yellow", -6, 0)],
    "Vola":      [("MX-E Violet/Blue", -12, -4), ("MX-E Red", -5, 0), ("MX-E Warm", -2, 10)],
    "Rode":      [("R20 Blue", -18, -8), ("R30 Violet", -10, -3), ("R40 Red", -5, 0), ("R50 Yellow", -1, 10)],
    "Holmenkol": [("Blue", -20, -8), ("Violet", -10, -2), ("Red", -4, 4)],
    "Maplus":    [("Green", -20, -8), ("Blue", -12, -4), ("Red", -5, 3), ("Yellow", 0, 10)],
    "Star":      [("Green", -20, -8), ("Blue", -12, -4), ("Violet", -8, -2), ("Red", -3, 3)],
    "HWK":       [("Cold", -20, -8), ("Mid", -10, -3), ("Warm", -2, 6)],
    "Rex":       [("Blue", -12, -5), ("Purple", -6, -1), ("Red", 0, 3)],
}

def pick_band(bands, t):
    for name, tmin, tmax in bands:
        if t >= tmin and t <= tmax:
            return name
    return bands[-1][0] if t > bands[-1][2] else bands[0][0]

def svg_data_uri(text, color):
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='200' height='36'><rect width='200' height='36' rx='6' fill='{color}'/><text x='12' y='24' font-size='16' font-weight='700' fill='white'>{text}</text></svg>"
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")

BRAND_COLORS = {
    "Swix": "#ef4444", "Toko": "#f59e0b", "Vola": "#3b82f6", "Rode": "#22c55e",
    "Holmenkol": "#60a5fa", "Maplus": "#38bdf8", "Star": "#f472b6",
    "HWK": "#93c5fd", "Rex": "#fb7185"
}

# =============== STRUCTURE DRAWINGS (stile Wintersteiger) ===============
# Generatore di strutture coerenti (lineare, diagonale, incrociata, onda, chevron)
def draw_structure(kind: str, density: str = "fine", angle_deg: float = 0.0, amplitude: float = 0.0):
    """
    Ritorna PNG bytes di una tavola 420x260 con pattern realistico:
    - kind: 'linear', 'diagonal', 'cross', 'wave', 'chevron'
    - density: 'fine' | 'medium' | 'coarse'
    """
    width, height = 4.2, 2.6  # aspect sci base
    fig = plt.figure(figsize=(width, height), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # Fondo scuro leggero
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor="#0e1116"))

    # Spaziatura in base alla densità
    if density == "fine":   step = 0.025
    elif density == "medium": step = 0.04
    else:                   step = 0.06

    line_kwargs = dict(color="#9aa4b2", linewidth=1.0, alpha=0.95)

    import numpy as np

    if kind == "linear":
        # linee verticali e sottili, parallele
        xs = np.arange(0, 1 + step, step)
        for x in xs:
            ax.plot([x, x], [0.05, 0.95], **line_kwargs)

    elif kind == "diagonal":
        # Diagonale uniforme (angolo 25–35°)
        theta = math.radians(angle_deg or 30)
        # disegniamo in spazio ruotato per avere segmenti rettilinei
        L = int(1 / step) + 3
        offs = np.linspace(-0.4, 1.4, L)
        for o in offs:
            x0 = o; y0 = 0
            x1 = o + math.tan(theta); y1 = 1
            ax.plot([x0, x1], [y0, y1], **line_kwargs)

    elif kind == "cross":
        # Incrociata: due famiglie diagonali
        theta = math.radians(30)
        L = int(1 / step) + 3
        offs = np.linspace(-0.4, 1.4, L)
        for o in offs:
            # /
            ax.plot([o, o + math.tan(theta)], [0, 1], **line_kwargs)
            # \
            ax.plot([o + math.tan(theta), o], [0, 1], **line_kwargs)

    elif kind == "wave":
        # Onda (rounded/broken wave): sinusoidi dolci
        xs = np.arange(0, 1 + step, step)
        amp = amplitude or 0.06
        freq = 6 if density == "fine" else (4 if density == "medium" else 3)
        for x in xs:
            yy = np.linspace(0.05, 0.95, 250)
            xx = x + amp * np.sin(2 * math.pi * freq * yy)
            ax.plot(xx, yy, **line_kwargs)

    elif kind == "chevron":
        # Chevron / a lisca (pattern a V ripetuta)
        freq = 12 if density == "fine" else (8 if density == "medium" else 6)
        yy = np.linspace(0.05, 0.95, 280)
        for k in np.arange(0, 1, step):
            # dente a V simmetrico
            xx = (k + 0.5 * (1 - np.abs((yy*freq) % 2 - 1)) * step)
            ax.plot(xx, yy, **line_kwargs)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=140, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return buf.getvalue()

def recommend_structure(t_surf_mean: float, wet: bool):
    """Regola semplice stile officina: freddo=linear, universale=cross, vicino 0/wet=wave/diagonal, molto bagnato=chevron/coarse"""
    if wet or t_surf_mean > 0.5:
        return ("diagonal", "coarse", 30, 0.06)   # caldo/umido → diagonale grossa per scarico
    if -2.0 <= t_surf_mean <= 0.5:
        return ("wave", "medium", 0, 0.05)       # vicino allo zero → onda
    if -6.0 <= t_surf_mean < -2.0:
        return ("cross", "medium", 0, 0.0)       # universale freddo medio
    if t_surf_mean < -10.0:
        return ("linear", "fine", 0, 0.0)        # molto freddo/ secco → lineare fine
    # default freddo “classico”
    return ("linear", "medium", 0, 0.0)

# =============== TUNING (edges) ===============
def tune_for(t_surf, discipline):
    # SIDE (gradi reali tipo 88.0°); BASE in gradi
    if t_surf <= -10:
        structure = "Fine (lineare)"; base = 0.5; side_map = {"SL": 88.5, "GS": 88.0, "SG": 87.5, "DH": 87.5}
    elif t_surf <= -3:
        structure = "Media (universale)"; base = 0.7; side_map = {"SL": 88.0, "GS": 88.0, "SG": 87.5, "DH": 87.0}
    else:
        structure = "Media-Grossa (scarico)"; base = 0.8 if t_surf <= 0.5 else 1.0; side_map = {"SL": 88.0, "GS": 87.5, "SG": 87.0, "DH": 87.0}
    return structure, side_map.get(discipline, 88.0), base

# =============== RUN ===============
colRun1, colRun2 = st.columns([1,2])
with colRun1:
    go = st.button("Scarica previsioni", type="primary")
with colRun2:
    upl = st.file_uploader("…oppure carica CSV (time,T2m,cloud,wind,sunup,prp_mmph,prp_type,td)", type=["csv"])

def window_slice(res, tzname, s, e):
    t = pd.to_datetime(res["time"]).dt.tz_localize(tz.gettz(tzname), nonexistent='shift_forward', ambiguous='NaT')
    D = res.copy(); D["dt"] = t
    today = pd.Timestamp.now(tz=tz.gettz(tzname)).date()
    W = D[(D["dt"].dt.date==today) & (D["dt"].dt.time>=s) & (D["dt"].dt.time<=e)]
    return W if not W.empty else D.head(7)

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

def wax_cards(t_med):
    cols = st.columns(3)
    items = list(WAX_BANDS.items())
    # mostriamo 9 brand in 3 righe
    for row in range(3):
        row_cols = st.columns(3) if row>0 else cols
        for i in range(3):
            idx = row*3 + i
            if idx >= len(items): break
            brand, bands = items[idx]
            rec = pick_band(bands, t_med)
            logo = svg_data_uri(brand.upper(), BRAND_COLORS.get(brand, "#64748b"))
            row_cols[i].markdown(
                f"<div class='brand'><img src='{logo}'/><div><div class='small'>{brand}</div><div style='font-weight:800'>{rec}</div></div></div>",
                unsafe_allow_html=True
            )

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

def show_structure_panel(t_med, wet):
    kind, density, ang, amp = recommend_structure(t_med, wet)
    png = draw_structure(kind, density, ang, amp)
    left, right = st.columns([1.2, 2.8])
    with left:
        st.image(png, caption=f"Struttura consigliata: {kind} / {density}", use_container_width=True)
    with right:
        st.markdown("**Perché questa struttura**")
        if kind == "linear":
            st.write("Lineare fine: neve fredda/asciutta. Riduce attrito meccanico con canali sottili e paralleli.")
        elif kind == "cross":
            st.write("Incrociata media: universale per freddo moderato. Compromesso tra scorrimento e scarico acqua.")
        elif kind == "diagonal":
            st.write("Diagonale grossa: caldo/umido. Aumenta il drenaggio longitudinale dell’acqua.")
        elif kind == "wave":
            st.write("Onda media: vicino a 0 °C. Canalizza e rompe il film d’acqua mantenendo scorrevolezza.")
        elif kind == "chevron":
            st.write("Chevron: forte scarico laterale su neve bagnata.")
        st.caption("Disegno stilizzato generato in-app sul formato soletta; ispirazione estetica: tavole macchine officina.")

def run_all(src, label):
    res = compute_snow_temperature(src, dt_hours=1.0)
    st.success(f"Previsioni per **{label}** pronte.")
    st.dataframe(res, use_container_width=True)
    f1, f2 = plots(res); st.pyplot(f1); st.pyplot(f2)
    st.download_button("Scarica CSV risultato", data=res.to_csv(index=False), file_name="forecast_with_snowT.csv", mime="text/csv")

    # Blocchi A/B/C
    for L, (s, e) in {"A": (A_start, A_end), "B": (B_start, B_end), "C": (C_start, C_end)}.items():
        st.markdown(f"### Blocco {L}")
        W = window_slice(res, tzname, s, e)
        wet = badge(W)
        t_med = float(W["T_surf"].mean())
        st.markdown(f"**T_surf medio {L}: {t_med:.1f} °C**")
        wax_cards(t_med)
        show_structure_panel(t_med, wet)

        # Tuning lamine
        disc = st.multiselect(f"Discipline (Blocco {L})", ["SL","GS","SG","DH"], default=["SL","GS"], key=f"d_{L}")
        rows = []
        for d in disc:
            structure, side, base = tune_for(t_med, d)
            rows.append([d, structure, f"{side:.1f}°", f"{base:.1f}°"])
        if rows:
            df_tune = pd.DataFrame(rows, columns=["Disciplina", "Struttura consigliata", "Lamina SIDE (°)", "Lamina BASE (°)"])
            st.table(df_tune)

# Ingressi: CSV o fetch
if upl is not None:
    try:
        df_u = pd.read_csv(upl)
        run_all(df_u, place_label)
    except Exception as e:
        st.error(f"CSV non valido: {e}")
elif go:
    try:
        js = fetch_open_meteo(lat, lon, tzname)
        df_src = build_df(js, hours, tzname)
        run_all(df_src, place_label)
    except Exception as e:
        st.error(f"Errore: {e}")
else:
    st.info("Seleziona una località dalla tendina, definisci A/B/C e premi **Scarica previsioni** (oppure carica un CSV).")

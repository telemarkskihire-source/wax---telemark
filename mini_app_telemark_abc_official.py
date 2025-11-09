# Telemark · Pro Wax & Tune (ricerca tipo Meteoblue + strutture realistiche)
# Copia/incolla questo file intero in sostituzione del precedente.

import io
import math
from datetime import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dateutil import tz

# ----------------------------- UI & THEME -----------------------------
PRIMARY = "#10bfcf"; BG = "#0f172a"; CARD = "#111827"; TEXT = "#e5e7eb"
st.set_page_config(page_title="Telemark · Pro Wax & Tune", page_icon="❄️", layout="wide")
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
  background: linear-gradient(180deg,{BG} 0%,#111827 100%);
}}
.block-container {{ padding-top: 0.8rem; }}
h1,h2,h3,h4,h5,p,span,div {{ color:{TEXT}; }}
.badge {{ border:1px solid rgba(255,255,255,.15); padding:6px 10px; border-radius:999px; font-size:.78rem; opacity:.85; }}
.card {{ background:{CARD}; border:1px solid rgba(255,255,255,.08); border-radius:16px; padding:14px; box-shadow:0 8px 20px rgba(0,0,0,.25); }}
.brand {{ display:flex; align-items:center; gap:8px; padding:8px 10px; border-radius:12px; background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08); }}
.brand img {{ height:22px; }}
.btn-primary button {{ width:100%; background:{PRIMARY} !important; color:#002b30 !important; border:none; font-weight:700; border-radius:12px; }}
.sugg-item {{ padding:8px 10px; border-bottom:1px solid rgba(255,255,255,.06); cursor:pointer; }}
.sugg-item:hover {{ background:rgba(255,255,255,.06); }}
</style>
""", unsafe_allow_html=True)

st.markdown("### Telemark · Pro Wax & Tune")
st.markdown("<span class='badge'>Ricerca live · Geolocalizzazione · Blocchi A/B/C · Sciolina + Strutture + Angoli</span>", unsafe_allow_html=True)

# ----------------------------- WEATHER I/O -----------------------------
def fetch_open_meteo(lat, lon, timezone_str):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon, "timezone": timezone_str,
        "hourly": "temperature_2m,dew_point_2m,precipitation,rain,snowfall,cloudcover,windspeed_10m,is_day,weathercode",
        "forecast_days": 7,
    }
    r = requests.get(url, params=params, timeout=30); r.raise_for_status()
    return r.json()

def prp_type(df):
    snow_codes = {71,73,75,77,85,86}; rain_codes = {51,53,55,61,63,65,80,81,82}
    def f(row):
        prp = row.precipitation
        rain = getattr(row, "rain", 0.0)
        snow = getattr(row, "snowfall", 0.0)
        if prp <= 0 or pd.isna(prp): return "none"
        if rain > 0 and snow > 0: return "mixed"
        if snow > 0 and rain == 0: return "snow"
        if rain > 0 and snow == 0: return "rain"
        code = int(getattr(row, "weathercode", 0)) if pd.notna(getattr(row, "weathercode", None)) else 0
        if code in snow_codes: return "snow"
        if code in rain_codes: return "rain"
        return "mixed"
    return df.apply(f, axis=1)

def build_df(js, hours, tzname):
    h = js["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])  # naive
    now = pd.Timestamp.now().floor("H")
    df = df[df["time"] >= now].head(hours).reset_index(drop=True)

    out = pd.DataFrame()
    out["time"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    out["T2m"] = df["temperature_2m"].astype(float)
    out["cloud"] = (df["cloudcover"].astype(float) / 100).clip(0, 1)
    out["wind"] = (df["windspeed_10m"].astype(float) / 3.6).round(3)  # m/s
    out["sunup"] = df["is_day"].astype(int)
    out["prp_mmph"] = df["precipitation"].astype(float)
    extra = df[["precipitation", "rain", "snowfall", "weathercode"]].copy()
    out["prp_type"] = prp_type(extra)
    out["td"] = df["dew_point_2m"].astype(float)
    return out

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
        rain
        | (df["T2m"] > 0)
        | (sunup & (df["cloud"] < 0.3) & (df["T2m"] >= -3))
        | (snow & (df["T2m"] >= -1))
        | (snow & tw.ge(-0.5).fillna(False))
    )
    T_surf = pd.Series(index=df.index, dtype=float); T_surf.loc[wet] = 0.0

    dry = ~wet
    clear = (1.0 - df["cloud"]).clip(0, 1)
    windc = df["wind"].clip(upper=6.0)
    drad = (1.5 + 3.0 * clear - 0.3 * windc).clip(lower=0.5, upper=4.5)
    T_surf.loc[dry] = df["T2m"][dry] - drad[dry]

    sunny_cold = sunup & dry & df["T2m"].between(-10, 0, inclusive="both")
    T_surf.loc[sunny_cold] = pd.concat(
        [(df["T2m"] + 0.5*(1.0 - df["cloud"]))[sunny_cold], pd.Series(-0.5, index=df.index)[sunny_cold]],
        axis=1
    ).min(axis=1)

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

def window_slice(res, tzname, s, e):
    t = pd.to_datetime(res["time"]).dt.tz_localize(tz.gettz(tzname), nonexistent="shift_forward", ambiguous="NaT")
    D = res.copy(); D["dt"] = t
    today = pd.Timestamp.now(tz=tz.gettz(tzname)).date()
    W = D[(D["dt"].dt.date == today) & (D["dt"].dt.time >= s) & (D["dt"].dt.time <= e)]
    return W if not W.empty else D.head(7)

# ----------------------------- PLACE SEARCH -----------------------------
# Obiettivo: comportamento “alla Meteoblue”: mentre digiti, sotto compaiono subito i suggerimenti.
# Evitiamo set diretti rischiosi a session_state: ritorniamo semplicemente la scelta.

def search_places_nominatim(query: str, limit: int = 8):
    if not query or len(query) < 2:
        return []
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": limit, "addressdetails": 1},
            headers={"User-Agent": "telemark-wax/1.0"},
            timeout=8,
        )
        r.raise_for_status()
        js = r.json()
        return [{"label": j.get("display_name",""), "lat": float(j["lat"]), "lon": float(j["lon"])} for j in js]
    except Exception:
        return []

def geo_ip():
    try:
        r = requests.get("https://ipapi.co/json", timeout=6)
        if r.ok:
            j = r.json()
            return float(j.get("latitude", 0)), float(j.get("longitude", 0)), j.get("city", "La tua posizione")
    except Exception:
        pass
    return None, None, "La tua posizione"

# --- barra di ricerca
left, right = st.columns([3, 1])
with left:
    q = st.text_input("Cerca località", placeholder="Es. Champoluc, Cervinia, Sestriere…", label_visibility="visible")
with right:
    use_me = st.button("📍 Usa la mia posizione", use_container_width=True)

chosen = None
if use_me:
    lat, lon, label = geo_ip()
    if lat is None:
        st.error("Geolocalizzazione non disponibile.")
    else:
        chosen = {"label": label, "lat": lat, "lon": lon}

if chosen is None:
    suggs = search_places_nominatim(q, limit=8)
    if suggs:
        st.markdown("**Suggerimenti**")
        # Mostriamo una lista “aperta” come un menù: un radio è perfetto perché si aggiorna ad ogni carattere.
        labels = [s["label"] for s in suggs]
        pick = st.radio("Seleziona", labels, label_visibility="collapsed", index=0)
        chosen = next(s for s in suggs if s["label"] == pick)

# Fallback iniziale se non hai ancora digitato
if chosen is None:
    chosen = {"label": "Champoluc (Ramey)", "lat": 45.831, "lon": 7.730}

st.markdown(f"**Località:** {chosen['label']}  ")
tzname = st.selectbox("Timezone", ["Europe/Rome", "UTC"], index=0)
hours = st.slider("Ore previsione", 12, 168, 72, 12)

# ----------------------------- A/B/C WINDOWS -----------------------------
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

go = st.button("Scarica previsioni Open-Meteo", type="primary", use_container_width=True)
upl = st.file_uploader("…oppure carica un CSV (time,T2m,cloud,wind,sunup,prp_mmph,prp_type,td)", type=["csv"])

# ----------------------------- WAX BANDS -----------------------------
SWIX = [("PS5 Turquoise", -18, -10), ("PS6 Blue", -12, -6), ("PS7 Violet", -8, -2), ("PS8 Red", -4, 4), ("PS10 Yellow", 0, 10)]
TOKO = [("Blue", -30, -9), ("Red", -12, -4), ("Yellow", -6, 0)]
VOLA = [("MX-E Violet/Blue", -12, -4), ("MX-E Red", -5, 0), ("MX-E Warm", -2, 10)]
RODE = [("R20 Blue", -18, -8), ("R30 Violet", -10, -3), ("R40 Red", -5, 0), ("R50 Yellow", -1, 10)]
HOLMENKOL = [("Beta Mix Cold", -12, -6), ("Beta Mix Mid", -8, -2), ("Beta Mix Warm", -3, 2)]
MAPLUS = [("LP2 Green", -20, -10), ("LP2 Blue", -12, -6), ("LP2 Violet", -8, -2), ("LP2 Red", -4, 3)]
BRIKO = [("Maplus/Briko Blue", -12, -6), ("Maplus/Briko Violet", -8, -2), ("Maplus/Briko Red", -3, 3)]

def pick_band(bands, t):
    for name, tmin, tmax in bands:
        if t >= tmin and t <= tmax:
            return name
    return bands[-1][0] if t > bands[-1][2] else bands[0][0]

# ----------------------------- STRUCTURE DRAWINGS -----------------------------
# Mini-render “alla catalogo macchine”: immagini 300x180 px con pattern puliti.

def _fig_to_png_bytes():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    buf.seek(0)
    return buf

def draw_linear(n=14):
    plt.figure(figsize=(3,1.8))
    ax = plt.gca(); ax.axis("off")
    for x in np.linspace(0.1, 0.9, n):
        plt.plot([x, x], [0.1, 0.9])
    return _fig_to_png_bytes()

def draw_wave(n=10, amp=0.18, freq=3.5):
    plt.figure(figsize=(3,1.8))
    ax = plt.gca(); ax.axis("off")
    xs = np.linspace(0.1, 0.9, 300)
    for i in range(n):
        phase = i * 0.15
        ys = 0.5 + amp * np.sin(2*np.pi*(freq*xs + phase))
        plt.plot(xs, ys)
    return _fig_to_png_bytes()

def draw_diagonal(n=11, slope=0.6):
    plt.figure(figsize=(3,1.8))
    ax = plt.gca(); ax.axis("off")
    for i in range(n):
        x0 = 0.05 + i*0.08
        plt.plot([x0, x0+slope], [0.1, 0.9])
    return _fig_to_png_bytes()

def draw_chevron(rows=8, cols=10):
    plt.figure(figsize=(3,1.8))
    ax = plt.gca(); ax.axis("off")
    xs = np.linspace(0.1, 0.9, cols)
    ys = np.linspace(0.2, 0.8, rows)
    for y in ys:
        for x in xs:
            plt.plot([x-0.03, x, x+0.03], [y-0.06, y, y-0.06])
    return _fig_to_png_bytes()

def draw_cross(spacing=0.08):
    plt.figure(figsize=(3,1.8))
    ax = plt.gca(); ax.axis("off")
    xs = np.arange(0.1, 0.9, spacing)
    ys = np.arange(0.1, 0.9, spacing)
    for x in xs: plt.plot([x, x], [0.1, 0.9])
    for y in ys: plt.plot([0.1, 0.9], [y, y])
    return _fig_to_png_bytes()

def structure_for_temp(t_surf_mean, wet_like):
    """
    Restituisce: nome, funzione disegno, descrizione, base/side angoli consigliati
    """
    if wet_like or t_surf_mean > 0:
        return ("Scarico laterale (diagonale)", draw_diagonal, "Neve umida/calda · drenaggio acqua", 1.0, {"SL":88.0,"GS":87.5,"SG":87.0,"DH":87.0})
    if t_surf_mean > -3:
        return ("Onda/Universale", draw_wave, "Range ampio, buon glide multi-condizione", 0.8, {"SL":88.0,"GS":88.0,"SG":87.5,"DH":87.5})
    if t_surf_mean > -10:
        return ("Lineare fine", draw_linear, "Neve fredda/asciutta · scorrimento pulito", 0.6, {"SL":88.5,"GS":88.0,"SG":87.5,"DH":87.5})
    return ("Chevron + Lineare", draw_chevron, "Freddo secco estremo · mordente ma scorrevole", 0.5, {"SL":88.5,"GS":88.0,"SG":87.5,"DH":87.5})

# ----------------------------- RENDER HELPERS -----------------------------
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

def badge(win):
    tmin = float(win["T_surf"].min()); tmax = float(win["T_surf"].max())
    wet = bool(((win["prp_type"].isin(["rain","mixed"])) | (win["prp_mmph"]>0.5)).any()) or (tmax > 0.5)
    if wet:
        color, title, desc = "#ef4444", "CRITICAL", "Possibile neve bagnata/pioggia · struttura grossa/diagonale"
    elif tmax > -1.0:
        color, title, desc = "#f59e0b", "WATCH", "Vicino a 0°C · cere medio-morbide"
    else:
        color, title, desc = "#22c55e", "OK", "Freddo/asciutto · cere dure"
    st.markdown(f"""
    <div class='card' style='border-color:{color}'>
      <div style='font-weight:800;color:{color};margin-bottom:4px'>{title}</div>
      <div style='opacity:.95'>{desc}</div>
      <div style='font-size:12px;opacity:.7;margin-top:6px'>T_surf min {tmin:.1f}°C / max {tmax:.1f}°C</div>
    </div>""", unsafe_allow_html=True)
    return wet

def wax_cards(t):
    brands = [
        ("Swix", SWIX), ("Toko", TOKO), ("Vola", VOLA), ("Rode", RODE),
        ("Holmenkol", HOLMENKOL), ("Maplus", MAPLUS), ("Briko", BRIKO),
    ]
    cols = st.columns(4)
    for i,(b,band) in enumerate(brands[:4]):
        cols[i].markdown(
            f"<div class='brand'><img src='https://dummyimage.com/200x36/333/fff&text={b.upper()}'/>"
            f"<div><div style='font-size:.8rem;opacity:.8'>{b}</div><div style='font-weight:800'>{pick_band(band, t)}</div></div></div>",
            unsafe_allow_html=True
        )
    cols2 = st.columns(3)
    for i,(b,band) in enumerate(brands[4:]):
        cols2[i].markdown(
            f"<div class='brand'><img src='https://dummyimage.com/200x36/333/fff&text={b.upper()}'/>"
            f"<div><div style='font-size:.8rem;opacity:.8'>{b}</div><div style='font-weight:800'>{pick_band(band, t)}</div></div></div>",
            unsafe_allow_html=True
        )

# ----------------------------- MAIN FLOW -----------------------------
def run_on(df_source, place_label):
    res = compute_snow_temperature(df_source, dt_hours=1.0)
    st.success(f"Dati pronti per **{place_label}**")
    st.dataframe(res, use_container_width=True)
    f1, f2 = plots(res); st.pyplot(f1); st.pyplot(f2)
    st.download_button("Scarica CSV risultato", res.to_csv(index=False), "forecast_with_snowT.csv", "text/csv")

    blocks = {"A": (A_start, A_end), "B": (B_start, B_end), "C": (C_start, C_end)}
    for L, (s, e) in blocks.items():
        st.markdown(f"### Blocco {L}")
        W = window_slice(res, tzname, s, e)
        wet = badge(W)
        t_med = float(W["T_surf"].mean())
        st.markdown(f"**T_surf medio {L}: {t_med:.1f} °C**")

        # wax
        wax_cards(t_med)

        # struttura & angoli
        name, drawer, desc, base_angle, side_map = structure_for_temp(t_med, wet)
        img_bytes = drawer()
        st.image(img_bytes, caption=f"Struttura consigliata: {name} — {desc}", use_column_width=False)

        pick_disc = st.multiselect(f"Discipline (Blocco {L})", ["SL","GS","SG","DH"], default=["SL","GS"], key=f"disc_{L}")
        rows = []
        for d in pick_disc:
            side = side_map.get(d, 88.0)
            rows.append([d, name, f"{side:.1f}°", f"{base_angle:.1f}°"])
        if rows:
            st.table(pd.DataFrame(rows, columns=["Disciplina","Struttura soletta","Lamina SIDE (°)","Lamina BASE (°)"]))

# --- Input data
if upl is not None:
    try:
        df_u = pd.read_csv(upl)
        run_on(df_u, chosen["label"])
    except Exception as e:
        st.error(f"CSV non valido: {e}")

if go:
    try:
        js = fetch_open_meteo(chosen["lat"], chosen["lon"], tzname)
        df = build_df(js, hours, tzname)
        run_on(df, chosen["label"])
    except Exception as e:
        st.error(f"Errore: {e}")

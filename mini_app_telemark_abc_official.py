# app.py — Telemark · Pro Wax & Tune (ricerca stile Meteoblue + strutture realistiche)

import streamlit as st
import pandas as pd
import requests
import base64
import io
import math
import matplotlib.pyplot as plt
from datetime import time
from dateutil import tz

# ------------------------ LOOK & FEEL ------------------------
PRIMARY = "#10bfcf"; BG = "#0f172a"; CARD = "#111827"; TEXT = "#e5e7eb"
st.set_page_config(page_title="Telemark · Pro Wax & Tune", page_icon="❄️", layout="wide")
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
  background: linear-gradient(180deg, {BG} 0%, #111827 100%);
}}
.block-container {{ padding-top: 1rem; }}
h1,h2,h3,h4,h5,p,span,div {{ color:{TEXT}; }}
.card {{
  background:{CARD}; border:1px solid rgba(255,255,255,.08);
  border-radius:16px; padding:14px; box-shadow:0 8px 20px rgba(0,0,0,.25);
}}
.badge {{
  border:1px solid rgba(255,255,255,.15); padding:6px 10px; border-radius:999px;
  font-size:.78rem; opacity:.85;
}}
.brand {{
  display:flex; align-items:center; gap:8px; padding:8px 10px; border-radius:12px;
  background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08);
}}
.brand img {{ height:22px; }}
.kpi {{
  display:flex; gap:8px; align-items:center; background:rgba(16,191,207,.06);
  border:1px dashed rgba(16,191,207,.45); padding:10px 12px; border-radius:12px;
}}
.kpi .lab {{ font-size:.8rem; color:#93c5fd; }}
.kpi .val {{ font-size:1rem; font-weight:800; color:{TEXT}; }}
hr {{ border-color: rgba(255,255,255,.1) }}
</style>
""", unsafe_allow_html=True)

st.markdown("### Telemark · Pro Wax & Tune")
st.markdown("<span class='badge'>Ricerca locale stile Meteoblue · Geolocalizzazione · Blocchi A/B/C · Sciolina + Struttura + Angoli</span>", unsafe_allow_html=True)

# ------------------------ UTILS ------------------------
def flag_emoji(country_code: str) -> str:
    """Converte ISO 3166-1 alpha-2 (es. 'it') nella bandierina emoji."""
    if not country_code or len(country_code) != 2: return ""
    cc = country_code.upper()
    return chr(0x1F1E6 + ord(cc[0]) - 65) + chr(0x1F1E6 + ord(cc[1]) - 65)

def geocode_autocomplete(query: str, limit: int = 10):
    """
    Autocomplete tipo Meteoblue: chiamiamo Nominatim a ogni digitazione.
    Ritorna una lista di opzioni corte 'Città, Regione (Paese)' + bandierina.
    """
    if not query or len(query) < 2:
        return []
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": limit, "addressdetails": 1},
            headers={"User-Agent": "telemark-pro-wax/1.0"},
            timeout=8
        )
        r.raise_for_status()
        data = r.json()
        options = []
        for item in data:
            addr = item.get("address", {})
            city = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("municipality") or ""
            state = addr.get("state") or addr.get("region") or ""
            country = addr.get("country") or ""
            cc = addr.get("country_code") or ""
            name_short = ", ".join([p for p in [city or addr.get("county"), state] if p]) or item.get("display_name","")
            label = f"{flag_emoji(cc)} {name_short} ({country})".strip()
            options.append({
                "label": label,
                "lat": float(item["lat"]),
                "lon": float(item["lon"]),
            })
        return options
    except Exception:
        return []

def ip_geolocate():
    try:
        r = requests.get("https://ipapi.co/json", timeout=8)
        if r.ok:
            j = r.json()
            return float(j.get("latitude", 0)), float(j.get("longitude", 0)), j.get("city", "")
    except Exception:
        pass
    return None, None, ""

# ------------------------ LOCATION SEARCH (Meteoblue-like) ------------------------
left, right = st.columns([3,1])
with left:
    q = st.text_input("Cerca località", placeholder="Inizia a digitare (es. Champoluc, Cervinia, Sestriere…)", value=st.session_state.get("q",""))
    suggestions = geocode_autocomplete(q, limit=12)
    # Mostra selectbox *dinamico* che si aggiorna ad ogni carattere
    if suggestions:
        labels = [s["label"] for s in suggestions]
        # Se c'è già una scelta precedente, selezionala; altrimenti la prima
        default_idx = 0
        if "chosen_label" in st.session_state and st.session_state["chosen_label"] in labels:
            default_idx = labels.index(st.session_state["chosen_label"])
        choice = st.selectbox("Suggerimenti", labels, index=default_idx if labels else 0)
        # Aggiorna coordinate quando cambia la selectbox
        sel = suggestions[labels.index(choice)]
        st.session_state["chosen_label"] = sel["label"]
        st.session_state["lat"] = sel["lat"]
        st.session_state["lon"] = sel["lon"]
        st.caption("Suggerimenti aggiornati automaticamente mentre scrivi.")
    else:
        st.info("Digita almeno 2 caratteri per vedere i suggerimenti (aggiornati in tempo reale).")

with right:
    if st.button("📍 Usa la mia posizione"):
        lat, lon, city = ip_geolocate()
        if lat is not None:
            st.session_state["lat"] = lat
            st.session_state["lon"] = lon
            st.session_state["chosen_label"] = city or "La tua posizione"
            st.session_state["q"] = city or ""
            st.success("Posizione impostata.")
        else:
            st.error("Geolocalizzazione non disponibile.")

# Fallback iniziale se non ancora scelto nulla
lat = st.session_state.get("lat", 45.831)
lon = st.session_state.get("lon", 7.730)
label = st.session_state.get("chosen_label", "Champoluc (Ramey)")

# ------------------------ TIME WINDOWS A/B/C ------------------------
st.markdown("#### Blocchi orari A · B · C (oggi)")
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

tzname = st.selectbox("Timezone", ["Europe/Rome","UTC"], index=0)
hours = st.slider("Ore previsione", 12, 168, 72, 12)

go_col, upl_col = st.columns([1,2])
with go_col:
    go = st.button("Scarica previsioni per la località", type="primary")
with upl_col:
    upl = st.file_uploader("…oppure carica CSV (time,T2m,cloud,wind,sunup,prp_mmph,prp_type,td)", type=["csv"])

# ------------------------ WEATHER FETCH + MODEL ------------------------
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
    df["time"] = pd.to_datetime(df["time"])    # naive
    now_naive = pd.Timestamp.now().floor("H")  # naive: evita conflitti tz-aware
    df = df[df["time"] >= now_naive].head(hours).reset_index(drop=True)

    out = pd.DataFrame()
    out["time"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    out["T2m"] = df["temperature_2m"].astype(float)
    out["cloud"] = (df["cloudcover"].astype(float) / 100.0).clip(0, 1)
    out["wind"] = (df["windspeed_10m"].astype(float) / 3.6).round(3)
    out["sunup"] = df["is_day"].astype(int)
    out["prp_mmph"] = df["precipitation"].astype(float)
    extra = df[["precipitation","rain","snowfall","weathercode"]].copy()
    out["prp_type"] = prp_type(extra)
    out["td"] = df["dew_point_2m"].astype(float)
    return out

def compute_snow_temperature(df, dt_hours=1.0):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    required = ["T2m","cloud","wind","sunup","prp_mmph","prp_type"]
    for c in required:
        if c not in df.columns: raise ValueError(f"Missing column: {c}")
    if "td" not in df.columns: df["td"] = float("nan")
    df = df.sort_values("time").reset_index(drop=True)

    rain = df["prp_type"].str.lower().isin(["rain","mixed"])
    snow = df["prp_type"].str.lower().eq("snow")
    sunup = df["sunup"].astype(int) == 1
    tw = (df["T2m"] + df["td"]) / 2.0
    wet = (rain | (df["T2m"] > 0) | (sunup & (df["cloud"] < 0.3) & (df["T2m"] >= -3)) |
           (snow & (df["T2m"] >= -1)) | (snow & tw.ge(-0.5).fillna(False)))

    T_surf = pd.Series(index=df.index, dtype=float); T_surf.loc[wet] = 0.0
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

def local_slice(res, tzname, s, e):
    t = pd.to_datetime(res["time"]).dt.tz_localize(tz.gettz(tzname), nonexistent='shift_forward', ambiguous='NaT')
    D = res.copy(); D["dt"] = t
    today = pd.Timestamp.now(tz=tz.gettz(tzname)).date()
    W = D[(D["dt"].dt.date==today) & (D["dt"].dt.time>=s) & (D["dt"].dt.time<=e)]
    return W if not W.empty else D.head(7)

# ------------------------ WAX BRANDS ------------------------
BRANDS = {
    "Swix": [("PS5 Turquoise", -18,-10), ("PS6 Blue",-12,-6), ("PS7 Violet",-8,-2), ("PS8 Red",-4,4), ("PS10 Yellow",0,10)],
    "Toko": [("Blue",-30,-9), ("Red",-12,-4), ("Yellow",-6,0)],
    "Vola": [("MX-E Violet/Blue",-12,-4), ("MX-E Red",-5,0), ("MX-E Warm",-2,10)],
    "Rode": [("R20 Blue",-18,-8), ("R30 Violet",-10,-3), ("R40 Red",-5,0), ("R50 Yellow",-1,10)],
    "Holmenkol": [("Ultra Mix Blue",-20,-8), ("BetaMix Red",-14,-4), ("AlphaMix Yellow",-4,5)],
    "Maplus": [("Universal Cold",-12,-6), ("Universal Medium",-7,-2), ("Universal Warm",-3,6)],
    "Start": [("SG Blue",-12,-6), ("SG Purple",-8,-2), ("SG Red",-3,7)],
    "Skigo": [("Paraffin Blue",-12,-6), ("Paraffin Violet",-8,-2), ("Paraffin Red",-3,2)],
}
BADGE_COLORS = {
    "Swix": "#ef4444", "Toko": "#f59e0b", "Vola": "#3b82f6", "Rode": "#22c55e",
    "Holmenkol": "#0ea5e9", "Maplus": "#22d3ee", "Start": "#fb7185", "Skigo": "#a78bfa"
}

def pick_band(bands, t):
    for name, tmin, tmax in bands:
        if t >= tmin and t <= tmax:
            return name
    return bands[-1][0] if t > bands[-1][2] else bands[0][0]

def brand_badge(name, rec):
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='88' height='24'><rect width='88' height='24' rx='6' fill='{BADGE_COLORS.get(name,'#999')}'/><text x='10' y='16' font-size='12' font-weight='700' fill='white'>{name.upper()}</text></svg>"
    data = "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"<div class='brand'><img src='{data}'/><div><div style='font-size:.8rem;opacity:.85'>{name}</div><div style='font-weight:800'>{rec}</div></div></div>"

# ------------------------ STRUCTURE DRAWINGS ------------------------
def draw_structure(kind: str, coarseness: str = "medium", width=500, height=120):
    """
    Disegni in stile realistico: linee sottili grigie, ripetitive come sui cataloghi macchine.
    kind: 'linear','chevron','cross','wave','lateral'
    coarseness: 'fine' | 'medium' | 'coarse'
    """
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    ax = plt.gca()
    ax.set_facecolor("#f5f5f5")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 20)
    ax.axis("off")
    # passo
    step = {"fine": 1.8, "medium": 2.6, "coarse": 3.4}[coarseness]
    color = "#444444"
    lw = {"fine": 0.8, "medium": 1.1, "coarse": 1.4}[coarseness]

    if kind == "linear":
        x = 0
        while x <= 100:
            ax.plot([x, x], [0, 20], color=color, linewidth=lw)
            x += step

    elif kind == "chevron":
        # serie di V ripetute
        x = 0
        angle = 20  # gradi simbolici (apertura V)
        while x <= 100:
            ax.plot([x-2, x, x+2], [0, 10, 0], color=color, linewidth=lw)
            ax.plot([x-2, x, x+2], [20, 10, 20], color=color, linewidth=lw)
            x += step*2.2

    elif kind == "cross":
        # reticolo diagonale
        x = -20
        while x <= 100:
            ax.plot([x, x+40], [0, 20], color=color, linewidth=lw*0.9)
            ax.plot([x+40, x], [0, 20], color=color, linewidth=lw*0.9)
            x += step*2.2

    elif kind == "wave":
        import numpy as np
        for y0 in [5, 10, 15]:
            xs = np.linspace(0, 100, 400)
            ys = y0 + (2.0 if coarseness!="fine" else 1.2)*np.sin(xs/4.5)
            ax.plot(xs, ys, color=color, linewidth=lw)

    elif kind == "lateral":
        # canali laterali + lineare centrale
        for band in [(0,2.5), (17.5,20)]:
            x = 0
            while x <= 100:
                ax.plot([x, x], [band[0], band[1]], color=color, linewidth=lw*1.2)
                x += step
        x = 0
        while x <= 100:
            ax.plot([x, x], [3.5, 16.5], color="#666666", linewidth=lw*0.9)
            x += step*1.4

    buf = io.BytesIO()
    fig.tight_layout(pad=0)
    plt.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

def structure_recommendation(t_surf, wet=False):
    """Restituisce (nome, kind, coarseness)."""
    if t_surf <= -10:
        return ("Lineare Fine", "linear", "fine")
    if t_surf <= -3:
        return ("Lineare Media (o Cross leggero)", "linear", "medium")
    # caldo / vicino a 0: strutture più pronunciate
    if wet or t_surf > -1:
        return ("Chevron / Scarico laterale", "lateral", "coarse")
    return ("Wave Media", "wave", "medium")

# ------------------------ EDGES TUNING ------------------------
def tune_for(t_surf, discipline):
    # SIDE (88°, 87.5°…) e BASE (0.5–1.0°)
    if t_surf <= -10:
        structure = "Fine (lineare)"
        base = 0.5
        side_map = {"SL": 88.5, "GS": 88.0, "SG": 87.5, "DH": 87.5}
    elif t_surf <= -3:
        structure = "Media (universale)"
        base = 0.7
        side_map = {"SL": 88.0, "GS": 88.0, "SG": 87.5, "DH": 87.0}
    else:
        structure = "Media-Grossa"
        base = 0.8 if t_surf <= 0.5 else 1.0
        side_map = {"SL": 88.0, "GS": 87.5, "SG": 87.0, "DH": 87.0}
    return structure, side_map.get(discipline, 88.0), base

# ------------------------ PLOTS ------------------------
def plot_series(res):
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

def status_badge(win):
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

# ------------------------ MAIN FLOW ------------------------
def run_all(src, label):
    res = compute_snow_temperature(src, dt_hours=1.0)
    st.success(f"Dati per **{label}** pronti.")
    st.dataframe(res, use_container_width=True)
    f1, f2 = plot_series(res); st.pyplot(f1); st.pyplot(f2)
    st.download_button("Scarica CSV risultato", data=res.to_csv(index=False), file_name="forecast_with_snowT.csv", mime="text/csv")

    blocks = {"A": (A_start, A_end), "B": (B_start, B_end), "C": (C_start, C_end)}
    for L, (s, e) in blocks.items():
        st.markdown(f"### Blocco {L}")
        W = local_slice(res, tzname, s, e)
        wet = status_badge(W)
        t_med = float(W["T_surf"].mean())
        st.markdown(f"**T_surf medio {L}: {t_med:.1f} °C**")

        # Wax cards (8 brand)
        cols = st.columns(4)
        items = list(BRANDS.items())
        for i in range(4):
            name, bands = items[i]
            rec = pick_band(bands, t_med)
            cols[i].markdown(brand_badge(name, rec), unsafe_allow_html=True)
        cols = st.columns(4)
        for i in range(4, 8):
            name, bands = items[i]
            rec = pick_band(bands, t_med)
            cols[i-4].markdown(brand_badge(name, rec), unsafe_allow_html=True)

        # Structure recommendation + preview
        s_name, s_kind, s_coarse = structure_recommendation(t_med, wet=wet)
        img = draw_structure(s_kind, s_coarse)
        st.markdown(f"**Struttura consigliata:** {s_name}")
        st.image(img, caption=f"{s_kind} · {s_coarse}", use_column_width=True)

        # Edge tuning
        disc = st.multiselect(f"Discipline (Blocco {L})", ["SL","GS","SG","DH"], default=["SL","GS"], key=f"d_{L}")
        rows = []
        for d in disc:
            _, side, base = tune_for(t_med, d)
            rows.append([d, f"{side:.1f}°", f"{base:.1f}°"])
        if rows:
            df_tune = pd.DataFrame(rows, columns=["Disciplina","Lamina SIDE (°)","Lamina BASE (°)"])
            st.table(df_tune)

# Input: CSV o fetch
if upl is not None:
    try:
        df_u = pd.read_csv(upl)
        run_all(df_u, label)
    except Exception as e:
        st.error(f"CSV non valido: {e}")

if go:
    try:
        js = fetch_open_meteo(lat, lon, tzname)
        df_src = build_df(js, hours, tzname)
        run_all(df_src, label)
    except Exception as e:
        st.error(f"Errore: {e}")

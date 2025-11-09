# telemark_pro_app.py
import streamlit as st
import pandas as pd
import requests, base64, math
import numpy as np
import matplotlib.pyplot as plt
from datetime import time
from dateutil import tz
from streamlit_searchbox import st_searchbox  # dropdown live, stile meteoblue

# ------------------------ PAGE & THEME ------------------------
PRIMARY = "#10bfcf"; BG = "#0f172a"; CARD = "#0f172a"; TEXT = "#eef2ff"
st.set_page_config(page_title="Telemark · Pro Wax & Tune", page_icon="❄️", layout="wide")
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
  background: linear-gradient(180deg, {BG} 0%, #111827 100%);
}}
.block-container {{ padding-top: 0.8rem; }}
h1,h2,h3,h4,h5, label, p, span, div {{ color:{TEXT}; }}
.badge {{ border:1px solid rgba(255,255,255,.15); padding:6px 10px; border-radius:999px; font-size:.78rem; opacity:.85; }}
.card {{ background:{CARD}; border:1px solid rgba(255,255,255,.12); border-radius:16px; padding:14px; box-shadow:0 10px 22px rgba(0,0,0,.25); }}
.brand {{ display:flex; align-items:center; gap:10px; padding:8px 10px; border-radius:12px;
         background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08); }}
.brand img {{ height:22px; }}
.kpi {{ display:flex; gap:8px; align-items:center; background:rgba(16,191,207,.06);
       border:1px dashed rgba(16,191,207,.45); padding:10px 12px; border-radius:12px; }}
.kpi .lab {{ font-size:.78rem; color:#93c5fd; }}
.kpi .val {{ font-size:1rem; font-weight:800; }}
.note {{ font-size:.78rem; opacity:.8; }}
</style>
""", unsafe_allow_html=True)

st.markdown("### Telemark · Pro Wax & Tune")
st.markdown("<span class='badge'>Ricerca tipo Meteoblue · Blocchi A/B/C · Sciolina + Struttura + Angoli (SIDE)</span>", unsafe_allow_html=True)

# ------------------------ UTILS ------------------------
def flag_emoji(country_code: str) -> str:
    """Convert ISO-2 country code to emoji flag."""
    try:
        cc = country_code.upper()
        return chr(127397 + ord(cc[0])) + chr(127397 + ord(cc[1]))
    except Exception:
        return "🏳️"

# Search function for st_searchbox (chiamata a ogni carattere)
def nominatim_search(search: str):
    if not search or len(search) < 2:
        return []
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": search, "format": "json", "limit": 10, "addressdetails": 1},
            headers={"User-Agent": "telemark-wax-app/1.0"},
            timeout=8
        )
        r.raise_for_status()
        out = []
        st.session_state._geo_map = {}
        for i, item in enumerate(r.json()):
            name = item.get("display_name", "")
            lat = float(item.get("lat", 0)); lon = float(item.get("lon", 0))
            cc = (item.get("address", {}) or {}).get("country_code", "") or ""
            label = f"{flag_emoji(cc)}  {name}"
            key = f"{label}|||{lat:.6f},{lon:.6f}"
            st.session_state._geo_map[key] = (lat, lon, label)
            out.append(key)
        return out
    except Exception:
        return []

# ------------------------ LOCATION (Meteoblue-like) ------------------------
st.markdown("#### 1) Cerca località")
selected = st_searchbox(
    nominatim_search,
    key="place",
    placeholder="Digita e scegli… (es. Champoluc, Cervinia, Sestriere)",
    clear_on_submit=False,
    default=None
)

# decode selection -> lat,lon,label
if selected and "|||" in selected and "_geo_map" in st.session_state:
    lat, lon, label = st.session_state._geo_map.get(selected, (45.831, 7.730, "Champoluc (Ramey)"))
    st.session_state.sel_lat, st.session_state.sel_lon, st.session_state.sel_label = lat, lon, label

# Fallback default se non c'è selezione
lat = st.session_state.get("sel_lat", 45.831)
lon = st.session_state.get("sel_lon", 7.730)
label = st.session_state.get("sel_label", "Champoluc (Ramey)")

coltz, colh = st.columns([1,2])
with coltz:
    tzname = st.selectbox("Timezone", ["Europe/Rome", "UTC"], index=0)
with colh:
    hours = st.slider("Ore previsione", 12, 168, 72, 12)

# ------------------------ WINDOWS A/B/C ------------------------
st.markdown("#### 2) Finestre orarie A · B · C (oggi)")
c1, c2, c3 = st.columns(3)
with c1:
    A_start = st.time_input("Inizio A", time(9, 0), key="A_s")
    A_end   = st.time_input("Fine A",   time(11, 0), key="A_e")
with c2:
    B_start = st.time_input("Inizio B", time(11, 0), key="B_s")
    B_end   = st.time_input("Fine B",   time(13, 0), key="B_e")
with c3:
    C_start = st.time_input("Inizio C", time(13, 0), key="C_s")
    C_end   = st.time_input("Fine C",   time(16, 0), key="C_e")

# ------------------------ DATA PIPELINE ------------------------
def fetch_open_meteo(lat, lon, timezone_str):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon, "timezone": timezone_str,
        "hourly": "temperature_2m,dew_point_2m,precipitation,rain,snowfall,cloudcover,windspeed_10m,is_day,weathercode",
        "forecast_days": 7,
    }
    r = requests.get(url, params=params, timeout=30); r.raise_for_status()
    return r.json()

def _prp_type(df):
    snow_codes = {71,73,75,77,85,86}
    rain_codes = {51,53,55,61,63,65,80,81,82}
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
    h = js["hourly"]; df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])         # naive
    now0 = pd.Timestamp.now().floor("H")
    df = df[df["time"] >= now0].head(hours).reset_index(drop=True)
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
    rain = df["prp_type"].str.lower().isin(["rain","mixed"])
    snow = df["prp_type"].str.lower().eq("snow")
    sunup = df["sunup"].astype(int) == 1
    tw = (df["T2m"] + df["td"]) / 2.0
    wet = (rain | (df["T2m"]>0) | (sunup & (df["cloud"]<0.3) & (df["T2m"]>=-3))
           | (snow & (df["T2m"]>=-1)) | (snow & tw.ge(-0.5).fillna(False)))
    T_surf = pd.Series(index=df.index, dtype=float); T_surf.loc[wet] = 0.0
    dry = ~wet
    clear = (1.0 - df["cloud"]).clip(0,1); windc = df["wind"].clip(upper=6.0)
    drad = (1.5 + 3.0*clear - 0.3*windc).clip(0.5, 4.5)
    T_surf.loc[dry] = df["T2m"][dry] - drad[dry]
    sunny_cold = sunup & dry & df["T2m"].between(-10,0, inclusive="both")
    T_surf.loc[sunny_cold] = pd.concat([
        (df["T2m"] + 0.5*(1.0 - df["cloud"]))[sunny_cold],
        pd.Series(-0.5, index=df.index)[sunny_cold]
    ], axis=1).min(axis=1)
    T_top5 = pd.Series(index=df.index, dtype=float)
    tau = pd.Series(6.0, index=df.index, dtype=float)
    tau.loc[rain | snow | (df["wind"]>=6)] = 3.0
    tau.loc[(~sunup) & (df["wind"]<2) & (df["cloud"]<0.3)] = 8.0
    alpha = 1.0 - (math.e ** (-dt_hours / tau))
    if len(df)>0:
        T_top5.iloc[0] = min(df["T2m"].iloc[0], 0.0)
        for i in range(1, len(df)):
            T_top5.iloc[i] = T_top5.iloc[i-1] + alpha.iloc[i] * (T_surf.iloc[i] - T_top5.iloc[i-1])
    df["T_surf"] = T_surf; df["T_top5"] = T_top5; return df

def window_slice(res, tzname, s, e):
    t = pd.to_datetime(res["time"]).dt.tz_localize(tz.gettz(tzname), nonexistent='shift_forward', ambiguous='NaT')
    D = res.copy(); D["dt"] = t
    today = pd.Timestamp.now(tz=tz.gettz(tzname)).date()
    W = D[(D["dt"].dt.date==today) & (D["dt"].dt.time>=s) & (D["dt"].dt.time<=e)]
    return W if not W.empty else D.head(7)

# ------------------------ WAX BANDS ------------------------
SWIX = [("PS5 Turquoise", -18,-10), ("PS6 Blue",-12,-6), ("PS7 Violet",-8,-2), ("PS8 Red",-4,4), ("PS10 Yellow",0,10)]
TOKO = [("Blue",-30,-9), ("Red",-12,-4), ("Yellow",-6,0)]
VOLA = [("MX-E Violet/Blue",-12,-4), ("MX-E Red",-5,0), ("MX-E Warm",-2,10)]
RODE = [("R20 Blue",-18,-8), ("R30 Violet",-10,-3), ("R40 Red",-5,0), ("R50 Yellow",-1,10)]
def pick(bands, t):
    for n,tmin,tmax in bands:
        if t>=tmin and t<=tmax: return n
    return bands[-1][0] if t>bands[-1][2] else bands[0][0]

# ------------------------ STRUTTURE STILE WINTERSTEIGER ------------------------
# Preset con etichette chiare (come i pannelli macchina)
STRUCTURE_PRESETS = {
    "Lineare fine (Freddo/Secco)"  : ("linear_fine",  "Solchi paralleli sottili per attrito ridotto su neve fredda e secca"),
    "Onda convessa (Universale)"   : ("wave_convex",  "Archi morbidi a raggio lungo per scorrimento polivalente"),
    "Scarico diagonale (Caldo/Umido)" : ("diagonal_drain", "Canali inclinati per evacuazione acqua libera"),
    "Cross incrociata"             : ("cross",        "Diagonali incrociate per neve variabile/trasformata"),
    "Chevron (freccia)"            : ("chevron",      "V solcati orientati → più direzionalità e rilascio acqua"),
    "Broken-wave (stone medium)"   : ("broken_wave",  "Onde spezzate/segmentate tipiche pietra media"),
}

def auto_preset(t_surf: float, wet: bool) -> str:
    """Selezione automatica simile alle tabelle officina."""
    if wet or t_surf > -1.0:
        return "diagonal_drain"
    if -8.0 <= t_surf <= -1.0:
        return "wave_convex"
    if t_surf <= -12.0:
        return "linear_fine"
    # fallback
    return "cross"

def draw_structure(kind: str, title: str):
    """
    Render realistico: base grigio chiaro, righe scure con spessori e raggio simili a schede Wintersteiger.
    Niente seaborn, un solo plot come richiesto.
    """
    fig = plt.figure(figsize=(3.6, 2.2), dpi=160)
    ax  = plt.gca()
    ax.set_facecolor("#d9d9de")                 # soletta
    groove = "#5d6066"                          # colore righe utensile
    ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")

    if kind == "linear_fine":
        # 0.5–0.7 mm passo equivalente → tante linee sottili
        for x in np.arange(8, 92, 4.5):
            ax.plot([x, x], [6, 54], linewidth=2.0, color=groove, solid_capstyle="round")

    elif kind == "wave_convex":
        # archi dolci con offset (simula rullo convesso)
        xs = np.linspace(6, 94, 7)
        y  = np.linspace(6, 54, 180)
        for i, cx in enumerate(xs):
            amp = 10 + 1.2*i
            curve = 30 + amp*np.sin(np.linspace(-np.pi, np.pi, y.size))
            ax.plot(np.full_like(y, cx), curve, linewidth=2.6, color=groove, solid_capstyle="round")

    elif kind == "diagonal_drain":
        # canali inclinati larghi (scarico)
        for x in np.arange(-20, 120, 10):
            ax.plot([x, x+55], [4, 56], linewidth=3.2, color=groove, solid_capstyle="round")

    elif kind == "cross":
        # incrocio diagonale fine
        for x in np.arange(-25, 125, 10):
            ax.plot([x, x+55], [4, 56], linewidth=2.4, color=groove, alpha=.95, solid_capstyle="round")
            ax.plot([x+55, x], [4, 56], linewidth=2.0, color=groove, alpha=.65, solid_capstyle="round")

    elif kind == "chevron":
        # V ripetuti (micro-chevron)
        for x in np.arange(5, 95, 8):
            ax.plot([x-6, x], [10, 32], linewidth=2.6, color=groove, solid_capstyle="round")
            ax.plot([x, x+6], [32, 10], linewidth=2.6, color=groove, solid_capstyle="round")
            ax.plot([x-6, x], [50, 28], linewidth=2.6, color=groove, solid_capstyle="round")
            ax.plot([x, x+6], [28, 50], linewidth=2.6, color=groove, solid_capstyle="round")

    elif kind == "broken_wave":
        # onde spezzate (segmenti curvi)
        cx = np.linspace(10, 90, 6)
        for c in cx:
            for k in range(6):
                t0 = -np.pi + k*0.9
                t1 = t0 + 0.55
                t  = np.linspace(t0, t1, 30)
                y  = 30 + 14*np.sin(t) + (k%2)*1.2
                x  = np.full_like(y, c)
                ax.plot(x, y, linewidth=2.4, color=groove, solid_capstyle="round")

    ax.set_title(title, fontsize=10, pad=4)
    st.pyplot(fig)

# Tuning (angoli + suggerimento struttura)
def tune_for(t_surf, discipline, wet: bool):
    # SIDE (gradi) + BASE (gradi) + preset struttura
    if wet or t_surf > -1:
        preset = "diagonal_drain"; base = 0.8 if t_surf <= 0.5 else 1.0
        side_map = {"SL":88.0, "GS":87.5, "SG":87.0, "DH":87.0}
        desc = "Caldo/Umido · Scarico diagonale"
    elif t_surf <= -10:
        preset = "linear_fine"; base = 0.5
        side_map = {"SL":88.5, "GS":88.0, "SG":87.5, "DH":87.5}
        desc = "Freddo/Secco · Lineare fine"
    else:
        preset = "wave_convex"; base = 0.7
        side_map = {"SL":88.0, "GS":88.0, "SG":87.5, "DH":87.0}
        desc = "Universale · Onda convessa"
    return desc, side_map.get(discipline, 88.0), base, preset

# ------------------------ LOGHI WAX ------------------------
def logo(text, color):
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='160' height='36'><rect width='160' height='36' rx='6' fill='{color}'/><text x='12' y='24' font-size='16' font-weight='700' fill='white'>{text}</text></svg>"
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")

BRANDS = {
    "Swix": ("#ef4444", SWIX),
    "Toko": ("#f59e0b", TOKO),
    "Vola": ("#3b82f6", VOLA),
    "Rode": ("#22c55e", RODE),
}

# ------------------------ RUN ------------------------
st.markdown("#### 3) Scarica dati meteo & calcola")
go = st.button("Scarica previsioni per la località selezionata", type="primary")

if go:
    try:
        js = fetch_open_meteo(lat, lon, tzname)
        src = build_df(js, hours)
        res = compute_snow_temperature(src, dt_hours=1.0)
        st.success(f"Dati per **{label}** caricati.")
        st.dataframe(res, use_container_width=True)

        # grafici
        t = pd.to_datetime(res["time"])
        fig1 = plt.figure(); plt.plot(t,res["T2m"],label="T2m"); plt.plot(t,res["T_surf"],label="T_surf"); plt.plot(t,res["T_top5"],label="T_top5")
        plt.legend(); plt.title("Temperature"); plt.xlabel("Ora"); plt.ylabel("°C"); st.pyplot(fig1)
        fig2 = plt.figure(); plt.bar(t,res["prp_mmph"]); plt.title("Precipitazione (mm/h)"); plt.xlabel("Ora"); plt.ylabel("mm/h"); st.pyplot(fig2)
        st.download_button("Scarica CSV risultato", data=res.to_csv(index=False), file_name="forecast_with_snowT.csv", mime="text/csv")

        # blocchi A/B/C
        for L,(s,e) in {"A":(A_start,A_end),"B":(B_start,B_end),"C":(C_start,C_end)}.items():
            st.markdown(f"### Blocco {L}")
            W = window_slice(res, tzname, s, e)
            wet = bool(((W["prp_type"].isin(["rain","mixed"])) | (W["prp_mmph"]>0.5)).any())
            t_med = float(W["T_surf"].mean())
            st.markdown(f"**T_surf medio {L}: {t_med:.1f}°C**")

            # Wax cards + loghi
            cols = st.columns(len(BRANDS))
            for i,(brand,(col,bands)) in enumerate(BRANDS.items()):
                rec = pick(bands, t_med)
                cols[i].markdown(
                    f"<div class='brand'><img src='{logo(brand.upper(), col)}'/>"
                    f"<div><div style='font-size:.8rem;opacity:.85'>{brand}</div>"
                    f"<div style='font-weight:800'>{rec}</div></div></div>", unsafe_allow_html=True
                )

            # ---------------- PRESET STRUTTURA ----------------
            # Suggerimento automatico + override manuale da tendina
            _, _, _, auto_kind = tune_for(t_med, "GS", wet)
            preset_names = ["Auto"] + list(STRUCTURE_PRESETS.keys())
            choice = st.selectbox(
                f"Preset struttura (Blocco {L})",
                preset_names,
                index=0, key=f"preset_{L}",
                help="Auto sceglie in base a temperatura neve e bagnato; puoi forzare un preset."
            )

            if choice == "Auto":
                kind = auto_kind
                # titolo da dict inverso
                inv = {v[0]: k for k,v in STRUCTURE_PRESETS.items()}
                title = inv.get(kind, "Preset automatico")
                descr = STRUCTURE_PRESETS.get(title, ("",""))[1] if title in STRUCTURE_PRESETS else ""
            else:
                kind, descr = STRUCTURE_PRESETS[choice]
                title = choice

            draw_structure(kind, title)
            st.markdown(f"<div class='note'>{descr}</div>", unsafe_allow_html=True)

            # Tuning per discipline (angoli side/base)
            disc = st.multiselect(f"Discipline (Blocco {L})", ["SL","GS","SG","DH"], default=["SL","GS"], key=f"disc_{L}")
            rows = []
            for d in disc:
                sdesc, side_d, base_d, _ = tune_for(t_med, d, wet)
                rows.append([d, sdesc, f"{side_d:.1f}°", f"{base_d:.1f}°"])
            if rows:
                st.table(pd.DataFrame(rows, columns=["Disciplina","Struttura consigliata","Lamina SIDE (°)","Lamina BASE (°)"]))
    except Exception as e:
        st.error(f"Errore: {e}")

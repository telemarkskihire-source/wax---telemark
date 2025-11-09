# telemark_pro_app.py
import streamlit as st
import pandas as pd
import requests, base64, math
import matplotlib.pyplot as plt
from datetime import time
from dateutil import tz

# ------------------------ PAGE & THEME ------------------------
PRIMARY = "#10bfcf"; BG = "#0f172a"; CARD = "#0f172a"; TEXT = "#eef2ff"
st.set_page_config(page_title="Telemark · Pro Wax & Tune", page_icon="❄️", layout="wide")
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
  background: linear-gradient(180deg, {BG} 0%, #111827 100%);
}}
.block-container {{ padding-top: .8rem; }}
h1,h2,h3,h4,h5,label,p,span,div {{ color:{TEXT}; }}
.badge {{ border:1px solid rgba(255,255,255,.15); padding:6px 10px; border-radius:999px; font-size:.78rem; opacity:.85; }}
.card {{ background:{CARD}; border:1px solid rgba(255,255,255,.12); border-radius:16px; padding:14px; box-shadow:0 10px 22px rgba(0,0,0,.25); }}
.brand {{ display:flex; align-items:center; gap:10px; padding:8px 10px; border-radius:12px;
         background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08); }}
.brand img {{ height:22px; }}
.kpi {{ display:flex; gap:8px; align-items:center; background:rgba(16,191,207,.06);
       border:1px dashed rgba(16,191,207,.45); padding:10px 12px; border-radius:12px; }}
.kpi .lab {{ font-size:.78rem; color:#93c5fd; }}
.kpi .val {{ font-size:1rem; font-weight:800; }}
.select-like > div[data-baseweb="select"] {{ background: rgba(255,255,255,.06); border-radius:12px; }}
</style>
""", unsafe_allow_html=True)

st.markdown("### Telemark · Pro Wax & Tune")
st.markdown("<span class='badge'>Ricerca rapida stile Meteoblue · Blocchi A/B/C · Sciolina + Struttura + Angoli (SIDE)</span>", unsafe_allow_html=True)

# ------------------------ UTILS ------------------------
def flag_emoji(country_code: str) -> str:
    try:
        cc = country_code.upper()
        return chr(127397 + ord(cc[0])) + chr(127397 + ord(cc[1]))
    except Exception:
        return "🏳️"

def svg_logo(text, color):
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='160' height='36'><rect width='160' height='36' rx='6' fill='{color}'/><text x='12' y='24' font-size='16' font-weight='700' fill='white'>{text}</text></svg>"
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")

# ------------------------ LIVE SEARCH (Meteoblue-like) ------------------------
st.markdown("#### 1) Cerca località")
if "_prev_q" not in st.session_state: st.session_state._prev_q = ""
if "_geo_opts" not in st.session_state: st.session_state._geo_opts = []
if "sel_lat" not in st.session_state: st.session_state.sel_lat, st.session_state.sel_lon, st.session_state.sel_label = 45.831, 7.730, "Champoluc (Ramey)"

q = st.text_input("Digita la località… (es. Champoluc, Cervinia, Sestriere)", value=st.session_state._prev_q, label_visibility="collapsed")

def nominatim(query: str):
    if not query or len(query) < 2: return []
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 10, "addressdetails": 1},
            headers={"User-Agent": "telemark-wax-app/1.0"},
            timeout=8
        )
        r.raise_for_status()
        out = []
        for it in r.json():
            name = it.get("display_name","")
            cc = (it.get("address",{}) or {}).get("country_code","") or ""
            label = f"{flag_emoji(cc)}  {name}"
            out.append({
                "label": label,
                "lat": float(it.get("lat", 0)),
                "lon": float(it.get("lon", 0)),
            })
        return out
    except Exception:
        return []

# aggiorna suggerimenti ad ogni carattere (no Enter)
if q != st.session_state._prev_q:
    st.session_state._prev_q = q
    st.session_state._geo_opts = nominatim(q)

# dropdown interattiva con risultati (stile meteoblue)
labels = [opt["label"] for opt in st.session_state._geo_opts] or [st.session_state.sel_label]
with st.container():
    st.markdown("<div class='select-like'>", unsafe_allow_html=True)
    choice = st.selectbox("Scegli", labels, index=0, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

# salva selezione
for opt in st.session_state._geo_opts:
    if opt["label"] == choice:
        st.session_state.sel_lat = opt["lat"]; st.session_state.sel_lon = opt["lon"]; st.session_state.sel_label = opt["label"]
        break

lat, lon, label = st.session_state.sel_lat, st.session_state.sel_lon, st.session_state.sel_label

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
    df = df.copy(); df["time"] = pd.to_datetime(df["time"])
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

# ------------------------ WAX BANDS (più marchi) ------------------------
SWIX = [("PS5 Turquoise", -18,-10), ("PS6 Blue",-12,-6), ("PS7 Violet",-8,-2), ("PS8 Red",-4,4), ("PS10 Yellow",0,10)]
TOKO = [("Blue",-30,-9), ("Red",-12,-4), ("Yellow",-6,0)]
VOLA = [("MX-E Violet/Blue",-12,-4), ("MX-E Red",-5,0), ("MX-E Warm",-2,10)]
RODE = [("R20 Blue",-18,-8), ("R30 Violet",-10,-3), ("R40 Red",-5,0), ("R50 Yellow",-1,10)]
HOLM = [("Ultra Mix Cold",-20,-8), ("Ultra Mix Mid",-10,-2), ("Ultra Mix Warm",-3,6)]
MAPL = [("Green Cold",-22,-10), ("Blue",-12,-4), ("Red",-6,0), ("Yellow",0,10)]
STAR = [("M20 Cold",-20,-8), ("M30 Med",-10,-2), ("M40 Warm",-3,6)]

BRANDS = {
    "Swix": ("#ef4444", SWIX),
    "Toko": ("#f59e0b", TOKO),
    "Vola": ("#3b82f6", VOLA),
    "Rode": ("#22c55e", RODE),
    "Holmenkol": ("#0ea5e9", HOLM),
    "Maplus": ("#a855f7", MAPL),
    "Star": ("#fb7185", STAR),
}

def pick(bands, t):
    for n,tmin,tmax in bands:
        if t>=tmin and t<=tmax: return n
    return bands[-1][0] if t>bands[-1][2] else bands[0][0]

# ------------------------ STRUCTURE & EDGES (preset stile Wintersteiger) ------------------------
def tune_for(t_surf, discipline):
    # SIDE (gradi) + BASE (gradi) e struttura consigliata
    if t_surf <= -10:
        structure = "Freddo/Secco · Lineare fine"
        base = 0.5; side_map = {"SL":88.5, "GS":88.0, "SG":87.5, "DH":87.5}
    elif t_surf <= -3:
        structure = "Universale · Onda/convessa"
        base = 0.7; side_map = {"SL":88.0, "GS":88.0, "SG":87.5, "DH":87.0}
    else:
        structure = "Caldo/Umido · Scarico diagonale / Cross"
        base = 0.8 if t_surf <= 0.5 else 1.0
        side_map = {"SL":88.0, "GS":87.5, "SG":87.0, "DH":87.0}
    return structure, side_map.get(discipline, 88.0), base

def _ax_base(fig_w=3.4, fig_h=2.1):
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=160)
    ax = plt.gca(); ax.set_facecolor("#d1d5db")  # soletta grigio chiaro satinato
    ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis('off')
    return fig, ax

def draw_linear_fine(title):
    fig, ax = _ax_base()
    for x in range(10, 91, 5):
        ax.plot([x, x], [6, 54], linewidth=2.2, color="#1f2937", solid_capstyle="round")
    ax.set_title(title, fontsize=10, pad=4); st.pyplot(fig)

def draw_wave_convex(title):
    import numpy as np
    fig, ax = _ax_base()
    xs = np.linspace(8, 92, 8)
    for x in xs:
        y = 30 + 18*np.sin(np.linspace(-math.pi, math.pi, 80))
        ax.plot(np.full_like(y, x), y, linewidth=2.4, color="#1f2937", solid_capstyle="round")
    ax.set_title(title, fontsize=10, pad=4); st.pyplot(fig)

def draw_diagonal_drain(title):
    fig, ax = _ax_base()
    for x in range(-15, 115, 7):
        ax.plot([x, x+52], [6, 54], linewidth=3.0, color="#111827", solid_capstyle="round")
    ax.set_title(title, fontsize=10, pad=4); st.pyplot(fig)

def draw_cross(title):
    fig, ax = _ax_base()
    # diagonale /
    for x in range(-20, 120, 9):
        ax.plot([x, x+52], [6, 54], linewidth=2.2, color="#111827", alpha=.85)
    # diagonale \
    for x in range(10, 150, 9):
        ax.plot([x, x-52], [6, 54], linewidth=2.2, color="#111827", alpha=.85)
    ax.set_title(title, fontsize=10, pad=4); st.pyplot(fig)

def draw_chevron(title):
    fig, ax = _ax_base()
    # piccole V ripetute (chevron)
    for x in range(6, 96, 10):
        ax.plot([x, x+6], [20, 30], linewidth=2.2, color="#1f2937")
        ax.plot([x+6, x], [30, 40], linewidth=2.2, color="#1f2937")
        ax.plot([x, x+6], [40, 30], linewidth=2.2, color="#1f2937")
        ax.plot([x+6, x], [30, 20], linewidth=2.2, color="#1f2937")
    ax.set_title(title, fontsize=10, pad=4); st.pyplot(fig)

def draw_double_linear(title):
    fig, ax = _ax_base()
    for x in range(10, 91, 9):
        ax.plot([x-1.5, x-1.5], [6, 54], linewidth=2.0, color="#111827")
        ax.plot([x+1.5, x+1.5], [6, 54], linewidth=2.0, color="#111827")
    ax.set_title(title, fontsize=10, pad=4); st.pyplot(fig)

def render_structure(structure_name: str):
    if "Lineare" in structure_name:
        draw_linear_fine("Lineare fine (freddo/secco)")
    elif "Onda" in structure_name:
        draw_wave_convex("Onda / convessa (universale)")
    elif "Cross" in structure_name:
        draw_cross("Cross (incrociata)")
    elif "Chevron" in structure_name:
        draw_chevron("Chevron (scarico direz.)")
    elif "Doppia" in structure_name:
        draw_double_linear("Doppia lineare")
    else:
        draw_diagonal_drain("Scarico diagonale (caldo/umido)")

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
        fig1 = plt.figure()
        plt.plot(t,res["T2m"],label="T2m"); plt.plot(t,res["T_surf"],label="T_surf"); plt.plot(t,res["T_top5"],label="T_top5")
        plt.legend(); plt.title("Temperature"); plt.xlabel("Ora"); plt.ylabel("°C")
        st.pyplot(fig1)

        fig2 = plt.figure()
        plt.bar(t,res["prp_mmph"])
        plt.title("Precipitazione (mm/h)"); plt.xlabel("Ora"); plt.ylabel("mm/h")
        st.pyplot(fig2)

        st.download_button("Scarica CSV risultato", data=res.to_csv(index=False), file_name="forecast_with_snowT.csv", mime="text/csv")

        # blocchi A/B/C
        for L,(s,e) in {"A":(A_start,A_end),"B":(B_start,B_end),"C":(C_start,C_end)}.items():
            st.markdown(f"### Blocco {L}")
            W = window_slice(res, tzname, s, e)
            t_med = float(W["T_surf"].mean())
            st.markdown(f"**T_surf medio {L}: {t_med:.1f}°C**")

            # Wax cards + loghi
            cols = st.columns(len(BRANDS))
            for i,(brand,(col,bands)) in enumerate(BRANDS.items()):
                rec = pick(bands, t_med)
                cols[i].markdown(
                    f"<div class='brand'><img src='{svg_logo(brand.upper(), col)}'/>"
                    f"<div><div style='font-size:.8rem;opacity:.85'>{brand}</div>"
                    f"<div style='font-weight:800'>{rec}</div></div></div>", unsafe_allow_html=True
                )

            # Struttura consigliata (preset) + disegno
            structure, side, base = tune_for(t_med, "GS")  # default reference
            st.markdown(f"**Struttura consigliata:** {structure}  ·  **Lamina SIDE:** {side:.1f}°  ·  **BASE:** {base:.1f}°")

            # scelte struttura extra (per mostrare altri preset “tipo Wintersteiger”)
            preset = st.selectbox(
                f"Preset struttura (Blocco {L})",
                ["Auto (consigliata) — " + structure,
                 "Lineare fine", "Onda/convessa", "Scarico diagonale", "Cross (incrociata)", "Chevron", "Doppia lineare"],
                index=0, key=f"preset_{L}"
            )
            to_draw = structure if preset.startswith("Auto") else preset
            render_structure(to_draw)

            # Tuning per discipline
            disc = st.multiselect(f"Discipline (Blocco {L})", ["SL","GS","SG","DH"], default=["SL","GS"], key=f"disc_{L}")
            rows = []
            for d in disc:
                sname, side_d, base_d = tune_for(t_med, d)
                rows.append([d, sname, f"{side_d:.1f}°", f"{base_d:.1f}°"])
            if rows:
                st.table(pd.DataFrame(rows, columns=["Disciplina","Struttura","Lamina SIDE (°)","Lamina BASE (°)"]))
    except Exception as e:
        st.error(f"Errore: {e}")

# Telemark · Pro Wax & Tune (A/B/C) — ricerca tipo Meteoblue + strutture Wintersteiger-like
# Requisiti: streamlit, pandas, requests, matplotlib, python-dateutil

import streamlit as st
import pandas as pd
import requests, io, base64, math
import numpy as np
import matplotlib.pyplot as plt
from datetime import time
from dateutil import tz

# ------------------------------ PAGE SETUP ------------------------------
st.set_page_config(page_title="Telemark · Pro Wax & Tune", page_icon="❄️", layout="wide")

PRIMARY = "#10bfcf"; BG = "#0f172a"; CARD = "#111827"; TEXT = "#e5e7eb"
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] > .main {{ background: linear-gradient(180deg, {BG} 0%, #0b1224 100%); }}
.block-container {{ padding-top: 0.8rem; }}
h1,h2,h3,h4,h5,p,span,div {{ color:{TEXT}; }}
.badge {{ border:1px solid rgba(255,255,255,.15); padding:6px 10px; border-radius:999px; font-size:.78rem; opacity:.85; }}
.card {{ background:{CARD}; border:1px solid rgba(255,255,255,.08);
        border-radius:16px; padding:14px; box-shadow:0 8px 20px rgba(0,0,0,.25); }}
.btn-primary button {{ background:{PRIMARY}!important; color:#002b30!important; font-weight:700; border-radius:12px; }}
.opt {{ padding:6px 10px; border-radius:8px; border:1px solid rgba(255,255,255,.12); margin-top:6px; }}
.brand {{ display:flex;align-items:center;gap:8px;padding:8px 10px;border-radius:12px;
         background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08); }}
.brand img {{ height:22px; }}
.table tbody tr td, .table thead tr th {{ color:{TEXT}; }}
</style>
""", unsafe_allow_html=True)

st.markdown("### Telemark · Pro Wax & Tune")
st.markdown("<span class='badge'>Ricerca località tipo Meteoblue · Blocchi A/B/C · Sciolina + Strutture stile Wintersteiger · Angoli lamine</span>", unsafe_allow_html=True)

# ------------------------------ HELPERS ------------------------------
def svg_badge(text, color):
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='200' height='36'>\
<rect width='200' height='36' rx='6' fill='{color}'/>\
<text x='12' y='24' font-size='16' font-weight='700' fill='white'>{text}</text></svg>"
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")

BRAND_BADGES = {
    "Swix":      svg_badge("SWIX", "#ef4444"),
    "Toko":      svg_badge("TOKO", "#f59e0b"),
    "Vola":      svg_badge("VOLA", "#3b82f6"),
    "Rode":      svg_badge("RODE", "#22c55e"),
    "Holmenkol": svg_badge("HOLMENKOL", "#0ea5e9"),
    "Maplus":    svg_badge("MAPLUS", "#a855f7"),
    "Star":      svg_badge("STAR", "#e11d48"),
}

def country_flag_emoji(cc):
    # Nominatim returns lower-case ISO alpha2
    if not cc: return "🏳️"
    base = 127397
    cc = cc.upper()
    return chr(ord(cc[0])+base) + chr(ord(cc[1])+base)

# ------------------------------ SEARCH (Meteoblue-like) ------------------------------
# Digiti -> ogni carattere scatena una nuova ricerca Nominatim -> menu a tendina con bandierina
def nominatim(q, limit=8):
    if not q or len(q) < 2: return []
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": q, "format":"json", "limit": limit, "addressdetails": 1},
            headers={"User-Agent": "telemark-pro-wax/1.0"},
            timeout=8
        )
        r.raise_for_status()
        js = r.json()
        out = []
        for it in js:
            addr = it.get("address", {})
            cc = addr.get("country_code", "")
            flag = country_flag_emoji(cc) + " "
            label = flag + it.get("display_name","")[:120]
            out.append({
                "label": label,
                "lat": float(it.get("lat", 0.0)),
                "lon": float(it.get("lon", 0.0)),
                "cc": cc
            })
        return out
    except Exception:
        return []

colL, colR = st.columns([3, 1])
with colL:
    q = st.text_input("Cerca località (digita e scegli dalla tendina)", value=st.session_state.get("q","Champoluc"),
                      placeholder="Es. Champoluc, Cervinia, Sestriere…", key="q_input")
    # ad ogni carattere ricalcolo i suggerimenti
    suggestions = nominatim(q, limit=10)
    labels = [s["label"] for s in suggestions] or ["— nessun risultato —"]
    choice = st.selectbox("Suggerimenti", labels, index=0)
    if suggestions:
        sel = suggestions[labels.index(choice)]
        st.session_state["lat"]  = sel["lat"]
        st.session_state["lon"]  = sel["lon"]
        st.session_state["label"]= sel["label"]
        st.session_state["q"]    = q
with colR:
    tzname = st.selectbox("Timezone", ["Europe/Rome","UTC"], index=0)
    hours  = st.slider("Ore previsione", 12, 168, 72, 12)

lat   = st.session_state.get("lat", 45.831)
lon   = st.session_state.get("lon", 7.730)
label = st.session_state.get("label", "Champoluc, IT")

# ------------------------------ WINDOWS A/B/C ------------------------------
st.markdown("#### Finestre A · B · C (oggi)")
c1,c2,c3 = st.columns(3)
with c1:
    A_start = st.time_input("Inizio A", value=time(9,0))
    A_end   = st.time_input("Fine A",   value=time(11,0))
with c2:
    B_start = st.time_input("Inizio B", value=time(11,0))
    B_end   = st.time_input("Fine B",   value=time(13,0))
with c3:
    C_start = st.time_input("Inizio C", value=time(13,0))
    C_end   = st.time_input("Fine C",   value=time(16,0))

# ------------------------------ METEO + MODEL ------------------------------
def fetch_open_meteo(lat, lon, tzname):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon, "timezone": tzname,
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
    h = js["hourly"]; df = pd.DataFrame(h); df["time"] = pd.to_datetime(df["time"])  # naive
    now = pd.Timestamp.now().floor("H")
    df = df[df["time"] >= now].head(hours).reset_index(drop=True)
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
    required = ["T2m","cloud","wind","sunup","prp_mmph","prp_type"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    if "td" not in df.columns: df["td"] = np.nan
    rain = df["prp_type"].str.lower().isin(["rain","mixed"])
    snow = df["prp_type"].str.lower().eq("snow")
    sun  = df["sunup"].astype(int)==1
    tw = (df["T2m"]+df["td"])/2.0
    wet = (rain | (df["T2m"]>0) | (sun & (df["cloud"]<0.3) & (df["T2m"]>=-3)) |
           (snow & (df["T2m"]>=-1)) | (snow & tw.ge(-0.5).fillna(False)))
    T_surf = pd.Series(np.nan, index=df.index)
    T_surf.loc[wet] = 0.0
    dry = ~wet
    clear = (1.0-df["cloud"]).clip(0,1)
    windc = df["wind"].clip(upper=6.0)
    drad = (1.5 + 3.0*clear - 0.3*windc).clip(0.5,4.5)
    T_surf.loc[dry] = df["T2m"][dry] - drad[dry]
    sunny_cold = sun & dry & df["T2m"].between(-10,0, inclusive="both")
    T_surf.loc[sunny_cold] = np.minimum((df["T2m"] + 0.5*(1.0-df["cloud"]))[sunny_cold], -0.5)
    T_top5 = pd.Series(np.nan, index=df.index)
    tau = pd.Series(6.0, index=df.index)
    tau.loc[rain | snow | (df["wind"]>=6)] = 3.0
    tau.loc[(~sun) & (df["wind"]<2) & (df["cloud"]<0.3)] = 8.0
    alpha = 1.0 - np.exp(-dt_hours/tau)
    if len(df):
        T_top5.iloc[0] = min(df["T2m"].iloc[0], 0.0)
        for i in range(1,len(df)):
            T_top5.iloc[i] = T_top5.iloc[i-1] + alpha.iloc[i]*(T_surf.iloc[i]-T_top5.iloc[i-1])
    df["T_surf"] = T_surf; df["T_top5"] = T_top5
    return df

def slice_today(res, tzname, s, e):
    t = pd.to_datetime(res["time"]).dt.tz_localize(tz.gettz(tzname), nonexistent='shift_forward', ambiguous='NaT')
    D = res.copy(); D["dt"] = t
    today = pd.Timestamp.now(tz=tz.gettz(tzname)).date()
    W = D[(D["dt"].dt.date==today) & (D["dt"].dt.time>=s) & (D["dt"].dt.time<=e)]
    return W if not W.empty else D.head(6)

# ------------------------------ STRUCTURE RENDER (Wintersteiger-like) ------------------------------
def _blank_ax(w=640, h=180, dpi=160):
    fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax = fig.add_axes([0,0,1,1])
    ax.set_xlim(0,w); ax.set_ylim(0,h)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0,0), w, h, color="#1a202c"))
    return fig, ax

def img_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    return buf.getvalue()

def draw_linear(spacing=14, width=2, wobble=0.0):
    w,h=640,180; fig,ax=_blank_ax(w,h)
    xs = np.arange(spacing/2, w, spacing)
    for x in xs:
        y = np.linspace(10,h-10,200)
        xline = x + wobble*np.sin(np.linspace(0, np.pi*6, y.size))
        ax.plot(xline, y, color="#9ca3af", linewidth=width, solid_capstyle="round")
    return img_bytes(fig)

def draw_diagonal(angle_deg=20, spacing=14, width=2):
    w,h=640,180; fig,ax=_blank_ax(w,h)
    angle = np.deg2rad(angle_deg)
    # draw many parallel lines across canvas
    # project spacing along perpendicular
    step = spacing/np.cos(angle)
    for k in np.arange(-w, w*2, step):
        x0 = k; y0 = 0
        x1 = k + h*np.tan(angle); y1 = h
        ax.plot([x0,x1],[y0,y1], color="#9ca3af", linewidth=width, solid_capstyle="round")
    return img_bytes(fig)

def draw_chevron(spacing=20, width=2, amplitude=50):
    w,h=640,180; fig,ax=_blank_ax(w,h)
    centers = np.arange(spacing, w, spacing)
    for c in centers:
        t = np.linspace(-amplitude, amplitude, 200)
        y = (h/2) + 60*np.sin(t/120*np.pi)
        x = c + 0.6*t
        ax.plot(x, y, color="#9ca3af", linewidth=width)
    return img_bytes(fig)

def draw_wave(spacing=26, width=2, amp=22):
    w,h=640,180; fig,ax=_blank_ax(w,h)
    rows = [h*0.25, h*0.5, h*0.75]
    for y0 in rows:
        x = np.linspace(0, w, 700)
        y = y0 + amp*np.sin(2*np.pi*x/spacing)
        ax.plot(x, y, color="#9ca3af", linewidth=width)
    return img_bytes(fig)

def draw_diamond(spacing=18, width=2, angle=25):
    # cross structure: left + right diagonals
    left  = draw_diagonal(+angle, spacing, width)
    right = draw_diagonal(-angle, spacing, width)
    # overlay two images
    imgL = plt.imread(io.BytesIO(left))
    imgR = plt.imread(io.BytesIO(right))
    fig, ax = _blank_ax()
    ax.imshow(imgL, extent=(0,640,0,180))
    ax.imshow(imgR, extent=(0,640,0,180), alpha=1.0)
    return img_bytes(fig)

def draw_broken_wave(spacing=26, width=2, amp=26):
    # warm/wet coarse broken waves
    w,h=640,180; fig,ax=_blank_ax(w,h)
    x = np.linspace(0, w, 600)
    for row in [h*0.22, h*0.42, h*0.62, h*0.82]:
        y = row + amp*np.sin(2*np.pi*x/spacing)
        # break into segments
        for i in range(0, len(x), 60):
            if (i//60)%2==0:
                ax.plot(x[i:i+40], y[i:i+40], color="#9ca3af", linewidth=width+0.6)
    return img_bytes(fig)

# structure chooser by temperature / wetness (style similar alle “schede” Wintersteiger)
def choose_structure(t_surf_med, wet=False):
    if wet or t_surf_med > 0.5:
        return ("Warm/Wet — Broken Wave (coarse)", draw_broken_wave())
    if -3 <= t_surf_med <= 0.5:
        return ("Near 0°C — Chevron / Diamond", draw_diamond())
    if -8 <= t_surf_med < -3:
        return ("Allround — Diagonal 20°", draw_diagonal(20))
    if -12 <= t_surf_med < -8:
        return ("Cold — Linear fine", draw_linear(spacing=12, width=2))
    return ("Very cold — Linear extra-fine", draw_linear(spacing=10, width=1.8))

# ------------------------------ WAX BANDS ------------------------------
# Semplificate, no-fluoro
WAX = {
    "Swix": [("PS5 Turquoise",-18,-10),("PS6 Blue",-12,-6),("PS7 Violet",-8,-2),("PS8 Red",-4,4),("PS10 Yellow",0,10)],
    "Toko": [("Blue",-30,-9),("Red",-12,-4),("Yellow",-6,0)],
    "Vola": [("MX-E Violet/Blue",-12,-4),("MX-E Red",-5,0),("MX-E Warm",-2,10)],
    "Rode": [("R20 Blue",-18,-8),("R30 Violet",-10,-3),("R40 Red",-5,0),("R50 Yellow",-1,10)],
    "Holmenkol":[("Alpha Mix Cold",-15,-5),("Beta Mix",-8,-2),("Ultra Mix Warm",-2,8)],
    "Maplus":[("BP1 Hard",-15,-7),("BP2 Medium",-10,-3),("BP3 Soft",-5,5)],
    "Star":[("F15 Green",-18,-10),("F10 Blue",-12,-6),("F30 Red",-6,0)]
}
def pick_band(bands, t):
    for n,tmin,tmax in bands:
        if t>=tmin and t<=tmax: return n
    return bands[-1][0] if t>bands[-1][2] else bands[0][0]

def wax_cards(t_med):
    cols = st.columns(4)
    brands = list(WAX.keys())
    for i,b in enumerate(brands[:4]):
        rec = pick_band(WAX[b], t_med)
        cols[i].markdown(f"<div class='brand'><img src='{BRAND_BADGES[b]}'/><div><div style='opacity:.8;font-size:.8rem'>{b}</div><div style='font-weight:800'>{rec}</div></div></div>", unsafe_allow_html=True)
    cols2 = st.columns(3)
    for j,b in enumerate(brands[4:7]):
        rec = pick_band(WAX[b], t_med)
        cols2[j].markdown(f"<div class='brand'><img src='{BRAND_BADGES[b]}'/><div><div style='opacity:.8;font-size:.8rem'>{b}</div><div style='font-weight:800'>{rec}</div></div></div>", unsafe_allow_html=True)

# ------------------------------ TUNING (edges) ------------------------------
def tune_for(t_surf, discipline):
    # side angle (°) style richiesto; base (°)
    if t_surf <= -10:
        structure_note = "Fine/Linear"
        base = 0.5
        side_map = {"SL": 88.5, "GS": 88.0, "SG": 87.5, "DH": 87.5}
    elif t_surf <= -3:
        structure_note = "Universale/Diagonale"
        base = 0.7
        side_map = {"SL": 88.0, "GS": 88.0, "SG": 87.5, "DH": 87.0}
    else:
        structure_note = "Media-Grossa/Wet"
        base = 0.8 if t_surf <= 0.5 else 1.0
        side_map = {"SL": 88.0, "GS": 87.5, "SG": 87.0, "DH": 87.0}
    return structure_note, side_map.get(discipline, 88.0), base

def badge(win):
    tmin = float(win["T_surf"].min()); tmax = float(win["T_surf"].max())
    wet  = bool(((win["prp_type"].isin(["rain","mixed"])) | (win["prp_mmph"]>0.5)).any())
    if wet or tmax>0.5:
        color,title,desc = "#ef4444","CRITICAL","Possibile neve bagnata/pioggia · struttura grossa"
    elif tmax>-1.0:
        color,title,desc = "#f59e0b","WATCH","Vicino a 0°C · cere medio-morbide"
    else:
        color,title,desc = "#22c55e","OK","Neve fredda/asciutta · cere dure"
    st.markdown(f"""
    <div class='card' style='border-color:{color}'>
      <div style='font-weight:800;color:{color};margin-bottom:4px'>{title}</div>
      <div style='opacity:.95'>{desc}</div>
      <div style='font-size:12px;opacity:.7;margin-top:6px'>T_surf min {tmin:.1f}°C / max {tmax:.1f}°C</div>
    </div>""", unsafe_allow_html=True)
    return wet

# ------------------------------ RUN ------------------------------
colGo1, colGo2 = st.columns([1,2])
with colGo1:
    go = st.button("Scarica previsioni per la località", type="primary")
with colGo2:
    upl = st.file_uploader("…oppure carica CSV (time,T2m,cloud,wind,sunup,prp_mmph,prp_type,td)", type=["csv"])

def plots(res):
    t = pd.to_datetime(res["time"])
    fig1 = plt.figure(); plt.plot(t,res["T2m"],label="T2m"); plt.plot(t,res["T_surf"],label="T_surf"); plt.plot(t,res["T_top5"],label="T_top5")
    plt.legend(); plt.title("Temperature"); plt.xlabel("Ora"); plt.ylabel("°C")
    fig2 = plt.figure(); plt.bar(t,res["prp_mmph"]); plt.title("Precipitazione (mm/h)"); plt.xlabel("Ora"); plt.ylabel("mm/h")
    return fig1, fig2

def run_all(src):
    res = compute_snow_temperature(src, dt_hours=1.0)
    st.success(f"Dati pronti per **{label}**")
    st.dataframe(res, use_container_width=True)
    f1,f2 = plots(res); st.pyplot(f1); st.pyplot(f2)
    st.download_button("Scarica CSV risultato", data=res.to_csv(index=False), file_name="forecast_with_snowT.csv", mime="text/csv")

    # Blocchi A/B/C
    for L,(s,e) in {"A":(A_start,A_end),"B":(B_start,B_end),"C":(C_start,C_end)}.items():
        st.markdown(f"### Blocco {L}")
        W = slice_today(res, tzname, s, e)
        wet = badge(W)
        t_med = float(W["T_surf"].mean())

        # Struttura consigliata + render in stile Wintersteiger
        title, png = choose_structure(t_med, wet)
        st.markdown(f"**Struttura consigliata:** {title}  ·  **T_surf medio:** {t_med:.1f}°C")
        st.image(png, use_column_width=True)

        # Wax per brand
        wax_cards(t_med)

        # Lamine
        chosen = st.multiselect(f"Discipline (Blocco {L})", ["SL","GS","SG","DH"], default=["SL","GS"], key=f"d_{L}")
        rows=[]
        for d in chosen:
            note, side, base = tune_for(t_med, d)
            rows.append([d, note, f"{side:.1f}°", f"{base:.1f}°"])
        if rows:
            df_tune = pd.DataFrame(rows, columns=["Disciplina","Struttura (nota)","Lamina SIDE (°)","Lamina BASE (°)"])
            st.table(df_tune)

# run from CSV or API
if upl is not None:
    try:
        df_u = pd.read_csv(upl); run_all(df_u)
    except Exception as e:
        st.error(f"CSV non valido: {e}")
elif go:
    try:
        js = fetch_open_meteo(lat, lon, tzname); df_src = build_df(js, hours, tzname); run_all(df_src)
    except Exception as e:
        st.error(f"Errore: {e}")
else:
    st.info("Cerca la località, imposta A/B/C e premi **Scarica previsioni** (oppure carica un CSV).")

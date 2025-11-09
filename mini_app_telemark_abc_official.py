# telemark_pro_wax_app.py
import streamlit as st
import pandas as pd
import requests, base64, math
import matplotlib.pyplot as plt
import numpy as np
from datetime import time
from dateutil import tz

# -------------------- CONFIG / THEME --------------------
PRIMARY = "#10bfcf"; BG = "#0f172a"; CARD = "#111827"; TEXT = "#e5e7eb"
st.set_page_config(page_title="Telemark · Pro Wax & Tune", page_icon="❄️", layout="wide")
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] > .main {{ background: linear-gradient(180deg,{BG} 0%, #111827 100%); }}
.block-container {{ padding-top: .8rem; }}
h1,h2,h3,h4,h5,p,span,div {{ color:{TEXT}; }}
.badge {{
  border:1px solid rgba(255,255,255,.15);
  padding:6px 10px;border-radius:999px;font-size:.78rem;opacity:.85;
}}
.card {{
  background:{CARD}; border:1px solid rgba(255,255,255,.08);
  border-radius:16px; padding:14px; box-shadow:0 8px 20px rgba(0,0,0,.25);
}}
.brand {{ display:flex; align-items:center; gap:8px; padding:8px 10px;
  border-radius:12px; background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08); }}
.brand img {{ height:22px; }}
.small {{ font-size:.8rem; opacity:.8; }}
</style>
""", unsafe_allow_html=True)

st.markdown("### Telemark · Pro Wax & Tune")
st.markdown("<span class='badge'>Ricerca stile Meteoblue · Geolocalizzazione · Blocchi A/B/C · Sciolina + Struttura + Angoli</span>", unsafe_allow_html=True)

# -------------------- WEATHER MODEL --------------------
def compute_snow_temperature(df, dt_hours=1.0):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    req = ["T2m","cloud","wind","sunup","prp_mmph","prp_type"]
    for c in req:
        if c not in df.columns: raise ValueError(f"Missing column {c}")
    if "td" not in df.columns: df["td"] = float("nan")
    df = df.sort_values("time").reset_index(drop=True)

    rain = df["prp_type"].str.lower().isin(["rain","mixed"])
    snow = df["prp_type"].str.lower().eq("snow")
    sunup = df["sunup"].astype(int) == 1
    tw = (df["T2m"] + df["td"])/2.0

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
    drad = (1.5 + 3.0*clear - 0.3*windc).clip(lower=0.5, upper=4.5)
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
    if len(df):
        T_top5.iloc[0] = min(df["T2m"].iloc[0], 0.0)
        for i in range(1,len(df)):
            T_top5.iloc[i] = T_top5.iloc[i-1] + alpha.iloc[i]*(T_surf.iloc[i]-T_top5.iloc[i-1])

    df["T_surf"] = T_surf; df["T_top5"] = T_top5
    return df

# -------------------- OPEN-METEO & BUILD DF --------------------
def fetch_open_meteo(lat, lon, timezone_str):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {"latitude": lat, "longitude": lon, "timezone": timezone_str,
              "hourly":"temperature_2m,dew_point_2m,precipitation,rain,snowfall,cloudcover,windspeed_10m,is_day,weathercode",
              "forecast_days":7}
    r = requests.get(url, params=params, timeout=30); r.raise_for_status()
    return r.json()

def _prp_type(df):
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
    h = js["hourly"]; df = pd.DataFrame(h); df["time"]=pd.to_datetime(df["time"])
    now_naive = pd.Timestamp.now().floor("H")
    df = df[df["time"]>=now_naive].head(hours).reset_index(drop=True)
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

def window_slice(res, tzname, s, e):
    t = pd.to_datetime(res["time"]).dt.tz_localize(tz.gettz(tzname), nonexistent='shift_forward', ambiguous='NaT')
    D = res.copy(); D["dt"]=t
    today = pd.Timestamp.now(tz=tz.gettz(tzname)).date()
    w = D[(D["dt"].dt.date==today)&(D["dt"].dt.time>=s)&(D["dt"].dt.time<=e)]
    return w if not w.empty else D.head(7)

# -------------------- SEARCH UI (leave as-is since it works for you) --------------------
def flag_emoji(cc):
    # convert "it" -> 🇮🇹 etc.
    if not cc or len(cc)!=2: return ""
    base = 127397
    return chr(ord(cc[0].upper())+base) + chr(ord(cc[1].upper())+base)

def geocode_autocomplete(q, limit=10):
    if not q or len(q)<2: return []
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search",
                         params={"q":q,"format":"json","limit":limit,"addressdetails":1},
                         headers={"User-Agent":"telemark-pro-wax/1.0"}, timeout=10)
        r.raise_for_status()
        out=[]
        for it in r.json():
            label = it.get("display_name","")
            cc = (it.get("address") or {}).get("country_code","")
            out.append({
                "label": f"{flag_emoji(cc)}  {label}",
                "lat": float(it.get("lat",0)),
                "lon": float(it.get("lon",0)),
            })
        return out
    except Exception:
        return []

col1,col2 = st.columns([3,1])
with col1:
    q = st.text_input("Cerca località (digita e scegli)", placeholder="Es. Champoluc, Cervinia, Sestriere…", key="q")
    suggestions = geocode_autocomplete(q, limit=12)
    place = None
    if suggestions:
        # menu a tendina “stile meteoblue” con selectbox che si aggiorna mentre scrivi
        options = [s["label"] for s in suggestions]
        choice = st.selectbox("Suggerimenti", options, index=0)
        sel = suggestions[options.index(choice)]
        if st.button("Usa questa località"):
            st.session_state["chosen_lat"]=sel["lat"]
            st.session_state["chosen_lon"]=sel["lon"]
            st.session_state["chosen_label"]=choice
            st.session_state["q"]=choice
            st.rerun()
with col2:
    if st.button("📍 Usa la mia posizione"):
        try:
            r = requests.get("https://ipapi.co/json", timeout=8)
            if r.ok:
                j=r.json()
                st.session_state["chosen_lat"]=float(j.get("latitude",0))
                st.session_state["chosen_lon"]=float(j.get("longitude",0))
                city=j.get("city","La tua posizione")
                st.session_state["chosen_label"]=f"{flag_emoji(j.get('country_code',''))}  {city}"
                st.session_state["q"]=city
                st.success("Posizione impostata")
            else: st.error("Geolocalizzazione non disponibile.")
        except Exception:
            st.error("Geolocalizzazione non disponibile.")

lat = st.session_state.get("chosen_lat",45.831)
lon = st.session_state.get("chosen_lon",7.730)
label = st.session_state.get("chosen_label","🇮🇹  Champoluc (Ramey)")
tzname = st.selectbox("Timezone",["Europe/Rome","UTC"], index=0)
hours  = st.slider("Ore previsione",12,168,72,12)

st.markdown("#### Finestre A · B · C (oggi)")
cA,cB,cC = st.columns(3)
with cA:
    A_start = st.time_input("Inizio A", value=time(9,0), key="A_s")
    A_end   = st.time_input("Fine A",   value=time(11,0), key="A_e")
with cB:
    B_start = st.time_input("Inizio B", value=time(11,0), key="B_s")
    B_end   = st.time_input("Fine B",   value=time(13,0), key="B_e")
with cC:
    C_start = st.time_input("Inizio C", value=time(13,0), key="C_s")
    C_end   = st.time_input("Fine C",   value=time(16,0), key="C_e")

# -------------------- WAX BRANDS (non-fluoro) --------------------
#  Swix, Toko, Vola, Rode + Holmenkol, Maplus (Briko-Maplus), Star, Dominator, OneBall
SWIX      = [("PS5 Turquoise",-18,-10),("PS6 Blue",-12,-6),("PS7 Violet",-8,-2),("PS8 Red",-4,4),("PS10 Yellow",0,10)]
TOKO      = [("Blue",-30,-9),("Red",-12,-4),("Yellow",-6,0)]
VOLA      = [("MX-E Violet/Blue",-12,-4),("MX-E Red",-5,0),("MX-E Warm",-2,10)]
RODE      = [("R20 Blue",-18,-8),("R30 Violet",-10,-3),("R40 Red",-5,0),("R50 Yellow",-1,10)]
HOLMENKOL = [("Alpha Blue",-14,-8),("Alpha Violet",-8,-2),("Alpha Red",-4,4)]
MAPLUS    = [("P1 Blue",-18,-8),("P2 Violet",-10,-2),("P3 Red",-4,4),("P4 Yellow",0,10)]
STAR      = [("M Blue",-18,-8),("M Violet",-10,-2),("M Red",-4,4)]
DOMINATOR = [("Zoom Graphite",-30,-5),("Hyper-Zoom",-8,2)]
ONEBALL   = [("X-Wax Cold",-15,-6),("X-Wax All-Temp",-7,3)]

BRANDS = [
    ("Swix","#ef4444",SWIX),("Toko","#f59e0b",TOKO),("Vola","#3b82f6",VOLA),("Rode","#22c55e",RODE),
    ("Holmenkol","#2563eb",HOLMENKOL),("Maplus","#8b5cf6",MAPLUS),
    ("Star","#eab308",STAR),("Dominator","#f97316",DOMINATOR),("OneBall","#b45309",ONEBALL),
]

def svg_logo(text, color):
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='180' height='34'><rect width='180' height='34' rx='6' fill='{color}'/><text x='10' y='22' font-size='16' font-weight='700' fill='white'>{text}</text></svg>"
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()

def pick(bands, t):
    for n,tmin,tmax in bands:
        if t>=tmin and t<=tmax: return n
    return bands[-1][0] if t>bands[-1][2] else bands[0][0]

def wax_cards(t_med):
    cols = st.columns(3)
    blocks = [BRANDS[:3], BRANDS[3:6], BRANDS[6:]]
    for col, chunk in zip(cols, blocks):
        with col:
            for brand, color, items in chunk:
                rec = pick(items, t_med)
                col.markdown(
                    f"<div class='brand'><img src='{svg_logo(brand.upper(),color)}'/>"
                    f"<div><div class='small'>{brand}</div><div style='font-weight:800'>{rec}</div></div></div>",
                    unsafe_allow_html=True
                )

# -------------------- STRUCTURE DRAWINGS (Wintersteiger-style) --------------------
def _canvas():
    fig,ax = plt.subplots(figsize=(3.6,2.2), dpi=150)
    ax.set_xlim(0,100); ax.set_ylim(0,60)
    ax.set_facecolor("#e5e7eb");  # light base
    ax.axis('off')
    return fig,ax

def draw_linear(spacing=4, angle=0, length=60):
    fig,ax = _canvas()
    ang = np.deg2rad(angle)
    # draw many parallel grooves
    for x in np.arange(-20,120,spacing):
        x0 = x; y0 = 0
        x1 = x + length*np.tan(ang); y1 = length
        ax.plot([x0,x1],[y0,y1], lw=1.2, color="#111827")
    return fig

def draw_cross(spacing=5, angle=22):
    fig,ax = _canvas()
    ang = angle
    # set A
    for x in np.arange(-20,120,spacing):
        x0=x; y0=0; x1=x+60*np.tan(np.deg2rad( ang)); y1=60
        ax.plot([x0,x1],[y0,y1], lw=1.1, color="#111827")
    # set B
    for x in np.arange(-20,120,spacing):
        x0=x; y0=0; x1=x+60*np.tan(np.deg2rad(-ang)); y1=60
        ax.plot([x0,x1],[y0,y1], lw=1.1, color="#111827")
    return fig

def draw_herringbone(spacing=6, angle=28):
    fig,ax = _canvas()
    # chevron: alternate short diagonals meeting at center line
    for y in np.arange(5,60,spacing):
        ax.plot([30,50],[y,y+spacing], lw=1.3, color="#111827")
        ax.plot([70,50],[y,y+spacing], lw=1.3, color="#111827")
    for y in np.arange(0,60,spacing):
        ax.plot([30,50],[y+spacing,y], lw=1.3, color="#111827")
        ax.plot([70,50],[y+spacing,y], lw=1.3, color="#111827")
    return fig

def draw_wave(spacing=10, amp=6):
    fig,ax = _canvas()
    xs = np.linspace(0,100,400)
    for off in np.arange(0,spacing*1.2,spacing/2):
        ys = 30 + amp*np.sin((xs+off)/12.0)
        ax.plot(xs, ys, lw=1.3, color="#111827")
    return fig

def draw_lateral_drain(spacing=6, margin=8):
    fig,ax = _canvas()
    # dense vertical grooves + deeper channels near edges
    for x in np.arange(margin,100-margin,spacing):
        ax.plot([x,x],[5,55], lw=1.0, color="#111827")
    ax.plot([margin-2,margin-2],[5,55], lw=2.4, color="#111827")
    ax.plot([100-margin+2,100-margin+2],[5,55], lw=2.4, color="#111827")
    return fig

def structure_recommendation(t_surf, wet):
    """
    Map stile Wintersteiger:
    - freddo/asciutto: lineare fine
    - universale sottozero: lineare media / onda (broken)
    - vicino/oltre 0 o bagnato: cross (diagonale) o herringbone + scarico laterale
    """
    if wet or t_surf > -1.0:
        return "Cross/Herringbone + Scarico laterale", draw_cross(), draw_lateral_drain()
    if t_surf <= -10:
        return "Lineare fine (cold/dry)", draw_linear(spacing=3), draw_linear(spacing=3, angle=5)
    if t_surf <= -3:
        return "Lineare medio · onda (universal)", draw_linear(spacing=5), draw_wave()
    return "Lineare medio", draw_linear(spacing=5), draw_linear(spacing=5, angle=8)

# -------------------- TUNING (angles side format) --------------------
def tune_for(t_surf, discipline):
    if t_surf <= -10:
        structure = "Fine (lineare)"; base = 0.5; side_map = {"SL":88.5,"GS":88.0,"SG":87.5,"DH":87.5}
    elif t_surf <= -3:
        structure = "Media (universale)"; base = 0.7; side_map = {"SL":88.0,"GS":88.0,"SG":87.5,"DH":87.0}
    else:
        structure = "Media-Grossa"; base = 0.8 if t_surf <= 0.5 else 1.0; side_map = {"SL":88.0,"GS":87.5,"SG":87.0,"DH":87.0}
    return structure, side_map.get(discipline,88.0), base

# -------------------- RUN --------------------
c_run1,c_run2 = st.columns([1,2])
with c_run1:
    go = st.button("Scarica previsioni per la località", type="primary")
with c_run2:
    upl = st.file_uploader("…oppure carica CSV (time,T2m,cloud,wind,sunup,prp_mmph,prp_type,td)", type=["csv"])

def plots(res):
    fig1 = plt.figure(figsize=(6,2.2)); t = pd.to_datetime(res["time"])
    plt.plot(t,res["T2m"], label="T2m")
    plt.plot(t,res["T_surf"], label="T_surf")
    plt.plot(t,res["T_top5"], label="T_top5")
    plt.legend(); plt.title("Temperature vs tempo"); plt.xlabel("Ora"); plt.ylabel("°C")
    fig2 = plt.figure(figsize=(6,2.0))
    plt.bar(t,res["prp_mmph"]); plt.title("Precipitazione (mm/h)"); plt.xlabel("Ora"); plt.ylabel("mm/h")
    return fig1,fig2

def badge(win):
    tmin=float(win["T_surf"].min()); tmax=float(win["T_surf"].max())
    wet = bool(((win["prp_type"].isin(["rain","mixed"])) | (win["prp_mmph"]>0.5)).any())
    if wet or tmax>0.5:   color,title,desc = "#ef4444","CRITICAL","Possibile neve bagnata/pioggia · struttura grossa/cross"
    elif tmax>-1.0:       color,title,desc = "#f59e0b","WATCH","Vicino a 0°C · cere medio-morbide"
    else:                 color,title,desc = "#22c55e","OK","Freddo/asciutto · cere dure"
    st.markdown(f"""
    <div class='card' style='border-color:{color}'>
      <div style='font-weight:800;color:{color};margin-bottom:4px'>{title}</div>
      <div class='small'>{desc}</div>
      <div class='small' style='margin-top:6px'>T_surf min {tmin:.1f}°C / max {tmax:.1f}°C</div>
    </div>
    """, unsafe_allow_html=True)
    return wet

def run_all(src, spot_label):
    res = compute_snow_temperature(src, dt_hours=1.0)
    st.success(f"Previsioni per **{spot_label}** pronte.")
    st.dataframe(res, use_container_width=True)
    f1,f2 = plots(res); st.pyplot(f1); st.pyplot(f2)
    st.download_button("Scarica CSV risultato", data=res.to_csv(index=False), file_name="forecast_with_snowT.csv", mime="text/csv")

    for name,(s,e) in {"A":(A_start,A_end),"B":(B_start,B_end),"C":(C_start,C_end)}.items():
        st.markdown(f"### Blocco {name}")
        W = window_slice(res, tzname, s, e)
        wet = badge(W)
        t_med = float(W["T_surf"].mean())
        st.markdown(f"**T_surf medio {name}: {t_med:.1f}°C**")

        # Wax
        wax_cards(t_med)

        # Structures (Wintersteiger-style panels)
        title, figPrimary, figAlt = structure_recommendation(t_med, wet)
        st.markdown(f"**Struttura consigliata:** {title}")
        c1,c2 = st.columns(2)
        with c1: st.pyplot(figPrimary)
        with c2: st.pyplot(figAlt)

        # Edges table
        disc = st.multiselect(f"Discipline (Blocco {name})", ["SL","GS","SG","DH"], default=["SL","GS"], key=f"disc_{name}")
        rows=[]
        for d in disc:
            structure, side, base = tune_for(t_med, d)
            rows.append([d, structure, f"{side:.1f}°", f"{base:.1f}°"])
        if rows:
            st.table(pd.DataFrame(rows, columns=["Disciplina","Struttura soluzione","Lamina SIDE (°)","Lamina BASE (°)"]))

# CSV o Open-Meteo
if upl is not None:
    try:
        df = pd.read_csv(upl); run_all(df, label)
    except Exception as e:
        st.error(f"CSV non valido: {e}")

if go:
    try:
        js = fetch_open_meteo(lat, lon, tzname); src = build_df(js, hours, tzname); run_all(src, label)
    except Exception as e:
        st.error(f"Errore: {e}")

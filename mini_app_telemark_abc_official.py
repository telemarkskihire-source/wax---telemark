# telemark_pro_app.py  —  file unico

import streamlit as st
import pandas as pd
import requests, base64
import matplotlib.pyplot as plt
from datetime import time
from dateutil import tz

# ============== BASE LOOK ==============
PRIMARY = "#10bfcf"; BG = "#0f172a"; CARD = "#111827"; TEXT = "#e5e7eb"
st.set_page_config(page_title="Telemark · Pro Wax & Tune", page_icon="❄️", layout="wide")
st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
  background: linear-gradient(180deg, {BG} 0%, #111827 100%);
}}
.block-container {{ padding-top: 1rem; }}
h1,h2,h3,h4,h5,p,span,div {{ color:{TEXT}; }}
.badge {{
  border:1px solid rgba(255,255,255,.15); padding:6px 10px; border-radius:999px;
  font-size:.78rem; opacity:.85;
}}
.card {{
  background:{CARD}; border:1px solid rgba(255,255,255,.08); border-radius:16px;
  padding:14px; box-shadow: 0 8px 20px rgba(0,0,0,.25);
}}
.brand {{ display:flex; gap:10px; align-items:center; padding:10px 12px;
  border-radius:12px; background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08); }}
.brand img {{ height:22px; }}
.kpi {{ display:flex; gap:8px; align-items:center; background:rgba(16,191,207,.06);
  border:1px dashed rgba(16,191,207,.45); padding:10px 12px; border-radius:12px; }}
</style>
""", unsafe_allow_html=True)

st.markdown("### Telemark · Pro Wax & Tune")
st.markdown("<span class='badge'>Ricerca semplice · Geolocalizzazione · Blocchi A/B/C · Sciolina + Struttura + Angoli (SIDE)</span>", unsafe_allow_html=True)

# ============== CORE MODEL (compatto) ==============
def compute_snow_temperature(df, dt_hours=1.0):
    import math
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    req = ["T2m","cloud","wind","sunup","prp_mmph","prp_type"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    if "td" not in df.columns: df["td"] = float("nan")

    rain = df["prp_type"].str.lower().isin(["rain","mixed"])
    snow = df["prp_type"].str.lower().eq("snow")
    sunup = df["sunup"].astype(int)==1
    tw = (df["T2m"]+df["td"])/2.0

    wet = (rain | (df["T2m"]>0) |
           (sunup & (df["cloud"]<0.3) & (df["T2m"]>=-3)) |
           (snow & (df["T2m"]>=-1)) |
           (snow & tw.ge(-0.5).fillna(False)))

    T_surf = pd.Series(index=df.index, dtype=float); T_surf.loc[wet] = 0.0
    dry = ~wet
    clear = (1-df["cloud"]).clip(0,1); windc = df["wind"].clip(upper=6.0)
    drad = (1.5 + 3.0*clear - 0.3*windc).clip(0.5,4.5)
    T_surf.loc[dry] = df["T2m"][dry] - drad[dry]

    sunny_cold = sunup & dry & df["T2m"].between(-10,0, inclusive="both")
    T_surf.loc[sunny_cold] = pd.concat([
        (df["T2m"] + 0.5*(1-clear))[sunny_cold],
        pd.Series(-0.5, index=df.index)[sunny_cold]
    ], axis=1).min(axis=1)

    T_top5 = pd.Series(index=df.index, dtype=float)
    tau = pd.Series(6.0, index=df.index, dtype=float)
    tau.loc[rain | snow | (df["wind"]>=6)] = 3.0
    tau.loc[(~sunup) & (df["wind"]<2) & (df["cloud"]<0.3)] = 8.0
    alpha = 1.0 - (math.e**(-dt_hours/tau))
    if len(df):
        T_top5.iloc[0] = min(df["T2m"].iloc[0], 0.0)
        for i in range(1,len(df)):
            T_top5.iloc[i] = T_top5.iloc[i-1] + alpha.iloc[i]*(T_surf.iloc[i]-T_top5.iloc[i-1])

    df["T_surf"] = T_surf; df["T_top5"] = T_top5
    return df

# ============== RICERCA LOCALITÀ SEMPLIFICATA ==============
def geocode(q, limit=10):
    if not q or len(q)<2: return []
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q":q, "format":"json", "limit":limit, "addressdetails":1},
            headers={"User-Agent":"telemark-pro-wax/1.0"},
            timeout=10
        )
        r.raise_for_status()
        js = r.json()
        return [{"label": item["display_name"],
                 "lat": float(item["lat"]),
                 "lon": float(item["lon"])} for item in js]
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

# UI Ricerca super semplice:
colS, colG = st.columns([3,1])
with colS:
    q = st.text_input("Cerca località", placeholder="Es. Champoluc, Cervinia, Sestriere…")
    options = geocode(q, limit=8)
    labels = [o["label"] for o in options]
    picked = st.selectbox("Risultati", labels if labels else ["Champoluc (Ramey)"], index=0)
with colG:
    if st.button("📍 Usa posizione"):
        glat, glon, gcity = ip_geolocate()
        if glat is not None:
            st.session_state["geo"] = {"lat":glat,"lon":glon,"label": gcity or "La tua posizione"}
            st.success("Posizione impostata")
        else:
            st.error("Geolocalizzazione non disponibile")

# Scelta finale lat/lon:
if "geo" in st.session_state:
    SEL = st.session_state["geo"]
elif options:
    SEL = options[labels.index(picked)]
else:
    SEL = {"lat": 45.831, "lon": 7.730, "label":"Champoluc (Ramey)"}

colP = st.columns(4)
with colP[0]: lat = st.number_input("Lat", value=float(SEL["lat"]), format="%.6f")
with colP[1]: lon = st.number_input("Lon", value=float(SEL["lon"]), format="%.6f")
with colP[2]: tzname = st.selectbox("Timezone", ["Europe/Rome","UTC"], index=0)
with colP[3]: hours = st.slider("Ore previsione", 12, 168, 72, 12)

# ============== FINESTRE A/B/C ==============
st.markdown("#### Finestre (oggi) A · B · C")
ca, cb, cc = st.columns(3)
with ca:
    A_start = st.time_input("Inizio A", value=time(9,0))
    A_end   = st.time_input("Fine A",   value=time(11,0))
with cb:
    B_start = st.time_input("Inizio B", value=time(11,0))
    B_end   = st.time_input("Fine B",   value=time(13,0))
with cc:
    C_start = st.time_input("Inizio C", value=time(13,0))
    C_end   = st.time_input("Fine C",   value=time(16,0))

# ============== DATI METEO OPEN-METEO ==============
def fetch_open_meteo(lat, lon, tzname):
    url = "https://api.open-meteo.com/v1/forecast"
    p = {"latitude":lat,"longitude":lon,"timezone":tzname,
         "hourly":"temperature_2m,dew_point_2m,precipitation,rain,snowfall,cloudcover,windspeed_10m,is_day,weathercode",
         "forecast_days":7}
    r = requests.get(url, params=p, timeout=30); r.raise_for_status(); return r.json()

def prp_type(df):
    snow = {71,73,75,77,85,86}; rain = {51,53,55,61,63,65,80,81,82}
    def f(row):
        prp = row.precipitation; rn = getattr(row,"rain",0.0); sn = getattr(row,"snowfall",0.0)
        if prp<=0 or pd.isna(prp): return "none"
        if rn>0 and sn>0: return "mixed"
        if sn>0 and rn==0: return "snow"
        if rn>0 and sn==0: return "rain"
        code = int(getattr(row,"weathercode",0))
        if code in snow: return "snow"
        if code in rain: return "rain"
        return "mixed"
    return df.apply(f, axis=1)

def build_df(js, hours, tzname):
    h = js["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])              # tempi NAIVE → niente conflitti tz
    now = pd.Timestamp.now().floor("H")
    df = df[df["time"] >= now].head(hours).reset_index(drop=True)

    out = pd.DataFrame()
    out["time"] = df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    out["T2m"] = df["temperature_2m"].astype(float)
    out["cloud"] = (df["cloudcover"].astype(float)/100).clip(0,1)
    out["wind"] = (df["windspeed_10m"].astype(float)/3.6).round(3)
    out["sunup"] = df["is_day"].astype(int)
    out["prp_mmph"] = df["precipitation"].astype(float)
    out["prp_type"] = prp_type(df[["precipitation","rain","snowfall","weathercode"]].copy())
    out["td"] = df["dew_point_2m"].astype(float)
    return out

def win_slice(res, tzname, s, e):
    t = pd.to_datetime(res["time"]).dt.tz_localize(tz.gettz(tzname), nonexistent='shift_forward', ambiguous='NaT')
    D = res.copy(); D["dt"] = t
    today = pd.Timestamp.now(tz=tz.gettz(tzname)).date()
    W = D[(D["dt"].dt.date==today) & (D["dt"].dt.time>=s) & (D["dt"].dt.time<=e)]
    return W if not W.empty else D.head(7)

# ============== WAX CATALOG (8 brand) ==============
SWIX = [("PS5 Turquoise",-18,-10),("PS6 Blue",-12,-6),("PS7 Violet",-8,-2),("PS8 Red",-4,4),("PS10 Yellow",0,10)]
TOKO = [("Blue",-30,-9),("Red",-12,-4),("Yellow",-6,0)]
VOLA = [("MX-E Blue/Violet",-25,-4),("MX-E Red",-5,0),("MX-E Yellow",-2,8)]
RODE = [("R20 Blue",-18,-8),("R30 Violet",-10,-3),("R40 Red",-5,0),("R50 Yellow",-1,10)]
HOLM = [("UltraMix Blue",-20,-8),("BetaMix Red",-14,-4),("AlphaMix Yellow",-4,0)]
MAPL = [("Universal Cold",-12,-6),("Universal Med",-7,-2),("Universal Soft",-5,1)]
SKIGO= [("Blue",-12,-6),("Violet",-8,-2),("Red",-5,1)]
START= [("SG Blue",-12,-6),("SG Purple",-8,-2),("SG Red",-3,7)]

BRANDS = [("Swix",SWIX,"#ef4444"),("Toko",TOKO,"#f59e0b"),("Vola",VOLA,"#3b82f6"),
          ("Rode",RODE,"#22c55e"),("Holmenkol",HOLM,"#06b6d4"),("Maplus",MAPL,"#a3e635"),
          ("Skigo",SKIGO,"#d946ef"),("Start",START,"#fb7185")]

def band_pick(bands, t):
    for name,tmin,tmax in bands:
        if t>=tmin and t<=tmax: return name
    return bands[-1][0] if t>bands[-1][2] else bands[0][0]

def logo_svg(text, color):
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='140' height='28'><rect width='140' height='28' rx='6' fill='{color}'/><text x='10' y='19' font-size='14' font-weight='700' fill='white'>{text}</text></svg>"
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()

# Piccoli disegni struttura (SVG)
def struct_svg(kind="fine"):
    if kind=="fine":
        patt = "<path d='M5 4 L135 4 M10 10 L140 10 M15 16 L145 16' stroke='white' stroke-width='1' opacity='.55'/>"
    elif kind=="medium":
        patt = "<path d='M5 5 L140 20 M5 10 L140 25 M5 15 L140 30' stroke='white' stroke-width='1.5' opacity='.55'/>"
    else:  # coarse
        patt = "<path d='M4 6 L136 26 M4 12 L136 32' stroke='white' stroke-width='2.2' opacity='.55'/>"
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='150' height='36'><rect width='150' height='36' rx='8' fill='rgba(255,255,255,.06)'/>{patt}</svg>"
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()

# Tuning: struttura + angoli SIDE/BASE
def tune_for(t_surf, disc):
    if t_surf <= -10:
        structure_name, struct_kind = "Fine (lineare)", "fine"
        base = 0.5
        sides = {"SL":88.5, "GS":88.0, "SG":87.5, "DH":87.5}
    elif t_surf <= -3:
        structure_name, struct_kind = "Media (universale)", "medium"
        base = 0.7
        sides = {"SL":88.0, "GS":88.0, "SG":87.5, "DH":87.0}
    else:
        structure_name, struct_kind = "Media-Grossa (lisca/diamante)", "coarse"
        base = 0.8 if t_surf <= 0.5 else 1.0
        sides = {"SL":88.0, "GS":87.5, "SG":87.0, "DH":87.0}
    return structure_name, struct_kind, sides.get(disc,88.0), base

# ============== RUN ==============
colRun1, colRun2 = st.columns([1,2])
with colRun1:
    go = st.button("Scarica previsioni", type="primary")
with colRun2:
    upl = st.file_uploader("…oppure carica CSV (time,T2m,cloud,wind,sunup,prp_mmph,prp_type,td)", type=["csv"])

def draw_plots(res):
    t = pd.to_datetime(res["time"])
    fig1 = plt.figure()
    plt.plot(t, res["T2m"], label="T2m")
    plt.plot(t, res["T_surf"], label="T_surf")
    plt.plot(t, res["T_top5"], label="T_top5")
    plt.legend(); plt.title("Temperature"); plt.xlabel("Ora"); plt.ylabel("°C")
    fig2 = plt.figure()
    plt.bar(t, res["prp_mmph"])
    plt.title("Precipitazione (mm/h)"); plt.xlabel("Ora"); plt.ylabel("mm/h")
    return fig1, fig2

def badge(win):
    tmin = float(win["T_surf"].min()); tmax = float(win["T_surf"].max())
    wet = bool(((win["prp_type"].isin(["rain","mixed"])) | (win["prp_mmph"]>0.5)).any())
    if wet or tmax>0.5:
        color, title, desc = "#ef4444", "CRITICAL", "Possibile neve bagnata/pioggia · struttura grossa"
    elif tmax>-1.0:
        color, title, desc = "#f59e0b", "WATCH", "Vicino a 0°C · cere medio-morbide"
    else:
        color, title, desc = "#22c55e", "OK", "Neve fredda/asciutta · cere dure"
    st.markdown(f"""
    <div class='card' style='border-color:{color}'>
      <div style='font-weight:800;color:{color};margin-bottom:4px'>{title}</div>
      <div style='opacity:.95'>{desc}</div>
      <div style='font-size:12px;opacity:.7;margin-top:6px'>T_surf min {tmin:.1f}°C / max {tmax:.1f}°C</div>
    </div>
    """, unsafe_allow_html=True)
    return wet

def show_brand_cards(t_med):
    cols = st.columns(4)
    for i,(name,bands,color) in enumerate(BRANDS):
        rec = band_pick(bands, t_med)
        cols[i%4].markdown(
            f"<div class='brand'><img src='{logo_svg(name.upper(), color)}'/><div><div style='font-size:.8rem;opacity:.85'>{name}</div><div style='font-weight:800'>{rec}</div></div></div>",
            unsafe_allow_html=True
        )

def run_all(src, label):
    res = compute_snow_temperature(src, dt_hours=1.0)
    st.success(f"Dati pronti per **{label}**")
    st.dataframe(res, use_container_width=True)
    f1, f2 = draw_plots(res); st.pyplot(f1); st.pyplot(f2)
    st.download_button("Scarica CSV risultato", data=res.to_csv(index=False), file_name="forecast_with_snowT.csv", mime="text/csv")

    blocks = {"A":(A_start,A_end), "B":(B_start,B_end), "C":(C_start,C_end)}
    for L,(s,e) in blocks.items():
        st.markdown(f"### Blocco {L}")
        W = win_slice(res, tzname, s, e)
        wet = badge(W)
        t_med = float(W["T_surf"].mean())
        st.markdown(f"**T_surf medio {L}: {t_med:.1f} °C**")
        show_brand_cards(t_med)

        # Tuning (struttura + lamine SIDE)
        st.markdown("**Tuning consigliato**")
        disc = st.multiselect(f"Discipline", ["SL","GS","SG","DH"], default=["SL","GS"], key=f"disc_{L}")
        rows = []
        for d in disc:
            sname, skind, side, base = tune_for(t_med, d)
            rows.append([d, sname, f"{side:.1f}°", f"{base:.1f}°", skind])
        if rows:
            dfT = pd.DataFrame(rows, columns=["Disciplina","Struttura","Lamina SIDE (°)","Lamina BASE (°)","_k"])
            st.table(dfT[["Disciplina","Struttura","Lamina SIDE (°)","Lamina BASE (°)"]])
            # disegni struttura
            st.caption("Anteprima struttura soletta")
            cc1, cc2, cc3 = st.columns(3)
            for i,kind in enumerate(sorted(set(dfT["_k"]))):
                img = struct_svg(kind)
                (cc1 if i==0 else (cc2 if i==1 else cc3)).markdown(f"<img src='{img}'>", unsafe_allow_html=True)

# Ingressi
if upl is not None:
    try:
        df_u = pd.read_csv(upl)
        run_all(df_u, SEL.get("label","Località"))
    except Exception as e:
        st.error(f"CSV non valido: {e}")

if go:
    try:
        js = fetch_open_meteo(lat, lon, tzname)
        src = build_df(js, hours, tzname)
        run_all(src, SEL.get("label","Località"))
    except Exception as e:
        st.error(f"Errore: {e}")

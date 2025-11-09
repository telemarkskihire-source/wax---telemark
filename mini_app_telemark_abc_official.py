# telemark_pro_app.py
import streamlit as st
import pandas as pd
import requests, base64, math
import matplotlib.pyplot as plt
import numpy as np
from datetime import time
from dateutil import tz

# --- tenta la ricerca tipo meteoblue con st_searchbox; se manca, usa fallback live ---
try:
    from streamlit_searchbox import st_searchbox  # pip install streamlit-searchbox
    HAVE_SEARCHBOX = True
except Exception:
    HAVE_SEARCHBOX = False

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
.sugg-item {{ padding:6px 10px; border-bottom:1px solid rgba(255,255,255,.08); cursor:pointer }}
.sugg-item:hover {{ background:rgba(255,255,255,.06) }}
</style>
""", unsafe_allow_html=True)

st.markdown("### Telemark · Pro Wax & Tune")
st.markdown("<span class='badge'>Ricerca tipo Meteoblue · Preset · Blocchi A/B/C · 8 marchi sciolina · Strutture stile Wintersteiger · Lamine (SIDE/BASE)</span>", unsafe_allow_html=True)

# ------------------------ UTILS ------------------------
def flag_emoji(country_code: str) -> str:
    try:
        cc = country_code.upper()
        return chr(127397 + ord(cc[0])) + chr(127397 + ord(cc[1]))
    except Exception:
        return "🏳️"

def nominatim_search_backend(query: str, limit=10):
    if not query or len(query) < 1:
        return []
    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": limit, "addressdetails": 1},
            headers={"User-Agent": "telemark-wax-app/1.0"},
            timeout=8
        )
        r.raise_for_status()
        out = []
        geo_map = {}
        for item in r.json():
            name = item.get("display_name", "")
            lat = float(item.get("lat", 0)); lon = float(item.get("lon", 0))
            cc = (item.get("address", {}) or {}).get("country_code", "") or ""
            label = f"{flag_emoji(cc)}  {name}"
            key = f"{label}|||{lat:.6f},{lon:.6f}"
            geo_map[key] = (lat, lon, label)
            out.append((key, lat, lon, label))
        return out, geo_map
    except Exception:
        return [], {}

def searchbox_provider(text: str):
    # adapter per st_searchbox: ritorna lista di stringhe-chiave
    res, geo = nominatim_search_backend(text, limit=10)
    st.session_state._geo_map = {k: (lat, lon, lab) for (k, lat, lon, lab) in res}
    return [k for (k, _, _, _) in res]

def svg_logo(text, color):
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='160' height='36'><rect width='160' height='36' rx='6' fill='{color}'/><text x='12' y='24' font-size='16' font-weight='700' fill='white'>{text}</text></svg>"
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")

# ------------------------ PRESET LOCALITÀ ------------------------
PRESETS = {
    "🏔️ Champoluc (Ramey)": (45.831, 7.730),
    "🏔️ Gressoney (Stafal)": (45.824, 7.827),
    "🏔️ Alagna (Pianalunga)": (45.855, 7.941),
    "⛷️ Cervinia": (45.936, 7.630),
    "⛷️ Sestriere": (44.956, 6.879),
}

# ------------------------ LOCATION (Meteoblue-like) ------------------------
st.markdown("#### 1) Scegli la località")
c_preset, c_search = st.columns([1,2])
with c_preset:
    preset = st.selectbox("Preset rapidi", list(PRESETS.keys()))
    if st.button("Usa preset"):
        plat, plon = PRESETS[preset]
        st.session_state.sel_lat, st.session_state.sel_lon = plat, plon
        st.session_state.sel_label = preset

with c_search:
    if HAVE_SEARCHBOX:
        selected = st_searchbox(
            searchbox_provider,
            key="place",
            placeholder="Digita e scegli… (es. Champoluc, Cervinia, Sestriere)",
            clear_on_submit=False,
            default=None
        )
        if selected and "|||" in selected and "_geo_map" in st.session_state:
            lat, lon, label = st.session_state._geo_map.get(selected, (None, None, None))
            if lat is not None:
                st.session_state.sel_lat, st.session_state.sel_lon = lat, lon
                st.session_state.sel_label = label
    else:
        # Fallback "live" aggiornato ad ogni carattere, senza Enter
        q = st.text_input("Cerca località (scrivi e scegli)", placeholder="Champoluc, Cervinia, Sestriere…", key="q_live")
        res, geo = nominatim_search_backend(q, limit=8)
        if res:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            for (k, lat, lon, lab) in res:
                if st.button(lab, key=f"sugg_{k}", use_container_width=True):
                    st.session_state.sel_lat, st.session_state.sel_lon = lat, lon
                    st.session_state.sel_label = lab
                    st.session_state.q_live = lab
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

# valore scelto / fallback default
lat = st.session_state.get("sel_lat", PRESETS["🏔️ Champoluc (Ramey)"][0])
lon = st.session_state.get("sel_lon", PRESETS["🏔️ Champoluc (Ramey)"][1])
label = st.session_state.get("sel_label", "🏔️ Champoluc (Ramey)")

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

# ------------------------ WAX BANDS (8 MARCHE) ------------------------
WAX = {
    "Swix": [("PS5 Turquoise", -18,-10), ("PS6 Blue",-12,-6), ("PS7 Violet",-8,-2), ("PS8 Red",-4,4), ("PS10 Yellow",0,10)],
    "Toko": [("Blue",-30,-9), ("Red",-12,-4), ("Yellow",-6,0)],
    "Vola": [("MX-E Blue/Violet",-12,-4), ("MX-E Red",-5,0), ("MX-E Warm",-2,10)],
    "Rode": [("R20 Blue",-18,-8), ("R30 Violet",-10,-3), ("R40 Red",-5,0), ("R50 Yellow",-1,10)],
    "Holmenkol": [("UltraMix Blue",-20,-8), ("BetaMix Red",-14,-4), ("AlphaMix Yellow",-4,5)],
    "Maplus": [("Universal Cold",-12,-6), ("Universal Medium",-7,-2), ("Universal Warm",-3,6)],
    "Start": [("SG Blue",-12,-6), ("SG Purple",-8,-2), ("SG Red",-3,7)],
    "Skigo": [("Paraffin Blue",-12,-6), ("Paraffin Violet",-8,-2), ("Paraffin Red",-3,2)],
}
BRAND_COLORS = {
    "Swix":"#ef4444","Toko":"#f59e0b","Vola":"#3b82f6","Rode":"#22c55e",
    "Holmenkol":"#0ea5e9","Maplus":"#f97316","Start":"#a855f7","Skigo":"#06b6d4"
}
def pick(bands, t):
    for n,tmin,tmax in bands:
        if t>=tmin and t<=tmax: return n
    return bands[-1][0] if t>bands[-1][2] else bands[0][0]

# ------------------------ STRUTTURE (stile Wintersteiger) + LAMINE ------------------------
# tipi: linear_fine, wave, diagonal_drain, chevron, cross
def draw_structure(kind: str, title: str):
    fig = plt.figure(figsize=(3.5, 2.1), dpi=180)
    ax = plt.gca()
    # base soletta
    ax.set_facecolor("#d9d9df")
    ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis('off')

    groove = "#6b7280"      # scuro
    groove2 = "#475569"     # molto scuro
    lw = 2.4

    if kind == "linear_fine":
        for x in np.arange(6, 96, 5):
            ax.plot([x, x], [5, 55], color=groove, linewidth=lw, solid_capstyle="round")
    elif kind == "wave":
        xs = np.arange(8, 98, 10)
        y = np.linspace(6, 54, 190)
        for x in xs:
            ax.plot(x + 3*np.sin((y/8.0)), y, color=groove, linewidth=lw, solid_capstyle="round")
    elif kind == "diagonal_drain":
        for x in np.arange(-20, 130, 10):
            ax.plot([x, x+50], [4, 56], color=groove2, linewidth=3.0, solid_capstyle="round")
    elif kind == "chevron":
        # V a pettine / herringbone
        for x in np.arange(0, 100, 8):
            ax.plot([x-12, x+12], [6, 30], color=groove2, linewidth=2.8)
            ax.plot([x+12, x-12], [30, 54], color=groove2, linewidth=2.8)
    elif kind == "cross":
        # incrociata: lineari + diagonali leggere
        for x in np.arange(8, 96, 8):
            ax.plot([x, x], [5, 55], color=groove, linewidth=2.2)
        for x in np.arange(-10, 120, 14):
            ax.plot([x, x+50], [6, 56], color=groove, alpha=0.55, linewidth=2.0)

    ax.set_title(title, fontsize=10, color="#0f172a", pad=4)
    st.pyplot(fig)

def tune_for(t_surf, discipline):
    # SIDE (gradi) + BASE (gradi) + struttura consigliata
    if t_surf <= -10:
        structure = ("linear_fine", "Freddo/Secco · Lineare fine")
        base = 0.5; side_map = {"SL":88.5, "GS":88.0, "SG":87.5, "DH":87.5}
    elif t_surf <= -3:
        structure = ("wave", "Universale · Onda")
        base = 0.7; side_map = {"SL":88.0, "GS":88.0, "SG":87.5, "DH":87.0}
    else:
        # caldo/umido: scarico marcato (diagonale o chevron), incrociata in neve mista
        structure = ("diagonal_drain", "Caldo/Umido · Scarico diagonale")
        base = 0.8 if t_surf <= 0.5 else 1.0
        side_map = {"SL":88.0, "GS":87.5, "SG":87.0, "DH":87.0}
    return structure, side_map.get(discipline, 88.0), base

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

        # grafici rapidi
        t = pd.to_datetime(res["time"])
        fig1 = plt.figure(); plt.plot(t,res["T2m"],label="T2m"); plt.plot(t,res["T_surf"],label="T_surf"); plt.plot(t,res["T_top5"],label="T_top5")
        plt.legend(); plt.title("Temperature"); plt.xlabel("Ora"); plt.ylabel("°C"); st.pyplot(fig1)
        fig2 = plt.figure(); plt.bar(t,res["prp_mmph"]); plt.title("Precipitazione (mm/h)"); plt.xlabel("Ora"); plt.ylabel("mm/h"); st.pyplot(fig2)
        st.download_button("Scarica CSV risultato", data=res.to_csv(index=False), file_name="forecast_with_snowT.csv", mime="text/csv")

        # blocchi A/B/C
        for L,(s,e) in {"A":(A_start,A_end),"B":(B_start,B_end),"C":(C_start,C_end)}.items():
            st.markdown(f"### Blocco {L}")
            W = window_slice(res, tzname, s, e)
            t_med = float(W["T_surf"].mean())
            st.markdown(f"**T_surf medio {L}: {t_med:.1f}°C**")

            # Wax cards + loghi per tutti i marchi
            cols = st.columns(4)
            items = list(WAX.items())
            for i, (brand, bands) in enumerate(items):
                rec = pick(bands, t_med)
                col = BRAND_COLORS.get(brand, "#10bfcf")
                block = cols[i % 4]
                block.markdown(
                    f"<div class='brand'><img src='{svg_logo(brand.upper(), col)}'/>"
                    f"<div><div style='font-size:.8rem;opacity:.85'>{brand}</div>"
                    f"<div style='font-weight:800'>{rec}</div></div></div>", unsafe_allow_html=True
                )
                if (i % 4) == 3 and i < len(items)-1:
                    cols = st.columns(4)

            # Struttura consigliata + disegno
            # Mostra anche un'alternativa “cross”/“chevron” come riferimento visivo in caldo/umido
            struct, side, base_angle = tune_for(t_med, "GS")
            kind, title = struct
            st.markdown(f"**Struttura consigliata:** {title}  ·  **Lamina SIDE:** {side:.1f}°  ·  **BASE:** {base_angle:.1f}°")
            draw_structure(kind, title)

            if kind == "diagonal_drain":
                st.caption("Alternativa per neve mista/trasformata: **Chevron (Herringbone)** o **Cross**")
                cc1, cc2 = st.columns(2)
                with cc1: draw_structure("chevron", "Chevron / Herringbone")
                with cc2: draw_structure("cross", "Cross / Incrociata")

            # Tuning per discipline
            disc = st.multiselect(f"Discipline (Blocco {L})", ["SL","GS","SG","DH"], default=["SL","GS"], key=f"disc_{L}")
            rows = []
            for d in disc:
                (kind_d, title_d), side_d, base_d = tune_for(t_med, d)
                rows.append([d, title_d, f"{side_d:.1f}°", f"{base_d:.1f}°"])
            if rows:
                st.table(pd.DataFrame(rows, columns=["Disciplina","Struttura","Lamina SIDE (°)","Lamina BASE (°)"]))

    except Exception as e:
        st.error(f"Errore: {e}")

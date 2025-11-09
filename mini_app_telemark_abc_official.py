# telemark_pro_app.py
import streamlit as st
import pandas as pd
import requests, base64, math
import matplotlib.pyplot as plt
from datetime import time
from dateutil import tz

# ========== TRY SEARCHBOX (Meteoblue-like). FALLBACK IF MISSING ==========
_HAVE_SB = True
try:
    from streamlit_searchbox import st_searchbox  # pip install streamlit-searchbox
except Exception:
    _HAVE_SB = False

# ============================= PAGE & THEME ==============================
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
.sugg {{ font-size:.92rem; }}
</style>
""", unsafe_allow_html=True)

st.markdown("### Telemark · Pro Wax & Tune")
st.markdown("<span class='badge'>Ricerca stile Meteoblue · Preset veloci · Blocchi A/B/C · Sciolina + Struttura + Angoli (SIDE)</span>", unsafe_allow_html=True)

# ================================ UTILS =================================
def flag_emoji(country_code: str) -> str:
    try:
        cc = country_code.upper()
        return chr(127397 + ord(cc[0])) + chr(127397 + ord(cc[1]))
    except Exception:
        return "🏳️"

def logo_svg(text, color):
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='160' height='36'><rect width='160' height='36' rx='6' fill='{color}'/><text x='12' y='24' font-size='16' font-weight='700' fill='white'>{text}</text></svg>"
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")

# ========================= LOCATION SEARCH ==============================
# Live search at each keystroke (like Meteoblue)
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

st.markdown("#### 1) Cerca località")
col_search, col_presets = st.columns([2.2, 1.8])

with col_search:
    if _HAVE_SB:
        selected = st_searchbox(
            nominatim_search,
            key="place",
            placeholder="Digita e scegli… (es. Champoluc, Cervinia, Sestriere)",
            clear_on_submit=False,  # mantiene il testo come su meteoblue
            default=None
        )
    else:
        # Fallback: aggiorna a ogni carattere con dropdown simulato
        q = st.text_input("Ricerca (autocompletamento live)", placeholder="Es. Champoluc, Cervinia, Sestriere")
        opts = nominatim_search(q)
        selected = st.selectbox("Suggerimenti", opts, index=0 if opts else None, placeholder="—", label_visibility="collapsed")

with col_presets:
    st.caption("Preset rapidi")
    preset_clicked = None
    c1, c2 = st.columns(2)
    if c1.button("🏔️ Champoluc"):
        preset_clicked = (45.831, 7.730, "🇮🇹 Champoluc, Aosta Valley, Italy")
    if c2.button("🏔️ Gressoney"):
        preset_clicked = (45.824, 7.827, "🇮🇹 Gressoney-La-Trinité (Stafal), Italy")
    c3, c4 = st.columns(2)
    if c3.button("⛷️ Alagna"):
        preset_clicked = (45.855, 7.941, "🇮🇹 Alagna Valsesia (Pianalunga), Italy")
    if c4.button("🗻 Cervinia"):
        preset_clicked = (45.936, 7.630, "🇮🇹 Breuil-Cervinia, Italy")
    c5, c6 = st.columns(2)
    if c5.button("🎿 Sestriere"):
        preset_clicked = (44.955, 6.878, "🇮🇹 Sestriere, Italy")
    if c6.button("⛰️ Courmayeur"):
        preset_clicked = (45.792, 6.972, "🇮🇹 Courmayeur, Italy")

# Decode user selection OR preset
if preset_clicked:
    lat, lon, label = preset_clicked
    st.session_state.sel_lat, st.session_state.sel_lon, st.session_state.sel_label = lat, lon, label
elif selected and "|||" in str(selected) and "_geo_map" in st.session_state:
    lat, lon, label = st.session_state._geo_map.get(selected, (45.831, 7.730, "🇮🇹 Champoluc (Ramey), Italy"))
    st.session_state.sel_lat, st.session_state.sel_lon, st.session_state.sel_label = lat, lon, label

# Defaults
lat = st.session_state.get("sel_lat", 45.831)
lon = st.session_state.get("sel_lon", 7.730)
label = st.session_state.get("sel_label", "🇮🇹 Champoluc (Ramey), Italy")

# Basic controls (no lat/lon fields)
coltz, colh = st.columns([1,2])
with coltz:
    tzname = st.selectbox("Timezone", ["Europe/Rome", "UTC"], index=0)
with colh:
    hours = st.slider("Ore previsione", 12, 168, 72, 12)

# =========================== WINDOWS A/B/C ===============================
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

# ============================ DATA PIPELINE ==============================
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
        prp = row.precipitation; rain = getattr(row, "rain", 0.0); snow = getattr(row, "snowfall", 0.0)
        if prp <= 0 or pd.isna(prp): return "none"
        if rain > 0 and snow > 0: return "mixed"
        if snow > 0 and rain == 0: return "snow"
        if rain > 0 and snow == 0: return "rain"
        code = int(getattr(row, "weathercode", 0)) if pd.notna(getattr(row, "weathercode", None)) else 0
        if code in snow_codes: return "snow"
        if code in rain_codes: return "rain"
        return "mixed"
    return df.apply(f, axis=1)

def build_df(js, hours):
    h = js["hourly"]; df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"])   # naive timestamps
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

# ============================ WAX BANDS ===============================
# Ripristino marchi completi: Swix, Toko, Vola, Rode, Holmenkol, Maplus, Start, Skigo
SWIX = [("PS5 Turquoise", -18,-10), ("PS6 Blue",-12,-6), ("PS7 Violet",-8,-2), ("PS8 Red",-4,4), ("PS10 Yellow",0,10)]
TOKO = [("Blue",-30,-9), ("Red",-12,-4), ("Yellow",-6,0)]
VOLA = [("MX-E Blue/Violet",-25,-8), ("MX-E Violet (Mid)",-12,-4), ("MX-E Red (Warm)",-5,0), ("MX-E Yellow (Very Warm)",-2,6)]
RODE = [("R20 Blue",-18,-8), ("R30 Violet",-10,-3), ("R40 Red",-5,0), ("R50 Yellow",-1,10)]
HOLM = [("Ultra/Alpha Mix Blue",-20,-8), ("BetaMix Red",-14,-4), ("AlphaMix Yellow",-4,5)]
MAPL = [("Universal Cold",-12,-6), ("Universal Medium",-7,-2), ("Universal Warm",-3,6)]
START= [("SG Blue",-12,-6), ("SG Purple",-8,-2), ("SG Red",-3,7)]
SKIGO= [("Paraffin Blue",-12,-6), ("Paraffin Violet",-8,-2), ("Paraffin Red",-3,2)]

def pick(bands, t):
    for n,tmin,tmax in bands:
        if t>=tmin and t<=tmax: return n
    return bands[-1][0] if t>bands[-1][2] else bands[0][0]

BRANDS = {
    "Swix":      ("#ef4444", SWIX),
    "Toko":      ("#f59e0b", TOKO),
    "Vola":      ("#3b82f6", VOLA),
    "Rode":      ("#22c55e", RODE),
    "Holmenkol": ("#6366f1", HOLM),
    "Maplus":    ("#06b6d4", MAPL),
    "Start":     ("#fb7185", START),
    "Skigo":     ("#a3e635", SKIGO),
}

# =================== STRUCTURE & EDGES (improved draw) ===================
def tune_for(t_surf, discipline):
    # SIDE (gradi) + BASE (gradi) e struttura consigliata
    if t_surf <= -10:
        structure = ("linear_fine", "Freddo/Secco · Lineare fine")
        base = 0.5; side_map = {"SL":88.5, "GS":88.0, "SG":87.5, "DH":87.5}
    elif t_surf <= -3:
        structure = ("wave_mid", "Universale · Onda convessa")
        base = 0.7; side_map = {"SL":88.0, "GS":88.0, "SG":87.5, "DH":87.0}
    else:
        structure = ("drain_diag", "Caldo/Umido · Scarico diagonale")
        base = 0.8 if t_surf <= 0.5 else 1.0
        side_map = {"SL":88.0, "GS":87.5, "SG":87.0, "DH":87.0}
    return structure, side_map.get(discipline, 88.0), base

def draw_structure(kind: str, title: str):
    """Preview più chiara stile Wintersteiger: base grigia, incisioni scure con profondità e bordo."""
    import numpy as np
    fig = plt.figure(figsize=(3.6, 2.2), dpi=160)
    ax = plt.gca()
    ax.set_facecolor("#d9d9de")  # soletta
    ax.set_xlim(0, 120); ax.set_ylim(0, 60); ax.axis('off')
    # bordo soletta
    ax.add_patch(plt.Rectangle((2,2), 116, 56, fill=False, linewidth=1.2, edgecolor="#555"))

    if kind == "linear_fine":
        # molte righe sottili verticali
        for x in range(8, 112, 5):
            ax.plot([x, x], [6, 54], color="#444", linewidth=1.8, solid_capstyle="round")
    elif kind == "wave_mid":
        # colonne con andamento sinusoidale (convesso) più grosso
        xs = np.linspace(10, 110, 9)
        basey = np.linspace(6, 54, 120)
        for x in xs:
            yy = 30 + 20*np.sin(np.linspace(-math.pi, math.pi, len(basey)))
            ax.plot(np.full_like(yy, x), yy, color="#3a3a3a", linewidth=2.6, solid_capstyle="round")
    elif kind == "drain_diag":
        # scarico diagonale (taglio) più marcato
        for x in range(-20, 140, 10):
            ax.plot([x, x+60], [6, 54], color="#2f2f2f", linewidth=3.2, solid_capstyle="round")
    else:
        # fallback lineare
        for x in range(8, 112, 6):
            ax.plot([x, x], [6, 54], color="#444", linewidth=2.0, solid_capstyle="round")

    ax.set_title(title, fontsize=10, pad=4, color="#111")
    st.pyplot(fig)

# =============================== RUN ====================================
st.markdown("#### 3) Scarica dati meteo & calcola")
go = st.button("Scarica previsioni per la località selezionata", type="primary")

if go:
    try:
        js = fetch_open_meteo(lat, lon, tzname)
        src = build_df(js, hours)
        res = compute_snow_temperature(src, dt_hours=1.0)
        st.success(f"Dati per **{label}** caricati.")
        st.dataframe(res, use_container_width=True)

        # Grafici rapidi
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

        # Blocchi A / B / C
        for L,(s,e) in {"A":(A_start,A_end),"B":(B_start,B_end),"C":(C_start,C_end)}.items():
            st.markdown(f"### Blocco {L}")
            W = window_slice(res, tzname, s, e)
            t_med = float(W["T_surf"].mean())
            st.markdown(f"**T_surf medio {L}: {t_med:.1f}°C**")

            # Wax cards + loghi (TUTTI i marchi richiesti)
            cols = st.columns(4)
            items = list(BRANDS.items())
            # riga 1
            for i,(brand,(col,bands)) in enumerate(items[:4]):
                rec = pick(bands, t_med)
                cols[i].markdown(
                    f"<div class='brand'><img src='{logo_svg(brand.upper(), col)}'/>"
                    f"<div><div style='font-size:.8rem;opacity:.85'>{brand}</div>"
                    f"<div style='font-weight:800'>{rec}</div></div></div>", unsafe_allow_html=True
                )
            # riga 2
            cols2 = st.columns(4)
            for i,(brand,(col,bands)) in enumerate(items[4:8]):
                rec = pick(bands, t_med)
                cols2[i].markdown(
                    f"<div class='brand'><img src='{logo_svg(brand.upper(), col)}'/>"
                    f"<div><div style='font-size:.8rem;opacity:.85'>{brand}</div>"
                    f"<div style='font-weight:800'>{rec}</div></div></div>", unsafe_allow_html=True
                )

            # Struttura + angoli (SIDE + BASE) con disegno “alla Wintersteiger”
            (kind, title), side, base = tune_for(t_med, "GS")  # riferimento
            st.markdown(f"**Struttura consigliata:** {title}  ·  **Lamina SIDE:** {side:.1f}°  ·  **BASE:** {base:.1f}°")
            draw_structure(kind, title)

            # Tabelle per discipline
            disc = st.multiselect(f"Discipline (Blocco {L})", ["SL","GS","SG","DH"], default=["SL","GS"], key=f"disc_{L}")
            rows = []
            for d in disc:
                (kind_d, title_d), side_d, base_d = tune_for(t_med, d)
                rows.append([d, title_d, f"{side_d:.1f}°", f"{base_d:.1f}°"])
            if rows:
                st.table(pd.DataFrame(rows, columns=["Disciplina","Struttura","Lamina SIDE (°)","Lamina BASE (°)"]))

    except Exception as e:
        st.error(f"Errore: {e}")

# Nota: per il live dropdown “stile Meteoblue” serve la libreria:
#   pip install streamlit-searchbox
# Se non presente, parte il fallback automatico.

# telemark_pro_app.py
# Telemark · Pro Wax & Tune — mobile-first Streamlit app

import streamlit as st
import pandas as pd
import requests
import base64
import matplotlib.pyplot as plt
from datetime import time
from dateutil import tz

# =============== MODEL: snow temperature (surface & top 5mm) =================
def compute_snow_temperature(df: pd.DataFrame, dt_hours: float = 1.0) -> pd.DataFrame:
    import math

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])

    req = ["T2m", "cloud", "wind", "sunup", "prp_mmph", "prp_type"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
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
    clear = (1.0 - df["cloud"]).clip(0, 1)
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


# ============================ UI THEME =======================================
PRIMARY = "#10bfcf"
BG = "#0f172a"
CARD = "#111827"
TEXT = "#e5e7eb"

st.set_page_config(page_title="Telemark · Pro Wax & Tune", page_icon="❄️", layout="wide")
st.markdown(
    f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
  background: linear-gradient(180deg, {BG} 0%, #111827 100%);
}}
.block-container {{ padding-top: 1rem; }}
h1,h2,h3,h4,h5,p,span,div {{ color:{TEXT}; }}
.badge {{ border:1px solid rgba(255,255,255,.15); padding:6px 10px; border-radius:999px; font-size:.78rem; opacity:.85; }}
.card {{ background:{CARD}; border:1px solid rgba(255,255,255,.08); border-radius:16px; padding:14px; box-shadow: 0 8px 20px rgba(0,0,0,.25);}}
.brand {{ display:flex; align-items:center; gap:8px; padding:8px 10px; border-radius:12px;
          background:rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.08); }}
.brand img {{ height:22px; }}
.btn-primary button {{ width:100%; background:{PRIMARY} !important; color:#002b30 !important; border:none; font-weight:700; border-radius:12px; }}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown("### Telemark · Pro Wax & Tune")
st.markdown(
    "<span class='badge'>Autocomplete stile Meteo · Geolocalizzazione · Blocchi A/B/C · Sciolina + Struttura + Angoli</span>",
    unsafe_allow_html=True,
)

# ====================== LOCATION: autocomplete + geolocate ====================
def geocode_autocomplete(q: str, limit: int = 8):
    """Nominatim autocomplete (no API key)."""
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
        js = r.json()
        out = []
        for it in js:
            out.append(
                {
                    "label": it.get("display_name", ""),
                    "lat": float(it.get("lat", 0)),
                    "lon": float(it.get("lon", 0)),
                }
            )
        return out
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


# Search row
c_search, c_geo = st.columns([3, 1])
with c_search:
    q = st.text_input("Cerca località (digita e scegli)", placeholder="Es. Champoluc, Cervinia, Sestriere…")
    suggestions = geocode_autocomplete(q, limit=12)
    labels = [s["label"] for s in suggestions] or ["— Nessun risultato —"]
    sel = st.selectbox("Risultati", labels, index=0, disabled=(labels[0] == "— Nessun risultato —"))

with c_geo:
    use_geo = st.button("📍 Usa la mia posizione")
    geo_lat, geo_lon, geo_label = None, None, ""
    if use_geo:
        geo_lat, geo_lon, geo_label = ip_geolocate()
        if geo_lat is None:
            st.error("Geolocalizzazione non disponibile.")
        else:
            st.success("Posizione impostata.")

# Resolve chosen coordinates (no fragile session_state writes)
if use_geo and geo_lat is not None:
    chosen_lat, chosen_lon, chosen_label = geo_lat, geo_lon, (geo_label or "La tua posizione")
elif suggestions and sel != "— Nessun risultato —":
    hit = suggestions[labels.index(sel)]
    chosen_lat, chosen_lon, chosen_label = hit["lat"], hit["lon"], hit["label"]
else:
    chosen_lat, chosen_lon, chosen_label = 45.831, 7.730, "Champoluc (Ramey)"

# Show coords and controls
c_lat, c_lon, c_tz, c_h = st.columns([1, 1, 1.6, 1.4])
with c_lat:
    lat = st.number_input("Lat", value=float(chosen_lat), format="%.6f")
with c_lon:
    lon = st.number_input("Lon", value=float(chosen_lon), format="%.6f")
with c_tz:
    tzname = st.selectbox("Timezone", ["Europe/Rome", "UTC"], index=0)
with c_h:
    hours = st.slider("Ore previsione", 12, 168, 72, 12)

# ============================= A/B/C blocks ==================================
st.markdown("#### Finestre A · B · C (oggi)")
b1, b2, b3 = st.columns(3)
with b1:
    A_start = st.time_input("Inizio A", value=time(9, 0), key="A_s")
    A_end = st.time_input("Fine A", value=time(11, 0), key="A_e")
with b2:
    B_start = st.time_input("Inizio B", value=time(11, 0), key="B_s")
    B_end = st.time_input("Fine B", value=time(13, 0), key="B_e")
with b3:
    C_start = st.time_input("Inizio C", value=time(13, 0), key="C_s")
    C_end = st.time_input("Fine C", value=time(16, 0), key="C_e")

# =============================== Fetch weather ===============================
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
        if prp <= 0 or pd.isna(prp):
            return "none"
        if rain > 0 and snow > 0:
            return "mixed"
        if snow > 0 and rain == 0:
            return "snow"
        if rain > 0 and snow == 0:
            return "rain"
        code = int(getattr(row, "weathercode", 0)) if pd.notna(getattr(row, "weathercode", None)) else 0
        if code in snow_codes:
            return "snow"
        if code in rain_codes:
            return "rain"
        return "mixed"

    return df.apply(f, axis=1)


def build_df(js, hours, tzname):
    # keep times NAIVE to avoid tz-aware comparison bugs
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


def window_slice(res, tzname, s, e):
    t = pd.to_datetime(res["time"]).dt.tz_localize(tz.gettz(tzname), nonexistent="shift_forward", ambiguous="NaT")
    D = res.copy()
    D["dt"] = t
    today = pd.Timestamp.now(tz=tz.gettz(tzname)).date()
    win = D[(D["dt"].dt.date == today) & (D["dt"].dt.time >= s) & (D["dt"].dt.time <= e)]
    return win if not win.empty else D.head(7)


# ====================== WAX tables (non-fluoro baseline) =====================
SWIX = [("PS5 Turquoise", -18, -10), ("PS6 Blue", -12, -6), ("PS7 Violet", -8, -2), ("PS8 Red", -4, 4), ("PS10 Yellow", 0, 10)]
TOKO = [("Blue", -30, -9), ("Red", -12, -4), ("Yellow", -6, 0)]
VOLA = [("MX-E Violet/Blue", -12, -4), ("MX-E Red", -5, 0), ("MX-E Warm", -2, 10)]
RODE = [("R20 Blue", -18, -8), ("R30 Violet", -10, -3), ("R40 Red", -5, 0), ("R50 Yellow", -1, 10)]


def pick(bands, t):
    for n, tmin, tmax in bands:
        if t >= tmin and t <= tmax:
            return n
    return bands[-1][0] if t > bands[-1][2] else bands[0][0]


# =================== Tuning: structure + edge angles (SIDE/BASE) =============
def tune_for(t_surf: float, discipline: str):
    """
    Returns: (structure text, side angle in degrees like 88.0, base angle in degrees)
    """
    if t_surf <= -10:
        structure = "Fine (lineare/pitone fine)"
        base = 0.5
        side_map = {"SL": 88.5, "GS": 88.0, "SG": 87.5, "DH": 87.5}
    elif t_surf <= -3:
        structure = "Media (universale)"
        base = 0.7
        side_map = {"SL": 88.0, "GS": 88.0, "SG": 87.5, "DH": 87.0}
    else:
        structure = "Media–Grossa (lisca/diamante)"
        base = 0.8 if t_surf <= 0.5 else 1.0
        side_map = {"SL": 88.0, "GS": 87.5, "SG": 87.0, "DH": 87.0}
    return structure, side_map.get(discipline, 88.0), base


# ============================== Brand logos ==================================
def svg_data_uri(text, color):
    svg = f"<svg xmlns='http://www.w3.org/2000/svg' width='200' height='36'><rect width='200' height='36' rx='6' fill='{color}'/><text x='12' y='24' font-size='16' font-weight='700' fill='white'>{text}</text></svg>"
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("utf-8")


brand_svg = {
    "Swix": svg_data_uri("SWIX", "#ef4444"),
    "Toko": svg_data_uri("TOKO", "#f59e0b"),
    "Vola": svg_data_uri("VOLA", "#3b82f6"),
    "Rode": svg_data_uri("RODE", "#22c55e"),
}


def wax_cards(t_med: float):
    cols = st.columns(4)
    for i, (brand, bands) in enumerate([("Swix", SWIX), ("Toko", TOKO), ("Vola", VOLA), ("Rode", RODE)]):
        rec = pick(bands, t_med)
        cols[i].markdown(
            f"<div class='brand'><img src='{brand_svg[brand]}'/><div>"
            f"<div style='font-size:.8rem;opacity:.85'>{brand}</div>"
            f"<div style='font-weight:800'>{rec}</div></div></div>",
            unsafe_allow_html=True,
        )


def badge(win: pd.DataFrame):
    tmin = float(win["T_surf"].min())
    tmax = float(win["T_surf"].max())
    wet = bool(((win["prp_type"].isin(["rain", "mixed"])) | (win["prp_mmph"] > 0.5)).any())

    if wet or tmax > 0.5:
        color, title, desc = "#ef4444", "CRITICAL", "Possibile neve bagnata / pioggia — struttura grossa"
    elif tmax > -1.0:
        color, title, desc = "#f59e0b", "WATCH", "Vicino a 0 °C — cere medio-morbide"
    else:
        color, title, desc = "#22c55e", "OK", "Neve fredda/asciutta — cere dure"

    st.markdown(
        f"""
    <div class='card' style='border-color:{color}'>
      <div style='font-weight:800;color:{color};margin-bottom:4px'>{title}</div>
      <div style='opacity:.95'>{desc}</div>
      <div style='font-size:12px;opacity:.7;margin-top:6px'>
        T_surf min {tmin:.1f}°C / max {tmax:.1f}°C
      </div>
    </div>""",
        unsafe_allow_html=True,
    )
    return wet


def make_plots(res: pd.DataFrame):
    t = pd.to_datetime(res["time"])

    fig1 = plt.figure()
    plt.plot(t, res["T2m"], label="T2m")
    plt.plot(t, res["T_surf"], label="T_surf")
    plt.plot(t, res["T_top5"], label="T_top5")
    plt.legend()
    plt.title("Temperature vs tempo")
    plt.xlabel("Ora")
    plt.ylabel("°C")

    fig2 = plt.figure()
    plt.bar(t, res["prp_mmph"])
    plt.title("Precipitazione (mm/h)")
    plt.xlabel("Ora")
    plt.ylabel("mm/h")
    return fig1, fig2


# =============================== Run actions =================================
c_run1, c_run2 = st.columns([1, 2])
with c_run1:
    go = st.button("Scarica previsioni per la località", type="primary")
with c_run2:
    upl = st.file_uploader(
        "…oppure carica CSV (schema: time,T2m,cloud,wind,sunup,prp_mmph,prp_type,td)", type=["csv"]
    )


def run_all(src_df: pd.DataFrame, place_label: str):
    res = compute_snow_temperature(src_df, dt_hours=1.0)
    st.success(f"Previsioni per **{place_label}** pronte.")
    st.dataframe(res, use_container_width=True)
    f1, f2 = make_plots(res)
    st.pyplot(f1)
    st.pyplot(f2)
    st.download_button(
        "Scarica CSV risultato", data=res.to_csv(index=False), file_name="forecast_with_snowT.csv", mime="text/csv"
    )

    # Blocchi A/B/C
    for L, (s, e) in {"A": (A_start, A_end), "B": (B_start, B_end), "C": (C_start, C_end)}.items():
        st.markdown(f"### Blocco {L}")
        W = window_slice(res, tzname, s, e)
        wet = badge(W)
        t_med = float(W["T_surf"].mean())
        st.markdown(f"**T_surf medio {L}: {t_med:.1f} °C**")
        wax_cards(t_med)

        # Tuning (discipline -> struttura + angoli)
        disc = st.multiselect(
            f"Discipline (Blocco {L})", ["SL", "GS", "SG", "DH"], default=["SL", "GS"], key=f"disc_{L}"
        )
        rows = []
        for d in disc:
            structure, side_deg, base_deg = tune_for(t_med, d)
            rows.append([d, structure, f"{side_deg:.1f}°", f"{base_deg:.1f}°"])
        if rows:
            st.table(pd.DataFrame(rows, columns=["Disciplina", "Struttura", "Lamina SIDE (°)", "Lamina BASE (°)"]))


# File CSV caricato?
if upl is not None:
    try:
        df_u = pd.read_csv(upl)
        run_all(df_u, chosen_label)
    except Exception as e:
        st.error(f"CSV non valido: {e}")

# Fetch da Open-Meteo?
if go:
    try:
        js = fetch_open_meteo(lat, lon, tzname)
        df_src = build_df(js, hours, tzname)
        run_all(df_src, chosen_label)
    except Exception as e:
        st.error(f"Errore: {e}")

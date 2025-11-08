import streamlit as st
import pandas as pd, json
from datetime import time, datetime
from dateutil import tz
import matplotlib.pyplot as plt
from io import BytesIO
import requests

from core_snowtemp import compute_snow_temperature

st.set_page_config(page_title="Telemark · Snow Temp (A/B/C) + Official Wax Charts", page_icon="❄️", layout="wide")

@st.cache_data
def load_wax_catalog():
    with open("wax_catalog_official.json","r",encoding="utf-8") as f:
        return json.load(f)

CAT = load_wax_catalog()

def pick_band(items, t):
    for it in items:
        if t >= it["tmin"] and t <= it["tmax"]:
            return it["name"]
    # fallback
    items_sorted = sorted(items, key=lambda x:x["tmin"])
    return items_sorted[0]["name"] if t < items_sorted[0]["tmin"] else items_sorted[-1]["name"]

def suggest_all_brands(t_surf):
    out = {}
    for brand, spec in CAT.items():
        out[brand] = pick_band(spec["items"], t_surf)
    return out

# Sidebar
with st.sidebar:
    st.subheader("Località & Dati")
    spot = st.selectbox("Spot", ["Champoluc (Ramey)","Gressoney (Stafal)","Alagna (Pianalunga)","Custom"])
    if spot == "Champoluc (Ramey)":
        lat, lon = 45.831, 7.730
    elif spot == "Gressoney (Stafal)":
        lat, lon = 45.824, 7.827
    elif spot == "Alagna (Pianalunga)":
        lat, lon = 45.855, 7.941
    else:
        lat = st.number_input("Lat", value=45.831, format="%.6f")
        lon = st.number_input("Lon", value=7.730, format="%.6f")
    tzname = st.text_input("Timezone", value="Europe/Rome")
    hours = st.slider("Ore previsione", 12, 168, 72, 12)
    btn = st.button("Scarica Open-Meteo", type="primary")
    st.divider()
    upl = st.file_uploader("CSV alternativo", type=["csv"])

# Windows A/B/C
st.markdown("### Finestre orarie A · B · C (oggi)")
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

def fetch(lat, lon, tzname):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,"longitude": lon,"timezone": tzname,
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
    h = js["hourly"]; df = pd.DataFrame(h); df["time"] = pd.to_datetime(df["time"])
    now = pd.Timestamp.now(tz=tz.gettz(js.get("timezone", tzname)))
    df = df[df["time"] >= now.floor("H")].head(hours).reset_index(drop=True)
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

def window_slice(res, tzname, s, e):
    t = pd.to_datetime(res["time"]).dt.tz_localize(tz.gettz(tzname), nonexistent='shift_forward', ambiguous='NaT')
    D = res.copy(); D["dt"] = t
    today = pd.Timestamp.now(tz=tz.gettz(tzname)).date()
    win = D[(D["dt"].dt.date==today) & (D["dt"].dt.time>=s) & (D["dt"].dt.time<=e)]
    return win if not win.empty else D.head(7)

def plots(res):
    fig1 = plt.figure(); t = pd.to_datetime(res["time"])
    plt.plot(t,res["T2m"],label="T2m"); plt.plot(t,res["T_surf"],label="T_surf"); plt.plot(t,res["T_top5"],label="T_top5")
    plt.legend(); plt.title("Temperature"); plt.xlabel("Ora"); plt.ylabel("°C")
    fig2 = plt.figure(); plt.bar(t,res["prp_mmph"]); plt.title("Precipitazione (mm/h)"); plt.xlabel("Ora"); plt.ylabel("mm/h")
    return fig1,fig2

def run_all(src_df, spot_name):
    res = compute_snow_temperature(src_df, dt_hours=1.0)
    st.dataframe(res, use_container_width=True)
    f1,f2 = plots(res); st.pyplot(f1); st.pyplot(f2)

    for label, (s,e) in {"A":(A_start,A_end), "B":(B_start,B_end), "C":(C_start,C_end)}.items():
        win = window_slice(res, tzname, s, e)
        t_med = float(win["T_surf"].mean())
        sug = suggest_all_brands(t_med)
        st.markdown(f"#### Blocco {label} – T_surf medio **{t_med:.1f} °C**")
        cols = st.columns(4)
        bnames = list(sug.keys())
        # show first 8 as metrics across two rows
        for i,k in enumerate(bnames[:4]): cols[i].metric(k.split('_')[0], sug[k])
        cols2 = st.columns(4)
        for i,k in enumerate(bnames[4:8]): cols2[i].metric(k.split('_')[0], sug[k])
    # Cite sources section
    with st.expander("Fonti ufficiali (schede prodotto)"):
        for brand, spec in CAT.items():
            st.markdown(f"**{brand}** — {spec['source']}")
            for it in spec["items"]:
                st.markdown(f"- {it['name']}: {it['tmin']}…{it['tmax']} °C · {it['src']}")

# Input
if upl is not None:
    import pandas as pd
    try:
        df_upl = pd.read_csv(upl); run_all(df_upl, spot)
    except Exception as e:
        st.error(f"CSV non valido: {e}")
elif btn:
    try:
        js = fetch(lat, lon, tzname); df = build_df(js, hours, tzname); run_all(df, spot)
    except Exception as e:
        st.error(f"Errore: {e}")
else:
    st.info("Carica un CSV o scarica le previsioni per iniziare.")

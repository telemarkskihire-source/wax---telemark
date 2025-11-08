# streamlit_app.py
# Telemark · Snow Temps + Wax (mobile-first, A/B/C, UI moderna)
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import time
from dateutil import tz

# ---------------------- PAGE SETUP ----------------------
st.set_page_config(page_title="Telemark · Snow Temps + Wax", page_icon="❄️", layout="wide")

# ---------------------- CSS (look & feel) ----------------------
PRIMARY = "#10bfcf"      # Telemark turquoise
BG = "#0f172a"           # slate-900
CARD = "#111827"         # slightly lighter
TEXT = "#e5e7eb"

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
  background: linear-gradient(180deg, {BG} 0%, #111827 100%);
}}
.block-container {{ padding-top: 1rem; }}
.hero {{ display:flex;gap:14px;align-items:center;margin-bottom:6px; }}
.hero h1 {{ margin:0;font-size:1.6rem;color:{TEXT};letter-spacing:0.2px; }}
.badge {{
  border:1px solid rgba(255,255,255,.15);
  color:{TEXT};opacity:0.9;padding:4px 10px;border-radius:999px;font-size:.78rem;
  background: rgba(255,255,255,.03);
}}
.card {{
  background:{CARD}; border:1px solid rgba(255,255,255,.08);
  border-radius:16px; padding:14px; box-shadow: 0 8px 20px rgba(0,0,0,.25);
}}
.kpi {{
  display:flex; flex-direction:column; gap:2px;
  background:rgba(16,191,207,.06); border:1px dashed rgba(16,191,207,.45);
  padding:10px 12px; border-radius:12px;
}}
.kpi .label {{ font-size:.78rem; color:#93c5fd; }}
.kpi .value {{ font-size:1rem; font-weight:700; color:{TEXT}; }}
.tag {{
  display:inline-block; margin-right:6px; margin-top:6px;
  padding:3px 8px; border-radius:999px; font-size:.72rem;
  border:1px solid rgba(255,255,255,.12); color:{TEXT}; opacity:.9;
}}
.section-title {{ color:{TEXT}; opacity:.95; margin:8px 0 4px 2px; }}
hr {{ border-color: rgba(255,255,255,.1) }}
</style>
""", unsafe_allow_html=True)

# ---------------------- CORE MODEL ----------------------
def compute_snow_temperature(df: pd.DataFrame, dt_hours: float = 1.0) -> pd.DataFrame:
    """
    Stima T_surf (superficie neve) e T_top5 (0–5 cm) da previsioni orarie.
    Richiede colonne: time, T2m, cloud(0..1), wind(m/s), sunup(0/1), prp_mmph, prp_type, td (opz.).
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])

    req = ["T2m", "cloud", "wind", "sunup", "prp_mmph", "prp_type"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Manca la colonna: {c}")
    if "td" not in df.columns:
        df["td"] = float("nan")

    df = df.sort_values("time").reset_index(drop=True)
    rain = df["prp_type"].str.lower().isin(["rain", "mixed"])
    snow = df["prp_type"].str.lower().eq("snow")
    sunup = df["sunup"].astype(int) == 1

    tw = (df["T2m"] + df["td"]) / 2.0
    wet = (
        rain |
        (df["T2m"] > 0) |
        (sunup & (df["cloud"] < 0.3) & (df["T2m"] >= -# streamlit_app.py
# Telemark · Snow Temps + Wax (mobile-first, A/B/C, UI moderna)
import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import time
from dateutil import tz

# ---------------------- PAGE SETUP ----------------------
st.set_page_config(page_title="Telemark · Snow Temps + Wax", page_icon="❄️", layout="wide")

# ---------------------- CSS (look & feel) ----------------------
PRIMARY = "#10bfcf"      # Telemark turquoise
BG = "#0f172a"           # slate-900
CARD = "#111827"         # slightly lighter
TEXT = "#e5e7eb"

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
  background: linear-gradient(180deg, {BG} 0%, #111827 100%);
}}
.block-container {{ padding-top: 1rem; }}
.hero {{ display:flex;gap:14px;align-items:center;margin-bottom:6px; }}
.hero h1 {{ margin:0;font-size:1.6rem;color:{TEXT};letter-spacing:0.2px; }}
.badge {{
  border:1px solid rgba(255,255,255,.15);
  color:{TEXT};opacity:0.9;padding:4px 10px;border-radius:999px;font-size:.78rem;
  background: rgba(255,255,255,.03);
}}
.card {{
  background:{CARD}; border:1px solid rgba(255,255,255,.08);
  border-radius:16px; padding:14px; box-shadow: 0 8px 20px rgba(0,0,0,.25);
}}
.kpi {{
  display:flex; flex-direction:column; gap:2px;
  background:rgba(16,191,207,.06); border:1px dashed rgba(16,191,207,.45);
  padding:10px 12px; border-radius:12px;
}}
.kpi .label {{ font-size:.78rem; color:#93c5fd; }}
.kpi .value {{ font-size:1rem; font-weight:700; color:{TEXT}; }}
.tag {{
  display:inline-block; margin-right:6px; margin-top:6px;
  padding:3px 8px; border-radius:999px; font-size:.72rem;
  border:1px solid rgba(255,255,255,.12); color:{TEXT}; opacity:.9;
}}
.section-title {{ color:{TEXT}; opacity:.95; margin:8px 0 4px 2px; }}
hr {{ border-color: rgba(255,255,255,.1) }}
</style>
""", unsafe_allow_html=True)

# ---------------------- CORE MODEL ----------------------
def compute_snow_temperature(df: pd.DataFrame, dt_hours: float = 1.0) -> pd.DataFrame:
    """
    Stima T_surf (superficie neve) e T_top5 (0–5 cm) da previsioni orarie.
    Richiede colonne: time, T2m, cloud(0..1), wind(m/s), sunup(0/1), prp_mmph, prp_type, td (opz.).
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])

    req = ["T2m", "cloud", "wind", "sunup", "prp_mmph", "prp_type"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Manca la colonna: {c}")
    if "td" not in df.columns:
        df["td"] = float("nan")

    df = df.sort_values("time").reset_index(drop=True)
    rain = df["prp_type"].str.lower().isin(["rain", "mixed"])
    snow = df["prp_type"].str.lower().eq("snow")
    sunup = df["sunup"].astype(int) == 1

    tw = (df["T2m"] + df["td"]) / 2.0
    wet = (
        rain |
        (df["T2m"] > 0) |
        (sunup & (df["cloud"] < 0.3) & (df["T2m"] >= -

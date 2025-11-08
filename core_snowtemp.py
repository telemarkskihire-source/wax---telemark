
import math
import pandas as pd

def compute_snow_temperature(df, dt_hours=1.0):
    """
    Expects columns (case-sensitive):
      - time: ISO8601 or any parseable datetime string
      - T2m: °C (float)
      - cloud: 0..1 (float)  (use 0..100 if you set cloud_pct=True in caller and convert)
      - wind: m/s (float)
      - sunup: 0/1 (int)
      - prp_mmph: precipitation intensity (mm/h)
      - prp_type: one of ["none","snow","rain","mixed"]
      - td: dew point °C (float, optional; can be NaN)
    Returns a new DataFrame with T_surf and T_top5 added.
    """
    df = df.copy()
    # Clean/cast
    df["time"] = pd.to_datetime(df["time"])
    required = ["T2m","cloud","wind","sunup","prp_mmph","prp_type"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    if "td" not in df.columns:
        df["td"] = float("nan")
    # Ensure ordering
    df = df.sort_values("time").reset_index(drop=True)

    # Helper masks
    rain = df["prp_type"].str.lower().isin(["rain","mixed"])
    snow = df["prp_type"].str.lower().eq("snow")
    sunup = df["sunup"].astype(int) == 1

    # Approx wet-bulb proxy where td exists
    tw = (df["T2m"] + df["td"]) / 2.0

    # Wet regime
    wet = (
        rain |
        (df["T2m"] > 0) |
        (sunup & (df["cloud"] < 0.3) & (df["T2m"] >= -3)) |
        (snow & (df["T2m"] >= -1)) |
        (snow & tw.ge(-0.5).fillna(False))
    )

    # Start T_surf with NaN
    T_surf = pd.Series(index=df.index, dtype=float)

    # 1) Wet -> 0 °C
    T_surf.loc[wet] = 0.0

    # 2) Dry/cold regime where not wet
    dry = ~wet
    clear = (1.0 - df["cloud"]).clip(lower=0.0, upper=1.0)
    windc = df["wind"].clip(upper=6.0)
    drad = (1.5 + 3.0*clear - 0.3*windc).clip(lower=0.5, upper=4.5)
    T_surf.loc[dry] = df["T2m"][dry] - drad[dry]

    # 3) Sunny cold-day correction: if sunup and -10<=T2m<=0 and not wet
    sunny_cold = sunup & dry & df["T2m"].between(-10, 0, inclusive="both")
    T_surf.loc[sunny_cold] = pd.concat([
        (df["T2m"] + 0.5*(1.0 - df["cloud"]))[sunny_cold],
        pd.Series(-0.5, index=df.index)[sunny_cold]
    ], axis=1).min(axis=1)

    # 4) Top 5 cm inertia (exponential)
    T_top5 = pd.Series(index=df.index, dtype=float)
    # Choose tau per-row
    tau = pd.Series(6.0, index=df.index, dtype=float)
    tau.loc[rain | snow | (df["wind"] >= 6)] = 3.0
    tau.loc[(~sunup) & (df["wind"] < 2) & (df["cloud"] < 0.3)] = 8.0
    alpha = 1.0 - (math.e ** (-dt_hours / tau))

    # Initialize T_top5 at first row
    if len(df) > 0:
        T_top5.iloc[0] = min(df["T2m"].iloc[0], 0.0)
        for i in range(1, len(df)):
            T_top5.iloc[i] = T_top5.iloc[i-1] + alpha.iloc[i] * (T_surf.iloc[i] - T_top5.iloc[i-1])

    # Attach to df
    df["T_surf"] = T_surf
    df["T_top5"] = T_top5

    return df

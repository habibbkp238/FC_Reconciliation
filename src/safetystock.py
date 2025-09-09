# src/safetystock.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

# Konversi month-equivalent -> week-equivalent
WEEKS_PER_MONTH_EQUIV = 4.345  # ~52.14 / 12

# Lookup z untuk service level + interpolasi linear
_Z_TABLE = [
    (0.90, 1.2816),
    (0.95, 1.6449),
    (0.98, 2.0537),
    (0.99, 2.3263),
]


def _approx_z(service_level: float) -> float:
    """
    Aproksimasi inverse normal CDF untuk p∈[0.90..0.99] via interpolasi linear.
    Guard: clamp ke [0.50, 0.999].
    """
    p = float(service_level)
    p = max(0.50, min(0.999, p))
    # cek hit tepat
    for pp, zz in _Z_TABLE:
        if abs(p - pp) < 1e-12:
            return zz
    # interpolasi
    pts = sorted(_Z_TABLE, key=lambda x: x[0])
    for i in range(len(pts) - 1):
        p0, z0 = pts[i]
        p1, z1 = pts[i + 1]
        if p0 <= p <= p1:
            w = (p - p0) / (p1 - p0) if p1 > p0 else 0.0
            return z0 * (1.0 - w) + z1 * w
    # di luar range -> clamp
    return pts[0][1] if p < pts[0][0] else pts[-1][1]


def _ensure_month_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "month" not in out.columns:
        out["month"] = pd.to_datetime(out["date"]).dt.to_period("M").dt.to_timestamp()
    else:
        out["month"] = pd.to_datetime(out["month"]).dt.to_period("M").dt.to_timestamp()
    return out


def _sigma_month_last_window(hist_monthly: pd.DataFrame, ma_window: int) -> pd.DataFrame:
    """
    Estimasi sigma bulanan per (sku, wh) pada window 'ma_window' bulan terakhir.
    Jika <2 titik dalam window -> fallback pakai semua bulan; jika tetap <2 -> 0.
    """
    hm = _ensure_month_col(hist_monthly)
    required = ["month", "sku", "wh", "qty"]
    miss = [c for c in required if c not in hm.columns]
    if miss:
        raise ValueError(f"hist_monthly missing columns: {miss}")

    hm = hm[required].copy().sort_values(["sku", "wh", "month"])

    def _compute(g: pd.DataFrame) -> pd.Series:
        gg = g.sort_values("month")
        if ma_window and ma_window > 0:
            gg = gg.tail(int(ma_window))
        if len(gg) >= 2:
            s = float(np.std(gg["qty"].to_numpy(dtype=float), ddof=1))
        else:
            s = float(np.std(g["qty"].to_numpy(dtype=float), ddof=1)) if len(g) >= 2 else 0.0
        return pd.Series({"sigma_month": s})

    return (
        hm.groupby(["sku", "wh"], dropna=False)
          .apply(_compute)
          .reset_index()
    )


def _sigma_week_from_daily(hist_daily: pd.DataFrame, ma_window: int, week_anchor: str = "W-MON") -> pd.DataFrame:
    """
    Estimasi sigma mingguan langsung dari hist harian:
      1) Totalkan harian -> mingguan (resample W-MON, label/closed='left')
      2) Ambil N minggu terakhir (N ≈ ceil(ma_window * 4.345))
      3) Std sampel per (sku, wh)
    """
    hd = hist_daily.copy()
    for c in ["date", "sku", "wh", "qty"]:
        if c not in hd.columns:
            raise ValueError("hist_daily must contain ['date','sku','wh','qty']")
    hd["date"] = pd.to_datetime(hd["date"])
    hd["qty"] = pd.to_numeric(hd["qty"], errors="coerce").fillna(0.0)

    dims = ["sku", "wh"]
    weekly = (
        hd.set_index("date")
          .groupby(dims, dropna=False)["qty"]
          .resample(week_anchor, label="left", closed="left")
          .sum()
          .reset_index()
          .sort_values(dims + ["date"])
          .reset_index(drop=True)
    )

    weeks_window = int(np.ceil(max(int(ma_window), 1) * WEEKS_PER_MONTH_EQUIV)) if (ma_window and ma_window > 0) else 0

    def _compute(g: pd.DataFrame) -> pd.Series:
        gg = g.sort_values("date")
        if weeks_window and weeks_window > 0:
            gg = gg.tail(weeks_window)
        if len(gg) >= 2:
            s = float(np.std(gg["qty"].to_numpy(dtype=float), ddof=1))
        else:
            s = float(np.std(g["qty"].to_numpy(dtype=float), ddof=1)) if len(g) >= 2 else 0.0
        return pd.Series({"sigma_week": s})

    return (
        weekly.groupby(dims, dropna=False)
              .apply(_compute)
              .reset_index()
    )


def compute_safety_stock_wh(
    hist_monthly: pd.DataFrame,
    service_params: pd.DataFrame,
    ma_window: int = 3,
    round_up: bool = True,
    # opsional: pakai jalur harian jika tersedia
    hist_daily: Optional[pd.DataFrame] = None,
    week_anchor: str = "W-MON",
) -> pd.DataFrame:
    """
    Hitung Safety Stock (basis mingguan) per SKU–WH.
    ***Tidak*** mengubah atau bercampur dengan weekly split forecasting.

    Preferensi estimasi sigma:
      - Jika hist_daily diberikan -> gunakan _sigma_week_from_daily (disarankan)
      - Jika tidak -> fallback: sigma_week ≈ sigma_month / 4.345

    service_params:
      - Skema A (disarankan): ['sku','wh','lead_time_days','service_level_target']
      - Skema B (legacy)    : ['sku','lead_time_days','service_level_target'] -> dibroadcast ke semua WH

    Output kolom:
      ['sku','wh','sigma_week','z','lead_time_days','safety_stock_week','safety_stock_week_int']
    """
    # 1) Estimasi sigma mingguan
    if hist_daily is not None and not hist_daily.empty:
        sigw = _sigma_week_from_daily(hist_daily, ma_window=ma_window, week_anchor=week_anchor)
    else:
        sigm = _sigma_month_last_window(hist_monthly, ma_window=ma_window)
        sigw = sigm.copy()
        sigw["sigma_week"] = sigw["sigma_month"] / WEEKS_PER_MONTH_EQUIV
        sigw = sigw.drop(columns=["sigma_month"])

    # 2) Normalisasi service params ke level (sku, wh)
    sp = service_params.copy()
    required = ["sku", "lead_time_days", "service_level_target"]
    miss = [c for c in required if c not in sp.columns]
    if miss:
        raise ValueError("service_params must contain ['sku','lead_time_days','service_level_target'] (optional 'wh').")

    sp["sku"] = sp["sku"].astype(str)
    sp["lead_time_days"] = pd.to_numeric(sp["lead_time_days"], errors="coerce")
    sp["service_level_target"] = pd.to_numeric(sp["service_level_target"], errors="coerce")

    sku_wh_obs = sigw[["sku", "wh"]].drop_duplicates()
    if "wh" in sp.columns:
        sp_norm = sku_wh_obs.merge(sp, on=["sku", "wh"], how="left")
    else:
        sp_norm = sku_wh_obs.merge(sp[["sku", "lead_time_days", "service_level_target"]], on="sku", how="left")

    # Fallback nilai kosong
    sp_norm["lead_time_days"] = sp_norm["lead_time_days"].fillna(14).clip(lower=0)
    sp_norm["service_level_target"] = sp_norm["service_level_target"].fillna(0.95).clip(lower=0.50, upper=0.999)
    sp_norm["z"] = sp_norm["service_level_target"].map(_approx_z)

    # 3) Gabungkan + hitung SS (basis minggu)
    out = sigw.merge(sp_norm, on=["sku", "wh"], how="left")

    lt_weeks = (out["lead_time_days"].astype(float) / 7.0).clip(lower=0.0)
    out["safety_stock_week"] = out["z"].astype(float) * out["sigma_week"].astype(float) * np.sqrt(lt_weeks)

    # 4) Integerisasi (tidak memengaruhi weekly forecast)
    if round_up:
        out["safety_stock_week_int"] = np.ceil(out["safety_stock_week"]).fillna(0).astype(int)
    else:
        out["safety_stock_week_int"] = np.round(out["safety_stock_week"]).fillna(0).astype(int)

    cols = ["sku", "wh", "sigma_week", "z", "lead_time_days", "safety_stock_week", "safety_stock_week_int"]
    return out[cols].sort_values(["sku", "wh"]).reset_index(drop=True)

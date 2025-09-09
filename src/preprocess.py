# src/preprocess.py
from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd


# =============================================================================
# Konstanta kolom
# =============================================================================
ESSENTIAL_COLS = ["date", "sku", "wh", "qty"]
OPT_ATTRS = ["sku_name", "category", "class", "brand", "extension", "region", "kota"]


# =============================================================================
# Helpers
# =============================================================================
def _snap_wmon_start(s: pd.Series | pd.DatetimeIndex) -> pd.Series:
    """
    Snap semua timestamp ke awal minggu (Senin, W-MON). Idempotent.
    """
    s = pd.to_datetime(s)
    return s.dt.to_period("W-MON").apply(lambda p: p.start_time.normalize())


def _as_month(ts: pd.Series) -> pd.Series:
    """Konversi ke awal bulan (Timestamp naive)."""
    return pd.to_datetime(ts).dt.to_period("M").dt.to_timestamp()


def _ensure_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib hilang: {missing}. Ditemukan: {list(df.columns)}")


def _presence_lists(hist_monthly: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Presence preview:
      - wh_per_sku
      - region_per_sku_wh
      - kota_per_sku_wh_region
    Tahan banting jika kolom opsional tidak tersedia.
    """
    out: Dict[str, pd.DataFrame] = {}

    # WH per SKU
    if {"sku", "wh"}.issubset(hist_monthly.columns):
        g1 = (
            hist_monthly.groupby(["sku"], dropna=False)["wh"]
            .apply(lambda s: sorted(pd.Series(s).dropna().unique().tolist()))
            .reset_index(name="wh_list")
        )
        out["wh_per_sku"] = g1
    else:
        out["wh_per_sku"] = pd.DataFrame(columns=["sku", "wh_list"])

    # Region per (SKU, WH)
    if {"sku", "wh", "region"}.issubset(hist_monthly.columns):
        g2 = (
            hist_monthly.groupby(["sku", "wh"], dropna=False)["region"]
            .apply(lambda s: sorted(pd.Series(s).dropna().unique().tolist()))
            .reset_index(name="region_list")
        )
        out["region_per_sku_wh"] = g2
    else:
        out["region_per_sku_wh"] = pd.DataFrame(columns=["sku", "wh", "region_list"])

    # Kota per (SKU, WH, Region)
    if {"sku", "wh", "region", "kota"}.issubset(hist_monthly.columns):
        g3 = (
            hist_monthly.groupby(["sku", "wh", "region"], dropna=False)["kota"]
            .apply(lambda s: sorted(pd.Series(s).dropna().unique().tolist()))
            .reset_index(name="kota_list")
        )
        out["kota_per_sku_wh_region"] = g3
    else:
        out["kota_per_sku_wh_region"] = pd.DataFrame(columns=["sku", "wh", "region", "kota_list"])

    return out


def _dq_metrics(df_raw: pd.DataFrame, df_clean: pd.DataFrame, dims: List[str]) -> Dict[str, float | int | str]:
    """
    Ringkas DQ: ukuran, rentang tanggal, % zero qty, count qty negatif, duplikat per key.
    df_clean: hasil setelah coercion/filter minimal (belum agregasi).
    """
    out: Dict[str, float | int | str] = {}

    out["raw_rows"] = int(len(df_raw))
    out["clean_rows"] = int(len(df_clean))

    # rentang tanggal
    if len(df_clean):
        out["date_min"] = str(pd.to_datetime(df_clean["date"]).min().date())
        out["date_max"] = str(pd.to_datetime(df_clean["date"]).max().date())
    else:
        out["date_min"] = "-"
        out["date_max"] = "-"

    # qty
    qty = pd.to_numeric(df_clean.get("qty", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    out["pct_zero_qty_clean"] = float((qty == 0).mean()) if len(qty) else 0.0
    out["neg_qty_count_clean"] = int((qty < 0).sum()) if len(qty) else 0

    # duplikat per key (date + dims)
    key_cols = ["date"] + dims
    if all(c in df_clean.columns for c in key_cols):
        dups = df_clean.duplicated(subset=key_cols, keep=False).sum()
        out["duplicate_key_rows_clean"] = int(dups)
    else:
        out["duplicate_key_rows_clean"] = 0

    # unique key raw (untuk sense-check)
    if all(c in df_raw.columns for c in key_cols):
        out["unique_key_raw"] = int(df_raw.drop_duplicates(key_cols).shape[0])
    else:
        out["unique_key_raw"] = int(df_raw.shape[0])

    return out


# =============================================================================
# Public API
# =============================================================================
def preprocess_history(
    df_hist_raw: pd.DataFrame,
    last_n_months: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame], Dict, pd.DataFrame]:
    """
    Canonicalisasi histori:
      - Coerce kolom wajib
      - Bentuk DAILY (harian), WEEKLY (W-MON) & MONTHLY
      - Filter N bulan terakhir (opsional, inklusif bulan terbaru)
      - Presence preview (berbasis monthly)
      - DQ metrics (post-canonical minimal)

    Returns
    -------
    hist_weekly : DataFrame  ['date', dims..., 'qty']   (date = Senin, W-MON)
    hist_monthly: DataFrame  ['month', dims..., 'qty']  (month = awal bulan)
    presence_preview: dict   {'wh_per_sku', 'region_per_sku_wh', 'kota_per_sku_wh_region'}
    dq_post: dict            ringkasan DQ
    hist_daily : DataFrame   ['date', dims..., 'qty']   (harian, diagregasi per date+dims)
    """
    # --- 0) Early-return kalau kosong
    if df_hist_raw is None or df_hist_raw.empty:
        empty_daily  = pd.DataFrame(columns=["date", "sku", "wh", "qty"])
        empty_weekly = empty_daily.copy()
        empty_monthly= pd.DataFrame(columns=["month", "sku", "wh", "qty"])
        return (
            empty_weekly,
            empty_monthly,
            {"wh_per_sku": pd.DataFrame(), "region_per_sku_wh": pd.DataFrame(), "kota_per_sku_wh_region": pd.DataFrame()},
            {"raw_rows": 0, "clean_rows": 0},
            empty_daily,
        )

    # --- 1) Validasi minimal & coercion
    _ensure_columns(df_hist_raw, ESSENTIAL_COLS)
    raw = df_hist_raw.copy()

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["qty"]  = pd.to_numeric(raw["qty"], errors="coerce")

    # Buang baris tidak layak: NaT / key NaN; qty NaN -> 0
    df = raw.dropna(subset=["date", "sku", "wh"]).copy()
    df["qty"] = df["qty"].fillna(0.0)

    # Dimensi yang tersedia
    dims = [c for c in ["sku", "wh", "region", "kota"] if c in df.columns]

    # --- 2) DAILY canonical (agregasi per tanggal + dims)
    hist_daily = (
        df[["date"] + dims + ["qty"]]
        .groupby(["date"] + dims, dropna=False)["qty"]
        .sum()
        .reset_index()
        .sort_values(["date"] + dims)
        .reset_index(drop=True)
    )

    # --- 3) WEEKLY canonical (W-MON, label & closed = left) + snap
    hist_weekly = (
        hist_daily.set_index("date")
                  .groupby(dims, dropna=False)["qty"]
                  .resample("W-MON", label="left", closed="left")
                  .sum()
                  .reset_index()
    )
    hist_weekly["date"] = _snap_wmon_start(hist_weekly["date"])

    # --- 4) MONTHLY canonical
    hist_monthly = hist_daily.copy()
    hist_monthly["month"] = _as_month(hist_monthly["date"])
    hist_monthly = (
        hist_monthly
        .groupby(dims + ["month"], dropna=False)["qty"]
        .sum()
        .reset_index()
        .sort_values(["month"] + dims)
        .reset_index(drop=True)
    )

    # --- 5) Filter last_n_months (inklusif bulan terbaru) untuk daily/weekly/monthly
    if isinstance(last_n_months, (int, np.integer)) and last_n_months > 0 and len(hist_monthly):
        anchor: pd.Timestamp = hist_monthly["month"].max()  # Timestamp (awal bulan terbaru)
        cutoff = (anchor - pd.offsets.DateOffset(months=int(last_n_months) - 1)).to_period("M").to_timestamp()

        # Filter monthly
        hist_monthly = hist_monthly.loc[hist_monthly["month"] >= cutoff].copy()

        # Filter weekly (berdasarkan month dari tanggal Senin)
        hw_month = _as_month(hist_weekly["date"])
        hist_weekly = hist_weekly.loc[hw_month >= cutoff].copy()

        # Filter daily (berdasarkan month dari tanggal harian)
        hd_month = _as_month(hist_daily["date"])
        hist_daily = hist_daily.loc[hd_month >= cutoff].copy()

    # --- 6) Presence preview berbasis monthly
    presence_preview = _presence_lists(hist_monthly)

    # --- 7) DQ metrics (post-canonical minimal)  **FIX: argumen df_raw**
    dq_post = _dq_metrics(df_raw=raw, df_clean=df, dims=dims)

    # --- 8) Urut & kembalikan kolom sesuai skema
    # Weekly
    wk_cols = ["date"] + dims + ["qty"]
    hist_weekly = (
        hist_weekly[wk_cols]
        .sort_values(["date"] + dims)
        .reset_index(drop=True)
    )

    # Monthly
    mo_cols = ["month"] + dims + ["qty"]
    hist_monthly = (
        hist_monthly[mo_cols]
        .sort_values(["month"] + dims)
        .reset_index(drop=True)
    )

    # Daily
    dy_cols = ["date"] + dims + ["qty"]
    hist_daily = hist_daily[dy_cols].reset_index(drop=True)

    return hist_weekly, hist_monthly, presence_preview, dq_post, hist_daily

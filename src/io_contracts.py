# src/io_contracts.py
from __future__ import annotations
from typing import Tuple, List

import numpy as np
import pandas as pd

REQ_FORECAST_COLS = ["date", "sku", "qty"]
REQ_SERVICE_COLS  = ["sku", "lead_time_days", "service_level_target"]
REQ_HIST_COLS     = ["date","sku","sku_name","category","class","brand","extension","wh","region","kota","qty"]

def _read_any(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    name = getattr(file, "name", str(file)).lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

def _ensure_columns(df: pd.DataFrame, required: List[str], label: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: kolom wajib hilang: {missing}")

def _strip_and_blank_to_na(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if not cols:
        return df
    x = df.copy()
    for c in cols:
        if c in x.columns:
            x[c] = x[c].astype(str).str.strip()
    rep = {c: {"": pd.NA} for c in cols if c in x.columns}
    if rep:
        x.replace(rep, inplace=True)
    return x

def _coerce_num(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    x = df.copy()
    for c in cols:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    return x

def _coerce_dt(df: pd.DataFrame, col: str) -> pd.DataFrame:
    x = df.copy()
    if col in x.columns:
        x[col] = pd.to_datetime(x[col], errors="coerce")
    return x

def read_inputs(f_forecast, f_service, f_hist) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
    warns: List[str] = []

    df_f = _read_any(f_forecast)
    df_s = _read_any(f_service)
    df_h = _read_any(f_hist)

    _ensure_columns(df_f, REQ_FORECAST_COLS, "Forecast Nasional")
    _ensure_columns(df_s, REQ_SERVICE_COLS,  "Service Params")
    _ensure_columns(df_h, REQ_HIST_COLS,     "Historical + Attributes")

    df_f = _strip_and_blank_to_na(df_f, ["sku"])
    df_s = _strip_and_blank_to_na(df_s, ["sku"])
    df_h = _strip_and_blank_to_na(df_h, ["sku","sku_name","category","class","brand","extension","wh","region","kota"])

    df_f = _coerce_dt(df_f, "date")
    df_h = _coerce_dt(df_h, "date")

    df_f = _coerce_num(df_f, ["qty"])
    df_h = _coerce_num(df_h, ["qty"])
    df_s = _coerce_num(df_s, ["lead_time_days", "service_level_target"])

    if df_f["date"].isna().any():
        warns.append("Forecast: ada tanggal tidak valid (NaT).")
    if df_h["date"].isna().any():
        warns.append("Historical: ada tanggal tidak valid (NaT).")
    if df_f["qty"].fillna(0).lt(0).any():
        warns.append("Forecast: ada qty negatif.")
    if df_h["qty"].fillna(0).lt(0).any():
        warns.append("Historical: ada qty negatif.")

    df_f = df_f[REQ_FORECAST_COLS + [c for c in df_f.columns if c not in REQ_FORECAST_COLS]]
    df_s = df_s[REQ_SERVICE_COLS  + [c for c in df_s.columns if c not in REQ_SERVICE_COLS]]
    df_h = df_h[REQ_HIST_COLS     + [c for c in df_h.columns if c not in REQ_HIST_COLS]]

    return df_f, df_s, df_h, warns

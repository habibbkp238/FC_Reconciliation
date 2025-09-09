# src/templates.py
from __future__ import annotations

import io
from typing import Optional, Dict
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Public helpers
# ------------------------------------------------------------
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# ------------------------------------------------------------
# Templates (DataFrame)
# ------------------------------------------------------------
def make_forecast_template(sample: bool = True) -> pd.DataFrame:
    """
    Forecast nasional (bulanan) — minimal kolom: date (awal bulan), sku, qty
    """
    cols = ["date", "sku", "qty"]
    if not sample:
        return pd.DataFrame(columns=cols)

    # 10 SKU contoh, 2 bulan ke depan dari anchor (bulan sekarang)
    skus = [f"SKU{str(i).zfill(3)}" for i in range(1, 11)]
    anchor = pd.Timestamp.today().to_period("M").to_timestamp()
    months = pd.period_range(anchor, periods=2, freq="M").to_timestamp()
    rows = []
    for m in months:
        for s in skus:
            rows.append({"date": m.normalize(), "sku": s, "qty": 1000})
    return pd.DataFrame(rows, columns=cols)


def make_service_template(sample: bool = True) -> pd.DataFrame:
    """
    Service params — format baru **per (sku, wh)**.
    Kolom wajib: sku, wh, lead_time_days, service_level_target

    Catatan:
    - Jika Anda ingin pakai schema lama (per SKU saja), cukup hapus kolom 'wh'.
      Tool akan otomatis broadcast parameter itu ke semua WH yang ada di histori.
    - Namun untuk konsistensi safety stock per WH, disarankan isi per (sku, wh).
    """
    cols = ["sku", "wh", "lead_time_days", "service_level_target"]
    if not sample:
        return pd.DataFrame(columns=cols)

    skus = [f"SKU{str(i).zfill(3)}" for i in range(1, 11)]
    whs  = ["WH01", "WH02"]

    rows = []
    for s in skus:
        for w in whs:
            rows.append({"sku": s, "wh": w, "lead_time_days": 14, "service_level_target": 0.95})
    return pd.DataFrame(rows, columns=cols)


def make_historical_template(sample: bool = True) -> pd.DataFrame:
    """
    Historical + Attributes (HARlAN) — **harian**.

    Kolom minimal:
      date, sku, sku_name, category, class, brand, extension, wh, region, kota, qty
    """
    cols = ["date","sku","sku_name","category","class","brand","extension","wh","region","kota","qty"]
    if not sample:
        return pd.DataFrame(columns=cols)

    # Buat contoh kecil 2 hari, 2 SKU, 1 WH, 2 Region (masing2 1 kota)
    days = pd.date_range(pd.Timestamp.today().normalize() - pd.Timedelta(days=6), periods=2, freq="D")
    sample_rows = [
        # Day 1
        {"date": days[0], "sku":"SKU001", "sku_name":"Product 001", "category":"Food", "class":"A", "brand":"BrandX", "extension":"Sachet",
         "wh":"WH01", "region":"Region-A", "kota":"Kota-A-WH01-1", "qty":120},
        {"date": days[0], "sku":"SKU001", "sku_name":"Product 001", "category":"Food", "class":"A", "brand":"BrandX", "extension":"Sachet",
         "wh":"WH01", "region":"Region-B", "kota":"Kota-B-WH01-1", "qty":95},
        {"date": days[0], "sku":"SKU002", "sku_name":"Product 002", "category":"Home", "class":"B", "brand":"BrandY", "extension":"Bottle",
         "wh":"WH02", "region":"Region-A", "kota":"Kota-A-WH02-1", "qty":60},
        # Day 2
        {"date": days[1], "sku":"SKU001", "sku_name":"Product 001", "category":"Food", "class":"A", "brand":"BrandX", "extension":"Sachet",
         "wh":"WH01", "region":"Region-A", "kota":"Kota-A-WH01-1", "qty":130},
        {"date": days[1], "sku":"SKU001", "sku_name":"Product 001", "category":"Food", "class":"A", "brand":"BrandX", "extension":"Sachet",
         "wh":"WH01", "region":"Region-B", "kota":"Kota-B-WH01-1", "qty":88},
        {"date": days[1], "sku":"SKU002", "sku_name":"Product 002", "category":"Home", "class":"B", "brand":"BrandY", "extension":"Bottle",
         "wh":"WH02", "region":"Region-A", "kota":"Kota-A-WH02-1", "qty":70},
    ]
    return pd.DataFrame(sample_rows, columns=cols)


# ------------------------------------------------------------
# Example Excel (for the Download button)
# ------------------------------------------------------------
def make_output_example_xlsx() -> bytes:
    """
    Bangun contoh workbook Excel berisi 3 sheet:
      - Forecast_National (template)
      - Service_Params (template per sku, wh)
      - Historical_With_Attributes (template harian)
    """
    import xlsxwriter

    df_fc  = make_forecast_template(sample=True)
    df_sp  = make_service_template(sample=True)
    df_hist= make_historical_template(sample=True)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter", datetime_format="yyyy-mm-dd", date_format="yyyy-mm-dd") as writer:
        def _write(ws_name: str, df: pd.DataFrame):
            df.to_excel(writer, index=False, sheet_name=ws_name)
            ws = writer.sheets[ws_name]
            ws.freeze_panes(1, 0)
            # autosize
            for i, c in enumerate(df.columns):
                m = max(len(str(c)), *(df[c].astype(str).map(len).tolist() or [0]))
                ws.set_column(i, i, min(m + 2, 60))

        _write("Forecast_National", df_fc)
        _write("Service_Params", df_sp)
        _write("Historical_With_Attributes", df_hist)

        # README sheet
        readme = pd.DataFrame({
            "Field": [
                "Service Params format",
                "Service Params alt",
                "Historical grain",
                "Forecast grain",
                "Notes",
            ],
            "Value": [
                "Per (sku, wh): columns = sku, wh, lead_time_days, service_level_target",
                "If 'wh' column is omitted: params broadcast per-SKU to all WH seen in history",
                "Daily (date, ... , qty) — planner friendly for Month-Sliced DOW allocation",
                "Monthly national per SKU (date, sku, qty)",
                "All dates should be ISO format YYYY-MM-DD; month uses first day of month",
            ]
        })
        readme.to_excel(writer, index=False, sheet_name="README")
        ws = writer.sheets["README"]
        ws.freeze_panes(1, 0)
        for i, c in enumerate(readme.columns):
            m = max(len(str(c)), *(readme[c].astype(str).map(len).tolist() or [0]))
            ws.set_column(i, i, min(m + 2, 80))

    buf.seek(0)
    return buf.getvalue()

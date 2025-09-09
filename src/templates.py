# src/templates.py
from __future__ import annotations

import io
from typing import Optional, Dict
import pandas as pd


# ============================================================
# Public helpers
# ============================================================
def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Serialize DataFrame to CSV bytes (UTF-8, no index).
    """
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# ============================================================
# Templates (DataFrame)
# ============================================================
def make_forecast_template(sample: bool = True) -> pd.DataFrame:
    """
    Template untuk forecast nasional (bulanan).
    Kolom wajib: date (awal bulan), sku, qty
    """
    cols = ["date", "sku", "qty"]
    if not sample:
        return pd.DataFrame(columns=cols)

    # Contoh 10 SKU, 2 bulan ke depan dari anchor (awal bulan sekarang)
    skus = [f"SKU{str(i).zfill(3)}" for i in range(1, 11)]
    anchor = pd.Timestamp.today().to_period("M").to_timestamp()  # first day of current month
    months = pd.period_range(anchor, periods=2, freq="M").to_timestamp()
    rows = []
    for m in months:
        for s in skus:
            rows.append({"date": m.normalize(), "sku": s, "qty": 1000})
    return pd.DataFrame(rows, columns=cols)


def make_service_template(sample: bool = True) -> pd.DataFrame:
    """
    Template untuk service params (format baru).
    Disarankan per (sku, wh), namun kolom 'wh' opsional:
      - Jika 'wh' disertakan  -> parameter spesifik per (sku, wh)
      - Jika 'wh' dihilangkan -> parameter per SKU akan di-broadcast
                                 ke semua WH yang muncul di histori.

    Kolom: sku, wh (opsional), lead_time_days, service_level_target
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
    Template untuk Historical + Attributes (harian).
    Kolom minimal:
      date, sku, sku_name, category, class, brand, extension, wh, region, kota, qty
    """
    cols = ["date","sku","sku_name","category","class","brand","extension","wh","region","kota","qty"]
    if not sample:
        return pd.DataFrame(columns=cols)

    # Contoh kecil 2 hari, 2 SKU, 1 WH, 2 region (masing2 1 kota)
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


# ============================================================
# Example Excel (untuk tombol Download di UI)
# ============================================================
def make_output_example_xlsx() -> bytes:
    """
    Bangun workbook contoh berisi 4 sheet:
      - Forecast_National              (template bulanan)
      - Service_Params                 (template per sku, wh)
      - Historical_With_Attributes     (template harian)
      - README                         (penjelasan format)

    Catatan: engine Excel dicoba berurutan: 'openpyxl' -> 'xlsxwriter'.
             Jika keduanya tidak tersedia, akan raise error yang jelas.
    """
    df_fc  = make_forecast_template(sample=True)
    df_sp  = make_service_template(sample=True)
    df_hist= make_historical_template(sample=True)

    readme = pd.DataFrame({
        "Field": [
            "Service Params format",
            "Service Params alt",
            "Historical grain",
            "Forecast grain",
            "Dates",
            "Notes",
        ],
        "Value": [
            "Per (sku, wh): columns = sku, wh, lead_time_days, service_level_target",
            "If 'wh' column is omitted: params broadcast per-SKU to all WH seen in history",
            "Daily (date, ... , qty) — planner friendly for Month-Sliced DOW allocation",
            "Monthly national per SKU (date, sku, qty)",
            "Use ISO date YYYY-MM-DD; month uses first day of month",
            "Safety Stock dihitung di level SKU–WH dan tidak memengaruhi weekly split",
        ]
    })

    buf = io.BytesIO()

    def _write_sheet(writer, name: str, df: pd.DataFrame):
        df.to_excel(writer, index=False, sheet_name=name)
        ws = writer.sheets[name]
        # freeze row header (jika engine mendukung)
        try:
            ws.freeze_panes(1, 0)
        except Exception:
            pass
        # autosize kolom (xlsxwriter mendukung; openpyxl tidak)
        for i, c in enumerate(df.columns):
            try:
                m = max(len(str(c)), *(df[c].astype(str).map(len).tolist() or [0]))
                ws.set_column(i, i, min(m + 2, 60))
            except Exception:
                # engine tidak mendukung set_column → diabaikan
                pass

    last_err: Optional[Exception] = None
    for eng in ("openpyxl", "xlsxwriter"):
        try:
            buf.seek(0); buf.truncate(0)
            with pd.ExcelWriter(
                buf, engine=eng, datetime_format="yyyy-mm-dd", date_format="yyyy-mm-dd"
            ) as writer:
                _write_sheet(writer, "Forecast_National", df_fc)
                _write_sheet(writer, "Service_Params", df_sp)
                _write_sheet(writer, "Historical_With_Attributes", df_hist)
                _write_sheet(writer, "README", readme)
            break  # sukses
        except Exception as e:
            last_err = e
            continue

    if last_err is not None and buf.getbuffer().nbytes == 0:
        raise RuntimeError(
            "Tidak ada engine Excel yang tersedia. Install salah satu: 'openpyxl' atau 'xlsxwriter'. "
            f"Last error: {last_err}"
        )

    buf.seek(0)
    return buf.getvalue()

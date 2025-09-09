# src/exporter.py
from __future__ import annotations
import io
import json
from datetime import datetime, date
from decimal import Decimal
from typing import Any, Optional, List

import numpy as np
import pandas as pd


# ============================ Helpers ============================

EXCEL_SCALARS = (str, int, float, bool, type(None), datetime, date)

def _as_month(ts: Any) -> pd.Timestamp:
    return pd.to_datetime(ts).to_period("M").to_timestamp()

def _safe_sum(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").fillna(0.0)
    return float(x.sum())

def _as_jsonable(v: Any):
    if isinstance(v, (pd.Index, pd.RangeIndex, pd.MultiIndex)):
        return list(map(str, list(v)))
    if isinstance(v, pd.Series):
        return [ _to_excel_scalar(xx) for xx in v.tolist() ]
    if isinstance(v, pd.DataFrame):
        out = []
        cols = [str(c) for c in v.columns]
        for row in v.itertuples(index=False, name=None):
            out.append({ c: _to_excel_scalar(val) for c, val in zip(cols, row) })
        return out
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, range):
        return list(v)
    return str(v)

def _to_excel_scalar(v: Any) -> Any:
    """Konversi nilai ke scalar aman untuk Excel; selain itu -> string."""
    if v is None or v is pd.NA:
        return None
    if isinstance(v, float):
        try:
            if np.isnan(v):
                return None
        except Exception:
            pass
    if isinstance(v, np.generic):
        try:
            v = v.item()
        except Exception:
            v = str(v)
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    if isinstance(v, pd.Period):
        return str(v.to_timestamp())
    if isinstance(v, pd.Timedelta):
        return str(v)
    if isinstance(v, (np.datetime64, np.timedelta64)):
        try:
            return pd.to_datetime(v).to_pydatetime()
        except Exception:
            return str(v)
    if isinstance(v, Decimal):
        try:
            return float(v)
        except Exception:
            return str(v)
    if isinstance(v, EXCEL_SCALARS):
        return v
    try:
        if isinstance(v, (pd.Index, pd.RangeIndex, pd.MultiIndex, pd.Series, pd.DataFrame, np.ndarray, memoryview, range)):
            return json.dumps(_as_jsonable(v), default=str, ensure_ascii=False)
        if isinstance(v, (list, tuple, set, dict)):
            return json.dumps(v, default=str, ensure_ascii=False)
    except Exception:
        return str(v)
    return str(v)

def _sanitize_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Reset index, kolom->str, map semua sel -> scalar aman."""
    if df is None:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    x = df.copy().reset_index(drop=True)
    x.columns = [str(c) for c in x.columns]
    for c in x.columns:
        if pd.api.types.is_datetime64_any_dtype(x[c]):
            x[c] = x[c].where(~x[c].isna(), None)
        x[c] = x[c].map(_to_excel_scalar)
    return x

def _freeze_header_xlsxwriter(ws):
    try:
        ws.freeze_panes(1, 0)
    except Exception:
        pass

def _write_sheet(ws, df: pd.DataFrame, header_fmt=None, date_fmt=None, num_fmt=None):
    """Tulis DataFrame ke worksheet xlsxwriter SEL BARIS DEMI BARIS dengan konversi scalar."""
    # Header
    for j, col in enumerate(df.columns):
        ws.write(0, j, str(col), header_fmt)
    # Rows
    for i in range(len(df)):
        row = df.iloc[i]
        for j, val in enumerate(row):
            v = _to_excel_scalar(val)
            if isinstance(v, (int, float)) and v is not None:
                ws.write_number(i+1, j, float(v), num_fmt)
            elif isinstance(v, (datetime, date)):
                ws.write_datetime(i+1, j, v, date_fmt)
            elif isinstance(v, bool):
                ws.write_boolean(i+1, j, v)
            elif v is None:
                ws.write_blank(i+1, j, None)
            else:
                ws.write_string(i+1, j, str(v)[:32767])  # Excel cell limit
    _freeze_header_xlsxwriter(ws)

def _scan_indexlikes(df: pd.DataFrame, name: str, limit: int = 20) -> List[tuple]:
    """Diagnostik: cari sel yang masih Index/Series/ndarray (harusnya tidak ada)."""
    issues = []
    if df is None or df.empty:
        return issues
    for col in df.columns:
        for i, v in df[col].items():
            if isinstance(v, (pd.Index, pd.RangeIndex, pd.MultiIndex, pd.Series, np.ndarray)):
                issues.append((name, int(i), str(col), type(v).__name__, str(v)[:160]))
                if len(issues) >= limit:
                    return issues
    return issues


# ============================ Sheet Builders ============================

def build_sheet1_wh_month(wh_month: pd.DataFrame, sku_attrs: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = wh_month.copy()
    df["date"] = _as_month(df["date"])
    if sku_attrs is not None and "sku" in sku_attrs.columns:
        keep_cols = [c for c in ["sku","sku_name","category","class","brand","extension"] if c in sku_attrs.columns]
        if keep_cols:
            df = df.merge(sku_attrs[keep_cols].drop_duplicates("sku"), on="sku", how="left")
    col_order = [
        "date","sku","sku_name","category","class","brand","extension","wh",
        "forecast_reconciled","safety_stock_week_int"
    ]
    col_order = [c for c in col_order if c in df.columns]
    sort_cols = [c for c in ["date","sku","wh"] if c in col_order]
    return df[col_order].sort_values(sort_cols).reset_index(drop=True)

def build_sheet2_region_kota(
    scope: str,
    grain_output: str,
    reconciled_monthly_leaf: pd.DataFrame,
    weekly_df: Optional[pd.DataFrame],
    sku_attrs: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    if scope == "WH only":
        return pd.DataFrame({"note": ["Scope = WH only. Tidak ada level Region/Kota untuk diekspor."]})
    if str(grain_output).startswith("Weekly"):
        if weekly_df is None or weekly_df.empty:
            return pd.DataFrame({"note": ["Weekly dipilih, namun data weekly kosong."]})
        df = weekly_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        out_cols = ["date","sku","wh"]
        if "region" in df.columns: out_cols.append("region")
        if "kota"   in df.columns: out_cols.append("kota")
        out_cols.append("forecast_weekly")
        df = df[out_cols].rename(columns={"forecast_weekly": "forecast"})
    else:
        df = reconciled_monthly_leaf.copy()
        df["date"] = _as_month(df["date"])
        out_cols = ["date","sku","wh"]
        if scope in ("WH → Region", "WH → Region → Kota"): out_cols.append("region")
        if scope == "WH → Region → Kota": out_cols.append("kota")
        out_cols.append("forecast_reconciled")
        df = df[out_cols].rename(columns={"forecast_reconciled": "forecast"})
    if sku_attrs is not None and "sku" in sku_attrs.columns:
        keep_cols = [c for c in ["sku","sku_name","category","class","brand","extension"] if c in sku_attrs.columns]
        if keep_cols:
            df = df.merge(sku_attrs[keep_cols].drop_duplicates("sku"), on="sku", how="left")
    sort_cols = [c for c in ["date","sku","wh","region","kota"] if c in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True)

def build_sheet3_summary(
    df_forecast_nat: pd.DataFrame,
    reconciled_monthly_leaf: pd.DataFrame,
    scope: str,
    grain_output: str,
    params: dict,
    coverage_ratio: Optional[float] = None
) -> pd.DataFrame:
    nat_total = _safe_sum(df_forecast_nat["qty"])
    leaf_total = _safe_sum(reconciled_monthly_leaf.get("forecast_reconciled", pd.Series([], dtype=float)))
    fm = df_forecast_nat.copy(); fm["month"] = _as_month(fm["date"])
    rm = reconciled_monthly_leaf.copy(); rm["month"] = _as_month(rm["date"])
    chk = (
        fm.groupby(["month","sku"])["qty"].sum().reset_index()
          .merge(rm.groupby(["month","sku"])["forecast_reconciled"].sum().reset_index(),
                 on=["month","sku"], how="left")
    )
    ok_monthly = bool((chk["qty"] - chk["forecast_reconciled"]).abs().fillna(0).lt(1e-6).all())
    rows = [
        ("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("scope", str(scope)),
        ("grain_output", str(grain_output)),
        ("recon_method", str(params.get("recon_method"))),
    ]
    if str(params.get("recon_method","")).startswith("MinT"):
        rows.append(("mint_lambda", params.get("mint_lambda")))
    rows += [
        ("volatility_threshold_WH", params.get("vol_threshold")),
        ("weekly_method", str(params.get("weekly_method")) if str(grain_output).startswith("Weekly") else "-"),
        ("integerize", bool(params.get("int_toggle"))),
        ("integer_method", str(params.get("int_method")) if params.get("int_toggle") else "-"),
        ("ss_ma_window", params.get("ss_ma")),
        ("ss_round_up", bool(params.get("ss_round_up"))),
        ("coverage_month_sku", f"{float(coverage_ratio):.1%}" if coverage_ratio is not None else "-"),
        ("total_forecast_national", float(nat_total)),
        ("total_leaf_after_reconciliation", float(leaf_total)),
        ("coherence_sum_month_sku", bool(ok_monthly)),
    ]
    return pd.DataFrame(rows, columns=["metric","value"])


# ============================ Manual Writer ============================

def _write_workbook_xlsxwriter(sheets: List[tuple]) -> bytes:
    """
    sheets: list of (sheet_name, dataframe)
    Semua dataframe disanitasi lalu ditulis manual.
    """
    import xlsxwriter  # pastikan tersedia
    buf = io.BytesIO()
    wb = xlsxwriter.Workbook(buf, {'in_memory': True})

    # Formats
    fmt_header = wb.add_format({'bold': True, 'bg_color': '#F2F2F2', 'border': 1})
    fmt_date = wb.add_format({'num_format': 'yyyy-mm-dd'})
    fmt_num = wb.add_format()

    for name, df in sheets:
        df2 = _sanitize_df(df)
        # keamanan tambahan: scan sel index-like (seharusnya kosong)
        issues = _scan_indexlikes(df2, name, limit=5)
        if issues:
            sample = "\n".join([f"- {r[0]} r={r[1]} c='{r[2]}' type={r[3]} prev={r[4]}" for r in issues])
            wb.close()
            raise RuntimeError(f"Non-scalar cells detected in '{name}':\n{sample}")

        ws = wb.add_worksheet(str(name)[:31])  # Excel sheet name max 31 chars
        _write_sheet(ws, df2, header_fmt=fmt_header, date_fmt=fmt_date, num_fmt=fmt_num)

        # Optional: auto width (sederhana)
        for j, col in enumerate(df2.columns):
            width = max(len(str(col)), min(40, int(df2[col].astype(str).str.len().max() or 10)))
            try:
                ws.set_column(j, j, width)
            except Exception:
                pass

    wb.close()
    buf.seek(0)
    return buf.getvalue()


# ============================ Public API ============================

def make_export_xlsx(
    *,
    df_forecast_nat: pd.DataFrame,
    reconciled_monthly_leaf: pd.DataFrame,
    wh_with_ss: pd.DataFrame,
    scope: str,
    grain_output: str,
    weekly_df: Optional[pd.DataFrame],
    sku_attrs: Optional[pd.DataFrame],
    params: dict,
    coverage_ratio: Optional[float] = None,
    engine: str = "xlsxwriter"  # paksa xlsxwriter untuk manual writer
) -> bytes:
    # Build raw sheets
    raw1 = build_sheet1_wh_month(wh_with_ss, sku_attrs)
    raw2 = build_sheet2_region_kota(scope, grain_output, reconciled_monthly_leaf, weekly_df, sku_attrs)
    raw3 = build_sheet3_summary(df_forecast_nat, reconciled_monthly_leaf, scope, grain_output, params, coverage_ratio)

    # Susun & tulis manual
    sheets = [
        ("Alokasi WH", raw1),
        ("Alokasi Region & Kota", raw2),
        ("Summary Reports", raw3),
    ]
    # Tambah sheet debug ringkas (opsional; aman karena sudah stringified)
    dbg = []
    for nm, df in sheets:
        dbg.append({"sheet": nm, "rows": int(len(df)), "cols": int(len(df.columns))})
    debug_df = pd.DataFrame(dbg, columns=["sheet","rows","cols"])
    sheets.append(("Debug", debug_df))

    return _write_workbook_xlsxwriter(sheets)

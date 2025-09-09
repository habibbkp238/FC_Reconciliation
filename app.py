# app.py
import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# ==== Domain modules ====
from src.io_contracts import read_inputs
from src.preprocess import preprocess_history
from src.allocation import compute_allocation_keys
from src.topdown import topdown_split
from src.reconcile import reconcile_forecast
from src.weekly import weekly_split
from src.safetystock import compute_safety_stock_wh
from src.rounding import coherent_round
from src.templates import (
    make_forecast_template,
    make_service_template,
    make_historical_template,
    make_output_example_xlsx,
    df_to_csv_bytes,
)
from src.exporter import make_export_xlsx


# =============================================================================
# Page Config & Theme
# =============================================================================
st.set_page_config(
    page_title="Demand Forecasting — Steps 1–8",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Minimal pro styling (SAP/Kinaxis vibe)
st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 2.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background: #f6f7f9; border-radius: 8px; padding: 10px 12px; font-weight: 600;
        border: 1px solid #e5e7eb;
    }
    .stTabs [aria-selected="true"] {
        background: #ffffff !important; border-color: #d1d5db !important; color: #111827 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .metric-card {
        border: 1px solid #e5e7eb; border-radius: 14px; padding: 14px 16px; background: #fff;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .chip {
        display: inline-block; padding: 2px 10px; border-radius: 999px; border: 1px solid #e5e7eb;
        background: #f9fafb; font-size: 12px; font-weight: 600; color: #374151;
        margin-left: 8px;
    }
    .soft { color: #6b7280; font-size: 12px; margin-top: -6px; }
    .subtle { color: #4b5563; font-size: 13px; }
    .primary-btn button[kind="primary"] { border-radius: 12px !important; font-weight: 700 !important; }
    .danger-btn button {
        border-radius: 10px !important; background: #fee2e2 !important; color: #991b1b !important; border: 1px solid #fecaca !important;
    }
    .good { color: #059669; font-weight: 700; }
    .bad { color: #b91c1c; font-weight: 700; }
    .neutral { color: #6b7280; font-weight: 700; }
    hr { margin: 1.0rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Helpers (incl. Export sanitizers)
# =============================================================================
def _as_month(ts):
    return pd.to_datetime(ts).dt.to_period("M").dt.to_timestamp()


def build_presence_lists(hist_monthly: pd.DataFrame):
    g1 = (
        hist_monthly.groupby(["sku"], dropna=False)["wh"]
        .apply(lambda s: sorted(pd.Series(s).dropna().unique().tolist()))
        .reset_index(name="wh_list")
    )
    g2 = (
        hist_monthly.groupby(["sku", "wh"], dropna=False)["region"]
        .apply(lambda s: sorted(pd.Series(s).dropna().unique().tolist()))
        .reset_index(name="region_list")
    ) if "region" in hist_monthly.columns else pd.DataFrame(columns=["sku","wh","region_list"])
    g3 = (
        hist_monthly.groupby(["sku", "wh", "region"], dropna=False)["kota"]
        .apply(lambda s: sorted(pd.Series(s).dropna().unique().tolist()))
        .reset_index(name="kota_list")
    ) if set(["region","kota"]).issubset(hist_monthly.columns) else pd.DataFrame(columns=["sku","wh","region","kota_list"])
    return {"wh_per_sku": g1, "region_per_sku_wh": g2, "kota_per_sku_wh_region": g3}


def _kpi_row():
    ss = st.session_state
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown('<div class="metric-card">📄<br><span class="subtle">Forecast rows</span><h3 style="margin:2px 0;">{:,}</h3></div>'.format(
            len(ss.get("df_forecast", pd.DataFrame()))
        ), unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">📦<br><span class="subtle">Hist rows</span><h3 style="margin:2px 0;">{:,}</h3></div>'.format(
            len(ss.get("df_hist_raw", pd.DataFrame()))
        ), unsafe_allow_html=True)
    with col3:
        cov = ss.get("coverage_ratio")
        cov_txt = "-" if cov is None else f"{cov:.1%}"
        st.markdown(f'<div class="metric-card">🧩<br><span class="subtle">Coverage (month, sku)</span><h3 style="margin:2px 0;">{cov_txt}</h3></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card">🎯<br><span class="subtle">Scope</span><h4 style="margin:6px 0;">{ss.get("scope","-")}</h4></div>', unsafe_allow_html=True)
    with col5:
        grain = ss.get("grain_output","-")
        st.markdown(f'<div class="metric-card">📆<br><span class="subtle">Output grain</span><h4 style="margin:6px 0;">{grain}</h4></div>', unsafe_allow_html=True)
    with col6:
        int_on = ss.get("int_toggle")
        label = "ON" if int_on else "OFF"
        color = "good" if int_on else "neutral"
        st.markdown(f'<div class="metric-card">🔢<br><span class="subtle">Integerize</span><h4 class="{color}" style="margin:6px 0;">{label}</h4></div>', unsafe_allow_html=True)


def _guard_inputs(f_forecast, f_service, f_hist):
    missing = []
    if not f_forecast: missing.append("Forecast Nasional")
    if not f_service:  missing.append("Service Params")
    if not f_hist:     missing.append("Historical + Attributes")
    if missing:
        st.error("Mohon upload 3 file wajib: " + ", ".join(missing))
        st.stop()


def _coherence_close(a: pd.Series, b: pd.Series, tol: float) -> bool:
    return bool((a.fillna(0) - b.fillna(0)).abs().lt(tol).all())


# ---------- Export helpers: sanitize to avoid "unsupported Type RangeIndex" ----------
def _flatten_multiindex(obj):
    if isinstance(obj, pd.MultiIndex):
        return pd.Index([" ".join([str(x) for x in tup if x is not None]) for tup in obj.values])
    return obj


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    df = df.copy()
    df.index = range(len(df))
    df.reset_index(drop=True, inplace=True)
    df.columns = _flatten_multiindex(df.columns)

    for col in df.columns:
        s = df[col]
        if isinstance(s.dtype, pd.PeriodDtype):
            df[col] = s.astype(str); continue
        if np.issubdtype(s.dtype, np.timedelta64):
            df[col] = s.astype(str); continue
        if s.dtype.name == "category":
            df[col] = s.astype(str); continue
        if np.issubdtype(s.dtype, np.datetime64):
            try:
                if getattr(s.dt, 'tz', None) is not None:
                    df[col] = s.dt.tz_convert(None)
            except Exception:
                pass
            continue
        if s.dtype == object:
            def _x(v):
                if isinstance(v, (pd.Index, pd.RangeIndex, pd.MultiIndex, pd.Series, np.ndarray, list, dict, set, tuple)):
                    return str(v)
                return v
            df[col] = s.map(_x)
    return df


def sanitize_dict_of_dfs(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, pd.DataFrame):
            out[k] = sanitize_df(v)
    return out


def to_excel_bytes_xw(dfs: dict, meta: dict | None = None) -> bytes:
    """Simple fallback writer using xlsxwriter dengan autosize + freeze."""
    import xlsxwriter  # ensure engine available
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter", datetime_format="yyyy-mm-dd", date_format="yyyy-mm-dd") as writer:
        def _autosize(ws, df):
            ws.freeze_panes(1, 0)
            for i, c in enumerate(df.columns):
                m = max(len(str(c)), *(df[c].astype(str).map(len).tolist() or [0]))
                ws.set_column(i, i, min(m + 2, 60))
        if meta:
            mdf = sanitize_df(pd.DataFrame({"Field": list(meta.keys()), "Value": list(meta.values())}))
            mdf.to_excel(writer, index=False, sheet_name="_meta")
            _autosize(writer.sheets["_meta"], mdf)
        for name, df in dfs.items():
            if df is None or df.empty:
                continue
            safe = str(name)[:31].replace("/", "-").replace("\\", "-")
            sdf = sanitize_df(df)
            sdf.to_excel(writer, index=False, sheet_name=safe)
            _autosize(writer.sheets[safe], sdf)
    buf.seek(0)
    return buf.getvalue()


def preflight_indexlike(df: pd.DataFrame, name: str, limit: int = 20):
    bad = []
    if df is None or df.empty:
        return bad
    for col in df.columns:
        ser = df[col]
        for i, v in ser.items():
            if isinstance(v, (pd.Index, pd.RangeIndex, pd.MultiIndex, pd.Series, np.ndarray)):
                bad.append((name, int(i), str(col), type(v).__name__, str(v)[:160]))
                if len(bad) >= limit:
                    return bad
    return bad


# =============================================================================
# Header
# =============================================================================
st.title("📦Forecast Allocation/Reconciliation")
st.caption("Top-Down → Reconcile → Weekly (opsional) → Safety Stock → Export")

# Top bar chips
tstamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
env = "Production"
st.markdown(
    f'<span class="chip">⏱ {tstamp}</span> '
    f'<span class="chip">🧭 {env}</span>',
    unsafe_allow_html=True
)
st.markdown("<hr/>", unsafe_allow_html=True)


# =============================================================================
# Sidebar — Uploads, Parameters, Output
# =============================================================================
with st.sidebar:
    st.header("📥 Upload Data (Wajib)")
    st.caption("Historical harus **harian** untuk akurasi weekly split & safety stock.")
    f_forecast = st.file_uploader("Forecast Nasional — kolom: date (awal bulan), sku, qty", type=["csv", "xlsx"])
    f_service  = st.file_uploader("Service Params — kolom: sku, **wh (opsional)**, lead_time_days, service_level_target", type=["csv", "xlsx"])
    f_hist     = st.file_uploader("Historical + Attributes (harian) — kolom: date, sku, sku_name, category, class, brand, extension, wh, region, kota, qty", type=["csv", "xlsx"])

    st.markdown("---")
    st.subheader("⚙️ Parameters")
    last_n        = st.number_input("Filter histori (N bulan, 0=all)", min_value=0, value=int(st.session_state.get("last_n", 12)), step=1)
    vol_threshold = st.slider("Volatility threshold (WH share)", 0.00, 0.50, float(st.session_state.get("vol_threshold", 0.05)), 0.01)
    scope         = st.selectbox("Scope output", ["WH only", "WH → Region", "WH → Region → Kota"],
                                 index=["WH only","WH → Region","WH → Region → Kota"].index(st.session_state.get("scope","WH → Region → Kota")))

    st.markdown("---")
    recon_method  = st.selectbox("Reconciliation method", ["OLS", "WLS (diag)", "MinT (shrink)"],
                                 index=["OLS","WLS (diag)","MinT (shrink)"].index(st.session_state.get("recon_method","OLS")))
    mint_lambda   = st.slider("MinT λ (0=full sample Σ, 1=diagonal)", 0.00, 1.00, float(st.session_state.get("mint_lambda", 0.50)), 0.05)

    st.markdown("---")
    # Grain output + weekly modes
    grain_options = ["Monthly (no weekly split)", "Weekly (Month-Sliced, 5 angka)", "Weekly (W-MON)"]
    grain_default = st.session_state.get("grain_output", "Weekly (Month-Sliced, 5 angka)")
    grain_output  = st.selectbox("Grain output", grain_options, index=grain_options.index(grain_default) if grain_default in grain_options else 0)

    weekly_mode = "MonthSliced" if grain_output.startswith("Weekly (Month-Sliced") else ("W-MON" if grain_output.endswith("W-MON)") else None)

    # Strategy for Month-Sliced, and method for W-MON (kept for compat)
    weekly_strategy = None
    weekly_method = None
    if weekly_mode == "MonthSliced":
        weekly_strategy = st.selectbox("Month-Sliced strategy",
                                        ["DOW × Week-of-Month (robust)", "DOW (historical)", "Pro-rata Mon–Sat"],
                                        index=["DOW × Week-of-Month (robust)", "DOW (historical)", "Pro-rata Mon–Sat"].index(
                                            st.session_state.get("weekly_strategy","DOW × Week-of-Month (robust)")))
    elif weekly_mode == "W-MON":
        weekly_method = st.selectbox("Weekly method (W-MON)", ["Hybrid", "Pro-rata"],
                                     index=["Hybrid","Pro-rata"].index(st.session_state.get("weekly_method","Hybrid")))
    else:
        weekly_strategy = st.session_state.get("weekly_strategy")
        weekly_method = st.session_state.get("weekly_method")

    st.markdown("---")
    int_toggle = st.checkbox("Output as integers (coherent rounding)", value=bool(st.session_state.get("int_toggle", True)))
    int_method = st.selectbox("Integer rounding method", ["Largest remainder (Hamilton)", "Stochastic"],
                              index=["Largest remainder (Hamilton)","Stochastic"].index(st.session_state.get("int_method","Largest remainder (Hamilton)")))

    st.markdown("---")
    ss_ma = st.number_input("Safety Stock — MA window for σ (months)", min_value=2, max_value=12, value=int(st.session_state.get("ss_ma",3)), step=1)
    ss_round_up = st.checkbox("Safety Stock — round up (ceil)", value=bool(st.session_state.get("ss_round_up", True)))

    st.markdown("---")
    st.subheader("📤 Export Settings")
    engine_prefer = st.selectbox("Excel engine", ["Auto", "xlsxwriter", "openpyxl"], index=0)
    st.session_state["engine_prefer"] = engine_prefer
    run_btn = st.button("🚀 Run Steps 1–7", use_container_width=True, type="primary")
    reset_btn = st.button("♻️ Reset State", use_container_width=True)

    if reset_btn:
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.experimental_rerun()

    st.markdown("---")
    with st.expander("📥 Download Templates", expanded=False):
        st.caption("Pilih apakah ingin contoh baris data (bukan hanya header).")
        sample_rows = st.checkbox("Include sample rows", value=True)

        df_temp_forecast = make_forecast_template(sample=sample_rows)
        df_temp_service  = make_service_template(sample=sample_rows)  # per (sku, wh)
        df_temp_hist     = make_historical_template(sample=sample_rows)

        st.download_button("⬇️ Forecast Nasional — Template CSV",
                           data=df_to_csv_bytes(df_temp_forecast),
                           file_name="forecast_national_template.csv",
                           mime="text/csv",
                           use_container_width=True)
        st.download_button("⬇️ Service Params (SKU–WH) — Template CSV",
                           data=df_to_csv_bytes(df_temp_service),
                           file_name="service_params_template.csv",
                           mime="text/csv",
                           use_container_width=True)
        st.download_button("⬇️ Historical + Attributes (Daily) — Template CSV",
                           data=df_to_csv_bytes(df_temp_hist),
                           file_name="historical_with_attributes_template.csv",
                           mime="text/csv",
                           use_container_width=True)
        st.download_button("⬇️ Contoh Output Excel (3 sheet)",
                           data=make_output_example_xlsx(),
                           file_name="forecast_allocation_output_example.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)


# =============================================================================
# Tabs
# =============================================================================
tabs = st.tabs([
    "Step 1: Preview & DQ",
    "Step 2: Preprocess",
    "Step 3: Allocation Keys",
    "Step 4: Top-Down Split",
    "Step 5: Reconciliation",
    "Step 6: Weekly Split (opsional)",
    "Step 7: Safety Stock (WH)",
    "Step 8: Export",
])


# =============================================================================
# Pipeline (Run)
# =============================================================================
if run_btn:
    _guard_inputs(f_forecast, f_service, f_hist)

    progress = st.progress(0, text="Initializing…")

    try:
        # ---- Step 1
        progress.progress(5, text="Step 1 — Membaca & memvalidasi file…")
        df_forecast, df_service, df_hist_raw, warns = read_inputs(f_forecast, f_service, f_hist)

        dq_raw = {
            "forecast_unique_month_sku": int(df_forecast.drop_duplicates(["date", "sku"]).shape[0]),
            "hist_unique_key_raw": int(df_hist_raw.drop_duplicates(["date","sku","wh","region","kota"], keep="first").shape[0]) if set(["region","kota"]).issubset(df_hist_raw.columns) else int(df_hist_raw.drop_duplicates(["date","sku","wh"], keep="first").shape[0]),
            "hist_pct_zero_qty_raw": float((df_hist_raw["qty"] == 0).mean()),
        }

        # ---- Step 2
        progress.progress(22, text="Step 2 — Canonical daily/weekly/monthly + presence + DQ…")
        hist_weekly, hist_monthly, presence_preview, dq_post, hist_daily = preprocess_history(df_hist_raw, last_n_months=int(last_n))

        # Coverage (month, sku) yang punya histori > 0 (bulan yang sama)
        fm = df_forecast.copy()
        fm["month"] = _as_month(fm["date"])
        hm = hist_monthly.groupby(["month","sku"], dropna=False)["qty"].sum().reset_index()
        cov = fm.merge(hm, on=["month","sku"], how="left", suffixes=("_f","_h"))
        coverage = float((cov["qty_h"].fillna(0) > 0).mean()) if len(cov) else 0.0

        # ---- Step 3
        progress.progress(40, text="Step 3 — Allocation keys…")
        presence_full = build_presence_lists(hist_monthly)
        shares = compute_allocation_keys(
            hist_monthly=hist_monthly,
            presence=presence_full,
            vol_threshold=float(vol_threshold),
            window_12m=12,
            window_3m=3,
        )

        # ---- Step 4
        progress.progress(56, text=f"Step 4 — Top-Down split ({scope}) …")
        baseline = topdown_split(
            df_forecast=df_forecast,
            shares=shares,
            scope=scope,
            presence=presence_full,
            hist_monthly=hist_monthly,
        )

        # Conservation check (Σchild = National per (month, sku))
        fm2 = df_forecast.copy(); fm2["month"] = _as_month(fm2["date"])
        b2  = baseline.copy();    b2["month"]  = _as_month(b2["date"])
        sum_b = b2.groupby(["month","sku"], dropna=False)["forecast_baseline"].sum().reset_index()
        chk = fm2.merge(sum_b, on=["month","sku"], how="left")
        baseline_conservation_ok = _coherence_close(chk["qty"], chk["forecast_baseline"], 1e-6)

        # ---- Step 5
        progress.progress(70, text=f"Step 5 — Reconcile ({recon_method}, λ={float(mint_lambda):.2f}) …")
        recon = reconcile_forecast(
            df_forecast=df_forecast,
            baseline=baseline,
            hist_monthly=hist_monthly,
            scope=scope,
            method=recon_method,
            mint_lambda=float(mint_lambda),
        )
        rec_final = recon["final"].copy()

        # Coherence checks (NaN-aware)
        fm3 = df_forecast.copy(); fm3["month"] = _as_month(fm3["date"])

        if "wh" in recon and "wh" in recon["wh"].columns:
            wh = recon["wh"].copy(); wh["month"] = _as_month(wh["month"])
            sum_wh = wh.groupby(["month","sku"], dropna=False)["forecast_reconciled"].sum().reset_index()
            chk_wh = fm3.merge(sum_wh, on=["month","sku"], how="left")
            ok_wh = _coherence_close(chk_wh["qty"], chk_wh["forecast_reconciled"], 1e-6)
        else:
            ok_wh = True

        if "region" in recon and "wh" in recon:
            rr = recon["region"].copy(); rr["month"] = _as_month(rr["month"])
            sum_rr = rr.groupby(["month","sku","wh"], dropna=False)["forecast_reconciled"].sum().reset_index()
            tgt_rr = recon["wh"].rename(columns={"forecast_reconciled":"tgt"})[["month","sku","wh","tgt"]]
            chk_rr = tgt_rr.merge(sum_rr, on=["month","sku","wh"], how="left")
            ok_rr = _coherence_close(chk_rr["tgt"], chk_rr["forecast_reconciled"], 1e-6)
        else:
            ok_rr = (scope == "WH only")

        if "kota" in recon and "region" in recon:
            kk = recon["kota"].copy(); kk["month"] = _as_month(kk["month"])
            sum_kk = kk.groupby(["month","sku","wh","region"], dropna=False)["forecast_reconciled"].sum().reset_index()
            tgt_kk = recon["region"].rename(columns={"forecast_reconciled":"tgt"})[["month","sku","wh","region","tgt"]]
            chk_kk = tgt_kk.merge(sum_kk, on=["month","sku","wh","region"], how="left")
            ok_kk = _coherence_close(chk_kk["tgt"], chk_kk["forecast_reconciled"], 1e-6)
        else:
            ok_kk = (scope != "WH → Region → Kota")

        # Integerize monthly (optional)
        if bool(int_toggle):
            fm_int = df_forecast.copy()
            fm_int["date"] = _as_month(fm_int["date"])
            fm_int["qty_int_target"] = fm_int["qty"].round().astype(int)

            rec_final["date"] = _as_month(rec_final["date"])
            rec_final = rec_final.merge(
                fm_int[["date","sku","qty_int_target"]], on=["date","sku"], how="left"
            )
            method_map = {"Largest remainder (Hamilton)": "lrm", "Stochastic": "stochastic"}
            rec_final = coherent_round(
                df=rec_final,
                value_col="forecast_reconciled",
                group_cols=["date","sku"],
                target_col="qty_int_target",
                method=method_map[int_method],
                out_col="forecast_reconciled"
            ).drop(columns=["qty_int_target"])

        # ---- Step 6 (Weekly)
        if weekly_mode is not None and grain_output.startswith("Weekly"):
            label_mode = "Month-Sliced" if weekly_mode == "MonthSliced" else "W-MON"
            progress.progress(82, text=f"Step 6 — Weekly split ({label_mode})…")

            if weekly_mode == "MonthSliced":
                 # map ke kunci yang dipakai weekly.py
                if weekly_strategy.startswith("DOW × Week-of-Month"):
                    strat_key = "DOWxWOM"
                elif weekly_strategy.startswith("DOW"):
                    strat_key = "DOW"
                else:
                    strat_key = "ProRataWorkingDays"

                weekly_df = weekly_split(
                    reconciled_final=rec_final,
                    hist_weekly=hist_weekly,  # tidak dipakai di MonthSliced
                    method=weekly_method or "Hybrid",
                    lookback_months=int(last_n) if int(last_n) > 0 else None,
                    mode="MonthSliced",
                    strategy=strat_key,
                    hist_daily=hist_daily,
                    wom_alpha=0.6,        # kekuatan WOM (0..1)
                    wom_min_months=6,     # jumlah bulan untuk capai alpha penuh
                )

            # Σweekly = monthly (conservation)
            wf = weekly_df.copy(); wf["month"] = _as_month(wf["date"])
            rec_m = rec_final.copy(); rec_m["month"] = _as_month(rec_m["date"])
            rec_m = rec_m.rename(columns={"forecast_reconciled": "qty_month"})

            key = ["month","sku","wh"]
            if "region" in weekly_df.columns: key.append("region")
            if "kota"   in weekly_df.columns: key.append("kota")

            sum_w = wf.groupby(key, dropna=False)["forecast_weekly"].sum().reset_index().rename(columns={"forecast_weekly":"sum_week"})
            chk_w = rec_m.merge(sum_w, on=key, how="left")
            ok_week = _coherence_close(chk_w["qty_month"], chk_w["sum_week"], 1e-6)

            if bool(int_toggle):
                weekly_int = weekly_df.copy()
                weekly_int["month"] = _as_month(weekly_int["date"])
                target_leaf = rec_final.copy()
                target_leaf["month"] = _as_month(target_leaf["date"])
                target_leaf = target_leaf.rename(columns={"forecast_reconciled": "qty_int_target"})
                method_map = {"Largest remainder (Hamilton)": "lrm", "Stochastic": "stochastic"}
                weekly_int = weekly_int.merge(target_leaf[key + ["qty_int_target"]], on=key, how="left")
                weekly_int = coherent_round(
                    df=weekly_int,
                    value_col="forecast_weekly",
                    group_cols=key,
                    target_col="qty_int_target",
                    method=method_map[int_method],
                    out_col="forecast_weekly"
                ).drop(columns=["qty_int_target"])
                sum_wi = weekly_int.groupby(key, dropna=False)["forecast_weekly"].sum().reset_index(name="sum_week")
                chk_i = target_leaf[key + ["qty_int_target"]].merge(sum_wi, on=key, how="left")
                ok_week_int = _coherence_close(chk_i["qty_int_target"], chk_i["sum_week"], 1e-9)
                weekly_df = weekly_int
            else:
                ok_week_int = None
        else:
            weekly_df = None
            ok_week = None
            ok_week_int = None

        # ---- Step 7 — Safety Stock (SKU–WH; tidak memengaruhi weekly)
        progress.progress(92, text="Step 7 — Safety Stock (WH)…")
        ss_wh = compute_safety_stock_wh(
            hist_monthly=hist_monthly,
            service_params=df_service,
            ma_window=int(ss_ma),
            round_up=bool(ss_round_up),
            hist_daily=hist_daily,      # jalur harian (prioritas)
            week_anchor="W-MON",
        )
        # Apply ke monthly WH untuk display (tanpa mengubah weekly split)
        rec_wh = rec_final.groupby(["date","sku","wh"], dropna=False)["forecast_reconciled"].sum().reset_index()
        wh_month_with_ss = rec_wh.merge(ss_wh[["sku","wh","safety_stock_week_int"]], on=["sku","wh"], how="left")
        wh_month_with_ss["safety_stock_week_int"] = wh_month_with_ss["safety_stock_week_int"].fillna(0).astype(int)

        # ---- Persist state
        st.session_state.update({
            "df_forecast": df_forecast,
            "df_service": df_service,
            "df_hist_raw": df_hist_raw,
            "warns": warns,
            "dq_raw": dq_raw,

            "hist_daily": hist_daily,
            "hist_weekly": hist_weekly,
            "hist_monthly": hist_monthly,
            "presence_preview": presence_preview,
            "dq_post": dq_post,
            "coverage_ratio": coverage,

            "presence_full": presence_full,
            "shares": shares,
            "baseline": baseline,
            "baseline_conservation_ok": baseline_conservation_ok,

            "reconciled": rec_final,
            "ok_wh": ok_wh, "ok_rr": ok_rr, "ok_kk": ok_kk,

            "weekly": weekly_df,
            "ok_week": ok_week, "ok_week_int": ok_week_int,
            "weekly_method": weekly_method,
            "weekly_mode": weekly_mode,
            "weekly_strategy": weekly_strategy,

            "ss_wh": ss_wh,
            "wh_with_ss": wh_month_with_ss,

            "scope": scope,
            "recon_method": recon_method,
            "mint_lambda": float(mint_lambda),
            "vol_threshold": float(vol_threshold),
            "grain_output": grain_output,
            "int_toggle": bool(int_toggle),
            "int_method": int_method,
            "ss_ma": int(ss_ma),
            "ss_round_up": bool(ss_round_up),
            "last_n": int(last_n),

            "pipeline_ready": True,
            "last_run_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        progress.progress(100, text="Selesai ✅")
        st.success("✅ Step 7 selesai. Lanjut ke Export.")

    except Exception as e:
        st.error(f"❌ Error: {e}")


# =============================================================================
# Render tabs from state
# =============================================================================
def render_tabs_from_state(tabs):
    ss = st.session_state
    if not ss.get("pipeline_ready"):
        with tabs[0]:
            st.info("Upload data & jalankan **Run Steps 1–7** untuk melihat preview dan metrik.")
        return

    # KPI Row
    _kpi_row()

    # ------- Tab 1 -------
    with tabs[0]:
        st.subheader("📄 Forecast Nasional (date, sku, qty)")
        st.dataframe(ss["df_forecast"].head(20), use_container_width=True)
        st.caption(f"{len(ss['df_forecast']):,} rows • total qty: {float(ss['df_forecast']['qty'].sum()):,.0f}")

        st.subheader("🛠️ Service Params (SKU atau SKU–WH)")
        st.dataframe(ss["df_service"].head(20), use_container_width=True)
        st.caption(f"{len(ss['df_service']):,} rows • kolom wajib: sku, lead_time_days, service_level_target; kolom wh opsional")

        st.subheader("📦 Historical + Attributes (raw, daily)")
        st.dataframe(ss["df_hist_raw"].head(20), use_container_width=True)
        st.caption(f"{len(ss['df_hist_raw']):,} rows")

        st.divider()
        st.subheader("🔍 Data Quality (ringkas — raw)")
        st.json(ss.get("dq_raw", {}))
        for w in ss.get("warns", []): st.warning(w)

    # ------- Tab 2 -------
    with tabs[1]:
        st.subheader("📅 Hist Daily — preview")
        st.dataframe(ss["hist_daily"].head(20), use_container_width=True)
        st.caption(f"{len(ss['hist_daily']):,} rows")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📅 Hist Weekly (W-MON) — preview")
            st.dataframe(ss["hist_weekly"].head(20), use_container_width=True)
            st.caption(f"{len(ss['hist_weekly']):,} rows")
        with c2:
            st.subheader("🗓️ Hist Monthly — preview")
            st.dataframe(ss["hist_monthly"].head(20), use_container_width=True)
            st.caption(f"{len(ss['hist_monthly']):,} rows")

        st.divider()
        st.subheader("🧭 Presence Maps (ringkas)")
        if "presence_preview" in ss:
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.markdown("**WH per SKU**")
                st.dataframe(ss["presence_preview"]["wh_per_sku"].head(20), use_container_width=True)
            with col_b:
                st.markdown("**Region per SKU–WH**")
                st.dataframe(ss["presence_preview"]["region_per_sku_wh"].head(20), use_container_width=True)
            with col_c:
                st.markdown("**Kota per SKU–WH–Region**")
                st.dataframe(ss["presence_preview"]["kota_per_sku_wh_region"].head(20), use_container_width=True)

        st.divider()
        st.subheader("🧪 DQ Metrics (post-canonical)")
        st.json(ss.get("dq_post", {}))
        if "coverage_ratio" in ss:
            st.write(f"Coverage: **{ss['coverage_ratio']:.1%}** dari pasangan (month, sku) forecast memiliki histori pada bulan yang sama.")

    # ------- Tab 3 -------
    with tabs[2]:
        st.subheader("🔑 WH Shares per SKU")
        st.dataframe(ss["shares"]["wh"].head(50), use_container_width=True); st.caption(f"{len(ss['shares']['wh']):,} rows")
        if "region" in ss["shares"]:
            st.subheader("🌐 Region Shares per (SKU, WH)")
            st.dataframe(ss["shares"]["region"].head(50), use_container_width=True); st.caption(f"{len(ss['shares']['region']):,} rows")
        if "kota" in ss["shares"]:
            st.subheader("🏙️ Kota Shares per (SKU, WH, Region)")
            st.dataframe(ss["shares"]["kota"].head(50), use_container_width=True); st.caption(f"{len(ss['shares']['kota']):,} rows")
        st.divider()
        st.subheader("✅ Sanity Check (sum = 1 per parent)")
        wh_ok = ss["shares"]["wh"].groupby("sku")["share"].sum().round(6).between(0.999999,1.000001).all()
        reg_ok = ss["shares"].get("region", pd.DataFrame()).groupby(["sku","wh"])["share"].sum().round(6).between(0.999999,1.000001).all() if "region" in ss["shares"] else True
        kota_ok = ss["shares"].get("kota", pd.DataFrame()).groupby(["sku","wh","region"])["share"].sum().round(6).between(0.999999,1.000001).all() if "kota" in ss["shares"] else True
        st.write({"WH sum per SKU": bool(wh_ok), "Region sum per (SKU,WH)": bool(reg_ok), "Kota sum per (SKU,WH,Region)": bool(kota_ok)})

    # ------- Tab 4 -------
    with tabs[3]:
        st.subheader("📊 Baseline Forecast (hasil Top-Down)")
        st.dataframe(ss["baseline"].head(50), use_container_width=True)
        st.caption(f"{len(ss['baseline']):,} rows • scope: {ss['scope']}")
        st.write({"Conservation Σchild = National (per month, sku)": bool(ss.get("baseline_conservation_ok", True))})

    # ------- Tab 5 -------
    with tabs[4]:
        st.subheader("🤝 Reconciled Forecast (monthly)")
        st.dataframe(ss["reconciled"].head(50), use_container_width=True)
        lam = float(ss.get("mint_lambda", 0.0))
        st.caption(f"{len(ss['reconciled']):,} rows • scope: {ss['scope']} • method: {ss['recon_method']} • λ={lam:.2f}")
        st.divider()
        st.subheader("✅ Coherence Checks")
        st.write({
            "ΣWH = National": bool(ss.get("ok_wh", True)),
            "ΣRegion = WH": bool(ss.get("ok_rr", True)),
            "ΣKota = Region": bool(ss.get("ok_kk", True))
        })

    # ------- Tab 6 -------
    with tabs[5]:
        if ss.get("weekly") is not None:
            title = "📆 Weekly Forecast"
            mode = ss.get("weekly_mode") or ("W-MON" if ss.get("grain_output","").endswith("W-MON)") else "Month-Sliced")
            strat = ss.get("weekly_strategy")
            if mode == "MonthSliced":
                title += " — Month-Sliced (5 angka)"
                if strat:
                    title += f" • {strat}"
            else:
                wm = ss.get("weekly_method")
                title += f" — W-MON • method: {wm}"
            st.subheader(title)
            st.dataframe(ss["weekly"].head(50), use_container_width=True)
            st.caption(f"{len(ss['weekly']):,} rows")
            if "ok_week" in ss:
                st.write({"Conservation Σweekly = monthly reconciled": bool(ss["ok_week"])})
            if "ok_week_int" in ss and ss["ok_week_int"] is not None:
                st.write({"Conservation (integer) Σweekly = monthly": bool(ss["ok_week_int"])})
        else:
            st.info("Weekly split **tidak** dipilih. Output tetap **monthly**.")

    # ------- Tab 7 -------
    with tabs[6]:
        st.subheader("🛡️ Safety Stock per SKU–WH (basis mingguan, tidak memengaruhi weekly split)")
        ss_wh = ss["ss_wh"]
        cols_pref = ["sku","wh","sigma_week","z","lead_time_days","safety_stock_week","safety_stock_week_int"]
        cols_show = [c for c in cols_pref if c in ss_wh.columns]
        st.dataframe(ss_wh[cols_show].head(50), use_container_width=True)
        st.caption(
            f"{len(ss_wh):,} rows • SS_week = z × σ_week × √(LT_days/7) • "
            f"{'ceil' if ss['ss_round_up'] else 'round'} to int"
        )
        st.subheader("📄 Monthly WH + Safety Stock (display only)")
        st.dataframe(ss["wh_with_ss"].head(50), use_container_width=True)
        st.caption("Kolom: date (month), sku, wh, forecast_reconciled, safety_stock_week_int • SS tidak dicampur ke weekly forecast")

    # ------- Tab 8 -------
    with tabs[7]:
        st.header("📤 Step 8 — Export Excel (RangeIndex-safe)")
        st.caption("Membuat beberapa sheet. Export otomatis sanitize DataFrame: drop index, flatten kolom, dan stringifikasi objek index-like.")
        col1, col2 = st.columns([1, 4])
        with col1:
            export_btn = st.button("💾 Build Excel", use_container_width=True, type="primary")
        with col2:
            st.write(
                f"Terakhir run: **{ss.get('last_run_ts', '-')}** • "
                f"Scope: **{ss.get('scope','-')}** • Grain: **{ss.get('grain_output','-')}** • Engine pref: **{st.session_state.get('engine_prefer','Auto')}**"
            )

        if export_btn:
            try:
                # SKU attributes (optional)
                df_hist_raw = ss["df_hist_raw"]
                attr_cols = [c for c in ["sku","sku_name","category","class","brand","extension"] if c in df_hist_raw.columns]
                sku_attrs = df_hist_raw[attr_cols].drop_duplicates("sku").copy() if attr_cols else None

                # --- PREFLIGHT: deteksi sel yang berisi objek index-like ---
                issues = []
                issues += preflight_indexlike(ss["wh_with_ss"], "Alokasi WH")
                issues += preflight_indexlike(ss["reconciled"], "Reconciled (monthly)")
                if ss.get("weekly") is not None:
                    issues += preflight_indexlike(ss["weekly"], "Weekly")
                if issues:
                    st.warning("Preflight menemukan sel yang mengandung objek index-like. Akan di-stringify otomatis saat export. Berikut contoh (maks 20):")
                    st.dataframe(pd.DataFrame(issues, columns=["sheet","row","col","pytype","preview"]))

                # "Auto" -> prefer xlsxwriter
                eng_map = {"Auto": "xlsxwriter", "xlsxwriter": "xlsxwriter", "openpyxl": "openpyxl"}
                chosen_engine = eng_map.get(st.session_state.get("engine_prefer","Auto"), "xlsxwriter")

                # --- Coba exporter bawaan (dengan input yang sudah sanitized) ---
                try:
                    xlsx_bytes = make_export_xlsx(
                        df_forecast_nat=sanitize_df(ss["df_forecast"]),
                        reconciled_monthly_leaf=sanitize_df(ss["reconciled"]),
                        wh_with_ss=sanitize_df(ss["wh_with_ss"]),
                        scope=ss["scope"],
                        grain_output=ss["grain_output"],
                        weekly_df=sanitize_df(ss.get("weekly")) if ss.get("weekly") is not None else None,
                        sku_attrs=sanitize_df(sku_attrs) if sku_attrs is not None else None,
                        params={
                            "recon_method": ss["recon_method"],
                            "mint_lambda": ss["mint_lambda"],
                            "vol_threshold": ss["vol_threshold"],
                            "weekly_mode": ss.get("weekly_mode"),
                            "weekly_strategy": ss.get("weekly_strategy"),
                            "weekly_method": ss.get("weekly_method"),
                            "int_toggle": ss["int_toggle"],
                            "int_method": ss["int_method"],
                            "ss_ma": ss["ss_ma"],
                            "ss_round_up": ss["ss_round_up"],
                        },
                        coverage_ratio=ss.get("coverage_ratio"),
                        engine=chosen_engine,
                    )
                except Exception as e_exporter:
                    # --- Fallback: build workbook sederhana tapi lengkap ---
                    st.warning(f"Exporter bawaan gagal: {e_exporter}. Menggunakan fallback writer.")
                    sheets = {
                        "Forecast_National": ss["df_forecast"],
                        "Reconciled_Monthly": ss["reconciled"],
                        "WH_with_SS": ss["wh_with_ss"],
                    }
                    if ss.get("weekly") is not None:
                        sheets["Weekly"] = ss["weekly"]
                    if sku_attrs is not None:
                        sheets["SKU_Attributes"] = sku_attrs
                    sheets = sanitize_dict_of_dfs(sheets)
                    meta = {
                        "Generated At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Scope": ss.get("scope"),
                        "Grain": ss.get("grain_output"),
                        "Engine": chosen_engine,
                        "App Export": "rangeindex-safe-1.2",
                    }
                    xlsx_bytes = to_excel_bytes_xw(sheets, meta)

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"forecast_allocation_{ts}.xlsx"
                st.download_button(
                    "⬇️ Download Excel",
                    data=xlsx_bytes,
                    file_name=fname,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                st.success("File siap diunduh.")
            except Exception as e:
                st.error(f"Export gagal: {e}")


# ---- Render (even after rerun)
render_tabs_from_state(tabs)

# Footer
st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("© Demand Forecasting — FMCG • Built with Streamlit • Designed by Irsandi Habibie/Demnand Planning")

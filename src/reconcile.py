import numpy as np
import pandas as pd
from typing import Dict, Tuple

EPS = 1e-9

# ============== utilities ==============

def _ensure_month_col(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    x = df.copy()
    if "month" not in x.columns:
        if date_col in x.columns:
            x["month"] = pd.to_datetime(x[date_col]).dt.to_period("M").dt.to_timestamp()
        else:
            raise ValueError("Expected 'month' or a parsable date column.")
    else:
        x["month"] = pd.to_datetime(x["month"]).dt.to_period("M").dt.to_timestamp()
    return x

def _pivot_cov(hist_monthly: pd.DataFrame, parent_filter: Dict[str, str], child_col: str) -> pd.DataFrame:
    x = hist_monthly.copy()
    for k, v in parent_filter.items():
        x = x[x[k] == v]
    x = _ensure_month_col(x, "month" if "month" in x.columns else "date")
    pv = x.pivot_table(index="month", columns=child_col, values="qty", aggfunc="sum").sort_index()
    if pv.shape[0] < 2:
        # tidak cukup observasi -> kembalikan matriks nol
        return pd.DataFrame(np.zeros((pv.shape[1], pv.shape[1])), index=pv.columns, columns=pv.columns)
    cov = pv.cov(min_periods=2).fillna(0.0)
    return cov

def _mint_adjust(b: np.ndarray, T: float, Sigma: np.ndarray, method: str, lam: float = 0.5) -> np.ndarray:
    k = b.shape[0]
    if k == 1:
        return np.array([T], dtype=float)

    b = np.where(np.isfinite(b), b, 0.0)
    s_b = float(b.sum())
    if abs(T - s_b) <= 1e-12:
        return b

    if method == "OLS":
        Sigma_use = np.eye(k, dtype=float)
    elif method == "WLS (diag)":
        d = np.diag(Sigma) if Sigma.size else np.ones(k)
        d = np.where(np.isfinite(d) & (d > 0), d, 1.0)
        Sigma_use = np.diag(d)
    else:  # MinT (shrink)
        if Sigma.size == 0:
            Sigma = np.eye(k, dtype=float)
        D = np.diag(np.clip(np.diag(Sigma), a_min=EPS, a_max=None))
        Sigma_use = lam * D + (1.0 - lam) * Sigma + 1e-6 * np.eye(k, dtype=float)

    ones = np.ones((k, 1), dtype=float)
    denom = float(ones.T @ Sigma_use @ ones)
    if not np.isfinite(denom) or denom <= EPS:
        return b * (T / s_b) if s_b > 0 else np.ones(k) * (T / k)

    adj = (T - s_b) * (Sigma_use @ ones / denom).ravel()
    y = b + adj
    y = np.where(y < 0, 0.0, y)
    s = y.sum()
    return (y * (T / s)) if s > 0 else np.ones(k) * (T / k)

def _reconcile_one_parent(base_df: pd.DataFrame, child_col: str, target_total: float,
                          Sigma: pd.DataFrame, method: str, lam: float) -> pd.DataFrame:
    ch = base_df[child_col].tolist()
    b = base_df["base"].to_numpy(dtype=float)

    # Ambil Σ sesuai urutan child
    if Sigma is not None and len(Sigma) > 0:
        # extend jika ada child yang belum ada di Σ
        missing = [c for c in ch if c not in Sigma.index]
        if missing:
            ext = pd.DataFrame(np.zeros((len(missing), len(Sigma.columns))), index=missing, columns=Sigma.columns)
            Sigma = pd.concat([Sigma, ext], axis=0)
            ext2 = pd.DataFrame(np.zeros((len(Sigma.index), len(missing))), index=Sigma.index, columns=missing)
            Sigma = pd.concat([Sigma, ext2], axis=1)
            for m in missing:
                Sigma.loc[m, m] = 1.0
        Sigma = Sigma.loc[ch, ch]
    else:
        Sigma = pd.DataFrame(np.eye(len(ch)), index=ch, columns=ch)

    y = _mint_adjust(b, float(target_total), Sigma.values, method, lam)
    out = base_df.copy()
    out["reconciled"] = y
    return out

# ============== main ==============

def reconcile_forecast(
    df_forecast: pd.DataFrame,     # (date, sku, qty) monthly
    baseline: pd.DataFrame,        # hasil topdown (scope apa pun)
    hist_monthly: pd.DataFrame,    # canonical monthly from Step 2
    scope: str,
    method: str = "OLS",
    mint_lambda: float = 0.5,
) -> Dict[str, pd.DataFrame]:
    # -------- guards --------
    if df_forecast is None:
        raise ValueError("df_forecast is None (cek Step 1).")
    if baseline is None:
        raise ValueError("baseline is None (cek Step 4 — Top-Down Split).")
    if hist_monthly is None:
        raise ValueError("hist_monthly is None (cek Step 2).")

    out: Dict[str, pd.DataFrame] = {}

    # Pastikan month ada
    f = _ensure_month_col(df_forecast, "date").rename(columns={"qty": "target_nat"})
    b = baseline.copy()
    b["month"] = pd.to_datetime(b["date"]).dt.to_period("M").dt.to_timestamp()

    # ---------- 1) WH vs National ----------
    b_wh = b.groupby(["month", "sku", "wh"], dropna=False)["forecast_baseline"].sum().reset_index()
    target_nat = f[["month", "sku", "target_nat"]]

    # precompute cov antar WH per SKU
    cov_wh_map: Dict[str, pd.DataFrame] = {}
    for sku in b_wh["sku"].unique():
        cov_wh_map[sku] = _pivot_cov(hist_monthly[hist_monthly["sku"] == sku], {}, "wh")

    rows = []
    for (m, sku), g in b_wh.groupby(["month", "sku"]):
        T = float(target_nat.loc[(target_nat["month"] == m) & (target_nat["sku"] == sku), "target_nat"].sum())
        Sigma = cov_wh_map.get(sku, pd.DataFrame())
        out_g = _reconcile_one_parent(
            base_df=g.rename(columns={"forecast_baseline": "base"}),
            child_col="wh",
            target_total=T,
            Sigma=Sigma,
            method=method,
            lam=mint_lambda
        )
        rows.append(out_g)
    wh_rec = pd.concat(rows, ignore_index=True)
    wh_rec = wh_rec.rename(columns={"reconciled": "forecast_reconciled"}).drop(columns=["base"])
    out["wh"] = wh_rec.copy()

    if scope == "WH only":
        out["final"] = wh_rec[["month", "sku", "wh", "forecast_reconciled"]].rename(columns={"month": "date"})
        return out

    # ---------- 2) Region vs WH ----------
    if "region" in b.columns:
        b_reg = b.groupby(["month", "sku", "wh", "region"], dropna=False)["forecast_baseline"].sum().reset_index()
    else:
        out["final"] = wh_rec[["month", "sku", "wh", "forecast_reconciled"]].rename(columns={"month": "date"})
        return out

    target_wh = wh_rec.rename(columns={"forecast_reconciled": "target_wh"})[["month", "sku", "wh", "target_wh"]]

    cov_reg_map: Dict[Tuple[str, str], pd.DataFrame] = {}
    for (sku, wh), _ in b_reg.groupby(["sku", "wh"]):
        cov_reg_map[(sku, wh)] = _pivot_cov(hist_monthly, {"sku": sku, "wh": wh}, "region")

    rows = []
    for (m, sku, wh), g in b_reg.groupby(["month", "sku", "wh"]):
        T = float(target_wh.loc[(target_wh["month"] == m) & (target_wh["sku"] == sku) & (target_wh["wh"] == wh), "target_wh"].sum())
        Sigma = cov_reg_map.get((sku, wh), pd.DataFrame())
        out_g = _reconcile_one_parent(
            base_df=g.rename(columns={"forecast_baseline": "base"}),
            child_col="region",
            target_total=T,
            Sigma=Sigma,
            method=method,
            lam=mint_lambda
        )
        rows.append(out_g)
    reg_rec = pd.concat(rows, ignore_index=True)
    reg_rec = reg_rec.rename(columns={"reconciled": "forecast_reconciled"}).drop(columns=["base"])
    out["region"] = reg_rec.copy()

    if scope == "WH → Region":
        out["final"] = reg_rec[["month", "sku", "wh", "region", "forecast_reconciled"]].rename(columns={"month": "date"})
        return out

    # ---------- 3) Kota vs Region ----------
    if "kota" not in b.columns:
        out["final"] = reg_rec[["month", "sku", "wh", "region", "forecast_reconciled"]].rename(columns={"month": "date"})
        return out

    b_kota = b.groupby(["month", "sku", "wh", "region", "kota"], dropna=False)["forecast_baseline"].sum().reset_index()
    target_reg = reg_rec.rename(columns={"forecast_reconciled": "target_reg"})[["month", "sku", "wh", "region", "target_reg"]]

    cov_city_map: Dict[Tuple[str, str, str], pd.DataFrame] = {}
    for (sku, wh, reg), _ in b_kota.groupby(["sku", "wh", "region"]):
        cov_city_map[(sku, wh, reg)] = _pivot_cov(hist_monthly, {"sku": sku, "wh": wh, "region": reg}, "kota")

    rows = []
    for (m, sku, wh, reg), g in b_kota.groupby(["month", "sku", "wh", "region"]):
        T = float(target_reg.loc[
            (target_reg["month"] == m) & (target_reg["sku"] == sku) &
            (target_reg["wh"] == wh) & (target_reg["region"] == reg),
            "target_reg"
        ].sum())
        Sigma = cov_city_map.get((sku, wh, reg), pd.DataFrame())
        out_g = _reconcile_one_parent(
            base_df=g.rename(columns={"forecast_baseline": "base"}),
            child_col="kota",
            target_total=T,
            Sigma=Sigma,
            method=method,
            lam=mint_lambda
        )
        rows.append(out_g)
    kota_rec = pd.concat(rows, ignore_index=True)
    kota_rec = kota_rec.rename(columns={"reconciled": "forecast_reconciled"}).drop(columns=["base"])
    out["kota"] = kota_rec.copy()

    out["final"] = kota_rec[["month", "sku", "wh", "region", "kota", "forecast_reconciled"]].rename(columns={"month": "date"})
    return out

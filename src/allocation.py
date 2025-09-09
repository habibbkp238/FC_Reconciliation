import pandas as pd
import numpy as np
from typing import Dict, List

EPS = 1e-12

def _window_cutoff(max_month: pd.Timestamp, window_months: int) -> pd.Timestamp:
    # contoh: max=2025-06-01, window=12 -> cutoff 2024-07-01
    return (max_month.to_period("M") - (window_months - 1)).to_timestamp()

def _windowed_share(df_m: pd.DataFrame, parent_cols: List[str], child_col: str, window_months: int) -> pd.DataFrame:
    """Hitung share anak dalam jendela N bulan terakhir: qty_child / qty_parent."""
    x = df_m.copy()
    max_m = x["month"].max()
    if pd.isna(max_m):
        return pd.DataFrame(columns=parent_cols + [child_col, "share"])
    cutoff = _window_cutoff(max_m, window_months)
    x = x[x["month"] >= cutoff]

    # agregat qty per parent-child
    num = (
        x.groupby(parent_cols + [child_col], dropna=False)["qty"]
        .sum()
        .reset_index()
        .rename(columns={"qty": "qty_child"})
    )
    den = (
        x.groupby(parent_cols, dropna=False)["qty"]
        .sum()
        .reset_index()
        .rename(columns={"qty": "qty_parent"})
    )
    m = num.merge(den, on=parent_cols, how="left")
    m["share"] = np.where(m["qty_parent"] > 0, m["qty_child"] / (m["qty_parent"] + EPS), 0.0)
    return m[parent_cols + [child_col, "share"]]

def _normalize_or_equal(df: pd.DataFrame, group_cols: List[str], share_col: str = "share") -> pd.DataFrame:
    """Normalisasi share per parent; jika total 0/NaN -> equal split pada anak yang ada."""
    def _norm(g: pd.DataFrame):
        s = g[share_col].sum()
        if not np.isfinite(s) or s <= EPS:
            n = len(g)
            g[share_col] = 1.0 / n if n > 0 else np.nan
        else:
            g[share_col] = g[share_col] / s
        return g
    return df.groupby(group_cols, group_keys=False).apply(_norm)

def _ensure_children_from_presence(df_share: pd.DataFrame,
                                   parent_cols: List[str],
                                   child_col: str,
                                   presence_df: pd.DataFrame,
                                   presence_list_col: str) -> pd.DataFrame:
    """
    Pastikan setiap parent punya baris untuk SEMUA child yang pernah muncul (presence).
    presence_df harus mengandung parent_cols + [list] berisi daftar child.
    """
    rows = []
    # bangun dict -> list child
    pres_map = {}
    for _, r in presence_df.iterrows():
        key = tuple(r[c] for c in parent_cols)
        pres_map[key] = r[presence_list_col] if isinstance(r[presence_list_col], list) else []
    # generate baris yang hilang
    for key, childs in pres_map.items():
        # sub-share existing
        mask = np.logical_and.reduce([df_share[c].values == key[i] for i, c in enumerate(parent_cols)]) if len(df_share) else np.array([], dtype=bool)
        sub = df_share.loc[mask] if len(df_share) else pd.DataFrame(columns=df_share.columns)
        existing = set(sub[child_col].tolist())
        for c in childs:
            if c not in existing:
                rows.append(dict(**{parent_cols[i]: key[i] for i in range(len(parent_cols))}, **{child_col: c, "share": 0.0}))
    if rows:
        df_share = pd.concat([df_share, pd.DataFrame(rows)], ignore_index=True)
    return df_share

def compute_allocation_keys(
    hist_monthly: pd.DataFrame,
    presence: Dict[str, pd.DataFrame],
    vol_threshold: float = 0.05,
    window_12m: int = 12,
    window_3m: int = 3,
) -> Dict[str, pd.DataFrame]:
    """
    Hitung allocation keys sesuai blueprint:
      - WH: blend 12m/3m berbasis volatilitas (abs delta share >= threshold -> bobot 3m lebih besar)
      - Region & Kota: blend 50/50 (12m & 3m)
      - Fallback equal split pada child yang pernah muncul (pakai presence maps)
    Input:
      hist_monthly: kolom minimal ["month","sku","wh","region","kota","qty"]
      presence: dict dari Step 2: wh_per_sku, region_per_sku_wh, kota_per_sku_wh_region
    Output:
      dict: {"wh": df_wh, "region": df_region, "kota": df_kota}
    """
    dfm = hist_monthly.copy()
    # ========== WH shares per SKU ==========
    s12 = _windowed_share(dfm, ["sku"], "wh", window_12m)
    s03 = _windowed_share(dfm, ["sku"], "wh", window_3m)
    wh = s12.merge(s03, on=["sku", "wh"], how="outer", suffixes=("_12", "_3")).fillna(0.0)

    # Volatility proxy dan blending
    wh["vol"] = (wh["share_12"] - wh["share_3"]).abs()
    wh["w3"] = np.where(wh["vol"] >= vol_threshold, 0.7, 0.3)
    wh["share"] = wh["w3"] * wh["share_3"] + (1 - wh["w3"]) * wh["share_12"]
    wh = wh[["sku", "wh", "share"]]

    # Tambahkan child yang hilang berdasarkan presence (agar equal split bisa bekerja)
    pres_wh = presence["wh_per_sku"][["sku", "wh_list"]].rename(columns={"wh_list": "child_list"})
    wh = _ensure_children_from_presence(wh, ["sku"], "wh", pres_wh, "child_list")
    # Normalisasi / equal
    wh = _normalize_or_equal(wh, ["sku"], "share")

    # ========== Region shares per (SKU, WH) ==========
    r12 = _windowed_share(dfm, ["sku", "wh"], "region", window_12m)
    r03 = _windowed_share(dfm, ["sku", "wh"], "region", window_3m)
    reg = r12.merge(r03, on=["sku", "wh", "region"], how="outer", suffixes=("_12", "_3")).fillna(0.0)
    reg["share"] = 0.5 * reg["share_12"] + 0.5 * reg["share_3"]
    reg = reg[["sku", "wh", "region", "share"]]

    pres_reg = presence["region_per_sku_wh"][["sku", "wh", "region_list"]].rename(columns={"region_list": "child_list"})
    reg = _ensure_children_from_presence(reg, ["sku", "wh"], "region", pres_reg, "child_list")
    reg = _normalize_or_equal(reg, ["sku", "wh"], "share")

    # ========== Kota shares per (SKU, WH, Region) ==========
    k12 = _windowed_share(dfm, ["sku", "wh", "region"], "kota", window_12m)
    k03 = _windowed_share(dfm, ["sku", "wh", "region"], "kota", window_3m)
    kota = k12.merge(k03, on=["sku", "wh", "region", "kota"], how="outer", suffixes=("_12", "_3")).fillna(0.0)
    kota["share"] = 0.5 * kota["share_12"] + 0.5 * kota["share_3"]
    kota = kota[["sku", "wh", "region", "kota", "share"]]

    pres_kota = presence["kota_per_sku_wh_region"][["sku", "wh", "region", "kota_list"]].rename(columns={"kota_list": "child_list"})
    kota = _ensure_children_from_presence(kota, ["sku", "wh", "region"], "kota", pres_kota, "child_list")
    kota = _normalize_or_equal(kota, ["sku", "wh", "region"], "share")

    # (opsional) sort rapi
    wh   = wh.sort_values(["sku", "wh"]).reset_index(drop=True)
    reg  = reg.sort_values(["sku", "wh", "region"]).reset_index(drop=True)
    kota = kota.sort_values(["sku", "wh", "region", "kota"]).reset_index(drop=True)

    return {"wh": wh, "region": reg, "kota": kota}

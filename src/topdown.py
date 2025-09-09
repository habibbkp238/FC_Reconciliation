import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

EPS = 1e-12

def _equal_rows(parent_keys: List[Tuple], parent_cols: List[str], child_col: str, child_list_map: Dict, global_child_list: List[str]):
    """Bangun baris equal-split untuk parent yang tidak punya share anak."""
    rows = []
    for key in parent_keys:
        # ambil list child dari presence; kalau kosong -> pakai global list
        lst = child_list_map.get(key, None)
        if not lst:
            lst = global_child_list
        if not lst:
            continue
        w = 1.0 / len(lst)
        base = {parent_cols[i]: key[i] for i in range(len(parent_cols))}
        for c in lst:
            rows.append({**base, child_col: c, "share": w})
    return rows

def _complete_with_fallback(
    df_share: pd.DataFrame,
    parent_cols: List[str],
    child_col: str,
    presence_df: pd.DataFrame,
    presence_list_col: str,
    global_child_list: List[str],
    parent_universe: pd.DataFrame,
):
    """
    Lengkapi shares dengan fallback equal untuk parent tanpa baris,
    menggunakan daftar child dari presence (atau global list jika tidak ada).
    """
    # presence map: (parent tuple) -> [child,...]
    pres_map = {
        tuple(r[c] for c in parent_cols): r[presence_list_col]
        for _, r in presence_df.iterrows()
    }
    # cari parent yang ada di parent_universe tapi tidak ada baris di df_share
    have_parent = set(tuple(r) for r in df_share[parent_cols].drop_duplicates().itertuples(index=False, name=None))
    all_parent = set(tuple(r) for r in parent_universe[parent_cols].drop_duplicates().itertuples(index=False, name=None))
    missing = sorted(list(all_parent - have_parent))
    if missing:
        rows = _equal_rows(missing, parent_cols, child_col, pres_map, global_child_list)
        if rows:
            df_share = pd.concat([df_share, pd.DataFrame(rows)], ignore_index=True)
    # normalisasi (jaga equal kalau sum=0)
    def _norm(g):
        s = g["share"].sum()
        if not np.isfinite(s) or s <= EPS:
            g["share"] = 1.0 / len(g)
        else:
            g["share"] = g["share"] / s
        return g
    return df_share.groupby(parent_cols, group_keys=False).apply(_norm)

def topdown_split(
    df_forecast: pd.DataFrame,          # cols: date, sku, qty (monthly)
    shares: Dict[str, pd.DataFrame],    # dict: wh/region/kota dengan kolom 'share'
    scope: str,                         # 'WH only' | 'WH → Region' | 'WH → Region → Kota'
    presence: Dict[str, pd.DataFrame],  # presence lists dari Step 3
    hist_monthly: pd.DataFrame,         # untuk global fallback lists
) -> pd.DataFrame:
    """
    Hasilkan baseline forecast sesuai scope dengan mengalikan shares.
    Fallback equal-split bila suatu parent tak punya child share (SKU baru, dll).
    """
    # ---------- level WH ----------
    wh_share = shares["wh"].copy()  # sku, wh, share
    # parent universe untuk WH = semua sku di forecast
    parent_wh = df_forecast[["sku"]].drop_duplicates().copy()
    # global WH list
    global_wh = sorted(hist_monthly["wh"].dropna().unique().tolist())
    wh_presence = presence["wh_per_sku"].rename(columns={"wh_list": "child_list"})
    wh_share = _complete_with_fallback(
        wh_share, parent_cols=["sku"], child_col="wh",
        presence_df=wh_presence[["sku", "child_list"]],
        presence_list_col="child_list",
        global_child_list=global_wh,
        parent_universe=parent_wh
    )
    df_wh = (
        df_forecast.merge(wh_share, on="sku", how="left")
        .assign(forecast_wh=lambda d: d["qty"] * d["share"])
        .drop(columns=["share"])
    )

    if scope == "WH only":
        return df_wh.rename(columns={"forecast_wh": "forecast_baseline"})[["date","sku","wh","forecast_baseline"]]

    # ---------- level Region ----------
    reg_share = shares["region"].copy()  # sku, wh, region, share
    # parent universe untuk Region = (sku, wh) yang ada di df_wh
    parent_reg = df_wh[["sku","wh"]].drop_duplicates()
    # global region list per WH
    global_regions = (
        hist_monthly.groupby("wh", dropna=False)["region"]
        .apply(lambda s: sorted(pd.Series(s).dropna().unique().tolist()))
        .to_dict()
    )
    reg_presence = presence["region_per_sku_wh"].rename(columns={"region_list":"child_list"})
    # Per WH, kita butuh list global; bungkus helper untuk dipakai dalam _complete_with_fallback
    # kita perlebar parent_universe dengan kolom 'child_list_global' berupa list per baris
    parent_reg = parent_reg.copy()
    parent_reg["child_list_global"] = parent_reg["wh"].map(lambda w: global_regions.get(w, []))
    # _complete_with_fallback butuh satu global list, jadi kalau beda WH, kita lakukan per WH
    pieces = []
    for wh_code, parent_sub in parent_reg.groupby("wh"):
        sub_share = reg_share[reg_share["wh"] == wh_code].copy()
        pieces.append(_complete_with_fallback(
            sub_share, parent_cols=["sku","wh"], child_col="region",
            presence_df=reg_presence[reg_presence["wh"]==wh_code][["sku","wh","child_list"]],
            presence_list_col="child_list",
            global_child_list=global_regions.get(wh_code, []),
            parent_universe=parent_sub[["sku","wh"]]
        ))
    reg_share_filled = pd.concat(pieces, ignore_index=True) if pieces else reg_share
    df_reg = (
        df_wh.merge(reg_share_filled, on=["sku","wh"], how="left")
             .assign(forecast_region=lambda d: d["forecast_wh"] * d["share"])
             .drop(columns=["share"])
    )

    if scope == "WH → Region":
        return df_reg.rename(columns={"forecast_region":"forecast_baseline"})[["date","sku","wh","region","forecast_baseline"]]

    # ---------- level Kota ----------
    kota_share = shares["kota"].copy()  # sku, wh, region, kota, share
    parent_kota = df_reg[["sku","wh","region"]].drop_duplicates()
    # global kota list per (wh, region)
    global_kota = (
        hist_monthly.groupby(["wh","region"], dropna=False)["kota"]
        .apply(lambda s: sorted(pd.Series(s).dropna().unique().tolist()))
        .to_dict()
    )
    kota_presence = presence["kota_per_sku_wh_region"].rename(columns={"kota_list":"child_list"})
    pieces = []
    for (wh_code, region), parent_sub in parent_kota.groupby(["wh","region"]):
        sub = kota_share[(kota_share["wh"]==wh_code) & (kota_share["region"]==region)].copy()
        pieces.append(_complete_with_fallback(
            sub, parent_cols=["sku","wh","region"], child_col="kota",
            presence_df=kota_presence[(kota_presence["wh"]==wh_code) & (kota_presence["region"]==region)][["sku","wh","region","child_list"]],
            presence_list_col="child_list",
            global_child_list=global_kota.get((wh_code, region), []),
            parent_universe=parent_sub[["sku","wh","region"]]
        ))
    kota_share_filled = pd.concat(pieces, ignore_index=True) if pieces else kota_share
    df_kota = (
        df_reg.merge(kota_share_filled, on=["sku","wh","region"], how="left")
              .assign(forecast_kota=lambda d: d["forecast_region"] * d["share"])
              .drop(columns=["share"])
    )

    return df_kota.rename(columns={"forecast_kota":"forecast_baseline"})[
        ["date","sku","wh","region","kota","forecast_baseline"]
    ]

# src/rounding.py
from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd

EPS = 1e-12


def coherent_round(
    df: pd.DataFrame,
    value_col: str,
    group_cols: List[str],
    target_col: Optional[str] = None,     # target integer per grup; kalau None → round(sum(value_col))
    method: str = "lrm",                  # "lrm" (Largest Remainder / Hamilton) atau "stochastic"
    random_state: int = 42,
    out_col: Optional[str] = None         # default: overwrite value_col
) -> pd.DataFrame:
    """
    Bulatkan nilai ke integer per grup dengan total = target (koheren).
    - Jika target_col ada → gunakan integer pada kolom tsb (tiap grup satu nilai).
      Jika sebagian/semua target per grup NaN → fallback ke round(sum(value_col)) per grup.
    - Jika target_col None → target = round(sum(value_col)) per grup.

    Metode:
      - "lrm": alokasi via Largest Remainder (Hamilton) berdasarkan proporsi nilai.
      - "stochastic": multinomial sampling berdasarkan proporsi nilai (seedable).
    """
    if df is None or df.empty:
        return df

    out_col = out_col or value_col
    d = df.copy()

    # Pastikan numeric
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce").fillna(0.0)

    # Sum per grup (untuk fallback & proporsi)
    gsum = (
        d.groupby(group_cols, dropna=False)[value_col]
         .sum()
         .reset_index(name="_sum_val")
    )

    # Siapkan target per grup (DataFrame agar aman untuk join — no .loc with NaN in index)
    if target_col is not None and target_col in d.columns:
        gtgt = (
            d.groupby(group_cols, dropna=False)[target_col]
             .first()
             .reset_index(name="_target_raw")
        )
    else:
        gtgt = pd.DataFrame(columns=group_cols + ["_target_raw"])

    tgt = gsum.merge(gtgt, on=group_cols, how="left")
    # Fallback: jika _target_raw NaN → gunakan round(_sum_val)
    tgt["_target_float"] = pd.to_numeric(tgt["_target_raw"], errors="coerce")
    tgt["_target_float"] = tgt["_target_float"].where(tgt["_target_float"].notna(), tgt["_sum_val"].round())
    # Bersihkan nilai aneh
    tgt["_target_float"] = tgt["_target_float"].fillna(0.0)
    tgt["_target_float"] = np.where(np.isfinite(tgt["_target_float"]), tgt["_target_float"], 0.0)
    # Final target integer >= 0
    tgt["_target_int"] = np.clip(np.round(tgt["_target_float"]).astype(int), 0, None)
    tgt = tgt[group_cols + ["_target_int"]]

    # Gabungkan target ke baris
    d = d.merge(tgt, on=group_cols, how="left")
    d["_target_int"] = d["_target_int"].fillna(0).astype(int)

    # Alokasi per grup
    def _alloc_group(g: pd.DataFrame) -> pd.DataFrame:
        total = int(g["_target_int"].iloc[0])
        vals = g[value_col].to_numpy(dtype=float)
        n = len(vals)
        if n == 0:
            return g

        s = float(vals.sum())
        if s <= EPS:
            # Semua 0 → proporsi rata
            p = np.full(n, 1.0 / n, dtype=float)
        else:
            p = vals / s

        if method.lower().startswith("stoch"):
            rng = np.random.RandomState(random_state)  # deterministik untuk keseluruhan; bisa diubah per key jika perlu
            alloc = rng.multinomial(total, p) if total > 0 else np.zeros(n, dtype=int)
        else:
            # Largest Remainder (Hamilton)
            shares = p * total
            base = np.floor(shares).astype(int)
            remain = int(total - base.sum())
            if remain > 0:
                # Urutkan sisa terbesar; mergesort menjaga stabilitas
                order = np.argsort(-(shares - base), kind="mergesort")
                base[order[:remain]] += 1
            alloc = base

        g[out_col] = alloc
        return g

    d = (
        d.groupby(group_cols, dropna=False, as_index=False, sort=False)
         .apply(_alloc_group)
         .reset_index(drop=True)
    )

    # Rapikan kolom bantu
    d = d.drop(columns=[c for c in ["_sum_val", "_target_raw", "_target_float", "_target_int"] if c in d.columns], errors="ignore")
    return d

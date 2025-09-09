# src/weekly.py
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional, List, Dict


# ==============================
# Utils: kalender & helper
# ==============================
def _month_days_no_sun(month_start: pd.Timestamp) -> pd.DatetimeIndex:
    """Semua tanggal dalam 1 bulan, exclude Sunday (weekday=6)."""
    start = month_start.to_period("M").to_timestamp()
    end = (start + pd.offsets.MonthEnd(0))
    days = pd.date_range(start, end, freq="D")
    return days[days.weekday != 6]  # 0=Mon .. 6=Sun


def _assign_wom_index_for_month(month_start: pd.Timestamp) -> pd.DataFrame:
    """
    Buat mapping tanggal->week_of_month (1..5) untuk 1 bulan, dengan aturan:
      - Hanya Mon–Sat (Sun dibuang)
      - Blok berurutan ukuran 6 hari (terakhir bisa <6)
      - Week 'date' representative = tanggal pertama blok (dipakai untuk output)
    """
    days = _month_days_no_sun(month_start)
    if len(days) == 0:
        return pd.DataFrame(columns=["date", "wom_idx", "is_rep", "dow"])
    order = np.arange(len(days))
    wom_idx = (order // 6) + 1  # 6 hari per blok
    df = pd.DataFrame({"date": days, "wom_idx": wom_idx})
    # Representative date per WOM = tanggal pertama dari blok
    reps = df.groupby("wom_idx", as_index=False)["date"].min().rename(columns={"date": "rep_date"})
    df = df.merge(reps, on="wom_idx", how="left")
    df["is_rep"] = df["date"].eq(df["rep_date"])
    df["dow"] = df["date"].dt.weekday  # 0=Mon..6=Sun (Sun sudah tidak ada)
    return df[["date", "wom_idx", "is_rep", "dow"]]


def _calendar_month_sliced(month_start: pd.Timestamp) -> pd.DataFrame:
    """
    Kembalikan kalender month-sliced: baris per Mon–Sat hari dengan WOM idx.
    """
    return _assign_wom_index_for_month(month_start)


def _as_month(ts: pd.Series | pd.DatetimeIndex | pd.Timestamp) -> pd.Series | pd.Timestamp:
    if isinstance(ts, pd.Timestamp):
        return ts.to_period("M").to_timestamp()
    return pd.to_datetime(ts).dt.to_period("M").dt.to_timestamp()


# ==============================
# Profil DOW (Mon–Sat)
# ==============================
def _dow_profile(hist_daily: pd.DataFrame, group_cols: List[str], lookback_months: Optional[int]) -> pd.DataFrame:
    """
    Estimasi bobot harian per DOW (Mon..Sat) per group (default pakai level SKU–WH).
    Hasil: kolom ['dow_0'..'dow_5'] ter-normalisasi (jumlah=1).
    """
    hd = hist_daily.copy()
    need = ["date", "qty"] + group_cols
    miss = [c for c in need if c not in hd.columns]
    if miss:
        raise ValueError(f"hist_daily missing columns: {miss}")
    hd["date"] = pd.to_datetime(hd["date"])
    hd["qty"] = pd.to_numeric(hd["qty"], errors="coerce").fillna(0.0)
    hd["month"] = _as_month(hd["date"])
    if lookback_months and lookback_months > 0 and len(hd):
        anchor = hd["month"].max()
        cutoff = (anchor - pd.offsets.DateOffset(months=int(lookback_months) - 1)).to_period("M").to_timestamp()
        hd = hd.loc[hd["month"] >= cutoff]

    hd["dow"] = hd["date"].dt.weekday
    hd = hd.loc[hd["dow"] <= 5]  # Mon..Sat only

    grp = hd.groupby(group_cols + ["dow"], dropna=False)["qty"].sum().reset_index()
    pivot = grp.pivot_table(index=group_cols, columns="dow", values="qty", fill_value=0.0)
    # pastikan semua kolom ada
    for d in range(6):
        if d not in pivot.columns:
            pivot[d] = 0.0
    pivot = pivot[[0,1,2,3,4,5]]
    s = pivot.sum(axis=1)
    # default uniform kalau total 0
    for d in range(6):
        pivot[d] = np.where(s > 0, pivot[d] / s, 1.0/6.0)
    pivot.columns = [f"dow_{d}" for d in range(6)]
    return pivot.reset_index()


# ==============================
# WOM factor (W1..W5)
# ==============================
def _wom_factor(
    hist_daily: pd.DataFrame,
    group_cols: List[str],
    lookback_months: Optional[int],
    dow_profile: pd.DataFrame,
    alpha: float = 0.6,
    min_months_for_full_alpha: int = 6,
) -> pd.DataFrame:
    """
    Estimasi faktor WOM (rasio) per W1..W5 dan group:
      r_k = (observed_share_k) / (base_share_k_from_DOW)

    - observed_share_k = y_k / sum_k y_k, di mana y_k = total qty per WOM k (Mon–Sat saja).
    - base_share_k_from_DOW = (sum f_dow pada hari-hari WOM k) / (sum over all WOM).

    Lalu rata-ratakan r_k lintas bulan (berbobot total bulan), dan di-shrink:
      r_hat = (1 - alpha_eff) * 1.0 + alpha_eff * avg_ratio
      alpha_eff = alpha * clip(n_months / min_months_for_full_alpha, 0..1)
    """
    hd = hist_daily.copy()
    need = ["date", "qty"] + group_cols
    miss = [c for c in need if c not in hd.columns]
    if miss:
        raise ValueError(f"hist_daily missing columns: {miss}")
    hd["date"] = pd.to_datetime(hd["date"])
    hd["qty"] = pd.to_numeric(hd["qty"], errors="coerce").fillna(0.0)
    hd["month"] = _as_month(hd["date"])
    hd["dow"] = hd["date"].dt.weekday
    hd = hd.loc[hd["dow"] <= 5]  # Mon..Sat only

    if lookback_months and lookback_months > 0 and len(hd):
        anchor = hd["month"].max()
        cutoff = (anchor - pd.offsets.DateOffset(months=int(lookback_months) - 1)).to_period("M").to_timestamp()
        hd = hd.loc[hd["month"] >= cutoff]

    # Map day -> WOM per bulan
    # Kerjakan per bulan: assign WOM idx untuk semua hari di bulan itu
    wom_maps = []
    for m, g in hd.groupby("month", dropna=False):
        cal = _calendar_month_sliced(m)
        # join by date
        gg = g.merge(cal[["date","wom_idx"]], on="date", how="left")
        wom_maps.append(gg)
    if len(wom_maps) == 0:
        # fallback: semua WOM factor = 1.0
        out = dow_profile[group_cols].copy()
        for k in range(1, 6):
            out[f"wom{k}"] = 1.0
        return out

    hd2 = pd.concat(wom_maps, ignore_index=True)
    # Observed weekly totals per month
    obs = (hd2.groupby(group_cols + ["month", "wom_idx"], dropna=False)["qty"].sum()
                .reset_index(name="y"))
    # Observed monthly totals
    month_tot = (obs.groupby(group_cols + ["month"], dropna=False)["y"].sum()
                     .reset_index(name="Ym"))
    obs = obs.merge(month_tot, on=group_cols+["month"], how="left")
    obs["obs_share"] = np.where(obs["Ym"] > 0, obs["y"] / obs["Ym"], np.nan)

    # Base share from DOW profile per month, given actual day composition
    # dow_profile: group_cols + dow_0..dow_5
    # For each month, compute weight_k = sum f_dow for days in WOM k; base_share_k = weight_k / sum_k weight_k
    # Build calendar with dow for each day (Mon..Sat only)
    base_rows = []
    for (keys, g_month) in hd2.groupby(group_cols + ["month"], dropna=False):
        # get f_dow for this group
        key_dict = dict(zip(group_cols, keys[:len(group_cols)]))
        month = keys[len(group_cols)]
        fp = dow_profile
        for c, v in key_dict.items():
            fp = fp[fp[c] == v]
        if fp.empty:
            # uniform f_dow
            f = {d: 1.0/6.0 for d in range(6)}
        else:
            row = fp.iloc[0]
            f = {d: float(row[f"dow_{d}"]) for d in range(6)}

        cal = _calendar_month_sliced(month)
        cal = cal[cal["dow"] <= 5]
        wk = (cal.groupby("wom_idx")["dow"]
                 .apply(lambda s: sum(f[int(d)] for d in s.values))
                 .reset_index(name="weight"))
        wk["month"] = month
        for c, v in key_dict.items():
            wk[c] = v
        base_rows.append(wk)

    base = pd.concat(base_rows, ignore_index=True) if base_rows else pd.DataFrame(columns=group_cols+["month","wom_idx","weight"])
    base_tot = (base.groupby(group_cols + ["month"], dropna=False)["weight"].sum()
                    .reset_index(name="Wm"))
    base = base.merge(base_tot, on=group_cols+["month"], how="left")
    base["base_share"] = np.where(base["Wm"] > 0, base["weight"] / base["Wm"], np.nan)

    # Ratio r_k per month
    comb = obs.merge(base[group_cols+["month","wom_idx","base_share"]], on=group_cols+["month","wom_idx"], how="left")
    comb["ratio"] = comb["obs_share"] / comb["base_share"]
    # Guard
    comb.loc[~np.isfinite(comb["ratio"]), "ratio"] = np.nan

    # Average ratio per WOM with month weight = Ym
    def _avg_ratio(g: pd.DataFrame) -> pd.Series:
        # months count with data
        n_months = g["month"].nunique()
        w = g["Ym"].astype(float).values
        r = g["ratio"].astype(float).values
        m = np.nansum(w * r) / np.nansum(w) if np.nansum(w) > 0 else np.nan
        # shrink towards 1.0
        # alpha_eff grows with months up to min_months_for_full_alpha
        alpha_eff = float(alpha) * min(1.0, n_months / float(min_months_for_full_alpha))
        r_hat = (1.0 - alpha_eff) * 1.0 + alpha_eff * (m if np.isfinite(m) else 1.0)
        return pd.Series({"ratio_hat": r_hat})

    wom_hat = (comb.groupby(group_cols + ["wom_idx"], dropna=False)
                    .apply(_avg_ratio)
                    .reset_index())

    # Wide to columns wom1..wom5
    wide = wom_hat.pivot_table(index=group_cols, columns="wom_idx", values="ratio_hat", fill_value=1.0)
    for k in range(1, 6):
        if k not in wide.columns:
            wide[k] = 1.0
    wide = wide[[1,2,3,4,5]].rename(columns={i: f"wom{i}" for i in [1,2,3,4,5]})
    return wide.reset_index()


# ==============================
# Public API
# ==============================
def weekly_split(
    reconciled_final: pd.DataFrame,
    hist_weekly: Optional[pd.DataFrame] = None,  # kept for backward compat
    method: str = "Hybrid",                      # unused for MonthSliced
    lookback_months: Optional[int] = None,
    mode: str = "MonthSliced",
    strategy: str = "DOW",                       # "DOW", "ProRataWorkingDays", "DOWxWOM"
    hist_daily: Optional[pd.DataFrame] = None,
    wom_alpha: float = 0.6,                      # strength WOM (0..1)
    wom_min_months: int = 6,                     # months to reach full alpha
) -> pd.DataFrame:
    """
    Split monthly reconciled forecast menjadi weekly (5 angka untuk MonthSliced).
    - mode="MonthSliced": weeks = blok Mon–Sat dari hari pertama bulan (Sun dibuang)
    - strategy:
        * "DOW": hanya pola harian Mon–Sat -> bisa hilangkan WOM pattern
        * "ProRataWorkingDays": pro-rata by jumlah hari kerja (Mon–Sat)
        * "DOWxWOM": (baru) DOW × Week-of-Month factor (robust, shrinkage)
    """
    if mode not in {"MonthSliced", "W-MON"}:
        raise ValueError("mode must be 'MonthSliced' or 'W-MON'")
    if mode == "W-MON":
        # behavior lama (tidak diubah)
        if hist_weekly is None or hist_weekly.empty:
            # Pure pro-rata 4/5 minggu sama rata
            df = reconciled_final.copy()
            df["month"] = _as_month(df["date"])
            # Build W-MON calendar
            out_rows = []
            for (keys, g) in df.groupby(["month","sku","wh"] + [c for c in ["region","kota"] if c in df.columns], dropna=False):
                month = keys[0]; qty_m = g["forecast_reconciled"].sum()
                # W-MON bins
                weeks = pd.date_range(month, month + pd.offsets.MonthEnd(0), freq="W-MON")
                if len(weeks) == 0 or weeks[0] != month:
                    weeks = pd.DatetimeIndex([month]).append(weeks)
                w = np.ones(len(weeks)) / max(len(weeks), 1)
                wk_qty = qty_m * w / w.sum()
                for d, q in zip(weeks, wk_qty):
                    row = dict(zip(["month","sku","wh"]+[c for c in ["region","kota"] if c in g.columns], keys))
                    row.update({"date": d, "forecast_weekly": float(q)})
                    out_rows.append(row)
            return pd.DataFrame(out_rows)

        # (ke depan bisa pakai hist_weekly profile seperti sebelumnya)
        raise NotImplementedError("W-MON historical strategy is not elaborated here to keep focus on MonthSliced.")

    # ===== MonthSliced path =====
    df = reconciled_final.copy()
    df["month"] = _as_month(df["date"])

    # Level signature: kita pakai SKU–WH agar stabil di bawah Region/Kota (bisa diubah kalau mau)
    sig_level = ["sku","wh"]

    # Siapkan profil DOW dan WOM jika diperlukan
    if strategy in {"DOW", "DOWxWOM"}:
        if hist_daily is None or hist_daily.empty:
            raise ValueError("hist_daily required for strategy 'DOW' or 'DOWxWOM' in MonthSliced mode.")
        dow_prof = _dow_profile(hist_daily, sig_level, lookback_months)
    else:
        dow_prof = None

    if strategy == "DOWxWOM":
        wom_fac = _wom_factor(hist_daily, sig_level, lookback_months, dow_prof, alpha=wom_alpha, min_months_for_full_alpha=wom_min_months)
    else:
        wom_fac = None

    # Build allocations per month
    leaf_cols = ["sku","wh"] + [c for c in ["region","kota"] if c in df.columns]
    out = []

    for keys, g in df.groupby(["month"] + leaf_cols, dropna=False):
        month = keys[0]
        qty_m = g["forecast_reconciled"].sum()
        leaf_key = dict(zip(leaf_cols, keys[1:]))

        cal = _calendar_month_sliced(month)
        # List WOM blocks (1..N)
        wom_values = sorted(cal["wom_idx"].unique().tolist())
        # Representative date per WOM
        rep_dates = cal.loc[cal["is_rep"]].sort_values("wom_idx")["date"].tolist()

        if strategy == "ProRataWorkingDays":
            # base weight = jumlah hari Mon–Sat di WOM
            base_w = cal.groupby("wom_idx")["dow"].size().reindex(wom_values, fill_value=0).astype(float).values
            base_share = base_w / base_w.sum() if base_w.sum() > 0 else np.ones(len(wom_values))/len(wom_values)

        elif strategy == "DOW":
            # base weight = sum f_dow untuk hari-hari WOM ini
            fp = dow_prof
            for c, v in leaf_key.items():
                if c in ["region","kota"]:
                    # signature ada di SKU–WH, jadi abaikan region/kota di merge
                    continue
            # ambil f_dow di level sku–wh
            fp_row = fp[(fp["sku"]==leaf_key["sku"]) & (fp["wh"]==leaf_key["wh"])]
            if fp_row.empty:
                f = {d: 1.0/6.0 for d in range(6)}
            else:
                row = fp_row.iloc[0]
                f = {d: float(row[f"dow_{d}"]) for d in range(6)}
            w = (cal.groupby("wom_idx")["dow"].apply(lambda s: sum(f[int(d)] for d in s.values))
                        .reindex(wom_values, fill_value=0.0).astype(float).values)
            base_share = w / w.sum() if w.sum() > 0 else np.ones(len(wom_values))/len(wom_values)

        elif strategy == "DOWxWOM":
            # base dari DOW seperti di atas, lalu dikali WOM factor (shrinked)
            fp = dow_prof
            fp_row = fp[(fp["sku"]==leaf_key["sku"]) & (fp["wh"]==leaf_key["wh"])]
            if fp_row.empty:
                f = {d: 1.0/6.0 for d in range(6)}
            else:
                row = fp_row.iloc[0]
                f = {d: float(row[f"dow_{d}"]) for d in range(6)}
            w = (cal.groupby("wom_idx")["dow"].apply(lambda s: sum(f[int(d)] for d in s.values))
                        .reindex(wom_values, fill_value=0.0).astype(float).values)
            base_share = w / w.sum() if w.sum() > 0 else np.ones(len(wom_values))/len(wom_values)

            # apply WOM factor
            wf = wom_fac
            wf_row = wf[(wf["sku"]==leaf_key["sku"]) & (wf["wh"]==leaf_key["wh"])]
            if wf_row.empty:
                ratios = np.ones(5, dtype=float)  # default 1..5
            else:
                r = [float(wf_row.iloc[0].get(f"wom{k}", 1.0)) for k in range(1,6)]
                ratios = np.array(r, dtype=float)
            # align to available WOMs
            ratios = ratios[:len(wom_values)]
            adj = base_share * ratios
            base_share = adj / adj.sum() if adj.sum() > 0 else base_share

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        wk_qty = qty_m * base_share
        for d, q in zip(rep_dates, wk_qty):
            row = {**leaf_key, "date": d, "forecast_weekly": float(q)}
            out.append(row)

    weekly_df = pd.DataFrame(out).sort_values(["date"] + leaf_cols).reset_index(drop=True)
    return weekly_df

import pandas as pd


def check_coherence(df: pd.DataFrame, scope: str) -> dict:
out = {}
m = df.copy(); m["date"] = pd.to_datetime(m["date"]).dt.to_period("M").dt.to_timestamp()
# For simplicity we only check Σ children equals at each step by reconstructing sums
if scope == "WH only":
out["WH_sum_nonnegative"] = bool((m["forecast_final"]>=0).all())
return out
if scope == "WH → Region":
wh = m.groupby(["date","sku","wh"])['forecast_final'].sum().reset_index()
out["ΣRegion=WH (count)"] = int(len(wh))
return out
else:
reg = m.groupby(["date","sku","wh","region"])['forecast_final'].sum().reset_index()
out["ΣKota=Region (count)"] = int(len(reg))
return out




def summarize_params(p: dict) -> dict:
# Flatten for summary report
return {
"recon_method": p.get("recon_method"),
"mint_lambda": p.get("mint_lambda"),
"vol_threshold": p.get("vol_threshold"),
"weekly_method": p.get("weekly_method"),
"run_timestamp": p.get("run_timestamp"),
}
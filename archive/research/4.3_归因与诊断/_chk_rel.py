import sys, os
sys.argv=["x"]
exec(open("report_final.py", encoding="utf-8").read().split("# ---------------------------------------------------------------- 净值图")[0])
eq_s = np.exp(S["net"].cumsum()); eq_b = np.exp(bh_ret.cumsum()); rel = eq_s/eq_b
m = rel.resample("ME").last()
print(m.loc["2023-10":"2024-04"].round(3).to_string())
print()
print("策略月度:", (np.exp(S["net"].resample("ME").sum())-1).loc["2023-10":"2024-04"].round(4).to_string())
print("指数月度:", (np.exp(bh_ret.resample("ME").sum())-1).loc["2023-10":"2024-04"].round(4).to_string())
print("月均仓位:", d["PRAC"].resample("ME").mean().loc["2023-10":"2024-04"].round(3).to_string())

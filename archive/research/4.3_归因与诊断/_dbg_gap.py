# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
src = open("final_full_report.py", encoding="utf-8").read()
head = src.split("S, B, BD =")[0]
exec(head)

print("strat.sum  = %.4f" % strat.sum())
print("base_r.sum = %.4f" % base_r.sum())
print("diff       = %.4f" % (strat.sum() - base_r.sum()))
gap = (d["base"].shift(1) - d["pos"].shift(1)) * r - (turn - turn_b) * COST
print("gap.sum    = %.4f  (-gap = %.4f)" % (gap.sum(), -gap.sum()))
print("nan gap count:", int(gap.isna().sum()))
print("\nnonzero gap days:", int((gap.abs() > 1e-12).sum()), " trigger days:", int(d["trig"].sum()))
nz = gap.abs() > 1e-12
print("nonzero-gap but NOT trig:", int((nz & ~d["trig"].astype(bool)).sum()))
bad = d[(nz) & (~d["trig"].astype(bool))]
if len(bad):
    print(bad.head(20).to_string())
    print("\nsum of those:", gap[(nz) & (~d["trig"].astype(bool))].sum() * 100)

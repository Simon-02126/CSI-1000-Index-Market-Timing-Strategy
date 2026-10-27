# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
src = open("final_full_report.py", encoding="utf-8").read()
head = src.split("S, B, BD =")[0]
exec(head)

seg = d.loc["2020-07-06":"2020-10-02", ["pos", "base", "c", "trig"]].copy()
seg["r"] = np.log(seg["c"]).diff()
seg["bp"] = seg["base"].shift(1)
seg["pp"] = seg["pos"].shift(1)
seg["gain"] = -(seg["bp"] - seg["pp"]) * seg["r"]
print(seg.head(8).to_string())
print("\ntrig days:", int(seg["trig"].sum()))
print("sum gain on trig days: %.2f pct" % (seg.loc[seg["trig"], "gain"].sum() * 100))
print("base_prev mean on trig days: %.2f" % seg.loc[seg["trig"], "bp"].mean())
print("index ret on trig days: %.2f pct" % (seg.loc[seg["trig"], "r"].sum() * 100))
print("index ret full seg: %.2f pct" % (np.log(seg["c"].iloc[-1] / seg["c"].iloc[0]) * 100))

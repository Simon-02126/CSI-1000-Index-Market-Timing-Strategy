# -*- coding: utf-8 -*-
"""受控实验: 固定 SIG_DIM=64, 只改 label 期限(10日 vs 20日), 比较量价因子

用法:
    python train_ctrl.py <H> <years逗号分隔> [seeds]
例: python train_ctrl.py 20 2019,2022,2024 0
"""
import os as _os_path, sys as _sys_path
_d = _os_path.path.dirname(_os_path.path.abspath(__file__))
while _d != _os_path.path.dirname(_d) and not _os_path.path.isfile(
        _os_path.path.join(_d, "_paths.py")):
    _d = _os_path.path.dirname(_d)
_sys_path.path.insert(0, _d)
import _paths as _P  # noqa: E402  统一路径入口

import os
import sys
import numpy as np
import pandas as pd

H = int(sys.argv[1])
years = [int(x) for x in sys.argv[2].split(",")]
seeds = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["0"])]

if H != 10:
    os.environ["TM_H"] = str(H)
else:
    os.environ.pop("TM_H", None)

BASE = _P.ROOT
sys.path.insert(0, os.path.join(BASE, "model"))
import timing_model as tm  # noqa: E402

tm.SIG_DIM = 64          # 关键: 固定 64 通道, 隔离通道数差异
OUTDIR = _P.TRAIN
os.makedirs(OUTDIR, exist_ok=True)

fp = os.path.join(tm.INPUT, "中证1000_000852.SH.csv")

# 收盘价
raw = pd.read_csv(fp, encoding="gbk")
dd = [c for c in raw.columns if "日期" in c][0]
cc = [c for c in raw.columns if "收盘" in c][0]
close = pd.Series(pd.to_numeric(raw[cc], errors="coerce").values, index=pd.to_datetime(raw[dd]))
close = close[~close.index.duplicated(keep="last")].sort_index()


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x) < 10: return np.nan
    sx, sy = x.std(), y.std()
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)) if sx and sy else np.nan


print(f"=== 设置: label={H}日, SIG_DIM=64, years={years}, seeds={seeds} ===", flush=True)
results = []
for y in years:
    out = os.path.join(OUTDIR, f"h{H}_{y}.csv")
    if os.path.exists(out):
        os.remove(out)
    n = tm.run(fp, y, y, view="pv", out_fp=out, seeds=seeds)
    if n == 0 or not os.path.exists(out):
        print(f"  {y}: 样本不足, 跳过", flush=True)
        continue
    df = pd.read_csv(out, encoding="utf-8-sig", parse_dates=["日期"])
    cols = [c for c in df.columns if c.startswith("信号") and c != "信号均值"]
    s = df[cols].mean(axis=1)
    s.index = pd.to_datetime(df["日期"])
    s = s[~s.index.duplicated(keep="last")].sort_index()
    line = [f"  {y}: n={len(s)} 通道={len(cols)}"]
    for h in (10, 20):
        fwd = np.log(close.shift(-h) / close).reindex(s.index)
        d = pd.DataFrame({"s": s, "f": fwd}).dropna()
        line.append(f"IC(H={h})={pearson(d['s'].values, d['f'].values):+.4f}")
    results.append((y, line))
    print("  ".join(line), flush=True)

print("\n汇总:", flush=True)
for y, line in results:
    print("  ".join(line), flush=True)

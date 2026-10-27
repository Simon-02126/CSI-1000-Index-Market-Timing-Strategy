# -*- coding: utf-8 -*-
"""单10日、64通道量价因子的训练随机性测试：每次只跑一个 seed，信号单独存档

用法: python train_seed.py <seed> <years>
输出: _train_out/s<seed>_<year>.csv
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

SEED = int(sys.argv[1])
years = [int(x) for x in sys.argv[2].split(",")]

os.environ.pop("TM_H", None)
os.environ.pop("TM_V2", None)

BASE = _P.ROOT
sys.path.insert(0, os.path.join(BASE, "model"))
import torch  # noqa: E402
torch.set_num_threads(2)
import timing_model as tm  # noqa: E402

assert tm.HORIZONS == (10,), tm.HORIZONS
assert tm.SIG_DIM == 64, tm.SIG_DIM

TAG = f"s{SEED}"
OUTDIR = _P.TRAIN
os.makedirs(OUTDIR, exist_ok=True)
fp = os.path.join(tm.INPUT, "中证1000_000852.SH.csv")

print(f"=== seed={SEED} 单10日 64通道 years={years} ===", flush=True)
for y in years:
    out = os.path.join(OUTDIR, f"{TAG}_{y}.csv")
    if os.path.exists(out):
        os.remove(out)
    n = tm.run(fp, y, y, view="pv", out_fp=out, seeds=[SEED])
    print(f"  {y}: 写入 {n} 日", flush=True)
print(f"完成 seed={SEED}", flush=True)

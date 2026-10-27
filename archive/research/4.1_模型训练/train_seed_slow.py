# -*- coding: utf-8 -*-
"""slow(资金面/两融+规模) 因子：单 seed 训练，用于种子稳定性检验

用法: python train_seed_slow.py <seed> <years>
输出: _train_out/slow_s<seed>_<year>.csv
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

TAG = f"slow_s{SEED}"
OUTDIR = _P.TRAIN
os.makedirs(OUTDIR, exist_ok=True)
fp = os.path.join(tm.INPUT, "中证1000_000852.SH.csv")

print(f"=== slow seed={SEED} years={years} ===", flush=True)
for y in years:
    out = os.path.join(OUTDIR, f"{TAG}_{y}.csv")
    if os.path.exists(out):
        os.remove(out)
    n = tm.run(fp, y, y, view="slow", out_fp=out, seeds=[SEED])
    print(f"  {y}: {n} 日", flush=True)
print(f"完成 slow seed={SEED}", flush=True)

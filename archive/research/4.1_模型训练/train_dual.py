# -*- coding: utf-8 -*-
"""量价因子: 多期限标签 × 64 通道受控实验

用法:
    python train_dual.py "5,10" mse 2019,2020,2021,2022,2023,2024 0,1
    python train_dual.py "5,10" ic  ...
    python train_dual.py "5,10,20" ic ...

设置:
  - SIG_DIM 强制 64（与单10日基线同通道数，排除通道数混淆）
  - 候选期限 (5,10) / (5,10,20)
  - 损失:
      mse = 各期限 MSE 求和平均（多任务回归）
      ic  = ic_loss（各期限负IC均值，即 timing_model 的 V2 分支）
  - 早停口径统一用 10 日 label 的验证 IC（K10）
  - 输出到 _train_out/{tag}_{year}.csv, tag = d<horiz>_<loss>，如 d5_10_mse
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

HORIZ = tuple(int(x) for x in sys.argv[1].split(","))
LOSS = sys.argv[2]
years = [int(x) for x in sys.argv[3].split(",")]
seeds = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0, 1]
SUFFIX = sys.argv[5] if len(sys.argv) > 5 else ""

os.environ.pop("TM_H", None)
os.environ.pop("TM_V2", None)

BASE = _P.ROOT
sys.path.insert(0, os.path.join(BASE, "model"))
import torch  # noqa: E402
import timing_model as tm  # noqa: E402

# ---- 受控设置 ----
tm.HORIZONS = HORIZ
tm.K10 = HORIZ.index(10) if 10 in HORIZ else 0
tm.SIG_DIM = 64
tm.V2 = (LOSS == "ic")

TAG = f"d{'_'.join(str(h) for h in HORIZ)}_{LOSS}{SUFFIX}"
OUTDIR = _P.TRAIN
os.makedirs(OUTDIR, exist_ok=True)

# ---- 自建缓存（绕过 ensure_cache 的命名约定，标签列数随 HORIZONS 变化）----
_CACHE = {}


def my_ensure(fp):
    key = os.path.basename(fp)
    if key not in _CACHE:
        df = tm.load_index(fp)
        _CACHE[key] = tm.make_windows(df, tm.N)
    return _CACHE[key]


tm.ensure_cache = my_ensure


def train_dual_mse(Xtr, Str, ytr, Xva, Sva, yva, seed, use_fast=True, use_slow=True):
    """多期限 MSE：各期限分别算 MSE 后平均；其余与 timing_model.train_one_fold 完全一致"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = tm.TimingNet(in_dim=8 if use_fast else 0,
                       slow_dim=Str.shape[1] if use_slow else 0, out_dim=tm.SIG_DIM)
    opt = torch.optim.Adam(net.parameters(), lr=tm.LR, weight_decay=tm.WEIGHT_DECAY)
    yva_np = yva[:, tm.K10].ravel()
    best = {"ic": -9.0, "epoch": -1, "state": None, "bad": 0}
    g = torch.Generator().manual_seed(seed)
    for ep in range(tm.MAX_EPOCHS):
        net.train()
        perm = torch.randperm(len(Xtr), generator=g)
        nb = 0
        for i in range(0, len(Xtr), tm.BATCH):
            idx = perm[i:i + tm.BATCH]
            if len(idx) < 16:
                continue
            xb = torch.tensor(Xtr[idx.numpy()]).float()
            sb = torch.tensor(Str[idx.numpy()]).float()
            yb = torch.tensor(ytr[idx.numpy()]).float()          # (B, H)
            pred = net(xb, sb)                                    # (B, 64)
            loss = 0.0
            for h in range(yb.shape[1]):
                loss = loss + ((pred - yb[:, h:h + 1]) ** 2).mean()
            loss = loss / yb.shape[1] + tm.LAMBDA_CORR * tm.corr_penalty(pred)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            nb += 1
            if nb >= tm.MAX_BATCHES_PER_EPOCH:
                break
        net.eval()
        with torch.no_grad():
            pv = tm.predict(net, Xva, Sva).mean(axis=1)
        ic = np.corrcoef(pv, yva_np)[0, 1]
        if ic > best["ic"]:
            best = {"ic": ic, "epoch": ep,
                    "state": {k: v.clone() for k, v in net.state_dict().items()}, "bad": 0}
        else:
            best["bad"] += 1
            if best["bad"] >= tm.PATIENCE:
                break
    net.load_state_dict(best["state"])
    return net, best


if LOSS == "mse":
    tm.train_one_fold = train_dual_mse

fp = os.path.join(tm.INPUT, "中证1000_000852.SH.csv")
print(f"=== tag={TAG} 期限={HORIZ} 损失={LOSS} SIG_DIM={tm.SIG_DIM} years={years} seeds={seeds} ===", flush=True)

for y in years:
    out = os.path.join(OUTDIR, f"{TAG}_{y}.csv")
    if os.path.exists(out):
        os.remove(out)
    n = tm.run(fp, y, y, view="pv", out_fp=out, seeds=seeds)
    print(f"  {y}: 写入 {n} 日", flush=True)

print("完成", flush=True)

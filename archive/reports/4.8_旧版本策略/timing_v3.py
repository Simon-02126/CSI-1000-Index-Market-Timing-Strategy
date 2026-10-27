# -*- coding: utf-8 -*-
"""中证1000 择时 v3 实验台：现行规则 vs 三处修正 vs 三档(离散仓位)各实现

核心改动（相对 backtest_timing.py 现行规则）：
  1) 标准化: 扩张 z-score -> 滚动分位 rank（只用过去 W 天，不含当日）
  2) 映射:   三段阈值(钉在 z=0) -> 三档(分位 0.3 / 0.8) 或 线性
  3) 持仓:   minHold=20 硬锁 -> 离散保持型处理（众数/中位数/最短持有 h 天）或 5 日平滑

用法:
    python timing_v3.py                # 全量对比
    python timing_v3.py --save         # 额外写出仓位csv
依赖: numpy, pandas（不需要 torch）
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
import glob
import numpy as np
import pandas as pd

BASE = _P.ROOT
INPUT = os.path.join(BASE, "input")
SIGDIR = os.path.join(BASE, "output", "signals")
NAME = "中证1000"
ANN = 252
COST = 0.0003

# v3 参数
WIN, MINP = 750, 250
LOW, HIGH = 0.30, 0.80      # 三档阈值: pct<LOW 空仓; LOW~HIGH 半仓; >=HIGH 满仓
W_PV = 0.50
SMOOTH = 5                  # 连续版平滑天数
HOLD = 5                    # 三档 + 最短持有天数


# ---------------- 数据 ----------------
def read_any(fp):
    for enc in ("gbk", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(fp, encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    raise RuntimeError(fp)


def load_price(name):
    fp = glob.glob(os.path.join(INPUT, f"*{name}*.csv"))[0]
    df = read_any(fp)
    dcol = [c for c in df.columns if "日期" in c][0]
    ccol = [c for c in df.columns if "收盘" in c][0]
    s = pd.Series(pd.to_numeric(df[ccol], errors="coerce").values,
                  index=pd.to_datetime(df[dcol]), name="close")
    return s[~s.index.duplicated(keep="last")].sort_index()


def load_tower(name, view):
    pats = glob.glob(os.path.join(SIGDIR, f"{name}_*_{view}_signals.csv"))
    if not pats:
        raise FileNotFoundError(f"{SIGDIR}\\{name}_*_{view}_signals.csv 不存在（需先跑 run_all.py merge）")
    df = pd.read_csv(sorted(pats)[0], encoding="utf-8-sig", parse_dates=["日期"])
    cols = [c for c in df.columns if c.startswith("信号") and c != "信号均值"]
    s = pd.Series(df[cols].mean(axis=1).values, index=df["日期"], name=view)
    return s[~s.index.duplicated(keep="last")].sort_index()


# ---------------- 标准化 ----------------
def zexp(s, minp=60):
    mu = s.rolling(len(s), min_periods=minp).mean().shift(1)
    sd = s.rolling(len(s), min_periods=minp).std().shift(1).replace(0, np.nan)
    return (s - mu) / sd


def roll_pct(s, win=WIN, minp=MINP):
    v = s.values.astype(float)
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        if i < minp:
            continue
        lo = max(0, i - win)
        w = v[lo:i]
        out[i] = (w < v[i]).sum() / len(w)
    return pd.Series(out, index=s.index)


# ---------------- 仓位 ----------------
def tgt_band3(pct, lo=LOW, hi=HIGH):
    return pd.Series(np.where(pct < lo, 0.0, np.where(pct < hi, 0.5, 1.0)), index=pct.index).where(pct.notna())


def pos_minhold(tgt, hold):
    """最短持有 hold 个交易日；tgt 为 NaN 时保持原仓位"""
    pos = np.zeros(len(tgt))
    cur, last = 0.0, -10 ** 9
    tv = tgt.values
    for i in range(len(tv)):
        t = tv[i]
        if not np.isnan(t) and i - last >= hold and t != cur:
            cur, last = t, i
        pos[i] = cur
    return pd.Series(pos, index=tgt.index)


def pos_mode(tgt, k):
    """最近 k 天档位众数（保持离散）；平票时取较小档（保守）"""
    v = tgt.values
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        w = v[max(0, i - k + 1):i + 1]
        w = w[~np.isnan(w)]
        if len(w) == 0:
            continue
        vals, cnt = np.unique(w, return_counts=True)
        out[i] = vals[np.argmax(cnt)]
    return pd.Series(out, index=tgt.index)


def pos_median(tgt, k):
    return tgt.rolling(k, min_periods=1).median()


def pos_linear(pct, lo=LOW, span=HIGH - LOW, smooth=SMOOTH):
    raw = ((pct - lo) / span).clip(0, 1)
    return raw.rolling(smooth, min_periods=1).mean()


def pos_current(pv, sl):
    idx = pv.index.union(sl.index)
    z = (zexp(pv.reindex(idx)) + zexp(sl.reindex(idx))) / 2
    tgt = pd.Series(np.where(z > 0, 1.0, np.where(z > -0.5, 0.5, 0.0)), index=idx).where(z.notna())
    return pos_minhold(tgt, 20)


# ---------------- 回测 ----------------
def backtest(pos, close, start="2019-01-01", cost=COST):
    df = pd.DataFrame({"p": pos, "c": close}).dropna()
    df = df[df.index >= pd.Timestamp(start)]
    if len(df) < 150:
        return None
    ret = np.log(df["c"]).diff()
    turn = df["p"].diff().abs().fillna(0.0)
    strat = (df["p"].shift(1) * ret - turn * cost).dropna()
    if len(strat) < 150:
        return None
    eq = strat.cumsum()
    mdd = float(np.expm1((eq - eq.cummax()).min()))
    ann = float(strat.mean() * ANN)
    vol = float(strat.std() * np.sqrt(ANN))
    uniq = np.unique(df["p"].dropna().values)
    return dict(ann=ann, vol=vol, sharpe=ann / vol if vol else np.nan, mdd=mdd,
                calmar=ann / abs(mdd) if mdd else np.nan,
                avg=float(df["p"].mean()), turn=turn.sum() / (len(df) / ANN),
                chg=float((turn > 1e-9).sum()) / (len(df) / ANN), levels=len(uniq),
                byyear=strat.groupby(strat.index.year).sum(), n=len(df))


def fmt_row(label, st, label_w=26):
    if st is None:
        return f"{label:<{label_w}} 样本不足"
    return (f"{label:<{label_w}} 夏普{st['sharpe']:>5.2f} 年化{st['ann']:>7.1%} "
            f"波动{st['vol']:>6.1%} 回撤{st['mdd']:>7.1%} Calmar{st['calmar']:>5.2f} "
            f"均仓{st['avg']:>4.0%} 换仓{st['chg']:>5.1f}次/年 档位{st['levels']:>4}")


def main():
    save = "--save" in sys.argv
    close = load_price(NAME)
    pv, sl = load_tower(NAME, "pv"), load_tower(NAME, "slow")
    idx = pv.index.union(sl.index)
    print(f"[{NAME}] 价格 {close.index.min().date()}~{close.index.max().date()} | "
          f"pv {pv.index.min().date()}~{pv.index.max().date()} | slow {sl.index.min().date()}~{sl.index.max().date()}")

    pct = W_PV * roll_pct(pv.reindex(idx)) + (1 - W_PV) * roll_pct(sl.reindex(idx))
    t3 = tgt_band3(pct)

    p_cur = pos_current(pv, sl)
    p_lin = pos_linear(pct)
    cands = [
        ("现行: 扩张z+三段+锁20", p_cur),
        ("v3连续: 线性+平滑5", p_lin),
        ("三档 纯日频(无处理)", t3),
        ("三档 + 众数5", pos_mode(t3, 5)),
        ("三档 + 中位数5", pos_median(t3, 5)),
        ("三档 + 最短持有3", pos_minhold(t3, 3)),
        ("三档 + 最短持有5", pos_minhold(t3, 5)),
        ("三档 + 最短持有10", pos_minhold(t3, 10)),
        ("三档 + 最短持有20", pos_minhold(t3, 20)),
    ]

    for start, lbl in [("2019-01-01", "2019-01 ~ 2026-10（全样本）"),
                       ("2023-01-01", "2023-01 ~ 2026-10（留出期，参数未参与选择）")]:
        print(f"\n======= {lbl} =======")
        for nm, p in cands:
            print("  " + fmt_row(nm, backtest(p, close, start)))

    print("\n--- 逐年对数收益（2019+）---")
    picks = [nm for nm in ("现行: 扩张z+三段+锁20", "v3连续: 线性+平滑5",
                           "三档 + 众数5", "三档 + 最短持有5") ]
    pick_map = dict((nm, p) for nm, p in cands)
    res = {nm: backtest(pick_map[nm], close, "2019-01-01") for nm in picks}
    years = sorted(res[picks[0]]["byyear"].index)
    hdr = "  年份   " + "".join(f"{nm:>20}" for nm in picks)
    print(hdr)
    for y in years:
        print(f"  {y}   " + "".join(f"{res[nm]['byyear'].get(y, 0):>19.1%}" for nm in picks))

    print("\n--- 离散处理 vs 平滑：参数扫描（2019+ / 留出期夏普）---")
    print("  最短持有 h:   " + "  ".join(f"h={h}" for h in (0, 3, 5, 10, 20)))
    row = []
    for h in (0, 3, 5, 10, 20):
        a = backtest(pos_minhold(t3, h), close, "2019-01-01")
        b = backtest(pos_minhold(t3, h), close, "2023-01-01")
        row.append(f"{a['sharpe']:.2f}/{b['sharpe']:.2f}")
    print("  " + " " * 13 + "  ".join(row))
    print("  众数/中位 k:  " + "  ".join(f"k={k}" for k in (3, 5, 7, 10)))
    r1, r2 = [], []
    for k in (3, 5, 7, 10):
        a = backtest(pos_mode(t3, k), close, "2019-01-01")
        b = backtest(pos_mode(t3, k), close, "2023-01-01")
        r1.append(f"{a['sharpe']:.2f}/{b['sharpe']:.2f}")
    print("  众数 " + " " * 11 + "  ".join(r1))
    for k in (3, 5, 7, 10):
        a = backtest(pos_median(t3, k), close, "2019-01-01")
        b = backtest(pos_median(t3, k), close, "2023-01-01")
        r2.append(f"{a['sharpe']:.2f}/{b['sharpe']:.2f}")
    print("  中位 " + " " * 11 + "  ".join(r2))

    print("\n--- 阈值扫描（众数5 / 最短持有5 各自）---")
    for lo, hi in ((0.25, 0.75), (0.30, 0.80), (0.35, 0.85)):
        t = tgt_band3(pct, lo, hi)
        a1 = backtest(pos_mode(t, 5), close, "2019-01-01")
        b1 = backtest(pos_mode(t, 5), close, "2023-01-01")
        a2 = backtest(pos_minhold(t, 5), close, "2019-01-01")
        b2 = backtest(pos_minhold(t, 5), close, "2023-01-01")
        print(f"  阈值 {lo:.2f}/{hi:.2f}   众数5 {a1['sharpe']:.2f}/{b1['sharpe']:.2f}   "
              f"最短持有5 {a2['sharpe']:.2f}/{b2['sharpe']:.2f}   "
              f"（2019+ 年化 {a2['ann']:.1%} / 回撤 {a2['mdd']:.1%}）")

    print("\n--- 三档档位分布（全样本）---")
    v = t3.dropna().values
    for lv in (0.0, 0.5, 1.0):
        print(f"  仓位 {lv:.1f}: {(v == lv).mean():.0%}", end="   ")
    print()

    print("\n--- 成本敏感性（三档+最短持有5, 2019+）---")
    for c in (0, 0.0003, 0.001, 0.002):
        st = backtest(pos_minhold(t3, 5), close, "2019-01-01", cost=c)
        print(f"  单边 {c:.2%}  夏普 {st['sharpe']:.2f}  年化 {st['ann']:.1%}")

    if save:
        out = pd.DataFrame({"pct": pct, "tgt_band3": t3,
                            "pos_band3_hold5": pos_minhold(t3, 5),
                            "pos_band3_mode5": pos_mode(t3, 5),
                            "pos_linear": p_lin, "pos_current": p_cur,
                            "close": close}).dropna(subset=["pos_band3_hold5"])
        fp = os.path.join(SIGDIR, f"{NAME}_v3_position.csv")
        out.to_csv(fp, encoding="utf-8-sig")
        print(f"\n已写出 {fp}")


if __name__ == "__main__":
    main()

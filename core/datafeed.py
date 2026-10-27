# -*- coding: utf-8 -*-
"""数据读取（唯一真身）—— 回测与实盘共用，保证"喂给策略的数据"完全一致。

三块数据：
    量价   input/中证1000_000852.SH.csv      → 8 通道（模型输入）
    资金   data/两融数据.csv + data/市场规模数据.csv → 5 字段（模型输入）
    情绪   input/情绪_面板_日频.csv           → IF_会员多空比 一列（策略输入）
"""
from __future__ import annotations

import glob
import os
import sys as _sys

import numpy as np
import pandas as pd

# 向上找 _paths.py（保证从任何工作目录直接跑本文件都能定位项目根）
_d = os.path.dirname(os.path.abspath(__file__))
while _d != os.path.dirname(_d) and not os.path.isfile(os.path.join(_d, "_paths.py")):
    _d = os.path.dirname(_d)
_sys.path.insert(0, _d)
import _paths as _P  # noqa: E402  统一路径入口

SEEDS = 6          # 研究用的 6 个 seed（0~5）


# ---------------------------------------------------------------- 通用
def read_any(fp: str) -> pd.DataFrame:
    """按常见编码依次尝试读 CSV。"""
    for enc in ("gbk", "gb18030", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(fp, encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    raise RuntimeError(f"无法读取：{fp}")


def _pick(df: pd.DataFrame, key: str) -> str:
    """列名不固定：按包含关系取第一列（如 '日期' / '收盘'）。"""
    cand = [c for c in df.columns if key in str(c)]
    if not cand:
        raise KeyError(f"找不到包含 '{key}' 的列，现有列：{list(df.columns)[:12]}")
    return cand[0]


# ---------------------------------------------------------------- 指数
#: 数据源开关：默认 'db'（db/timing.db 为唯一事实来源），可设 'csv' 或环境变量 DSH_SOURCE
SOURCE = os.environ.get("DSH_SOURCE", "db").strip().lower()


def _load_index_csv(fp: str | None = None) -> pd.DataFrame:
    fp = fp or _P.INDEX_CSV
    if not os.path.isfile(fp):
        hits = glob.glob(os.path.join(_P.INPUT, "*中证1000*.csv"))
        if not hits:
            raise FileNotFoundError(f"主数据缺失：{fp}")
        fp = hits[0]
    df = read_any(fp)
    dcol = _pick(df, "日期")
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.dropna(subset=[dcol]).drop_duplicates(dcol, keep="last")
    return df.set_index(dcol).sort_index()


def load_index(fp: str | None = None, source: str | None = None) -> pd.DataFrame:
    """主数据（宽表）。返回以 DatetimeIndex 排序、去重的 DataFrame。

    source='db'（默认）优先读 `db/timing.db`；不可用则回落 CSV。
    显式传 `fp` 或 `source='csv'` 时强制走 CSV。
    """
    src = (source or SOURCE).lower()
    if fp is None and src == "db":
        try:
            import dbsource as _dbs
            if _dbs.available():
                return _dbs.load_features()
        except Exception:
            pass
    return _load_index_csv(fp)


def load_close(fp: str | None = None) -> pd.Series:
    """收盘价序列。"""
    df = load_index(fp)
    return pd.to_numeric(df[_pick(df, "收盘")], errors="coerce").dropna()


def load_ohlc(fp: str | None = None) -> pd.DataFrame:
    """开高低收 + 成交量 + 成交额（实盘换仓与净值核对用）。"""
    df = load_index(fp)
    want = {}
    for k, name in (("开盘", "open"), ("最高", "high"), ("最低", "low"), ("收盘", "close")):
        try:
            want[name] = pd.to_numeric(df[_pick(df, k)], errors="coerce")
        except KeyError:
            pass
    for k, name in (("成交量", "volume"), ("成交额", "amount")):
        try:
            want[name] = pd.to_numeric(df[_pick(df, k)], errors="coerce")
        except KeyError:
            pass
    return pd.DataFrame(want).dropna(subset=["close"])


def log_returns(close: pd.Series) -> pd.Series:
    """日对数收益。"""
    return np.log(close).diff()


# ---------------------------------------------------------------- 情绪
def load_sentiment(fp: str | None = None) -> pd.DataFrame:
    """情绪面板宽表（3591 × 29，情绪字段自 2015-02-09 起）。"""
    fp = fp or _P.SENTI_PANEL
    df = pd.read_csv(fp, encoding="utf-8-sig")
    dcol = _pick(df, "日期")
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    return df.dropna(subset=[dcol]).drop_duplicates(dcol, keep="last") \
             .set_index(dcol).sort_index()


def load_if_ratio(fp: str | None = None, source: str | None = None) -> pd.Series:
    """IF 会员多空比原始值（策略的情绪输入）。

    source='db'（默认）优先读 `db/timing.db` 的 `if_sentiment`；不可用则回落情绪面板 CSV。
    """
    src = (source or SOURCE).lower()
    if fp is None and src == "db":
        try:
            import dbsource as _dbs
            s = _dbs.load_if_ratio()
            if len(s):
                return s
        except Exception:
            pass
    p = load_sentiment(fp)
    if "IF_会员多空比" not in p.columns:
        raise KeyError("情绪面板缺少 'IF_会员多空比' 列")
    return pd.to_numeric(p["IF_会员多空比"], errors="coerce").dropna()


# ---------------------------------------------------------------- 历史信号缓存
def load_train_seed(tag: str, seed: int, train_dir: str | None = None) -> pd.Series | None:
    """读单个 seed 的逐年信号文件（每文件 64 维 → 行均值 → 跨年份拼接）。

    tag: 's'（量价）或 'slow_s'；文件名形如 s0_2019.csv / slow_s0_2019.csv
    """
    train_dir = train_dir or _P.TRAIN
    parts = []
    pat = os.path.join(train_dir, f"{tag}{seed}_*.csv")
    for p in sorted(glob.glob(pat)):
        x = pd.read_csv(p, encoding="utf-8-sig", parse_dates=["日期"])
        cols = [c for c in x.columns if c.startswith("信号") and c != "信号均值"]
        if not cols:
            continue
        s = pd.Series(x[cols].mean(axis=1).values, index=x["日期"])
        parts.append(s[~s.index.duplicated(keep="last")])
    if not parts:
        return None
    return pd.concat(parts).sort_index()


def load_train_avg(tag: str, seeds: int = SEEDS,
                   train_dir: str | None = None) -> pd.Series:
    """跨 seed 取均值的原始信号（实盘 bootstrap 与历史回测都用这条）。"""
    cols = [v for v in (load_train_seed(tag, sd, train_dir) for sd in range(seeds))
            if v is not None]
    if not cols:
        raise FileNotFoundError(f"{train_dir or _P.TRAIN} 下找不到 {tag}0_*.csv")
    return pd.concat(cols, axis=1).mean(axis=1).dropna()


def build_raw_signals(train_dir: str | None = None) -> pd.DataFrame:
    """组装策略的两条原始输入信号：pv / slow（6-seed 均值）。"""
    return pd.DataFrame({
        "pv": load_train_avg("s", SEEDS, train_dir),
        "slow": load_train_avg("slow_s", SEEDS, train_dir),
    }).dropna()


# ---------------------------------------------------------------- 一次性打包
def load_all() -> dict:
    """回测/实盘通用的数据包。"""
    close = load_close()
    raw = build_raw_signals()
    return dict(close=close, ret=log_returns(close), raw=raw,
                if_raw=load_if_ratio(), panel=load_sentiment())


if __name__ == "__main__":
    d = load_all()
    print(f"主数据    {d['close'].index[0].date()} ~ {d['close'].index[-1].date()}  "
          f"({len(d['close'])} 日)")
    print(f"原始信号  {d['raw'].index[0].date()} ~ {d['raw'].index[-1].date()}  "
          f"({len(d['raw'])} 日)  pv 均值 {d['raw']['pv'].mean():+.4f}  "
          f"slow 均值 {d['raw']['slow'].mean():+.4f}")
    print(f"情绪 IF   {d['if_raw'].index[0].date()} ~ {d['if_raw'].index[-1].date()}  "
          f"({len(d['if_raw'])} 日)  最新 {d['if_raw'].iloc[-1]:.4f}")

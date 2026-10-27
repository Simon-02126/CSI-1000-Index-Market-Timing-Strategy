# -*- coding: utf-8 -*-
"""年度模型登记表：检测 → 缺则训练保存 → 有则直接加载，并把每年的信号拼成连续序列。

用法：
    python live/model_registry.py status          # 看哪些年、哪些 seed 已有模型
    python live/model_registry.py ensure 2026     # 确保 2026 年模型齐备（有就跳过）
    python live/model_registry.py ensure-all      # 从 FIRST_TEST_YEAR 到最新年
"""
from __future__ import annotations

import os
import sys

import pandas as pd

_d = os.path.dirname(os.path.abspath(__file__))
while _d != os.path.dirname(_d) and not os.path.isfile(os.path.join(_d, "_paths.py")):
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import _paths as _P  # noqa: E402

import model_io as mio  # noqa: E402
import timing_model as tm  # noqa: E402

VIEWS = ("pv", "slow")
SEEDS = mio.SEEDS


def index_last_date() -> pd.Timestamp:
    """数据最后交易日：DB 优先（唯一事实来源），回落主数据 CSV。"""
    try:
        import dbsource as _dbs
        if _dbs.available():
            f = _dbs.load_features()
            if len(f):
                return pd.Timestamp(f.index[-1])
    except Exception:                                        # noqa: BLE001
        pass
    df = pd.read_csv(_P.INDEX_CSV, encoding="gbk")
    dc = [c for c in df.columns if "日期" in c][0]
    return pd.to_datetime(df[dc].iloc[-1])


def years_to_cover(start_year: int | None = None, end_year: int | None = None) -> list[int]:
    """要覆盖的年份：默认 FIRST_TEST_YEAR ~ 数据最后一年。"""
    last = index_last_date()
    y0 = start_year or tm.FIRST_TEST_YEAR
    y1 = end_year or int(last.year)
    return list(range(y0, y1 + 1))


def status(views=VIEWS, seeds=SEEDS) -> pd.DataFrame:
    rows = []
    for y in years_to_cover():
        for v in views:
            have = [s for s in seeds if os.path.isfile(mio.model_path(y, v, s))]
            rows.append(dict(year=y, view=v,
                             models=len(have), seeds=",".join(map(str, have)) or "-",
                             signal_csv=os.path.isfile(mio.signal_csv_path(y, v))))
    return pd.DataFrame(rows)


def ensure(years=None, views=VIEWS, seeds=SEEDS, force=False, verbose=True) -> dict:
    """确保指定年份的模型齐备。返回 {(year, view): dict(trained, skipped)}。"""
    years = years or years_to_cover()
    out = {}
    for y in years:
        for v in views:
            r = mio.ensure_year(None, y, v, seeds=seeds, force=force, verbose=verbose)
            out[(y, v)] = r
            if verbose:
                tag = "训练" if r["trained"] else "已存在"
                print(f"  {y} {v:<4} {tag}：新增 seed {r['trained'] or '-'}，"
                      f"复用 {r['skipped'] or '-'}，耗时 {r['seconds']:.1f}s")
    return out


def view_signals(view: str, years=None, seeds=SEEDS) -> pd.Series:
    """把该 view 每年的信号 CSV 拼成连续的日频原始信号。

    兼容两种缓存格式：
        日期 + 信号00..  （archive 风格的 64 维 → 取行均值）
        日期 + 信号      （live 训练产出，已是 6 seed × 64 维的均值）
    """
    years = years or years_to_cover()
    parts = []
    for y in years:
        fp = mio.signal_csv_path(y, view)
        if not os.path.isfile(fp):
            continue
        df = pd.read_csv(fp, encoding="utf-8-sig")
        dc = [c for c in df.columns if "日期" in c][0]
        if "信号" in df.columns and "信号00" not in df.columns:
            s = df["信号"].astype(float)
        else:
            cols = [c for c in df.columns if c.startswith("信号") and c != "信号均值"]
            s = df[cols].mean(axis=1) if cols else df["信号均值"]
        s = pd.Series(s.to_numpy(), index=pd.to_datetime(df[dc])).rename(view)
        parts.append(s)
    if not parts:
        return pd.Series(dtype=float, name=view)
    s = pd.concat(parts).sort_index()
    return s[~s.index.duplicated(keep="last")].rename(view)


def all_signals(years=None, views=VIEWS, seeds=SEEDS) -> pd.DataFrame:
    """pv / slow 两列原始信号（未做分位）。没有任何模型时返回空表而不是报错。"""
    frames = [view_signals(v, years, seeds) for v in views]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=list(views))
    return pd.concat(frames, axis=1).sort_index()


def _main(argv):
    if not argv or argv[0] == "status":
        df = status()
        miss = df[(df["models"] == 0)][["year", "view"]]
        print(df.to_string(index=False))
        print(f"\n模型齐备 {(df['models'] > 0).sum()}/{len(df)} 组；"
              f"完全缺失 {len(miss)} 组")
        if len(miss):
            print("缺失清单：" + ", ".join(f"{r.year}-{r.view}" for r in miss.itertuples()))
        return 0
    if argv[0] == "ensure":
        years = [int(x) for x in argv[1:]] or None
        ensure(years)
        return 0
    if argv[0] == "ensure-all":
        ensure()
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))

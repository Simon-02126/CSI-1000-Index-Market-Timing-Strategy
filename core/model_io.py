# -*- coding: utf-8 -*-
"""年度模型的训练 / 保存 / 加载（实盘用）—— 不修改 timing_model.py 的任何行为。

一个"模型"= 某一年 fold 的 (网络权重 + 三套标准化参数)。
fold 定义与 timing_model.run 完全一致：
    预测第 Y 年：训练 <Y-3-01-01（末尾 embargo 11 日）、验证 [Y-3, Y-1)、测试 Y 全年

实盘语义：
    有 → 直接加载，不重训（默认）
    无 → 训练一次并保存到 models/{view}/{view}_{Y}_s{seed}.pt
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import torch

_d = os.path.dirname(os.path.abspath(__file__))
while _d != os.path.dirname(_d) and not os.path.isfile(os.path.join(_d, "_paths.py")):
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import _paths as _P  # noqa: E402

import timing_model as tm  # noqa: E402

SEEDS = tuple(range(6))          # 与训练历史信号用的 6 个 seed 一致
_SIG_COLS = [f"信号{i:02d}" for i in range(tm.SIG_DIM)]


def features_source():
    """入模特征源：优先 `db/timing.db`（唯一事实来源），不可用才回落主数据 CSV。

    返回 DataFrame（DB）或 csv 路径（回落时）；`timing_model.ensure_cache` 两种都吃。
    """
    try:
        import dbsource as _dbs
        if _dbs.available():
            return _dbs.load_features()
    except Exception:                                        # noqa: BLE001
        pass
    return _P.INDEX_CSV


def resolve_fp(fp=None):
    """`fp=None` 表示"用默认源"（DB 优先）；否则尊重调用方给的路径/DataFrame。"""
    return features_source() if fp is None else fp


def data_fingerprint(fp, until=None) -> dict:
    """主数据指纹。

    until 给定时只对 `日期 < until` 的行做哈希 —— 这是判断"模型该不该重训"的正确口径：
    某个年份 Y 的 fold，训练/验证窗口全部在 (Y-1)-01-01 之前，往后再追加交易日
    **不会**改变模型权重，只有历史数据被修正才会。

    `fp` 可以是 csv 路径，也可以是 DataFrame（DB 数据源）。
    """
    import hashlib
    if isinstance(fp, pd.DataFrame):
        df = fp.reset_index() if fp.index.name else fp.copy()
        if "日期" not in df.columns:
            df = df.rename(columns={df.columns[0]: "日期"})
        d = pd.to_datetime(df["日期"], errors="coerce")
        if until is not None:
            m = d < pd.Timestamp(until)
            sub = df.loc[m]
            body = sub.to_csv(index=False).encode("utf-8")
            return dict(md5=hashlib.md5(body).hexdigest(), n_rows=int(m.sum()),
                        last_date=str(d[m].max().date()) if m.any() else "",
                        until=str(pd.Timestamp(until).date()))
        body = df.to_csv(index=False).encode("utf-8")
        return dict(md5=hashlib.md5(body).hexdigest(), n_rows=int(len(df)),
                    last_date=str(d.max().date()) if len(df) else "")
    df = pd.read_csv(fp, encoding="gbk")
    dc = [c for c in df.columns if "日期" in c][0]
    d = pd.to_datetime(df[dc], errors="coerce", format="mixed")
    if until is not None:
        m = d < pd.Timestamp(until)
        sub = df.loc[m]
        body = sub.to_csv(index=False).encode("utf-8")
        return dict(md5=hashlib.md5(body).hexdigest(), n_rows=int(m.sum()),
                    last_date=str(d[m].max().date()) if m.any() else "",
                    until=str(pd.Timestamp(until).date()))
    with open(fp, "rb") as f:
        raw = f.read()
    return dict(md5=hashlib.md5(raw).hexdigest(), n_rows=int(len(df)),
                last_date=str(d.max().date()) if len(df) else "")



# ---------------------------------------------------------------- fold 切分（与 run 逐字一致）
def fold_arrays(fp, year: int, view: str):
    """返回该年 fold 的全部数组与标准化参数（`fp=None` = 默认源，DB 优先）。"""
    fp = resolve_fp(fp)
    use_fast = view in ("pv", "dual")
    use_slow = view in ("slow", "dual")
    dates, X, slow, labels, valid, slow_ok = tm.ensure_cache(fp)
    d = pd.to_datetime(dates)

    t_start = np.datetime64(pd.Timestamp(year=year, month=1, day=1))
    t_valid = np.datetime64(pd.Timestamp(year=year - 3, month=1, day=1))
    t_valid_end = np.datetime64(pd.Timestamp(year=year - 1, month=1, day=1))
    ok = valid & slow_ok if use_slow else valid
    tr_idx = np.where(ok & (d < t_valid))[0][:-(max(tm.HORIZONS) + 1)]
    va_mask = ok & (d >= t_valid) & (d < t_valid_end)
    te_mask = ok & (d >= t_start) & (
        d < np.datetime64(pd.Timestamp(year=year + 1, month=1, day=1)))
    if tr_idx.size < 100 or va_mask.sum() < 100:
        raise ValueError(f"{year} 年训练/验证样本不足（train={tr_idx.size}, "
                         f"valid={int(va_mask.sum())}）")

    mu, sigma = labels[tr_idx].mean(0), labels[tr_idx].std(0) + 1e-8
    if use_fast:
        fmu = X[tr_idx].reshape(-1, tm.FAST_DIM).mean(0)
        fsd = X[tr_idx].reshape(-1, tm.FAST_DIM).std(0) + 1e-8
        Xtr = (X[tr_idx] - fmu) / fsd
        Xva = (X[va_mask] - fmu) / fsd
        Xte = (X[te_mask] - fmu) / fsd
    else:
        fmu = np.zeros(tm.FAST_DIM, np.float32)
        fsd = np.ones(tm.FAST_DIM, np.float32)
        Xtr = np.zeros((len(tr_idx), tm.N, tm.FAST_DIM), np.float32)
        Xva = np.zeros((int(va_mask.sum()), tm.N, tm.FAST_DIM), np.float32)
        Xte = np.zeros((int(te_mask.sum()), tm.N, tm.FAST_DIM), np.float32)
    if use_slow:
        smu = slow[tr_idx].mean(0)
        ssd = slow[tr_idx].std(0) + 1e-8
        Str = (slow[tr_idx] - smu) / ssd
        Sva = (slow[va_mask] - smu) / ssd
        Ste = (slow[te_mask] - smu) / ssd
    else:
        smu = np.zeros(0, np.float32)
        ssd = np.ones(0, np.float32)
        Str = np.zeros((len(tr_idx), 0), np.float32)
        Sva = np.zeros((int(va_mask.sum()), 0), np.float32)
        Ste = np.zeros((int(te_mask.sum()), 0), np.float32)

    return dict(
        dates=d, use_fast=use_fast, use_slow=use_slow,
        Xtr=Xtr, Xva=Xva, Xte=Xte, Str=Str, Sva=Sva, Ste=Ste,
        ytr_z=(labels[tr_idx] - mu) / sigma,
        yva_z=(labels[va_mask] - mu) / sigma,
        scalers=dict(mu=mu, sigma=sigma, fmu=fmu, fsd=fsd, smu=smu, ssd=ssd),
        te_idx=np.where(te_mask)[0],
        n_train=int(tr_idx.size), n_valid=int(va_mask.sum()), n_test=int(te_mask.sum()),
    )


# ---------------------------------------------------------------- 单 seed 训练 / 加载
def model_path(year: int, view: str, seed: int, save_dir: str | None = None) -> str:
    save_dir = save_dir or (getattr(_P, f"MODELS_{view.upper()}", None) or
                            os.path.join(_P.MODELS, view))
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, f"{view}_{year}_s{seed}.pt")


def save_model(net, fold: dict, year: int, view: str, seed: int,
               save_dir: str | None = None, fingerprint: dict | None = None) -> str:
    fp = model_path(year, view, seed, save_dir)
    use_fast = bool(fold["use_fast"])
    torch.save(dict(
        state_dict={k: v.cpu() for k, v in net.state_dict().items()},
        in_dim=(tm.FAST_DIM if use_fast else 0),
        slow_dim=(10 if fold["use_slow"] else 0),
        use_fast=use_fast, use_slow=bool(fold["use_slow"]),
        hid=tm.HID, mlp_hidden=tm.MLP_HIDDEN, dropout=tm.DROPOUT,
        year=year, view=view, seed=seed,
        data=fingerprint or {},
        scalers={k: np.asarray(v) for k, v in fold["scalers"].items()},
    ), fp)
    return fp


def load_model(year: int, view: str, seed: int, save_dir: str | None = None):
    fp = model_path(year, view, seed, save_dir)
    ck = torch.load(fp, map_location="cpu", weights_only=False)
    # 老检查点没有 use_fast 字段：按 view 推断（slow 视图没有 LSTM 分支）
    use_fast = bool(ck.get("use_fast", view in ("pv", "dual")))
    in_dim = tm.FAST_DIM if use_fast else 0
    net = tm.TimingNet(in_dim=in_dim, slow_dim=ck["slow_dim"],
                       hid=ck["hid"], mlp_hidden=ck["mlp_hidden"],
                       dropout=ck["dropout"], out_dim=tm.SIG_DIM)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck


# ---------------------------------------------------------------- 年度：缺则训练，有则跳过
def ensure_year(fp, year: int, view: str, seeds=SEEDS,
                save_dir: str | None = None, force: bool = False,
                verbose: bool = True) -> dict:
    """确保该年 view 的全部 seed 模型都在本地；返回 {year, view, trained:[...], skipped:[...]}。"""
    trained, skipped, preds = [], [], []
    fold = None
    t0 = time.time()
    # 训练窗口口径的指纹：该年 fold 的训练/验证数据全在 (Y-1)-01-01 之前，
    # 往后追加交易日不会改变权重，只有历史被修正才会 → 只有这种变化才提示重训。
    cutoff = pd.Timestamp(year=year - 1, month=1, day=1)
    fp_data = data_fingerprint(fp, until=cutoff)
    stale = []
    for seed in seeds:
        fold = fold or fold_arrays(fp, year, view)
        fp_m = model_path(year, view, seed, save_dir)
        if os.path.isfile(fp_m) and not force:
            net, ck = load_model(year, view, seed, save_dir)
            old = ck.get("data") or {}
            # 老格式检查点没有 until 字段，无法判断训练窗口，跳过告警
            if old.get("until") and old.get("md5") and old["md5"] != fp_data["md5"]:
                stale.append(f"{view}_{year}_s{seed}")
            skipped.append(seed)
        else:
            net, best = tm.train_one_fold(fold["Xtr"], fold["Str"], fold["ytr_z"],
                                          fold["Xva"], fold["Sva"], fold["yva_z"], seed,
                                          use_fast=fold["use_fast"],
                                          use_slow=fold["use_slow"])
            save_model(net, fold, year, view, seed, save_dir, fp_data)
            trained.append(seed)
            if verbose:
                print(f"    {view} {year} seed{seed} 训练完成：最优 epoch {best['epoch']} "
                      f"validIC {best['ic']:+.4f}")
        preds.append(tm.predict(net, fold["Xte"], fold["Ste"]))
    if stale:
        print(f"    [警告] {len(stale)} 个模型的训练窗口数据与当前文件不一致"
              f"（历史被修正过），建议 --retrain：{stale[:3]}")
    sig = np.mean(preds, axis=0) if preds else None
    if verbose and trained:
        print(f"    {view} {year}：训练 {len(trained)} 个、复用 {len(skipped)} 个，"
              f"耗时 {time.time() - t0:.0f}s")
    cache = signal_csv_path(year, view, save_dir)
    need_write = bool(trained) or not os.path.isfile(cache)
    if not need_write and force:
        need_write = True
    if sig is not None and not need_write:
        # 已有缓存，但若缓存没覆盖到该年最后一个可预测日（缺 label 的尾部），补写一次
        try:
            old = pd.read_csv(cache, encoding="utf-8-sig", parse_dates=["日期"])
            full = predict_series(fp, year, view, seeds=seeds, save_dir=save_dir)
            need_write = bool(len(full)) and old["日期"].max() < full.index.max()
        except Exception:                                     # noqa: BLE001
            need_write = True
    if sig is not None and need_write:
        # 缓存格式：日期 + 信号（该日 = 6 seed × 64 维的均值，与 archive 口径一致）。
        # 这里用 predict_series 而不是 fold 的 te_idx，是为了把"缺 label 的最新几天"也预测出来，
        # 否则实盘信号会比数据晚 max(HORIZONS)=20 个交易日。
        full = predict_series(fp, year, view, seeds=seeds, save_dir=save_dir)
        if len(full):
            pd.DataFrame({"日期": full.index, "信号": full.to_numpy()}).to_csv(
                cache, index=False, encoding="utf-8-sig")
            if verbose:
                print(f"    {view} {year} 信号已写入 {os.path.relpath(cache, _P.ROOT)}"
                      f"（{len(full)} 行 {full.index[0].date()}~{full.index[-1].date()}）")
    return dict(year=year, view=view, trained=trained, skipped=skipped,
                fold=fold, sig=sig, seconds=time.time() - t0, stale=stale,
                signal_csv=cache if sig is not None else None)


def window_mask(X, slow, use_fast: bool, use_slow: bool):
    """窗口完整（不需要 label）的交易日掩码。"""
    m = ~np.isnan(X).any(axis=(1, 2)) if use_fast else np.ones(len(X), bool)
    if use_slow:
        m = m & ~np.isnan(slow).any(axis=1)
    return m


def predict_series(fp, year: int, view: str, seeds=SEEDS,
                   save_dir: str | None = None) -> pd.Series:
    """用已保存的模型对该年**所有窗口完整**的交易日预测（含缺 label 的最新几天）。

    返回 Series（index=日期，值=该日信号 = 6 seed × 64 维的均值，与 archive 口径一致）。

    `fp=None` 表示用默认源（DB 优先），由 `resolve_fp` 解析 —— 调用方（`live/db_update.py`、
    `live/model_registry.py`）都传 None，別再把 None 直接喂给 `ensure_cache`。
    """
    fp = resolve_fp(fp)
    dates, X, slow, _labels, _valid, _slow_ok = tm.ensure_cache(fp)
    d = pd.to_datetime(dates)
    use_fast = view in ("pv", "dual")
    use_slow = view in ("slow", "dual")
    m = window_mask(X, slow, use_fast, use_slow)
    m &= (d >= pd.Timestamp(year=year, month=1, day=1))
    m &= (d < pd.Timestamp(year=year + 1, month=1, day=1))
    idx = np.where(m)[0]
    if not idx.size:
        return pd.Series(dtype=np.float32)
    preds = []
    for seed in seeds:
        fp_m = model_path(year, view, seed, save_dir)
        if not os.path.isfile(fp_m):
            continue
        net, ck = load_model(year, view, seed, save_dir)
        sc = ck["scalers"]
        if use_fast:
            Xq = (X[idx] - sc["fmu"]) / sc["fsd"]
        else:
            Xq = np.zeros((idx.size, tm.N, tm.FAST_DIM), np.float32)
        if use_slow:
            S = (slow[idx] - sc["smu"]) / sc["ssd"]
        else:
            S = np.zeros((idx.size, 0), np.float32)
        preds.append(tm.predict(net, Xq, S))
    if not preds:
        return pd.Series(dtype=np.float32)
    sig = np.mean(preds, axis=0).mean(axis=1)
    return pd.Series(sig, index=d[idx])


def signal_csv_path(year: int, view: str, save_dir: str | None = None) -> str:
    out_dir = save_dir or (getattr(_P, f"MODELS_{view.upper()}", None) or
                           os.path.join(_P.MODELS, view))
    return os.path.join(out_dir, f"{view}_{year}_signals.csv")


def year_signal_csv(fp, year: int, view: str, seeds=SEEDS,
                    save_dir: str | None = None, force: bool = False,
                    verbose: bool = True) -> pd.DataFrame:
    """返回该年的信号表（列：日期 + 信号00.. + label），并把结果缓存成 CSV。

    缓存位置：models/{view}/{view}_{year}_signals.csv（与 .pt 并列）
    `fp=None` 表示用默认源（DB 优先）。
    """
    fp = resolve_fp(fp)
    out_dir = save_dir or (getattr(_P, f"MODELS_{view.upper()}", None) or
                           os.path.join(_P.MODELS, view))
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, f"{view}_{year}_signals.csv")
    if os.path.isfile(cache) and not force:
        return pd.read_csv(cache, encoding="utf-8-sig", parse_dates=["日期"])

    info = ensure_year(fp, year, view, seeds, out_dir, force, verbose)
    fold, sig = info["fold"], info["sig"]
    te_idx = fold["te_idx"]
    rows = []
    for i, ti in enumerate(te_idx):
        rows.append([fold["dates"][ti]] + list(sig[i]))
    df = pd.DataFrame(rows, columns=["日期"] + _SIG_COLS)
    df = df.sort_values("日期").reset_index(drop=True)
    df.to_csv(cache, index=False, encoding="utf-8-sig")
    return df


def missing_years(fp, years, views=("pv", "slow"), seeds=SEEDS,
                  save_dir=None) -> dict:
    """扫描哪些 (view, year) 还没有模型，供实盘"检测-训练"提示。"""
    miss = {}
    for view in views:
        need = [y for y in years
                if not all(os.path.isfile(model_path(y, view, s, save_dir)) for s in seeds)]
        if need:
            miss[view] = need
    return miss


if __name__ == "__main__":
    fp = features_source()
    y = int(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp.today().year
    print(f"检测 {y} 年模型：")
    print("  缺失：", missing_years(fp, [y]) or "无")
    for view in ("pv", "slow"):
        df = year_signal_csv(fp, y, view)
        print(f"  {view} {y}: {len(df)} 日，{df['日期'].min().date()} ~ {df['日期'].max().date()}")

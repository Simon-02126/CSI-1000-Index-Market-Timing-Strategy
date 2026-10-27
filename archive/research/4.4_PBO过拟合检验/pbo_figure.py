# -*- coding: utf-8 -*-
"""把 pbo_cscv.py 的结果画成一张四联图：05_报告与图表/PBO_CSCV_过拟合检验.png"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
import _paths as _P  # noqa: E402
sys.path.insert(0, _HERE)

import pbo_cscv as pb  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SHIP = pb.cfg_key(pb.SHIPPED)
FIG = os.path.join(_P.REPORTS, "PBO_CSCV_过拟合检验.png")


def load(name):
    z = np.load(os.path.join(_HERE, name), allow_pickle=True)
    return z["R"].astype(np.float64), list(z["keys"])


def main():
    res = json.load(open(os.path.join(_HERE, "results_pbo.json"), encoding="utf-8"))
    R, keys = load("returns_matrix.npz")
    Rs, keyss = load("returns_matrix_struct.npz")

    keep = pb.dedupe(R)
    Rf = R[:, keep]
    kf = [keys[i] for i in keep]
    jf = kf.index(SHIP)
    keeps = pb.dedupe(Rs)
    Rsf = Rs[:, keeps]
    ksf = [keyss[i] for i in keeps]
    js = ksf.index(SHIP)

    c6 = pb.cscv(Rf, S=16, keep_ranks=[jf])
    cs = pb.cscv(Rsf, S=16, keep_ranks=[js])

    fig, ax = plt.subplots(2, 2, figsize=(13.2, 8.6))

    # ---- (1) PBO 随候选集扩大 ----
    a = ax[0][0]
    lv = res["levels"]
    names = ["L1", "L2", "L3", "L4", "L5", "L6 S=10", "L6 S=16", "L6 S=20"]
    labels = [f"{n}\nN={s['N']}" for n, s in zip(names, lv)]
    vals = [s["pbo"] for s in lv]
    a.bar(range(len(vals)), vals, color="#3b6ea5", width=0.62)
    sp = res["struct"]["levels"][1]["pbo"]
    a.bar([len(vals)], [sp], color="#c0504d", width=0.62)
    a.set_xticks(range(len(vals) + 1))
    a.set_xticklabels(labels + [f"结构层\nN={res['struct']['n_unique']}"], fontsize=8.5)
    a.axhline(0.2, ls=":", c="gray", lw=1)
    a.axhline(0.5, ls="--", c="#c0504d", lw=1)
    a.text(len(vals) + 0.45, 0.505, "0.5 过拟合警戒", ha="right", va="bottom",
           fontsize=8, color="#c0504d")
    for i, v in enumerate(vals + [sp]):
        a.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=8.5)
    a.set_ylim(0, 0.62)
    a.set_ylabel("PBO")
    a.set_title("(1) PBO 随候选集扩大而上升；跨到结构维度站上 0.30", fontsize=10.5)

    # ---- (2) logit(λ) 分布 ----
    a = ax[0][1]
    bins = np.linspace(-6, 9, 46)
    a.hist(c6["lam"], bins=bins, density=True, alpha=0.62, color="#3b6ea5",
           label=f"参数层 N={Rf.shape[1]}  PBO=0.119")
    a.hist(cs["lam"], bins=bins, density=True, alpha=0.55, color="#c0504d",
           label=f"结构层 N={Rsf.shape[1]}  PBO=0.297")
    a.axvline(0, c="k", lw=1.2)
    a.text(0.15, a.get_ylim()[1] * 0.92, "λ=0（样本外中位）", fontsize=8)
    a.set_xlabel(r"$\lambda=\mathrm{logit}(\omega)$")
    a.set_ylabel("密度")
    a.legend(fontsize=8, frameon=False)
    a.set_title("(2) 样本内最优在样本外的相对排名", fontsize=10.5)

    # ---- (3) IS 超额 vs OOS 超额（扣掉市场共同因子） ----
    a = ax[1][0]
    ie6 = (c6["is_sel"] - c6["is_med"]) * np.sqrt(252)
    oe6 = (c6["os_sel"] - c6["os_med"]) * np.sqrt(252)
    ies = (cs["is_sel"] - cs["is_med"]) * np.sqrt(252)
    oes = (cs["os_sel"] - cs["os_med"]) * np.sqrt(252)
    a.scatter(ie6, oe6, s=3, alpha=0.22, color="#3b6ea5", edgecolors="none")
    a.scatter(ies, oes, s=3, alpha=0.22, color="#c0504d", edgecolors="none")
    a.axhline(0, c="gray", lw=0.8)
    a.axvline(0, c="gray", lw=0.8)
    for x, y, lab, col in ((ie6, oe6, "参数层", "#3b6ea5"), (ies, oes, "结构层", "#c0504d")):
        b1, b0 = np.polyfit(x, y, 1)
        xs = np.linspace(np.percentile(x, 2), np.percentile(x, 98), 2)
        a.plot(xs, b0 + b1 * xs, c=col, lw=1.8, label=f"{lab} 斜率 {b1:+.2f}")
    a.set_xlabel("选中组合的 IS 超额夏普（年化，相对同切分中位）")
    a.set_ylabel("同一组合的 OOS 超额夏普（年化）")
    a.legend(fontsize=8, frameon=False, loc="lower left")
    a.set_title("(3) 扣掉市场共同因子后仍是负斜率（赢家诅咒）", fontsize=10.5)

    # ---- (4) 出厂配置的 OOS 相对排名 ----
    a = ax[1][1]
    a.hist(c6["ranks"][jf], bins=40, density=True, alpha=0.62, color="#3b6ea5",
           label=f"参数层 N={Rf.shape[1]}")
    a.hist(cs["ranks"][js], bins=40, density=True, alpha=0.55, color="#c0504d",
           label=f"结构层 N={Rsf.shape[1]}")
    a.axvline(0.5, ls="--", c="k", lw=1)
    a.text(0.505, a.get_ylim()[1] * 0.9, "样本外中位", fontsize=8)
    a.set_xlabel("出厂配置在 OOS 中的相对排名 ω")
    a.set_ylabel("密度")
    a.legend(fontsize=8, frameon=False)
    a.set_title(f"(4) 出厂配置几乎从不掉到中位以下（中位 ω="
                f"{np.median(c6['ranks'][jf]):.2f} / {np.median(cs['ranks'][js]):.2f}）",
                fontsize=10.5)

    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print("saved", FIG)

    out = dict(
        param=dict(pbo=round(float((c6["lam"] <= 0).mean()), 4),
                   rank_median=round(float(np.median(c6["ranks"][jf])), 4),
                   below=round(float((c6["ranks"][jf] <= 0.5).mean()), 4),
                   is_sharpe_mean=round(float(c6["is_sel"].mean() * np.sqrt(252)), 3),
                   oos_sharpe_mean=round(float(c6["os_sel"].mean() * np.sqrt(252)), 3)),
        struct=dict(pbo=round(float((cs["lam"] <= 0).mean()), 4),
                    rank_median=round(float(np.median(cs["ranks"][js])), 4),
                    below=round(float((cs["ranks"][js] <= 0.5).mean()), 4),
                    is_sharpe_mean=round(float(cs["is_sel"].mean() * np.sqrt(252)), 3),
                    oos_sharpe_mean=round(float(cs["os_sel"].mean() * np.sqrt(252)), 3)),
    )
    with open(os.path.join(_HERE, "figure_stats.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

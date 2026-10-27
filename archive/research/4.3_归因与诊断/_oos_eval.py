import os as _os_path, sys as _sys_path
_d = _os_path.path.dirname(_os_path.path.abspath(__file__))
while _d != _os_path.path.dirname(_d) and not _os_path.path.isfile(
        _os_path.path.join(_d, "_paths.py")):
    _d = _os_path.path.dirname(_d)
_sys_path.path.insert(0, _d)
import _paths as _P  # noqa: E402  统一路径入口

import os, glob, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
BASE = _P.ROOT
TRAIN = _P.TRAIN
ANN=252
def read_any(fp):
    for enc in ("gbk","utf-8-sig","utf-8"):
        try: return pd.read_csv(fp,encoding=enc)
        except (UnicodeDecodeError,LookupError): continue
fp=glob.glob(os.path.join(BASE,"input","*中证1000*.csv"))[0]; df=read_any(fp)
dc=[c for c in df.columns if "日期" in c][0]; cc=[c for c in df.columns if "收盘" in c][0]
close=pd.Series(pd.to_numeric(df[cc],errors="coerce").values,index=pd.to_datetime(df[dc]))
close=close[~close.index.duplicated(keep="last")].sort_index()
def load_tag(tag):
    parts=[]
    for p in sorted(glob.glob(os.path.join(TRAIN,f"{tag}_*.csv"))):
        x=pd.read_csv(p,encoding="utf-8-sig",parse_dates=["日期"])
        cols=[c for c in x.columns if c.startswith("信号") and c!="信号均值"]
        s=pd.Series(x[cols].mean(axis=1).values,index=x["日期"]); parts.append(s[~s.index.duplicated(keep="last")])
    return pd.concat(parts).sort_index() if parts else None
def pe(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);m=np.isfinite(x)&np.isfinite(y);x,y=x[m],y[m]
    if len(x)<10: return np.nan
    return float(((x-x.mean())*(y-y.mean())).mean()/(x.std()*y.std())) if x.std() and y.std() else np.nan
def roll_pct(s,win=750,minp=250):
    v=s.values.astype(float); o=np.full(len(v),np.nan)
    for i in range(len(v)):
        if i<minp: continue
        lo=max(0,i-win); o[i]=(v[lo:i]<v[i]).sum()/(i-lo)
    return pd.Series(o,index=s.index)
def tgt3(p,lo=.30,hi=.80): return pd.Series(np.where(p<lo,0.,np.where(p<hi,.5,1.)),index=p.index).where(p.notna())
def pos_mode(t,k):
    v=t.values; o=np.full(len(v),np.nan)
    for i in range(len(v)):
        w=v[max(0,i-k+1):i+1]; w=w[~np.isnan(w)]
        if not len(w): continue
        vals,cnt=np.unique(w,return_counts=True); o[i]=vals[np.argmax(cnt)]
    return pd.Series(o,index=t.index)
def bt(pos,start,end,cost=.0003):
    d=pd.DataFrame({"p":pos,"c":close}).dropna(); d=d[(d.index>=pd.Timestamp(start))&(d.index<=pd.Timestamp(end))]
    if len(d)<60: return None
    r=np.log(d["c"]).diff(); turn=d["p"].diff().abs().fillna(0)
    st=(d["p"].shift(1)*r-turn*cost).dropna()
    eq=st.cumsum(); mdd=float(np.expm1((eq-eq.cummax()).min()))
    ann=float(st.mean()*ANN); vol=float(st.std()*np.sqrt(ANN))
    return dict(sharpe=ann/vol if vol else np.nan,ann=ann,mdd=mdd,avg=float(d["p"].mean()),
                chg=float((turn>1e-9).sum())/(len(d)/ANN),cum=float(st.sum()))
S={"单10日":load_tag("h10"),"双期MSE":load_tag("d5_10_mse"),"三期IC":load_tag("d5_10_20_ic")}
print("=== 各信号覆盖区间（2025-2026 是我所有实验都没碰过的真样本外）===")
for nm,s in S.items(): print(f"  {nm:<8} {s.index.min().date()}~{s.index.max().date()}  {len(s)} 日")
print("\n=== A. 真样本外 IC（自算前瞻收益）===")
print(f"  {'信号':<8}{'年':>6}{'IC(H=10)':>10}{'IC(H=20)':>10}{'n':>6}")
for nm,s in S.items():
    for y in (2025,2026):
        row=[]
        for h in (10,20):
            fwd=np.log(close.shift(-h)/close)
            d=pd.DataFrame({"s":s,"f":fwd.reindex(s.index)}).dropna(); d=d[d.index.year==y]
            row.append(pe(d["s"].values,d["f"].values))
        d=pd.DataFrame({"s":s}).dropna(); d=d[d.index.year==y]
        print(f"  {nm:<8}{y:>6}{row[0]:>10.4f}{row[1]:>10.4f}{len(d):>6}")
print("\n=== B. 策略层（三档0.30/0.80 + 众数5），含分位warmup=750/250 ===")
bh=pd.Series(1.0,index=close.index)
print(f"  {'信号/窗口':<14}{'2025夏普':>9}{'2025年化':>9}{'2025收益':>9}{'均仓':>6}"
      f"{'2026夏普':>9}{'2026年化':>9}{'2026收益':>9}{'均仓':>6}")
r=bt(bh,"2025-01-01","2025-12-31"); r2=bt(bh,"2026-01-01","2099-12-31")
print(f"  {'买入持有':<14}{r['sharpe']:>9.2f}{r['ann']:>9.1%}{r['cum']:>9.1%}{r['avg']:>6.0%}"
      f"{r2['sharpe']:>9.2f}{r2['ann']:>9.1%}{r2['cum']:>9.1%}{r2['avg']:>6.0%}")
for nm,s in S.items():
    for W in (150,750):
        pos=pos_mode(tgt3(roll_pct(s,W)),5)
        a=bt(pos,"2025-01-01","2025-12-31"); b=bt(pos,"2026-01-01","2099-12-31")
        if a is None or b is None: continue
        print(f"  {nm+' W='+str(W):<14}{a['sharpe']:>9.2f}{a['ann']:>9.1%}{a['cum']:>9.1%}{a['avg']:>6.0%}"
              f"{b['sharpe']:>9.2f}{b['ann']:>9.1%}{b['cum']:>9.1%}{b['avg']:>6.0%}")
print("\n=== C. 分位覆盖度（2019-2026 全区间）===")
print(f"  {'信号/W':<14}{'q80':>7}{'q90':>7}{'满仓天数':>9}{'满仓%':>7}")
for nm,s in S.items():
    for W in (150,750):
        p=roll_pct(s,W).dropna()
        print(f"  {nm+' W='+str(W):<14}{p.quantile(.8):>7.2f}{p.quantile(.9):>7.2f}"
              f"{int((p>=.80).sum()):>9}{(p>=.80).mean():>7.0%}")

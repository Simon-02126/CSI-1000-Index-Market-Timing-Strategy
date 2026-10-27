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
ANN=252; START="2019-01-01"; END="2024-12-31"
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
def roll_pct(s,win=750,minp=250):
    v=s.values.astype(float); o=np.full(len(v),np.nan)
    for i in range(len(v)):
        if i<minp: continue
        lo=max(0,i-win); o[i]=(v[lo:i]<v[i]).sum()/(i-lo)
    return pd.Series(o,index=s.index)
def pe(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);m=np.isfinite(x)&np.isfinite(y);x,y=x[m],y[m]
    if len(x)<10: return np.nan
    return float(((x-x.mean())*(y-y.mean())).mean()/(x.std()*y.std())) if x.std() and y.std() else np.nan
S={"单10日":load_tag("h10"),"双期5+10 MSE":load_tag("d5_10_mse"),"双期5+10 IC":load_tag("d5_10_ic")}
print("=== A. 信号时序结构（2019-2024）===")
print(f"  {'信号':<14}{'std':>10}{'AR1':>8}{'AR5':>8}{'AR20':>8}{'各年std变异':>12}")
for nm,s in S.items():
    v=s.values
    ar1=pe(v[1:],v[:-1]); ar5=pe(v[5:],v[:-5]); ar20=pe(v[20:],v[:-20])
    yr=[g.std() for _,g in s.groupby(s.index.year)]
    print(f"  {nm:<14}{v.std():>10.4f}{ar1:>8.3f}{ar5:>8.3f}{ar20:>8.3f}{np.std(yr)/np.mean(yr):>12.2f}")
print("\n=== B. 滚动分位(pct)的分布与三档占比 ===")
print(f"  {'信号':<14}{'pct均值':>9}{'q10':>7}{'q50':>7}{'q90':>7}{'空仓':>7}{'半仓':>7}{'满仓':>7}")
P={}
for nm,s in S.items():
    p=roll_pct(s); P[nm]=p
    pv=p.dropna(); pv=pv[(pv.index>=START)&(pv.index<=END)]
    print(f"  {nm:<14}{pv.mean():>9.3f}{pv.quantile(.1):>7.2f}{pv.median():>7.2f}{pv.quantile(.9):>7.2f}"
          f"{(pv<0.30).mean():>7.0%}{((pv>=0.30)&(pv<0.80)).mean():>7.0%}{(pv>=0.80).mean():>7.0%}")
print("\n=== C. 分箱前瞻收益（非重叠, offset=0）===")
for h in (10,20):
    fwd=np.log(close.shift(-h)/close)
    print(f"  H={h}")
    for nm,s in S.items():
        d=pd.DataFrame({"s":roll_pct(s),"f":fwd.reindex(s.index)}).dropna()
        d=d[(d.index>=START)&(d.index<=END)].iloc[::h]
        q=pd.qcut(d["s"],5,labels=False,duplicates="drop")
        m=[d["f"][q==k].mean()*100 for k in range(5)]
        print(f"    {nm:<14}"+"".join(f"{v:>8.2f}" for v in m)+f"   多空{m[4]-m[0]:>6.2f}")
print("\n=== D. 逐年：档位分布 + 分位与前瞻10日收益的年内相关 ===")
fwd10=np.log(close.shift(-10)/close)
for nm in S:
    print(f"  --- {nm} ---")
    for y in range(2019,2025):
        p=P[nm]; pv=p[p.index.year==y].dropna()
        f=fwd10.reindex(pv.index)
        print(f"    {y}: pct均值{pv.mean():.3f} 空{(pv<.30).mean():>4.0%} 半{((pv>=.30)&(pv<.80)).mean():>4.0%} "
              f"满{(pv>=.80).mean():>4.0%}  年内IC={pe(pv.values,f.values):+.3f}")

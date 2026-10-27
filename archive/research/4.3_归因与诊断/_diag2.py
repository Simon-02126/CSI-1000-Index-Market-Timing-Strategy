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
BASE = _P.ROOT; TRAIN = _P.TRAIN
START,END="2019-01-01","2024-12-31"
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
S={"单10日":load_tag("h10"),"双期MSE":load_tag("d5_10_mse"),"双期IC":load_tag("d5_10_ic"),"三期IC":load_tag("d5_10_20_ic")}
print("=== 信号结构 + 分位分布 ===")
print(f"  {'信号':<9}{'AR1':>7}{'AR20':>7}{'q50':>6}{'q80':>6}{'q90':>6}{'满仓天数':>9}{'空仓':>6}{'半仓':>6}{'满仓':>6}")
for nm,s in S.items():
    v=s.values; ar1=pe(v[1:],v[:-1]); ar20=pe(v[20:],v[:-20])
    p=roll_pct(s).dropna(); p=p[(p.index>=START)&(p.index<=END)]
    print(f"  {nm:<9}{ar1:>7.3f}{ar20:>7.3f}{p.median():>6.2f}{p.quantile(.8):>6.2f}{p.quantile(.9):>6.2f}"
          f"{(p>=0.80).sum():>9}{(p<0.30).mean():>6.0%}{((p>=0.30)&(p<0.80)).mean():>6.0%}{(p>=0.80).mean():>6.0%}")
print("\n=== 分箱前瞻收益（非重叠 offset=0, 单位 %）===")
for h in (10,20):
    fwd=np.log(close.shift(-h)/close)
    print(f"  H={h}: " + "".join(f"{'Q'+str(k+1):>9}" for k in range(5))+"     多空")
    for nm,s in S.items():
        d=pd.DataFrame({"s":roll_pct(s),"f":fwd.reindex(s.index)}).dropna()
        d=d[(d.index>=START)&(d.index<=END)].iloc[::h]
        q=pd.qcut(d["s"],5,labels=False,duplicates="drop")
        m=[d["f"][q==k].mean()*100 for k in range(5)]
        print(f"    {nm:<9}"+"".join(f"{x:>9.2f}" for x in m)+f"{m[4]-m[0]:>9.2f}")

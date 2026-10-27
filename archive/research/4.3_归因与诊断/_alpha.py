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
ANN,COST=252,0.0003
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
def roll_pct(s,win,minp=250):
    v=s.values.astype(float);o=np.full(len(v),np.nan);mp=min(minp,max(60,win//3))
    for i in range(len(v)):
        if i<mp: continue
        lo=max(0,i-win); o[i]=(v[lo:i]<v[i]).sum()/(i-lo)
    return pd.Series(o,index=s.index)
def tgt3(p,lo=.30,hi=.80): return pd.Series(np.where(p<lo,0.,np.where(p<hi,.5,1.)),index=p.index).where(p.notna())
def stretch(p,lo,hi): return ((p-lo)/(hi-lo)).clip(0,1)
def pos_mode(t,k=5):
    v=t.values;o=np.full(len(v),np.nan)
    for i in range(len(v)):
        w=v[max(0,i-k+1):i+1];w=w[~np.isnan(w)]
        if not len(w): continue
        vals,cnt=np.unique(w,return_counts=True);o[i]=vals[np.argmax(cnt)]
    return pd.Series(o,index=t.index)
pv=pd.concat([load_tag(f"s{sd}") for sd in range(6)],axis=1).mean(axis=1).dropna()
sl=pd.concat([load_tag(f"slow_s{sd}") for sd in range(6)],axis=1).mean(axis=1).dropna()
m=pd.DataFrame({"p":roll_pct(pv,300),"s":roll_pct(sl,300)}).dropna()
pos=pos_mode(tgt3(0.5*m["p"]+0.5*stretch(m["s"],0.30,0.65),0.30,0.80),5)
d=pd.DataFrame({"p":pos,"c":close}).dropna(); d=d[d.index>="2019-06-01"]
r=np.log(d["c"]).diff().fillna(0); turn=d["p"].diff().abs().fillna(0)
st=d["p"].shift(1).fillna(0)*r-turn*COST
ex=st-r
print("="*100); print("择时 alpha 的量化：beta 调整后到底有多少"); print("="*100)
for lbl,a,b in [("全期 2019-06~2026-09","2019-06-01","2099"),("2019-2024","2019-06-01","2024-12-31"),("2025-2026","2025-01-01","2099")]:
    mask=(ex.index>=pd.Timestamp(a))&(ex.index<=pd.Timestamp(b))
    x=r[mask]; y=st[mask]
    if len(x)<100: continue
    beta=np.polyfit(x.values,y.values,1)[0]
    alpha_d=(y.values-beta*x.values).mean()
    exs=ex[mask]
    ir=exs.mean()/exs.std()*np.sqrt(ANN) if exs.std() else np.nan
    t=exs.mean()/exs.std()*np.sqrt(len(exs)) if exs.std() else np.nan
    yrs=len(x)/ANN
    print(f"\n  --- {lbl}（{len(x)}日 / {yrs:.1f}年）---")
    print(f"    策略年化 {(np.expm1(y.sum()/yrs)):.2%} | 指数年化 {(np.expm1(x.sum()/yrs)):.2%}")
    print(f"    实际 beta {beta:.3f} | 日均仓位 {d['p'][mask].mean():.1%}")
    print(f"    beta 调整后期望收益 {np.expm1(beta*x.sum()/yrs):.2%} → **年化 alpha {np.expm1(alpha_d*ANN)-1:.2%}**")
    print(f"    超额收益（年化）{np.expm1(exs.sum()/yrs)-1:.2%} | 超额波动 {exs.std()*np.sqrt(ANN):.2%}")
    print(f"    **信息比率 IR = {ir:.2f}** | 超额 t 值 = {t:.2f} | 超额日胜率 {(exs>0).mean():.1%}")
    print(f"    单位敞口收益 {np.expm1(y.sum()/yrs/d['p'][mask].mean())-1:.2%} vs 指数 {np.expm1(x.sum()/yrs)-1:.2%}")

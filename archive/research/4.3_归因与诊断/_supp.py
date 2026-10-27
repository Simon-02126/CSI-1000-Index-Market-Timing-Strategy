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
ANN=252
def read_any(fp):
    for enc in ("gbk","utf-8-sig","utf-8"):
        try: return pd.read_csv(fp,encoding=enc)
        except (UnicodeDecodeError,LookupError): continue
fp=glob.glob(os.path.join(BASE,"input","*中证1000*.csv"))[0]; df=read_any(fp)
dc=[c for c in df.columns if "日期" in c][0]; cc=[c for c in df.columns if "收盘" in c][0]
close=pd.Series(pd.to_numeric(df[cc],errors="coerce").values,index=pd.to_datetime(df[dc]))
close=close[~close.index.duplicated(keep="last")].sort_index(); fwd10=np.log(close.shift(-10)/close)
def load_tag(tag):
    parts=[]
    for p in sorted(glob.glob(os.path.join(TRAIN,f"{tag}_*.csv"))):
        x=pd.read_csv(p,encoding="utf-8-sig",parse_dates=["日期"])
        cols=[c for c in x.columns if c.startswith("信号") and c!="信号均值"]
        s=pd.Series(x[cols].mean(axis=1).values,index=x["日期"]); parts.append(s[~s.index.duplicated(keep="last")])
    return pd.concat(parts).sort_index() if parts else None
def load_dir(v):
    p=glob.glob(os.path.join(BASE,"output","signals",f"*中证1000*_{v}_signals.csv"))[0]
    x=pd.read_csv(p,encoding="utf-8-sig",parse_dates=["日期"]); cols=[c for c in x.columns if c.startswith("信号") and c!="信号均值"]
    x=x.drop_duplicates("日期").set_index("日期").sort_index(); return x[cols].mean(axis=1)
def pe(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float);m=np.isfinite(x)&np.isfinite(y);x,y=x[m],y[m]
    if len(x)<10: return np.nan
    return float(((x-x.mean())*(y-y.mean())).mean()/(x.std()*y.std())) if x.std() and y.std() else np.nan
def roll_pct(s,win=750,minp=250):
    v=s.values.astype(float);o=np.full(len(v),np.nan);mp=min(minp,max(60,win//3))
    for i in range(len(v)):
        if i<mp: continue
        lo=max(0,i-win); o[i]=(v[lo:i]<v[i]).sum()/(i-lo)
    return pd.Series(o,index=s.index)
def tgt3(p,lo=.30,hi=.80): return pd.Series(np.where(p<lo,0.,np.where(p<hi,.5,1.)),index=p.index).where(p.notna())
def pos_mode(t,k=5):
    v=t.values;o=np.full(len(v),np.nan)
    for i in range(len(v)):
        w=v[max(0,i-k+1):i+1];w=w[~np.isnan(w)]
        if not len(w): continue
        vals,cnt=np.unique(w,return_counts=True);o[i]=vals[np.argmax(cnt)]
    return pd.Series(o,index=t.index)
def bt(pos,start="2019-01-01",cost=.0003):
    d=pd.DataFrame({"p":pos,"c":close}).dropna(); d=d[d.index>=pd.Timestamp(start)]
    if len(d)<100: return np.nan
    r=np.log(d["c"]).diff();turn=d["p"].diff().abs().fillna(0)
    st=(d["p"].shift(1)*r-turn*cost).dropna(); ann=float(st.mean()*ANN);vol=float(st.std()*np.sqrt(ANN))
    return ann/vol if vol else np.nan
slow=load_dir("slow"); ps=roll_pct(slow)
print("=== slow 单因子基准:", round(bt(pos_mode(tgt3(ps))),3), "===")
print("\n=== A. seed 平均的副作用：分箱多空(H=10) 会随平均而缩小 ===")
print(f"  {'信号':<18}{'分箱多空%':>10}{'信号std':>9}{'AR1':>7}")
for nm,keys in [("单seed s0",["s0"]),("2seed平均(h10)",["h10"]),("6seed平均",[f"s{i}" for i in range(6)])]:
    ss=[load_tag(k) for k in keys]; ss=[s for s in ss if s is not None]
    if not ss: continue
    a=pd.concat(ss,axis=1).mean(axis=1).dropna()
    d=pd.DataFrame({"s":roll_pct(a),"f":fwd10.reindex(a.index)}).dropna(); d=d[d.index>="2019-01-01"].iloc[::10]
    q=pd.qcut(d["s"],5,labels=False,duplicates="drop"); m=[d["f"][q==k].mean()*100 for k in range(5)]
    v=a.values
    print(f"  {nm:<18}{m[4]-m[0]:>10.2f}{v.std():>9.4f}{pe(v[1:],v[:-1]):>7.3f}")
print("\n=== B. 样本外 pooled IC（2025-2026 合并）===")
print(f"  {'信号':<18}{'2025-26 pooled IC':>18}{'2025':>8}{'2026':>8}")
for nm,keys in [("单10日(6seed)",[f"s{i}" for i in range(6)]),("单10日(2seed h10)",["h10"]),
                ("双期MSE(5seed)",[f"d5_10_mse_s{i}" for i in range(5)]),("双期MSE(2seed)",["d5_10_mse"]),
                ("三期IC(2seed)",["d5_10_20_ic"])]:
    ss=[load_tag(k) for k in keys]; ss=[s for s in ss if s is not None]
    if not ss: continue
    a=pd.concat(ss,axis=1).mean(axis=1).dropna()
    d=pd.DataFrame({"s":a,"f":fwd10.reindex(a.index)}).dropna()
    oos=d[d.index>="2025-01-01"]
    print(f"  {nm:<18}{pe(oos['s'].values,oos['f'].values):>18.3f}"
          f"{pe(d[d.index.year==2025]['s'].values,d[d.index.year==2025]['f'].values):>8.3f}"
          f"{pe(d[d.index.year==2026]['s'].values,d[d.index.year==2026]['f'].values):>8.3f}")
print("\n=== C. 用 seed 平均信号做组合（与 slow 0.5/0.5，仓位层加权）===")
print(f"  {'信号':<18}{'分位平均':>10}{'仓位层加权':>11}{'各seed仓位层':>34}")
for nm,keys in [("单10日",[f"s{i}" for i in range(6)]),("双期MSE",[f"d5_10_mse_s{i}" for i in range(5)]),
                ("三期IC",["d5_10_20_ic"]),("双期IC",["d5_10_ic"])]:
    ss=[load_tag(k) for k in keys]; ss=[s for s in ss if s is not None]
    if not ss: continue
    a=pd.concat(ss,axis=1).mean(axis=1).dropna()
    m1=pd.DataFrame({"pv":roll_pct(a),"sl":ps}).dropna()
    pct=(0.5*m1["pv"]+0.5*m1["sl"])
    pA=bt(pos_mode(tgt3(pct)))
    # 仓位层加权: 各自映射后加权
    pos=(0.5*tgt3(roll_pct(a)).fillna(0)+0.5*tgt3(ps).fillna(0))
    pB=bt(pos)
    cs=[]
    for s in ss:
        mm=pd.DataFrame({"pv":roll_pct(s),"sl":ps}).dropna()
        posi=0.5*tgt3(mm["pv"]).fillna(0)+0.5*tgt3(mm["sl"]).fillna(0)
        cs.append(bt(posi))
    cs=np.array(cs)
    print(f"  {nm:<18}{pA:>10.3f}{pB:>11.3f}   " + " ".join(f"{c:.2f}" for c in cs) + f"  min{cs.min():.2f}")

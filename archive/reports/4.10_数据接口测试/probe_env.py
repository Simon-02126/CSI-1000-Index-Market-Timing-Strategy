# -*- coding: utf-8 -*-
"""0. 环境自检：数据源相关包是否齐全"""
import importlib
import sys

print(f"Python {sys.version.split()[0]}  {sys.executable}")
for m in ["baostock", "akshare", "requests", "tushare", "efinance", "adata", "pandas"]:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "?")
        print(f"  {m:10} OK       version={ver}")
    except Exception as e:
        print(f"  {m:10} MISSING  ({type(e).__name__}: {e})")

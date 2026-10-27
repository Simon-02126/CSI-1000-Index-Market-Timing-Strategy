// 择时数据看板后端 (Node 版, 零第三方依赖, 仅内置 node:sqlite)
// 用法: node server.js [--host 127.0.0.1] [--port 7100]
const http = require('http');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const { DatabaseSync } = require('node:sqlite');

const args = process.argv.slice(2);
let HOST = '127.0.0.1', PORT = 7100;
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--host') HOST = args[i + 1];
  if (args[i] === '--port') PORT = parseInt(args[i + 1], 10) || PORT;
}
// npm 追加参数时后者覆盖前者
if (args.filter(a => a === '--port').length > 1) {
  PORT = parseInt(args[args.lastIndexOf('--port') + 1], 10) || PORT;
}

const DB = path.join(__dirname, '..', 'db', 'timing.db');
const ROOT = path.join(__dirname, '..');
const INDEX_HTML = path.join(__dirname, 'index.html');

// Python 解释器: 优先用环境变量 PYTHON（或旧的 DSH_PYTHON），否则用 PATH 上的 python。
// Windows 上如果 PATH 里的 python 是 Microsoft Store 的转发桩（执行即 exit 9009），
// 请设置 PYTHON 指向真实解释器，例如：set PYTHON=C:\Python312\python.exe
const PYTHON_EXE = process.env.PYTHON || process.env.DSH_PYTHON || 'python';

if (!fs.existsSync(DB)) { console.error('db not found:', DB); process.exit(1); }

const db = new DatabaseSync(DB);
db.exec('PRAGMA query_only = ON');   // 看板绝不可能写库

const DATE_COL = {
  ledger: 'exec_date',
  signal_freeze: 'signal_date',
  model_files: null, ingest_log: null, quality_check: null,
};

function q(sql) {
  try { return db.prepare(sql).all(); } catch (e) { return { error: String(e.message || e) }; }
}

let updating = false;   // 更新任务串行化 (内存锁)

const server = http.createServer((req, res) => {
  const u = new URL(req.url, `http://${req.headers.host}`);
  res.setHeader('Content-Type', 'application/json; charset=utf-8');

  // -------- 收益/风险: 样本外绩效, 实时从 ledger 计算 (参数 ?rf=年化无风险利率, 默认0)
  if (u.pathname === '/api/perf') {
    const rf = parseFloat(u.searchParams.get('rf') || '0') || 0;
    const rows = q(`SELECT exec_date, daily_ret, position, nav, target_pos, cost
                    FROM ledger ORDER BY exec_date`);
    if (!rows || rows.error || !rows.length) {
      return res.end(JSON.stringify({ error: 'ledger 为空, 请先重建样本外账本' }));
    }
    const etfRows = q(`SELECT date, close FROM etf_daily WHERE code='512100.SH'
                       AND status='final' ORDER BY date`) || [];
    const idxRows = q(`SELECT date, close FROM index_daily WHERE code='000852.SH'
                       AND status='final' ORDER BY date`) || [];
    // 收盘价查找: 网格日 d 的收益 = close(d)/close(网格上一日) - 1
    const gridRet = (bars) => {
      const m = new Map(bars.map(b => [b.date, b.close]));
      return (dates) => dates.map((d, i) => {
        const c = m.get(d); if (!c) return null;
        if (i > 0 && m.get(dates[i - 1])) return c / m.get(dates[i - 1]) - 1;
        const j = bars.findIndex(b => b.date === d);
        return j > 0 ? c / bars[j - 1].close - 1 : null;   // 首日: 用前一自然日收盘
      });
    };
    const dates = rows.map(r => r.exec_date);
    const ret = rows.map(r => r.daily_ret || 0);
    const nav = rows.map(r => r.nav);
    const pos = rows.map(r => r.position);
    const benchRet = gridRet(etfRows)(dates);
    const idxRet = gridRet(idxRows)(dates);
    const cum = (rs) => { let v = 1; return rs.map(r => r === null ? null : (v *= 1 + r)); };
    const benchNav = cum(benchRet);
    const idxNav = cum(idxRet);

    // ---- 序列指标
    const mean = a => a.reduce((s, x) => s + x, 0) / a.length;
    const std = a => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / a.length); };
    const ann = Math.sqrt(252);
    const total = nav[nav.length - 1] / nav[0] - 1;
    const years = dates.length / 252;
    const annRet = Math.pow(1 + total, 1 / years) - 1;
    const vol = std(ret) * ann;
    const dside = std(ret.filter(x => x < 0)) || 1e-9;
    const ddSeries = (ns) => { let pk = -1e9; return ns.map(v => { if (v === null) return null; pk = Math.max(pk, v); return v / pk - 1; }); };
    const dd = ddSeries(nav), benchDd = ddSeries(benchNav);
    const minDD = (s) => { let m = 0, at = -1; s.forEach((v, i) => { if (v !== null && v < m) { m = v; at = i; } }); return { mdd: m, at: at < 0 ? null : dates[at] }; };
    const roll = (rs, w) => rs.map((_, i) => i + 1 < w ? null : std(rs.slice(i + 1 - w, i + 1)) * ann);
    const roll20 = roll(ret, 20), benchRoll20 = roll(benchRet.map(x => x ?? 0), 20);

    // ---- 超额 (相对满仓ETF)
    const excess = ret.map((r, i) => (benchRet[i] === null ? null : r - benchRet[i]));
    const exV = excess.filter(x => x !== null);
    const exNav = []; { let v = 1; excess.forEach(x => exNav.push(x === null ? null : (v *= 1 + x))); }
    const exTotal = exNav[exNav.length - 1] - 1;

    // ---- 直方图 (20 bins, 策略 vs 基准)
    const hv = ret.concat(benchRet.filter(x => x !== null));
    const lo = Math.min(...hv), hi = Math.max(...hv), NB = 20, bw = (hi - lo) / NB || 1e-9;
    const hist = { bins: Array.from({ length: NB }, (_, i) => lo + (i + 0.5) * bw), strat: new Array(NB).fill(0), bench: new Array(NB).fill(0) };
    const binOf = x => Math.max(0, Math.min(NB - 1, Math.floor((x - lo) / bw)));
    ret.forEach(x => hist.strat[binOf(x)]++);
    benchRet.forEach(x => { if (x !== null) hist.bench[binOf(x)]++; });

    // ---- 月度收益
    const monthly = {};
    dates.forEach((d, i) => { const m = d.slice(0, 7); (monthly[m] ||= { s: 1, b: 1 }); monthly[m].s *= 1 + ret[i]; if (benchRet[i] !== null) monthly[m].b *= 1 + benchRet[i]; });
    const monthlyArr = Object.entries(monthly).map(([m, v]) => ({ m, strat: v.s - 1, bench: v.b - 1 }));

    // ---- 极端行情 (指数口径): 大跌日防守 / 大涨日捕获
    const dn = [], up = [];
    idxRet.forEach((r, i) => { if (r === null) return; if (r <= -0.02) dn.push(i); if (r >= 0.02) up.push(i); });
    const avg = (arr, f) => arr.length ? arr.reduce((s, i) => s + f(i), 0) / arr.length : null;
    const avgIdx = arr => avg(arr, i => idxRet[i]) * 100, avgSt = arr => avg(arr, i => ret[i]) * 100;

    // ---- 仓位分组 vs 次日ETF收益
    const buckets = [[0, 0.3], [0.3, 0.7], [0.7, 1.01]];
    const posNext = buckets.map(([a, b]) => {
      const xs = [];
      for (let i = 0; i < pos.length - 1; i++)
        if (pos[i] >= a && pos[i] < b && benchRet[i + 1] !== null) xs.push(benchRet[i + 1]);
      return { n: xs.length, avg: xs.length ? mean(xs) * 100 : null };
    });

    // ---- 其他
    let streak = 0, maxStreak = 0;
    ret.forEach(r => { if (r < 0) { streak++; maxStreak = Math.max(maxStreak, streak); } else streak = 0; });
    const trades = rows.filter((r, i) => i === 0 || Math.abs(r.target_pos - rows[i - 1].target_pos) > 1e-9);
    const turnover = rows.reduce((s, r, i) => s + (i ? Math.abs(r.target_pos - rows[i - 1].target_pos) : r.target_pos), 0) / 2;
    const srt = ret.slice().sort((a, b) => a - b);
    const var95 = -srt[Math.floor(0.05 * srt.length)] * 100;
    const tail = srt.slice(0, Math.max(1, Math.floor(0.05 * srt.length)));
    const es95 = -mean(tail) * 100;

    const mddS = minDD(dd), mddB = minDD(benchDd), mddE = minDD(ddSeries(exNav));
    const winRate = ret.filter(x => x > 0).length / ret.length;
    return res.end(JSON.stringify({
      dates, nav, bench_nav: benchNav, idx_nav: idxNav, pos, dd, bench_dd: benchDd,
      roll20, bench_roll20: benchRoll20, hist, monthly: monthlyArr,
      rf,
      metrics: {
        n_days: dates.length, from: dates[0], to: dates[dates.length - 1],
        total, ann_ret: annRet, vol, sharpe: (mean(ret) - rf / 252) / (std(ret) || 1e-9) * ann,
        sortino: (mean(ret) - rf / 252) / dside * ann,
        calmar: (annRet - rf) / (Math.abs(mddS.mdd) || 1e-9),
        mdd: mddS.mdd, mdd_date: mddS.at, win_rate: winRate,
        cost: rows.reduce((s, r) => s + (r.cost || 0), 0),
        n_trades: trades.length, turnover,
        var95, es95, max_lose_streak: maxStreak,
      },
      excess: {
        total: exTotal, ann: Math.pow(1 + exTotal, 1 / years) - 1,
        ir: std(exV) ? mean(exV) / std(exV) * ann : null,
        mdd: mddE.mdd, mdd_date: mddE.at, nav: exNav,
      },
      bench_metrics: { total: benchNav[benchNav.length - 1] - 1, mdd: mddB.mdd, mdd_date: mddB.at },
      extreme: {
        down_n: dn.length, down_idx: avgIdx(dn), down_strat: avgSt(dn),
        up_n: up.length, up_idx: avgIdx(up), up_strat: avgSt(up),
      },
      pos_next: posNext,
    }));
  }

  // -------- 交易更新: 状态卡 (各源最新日 / T_max / 账本现状 / 待执行)
  if (u.pathname === '/api/state') {
    const g = (sql) => { const r = q(sql); return (r && r[0]) ? r[0].v : null; };
    const src = {
      index:  g(`SELECT MAX(date) v FROM index_daily WHERE code='000852.SH' AND status='final'`),
      etf:    g(`SELECT MAX(date) v FROM etf_daily WHERE code='512100.SH' AND status='final'`),
      if:     g(`SELECT MAX(date) v FROM if_sentiment WHERE variety='IF'`),
      margin: g(`SELECT MAX(effective_date) v FROM margin_daily WHERE exch='ALL'`),
      size:   g(`SELECT MAX(effective_date) v FROM market_size_daily WHERE sgn_code=11`),
    };
    const dates = Object.values(src).filter(Boolean);
    const tmax = dates.length ? dates.sort()[0] : null;
    const ledger = q(`SELECT * FROM ledger ORDER BY exec_date DESC LIMIT 1`)[0] || null;
    const pend = q(`SELECT COUNT(*) c FROM signal_freeze
                    WHERE signal_date > (SELECT COALESCE(MAX(signal_date),'') FROM ledger)`);
    return res.end(JSON.stringify({
      src, t_max: tmax,
      ledger_last: ledger ? {
        exec_date: ledger.exec_date, signal_date: ledger.signal_date,
        target_pos: ledger.target_pos, position: ledger.position,
        nav: ledger.nav === null ? null : Math.round(ledger.nav * 1e4) / 1e4,
        equity: ledger.equity, daily_ret: ledger.daily_ret,
      } : null,
      pending_count: (pend && pend[0]) ? pend[0].c : 0,
      pending: q(`SELECT signal_date, target_pos FROM signal_freeze
                  WHERE signal_date > (SELECT COALESCE(MAX(signal_date),'') FROM ledger)
                  ORDER BY signal_date`),
      last_signals: q(`SELECT * FROM signal_freeze ORDER BY signal_date DESC LIMIT 10`),
      last_fills: q(`SELECT * FROM ledger ORDER BY exec_date DESC LIMIT 10`),
    }));
  }

  // -------- 交易更新: 一键更新 (抓数->重建->信号->记账, Python 全链路)
  if (u.pathname === '/api/update' && req.method === 'POST') {
    if (updating) { res.statusCode = 409; return res.end(JSON.stringify({ error: '已有更新任务在运行, 请等待完成' })); }
    updating = true;
    // 解释器必须显式指定: PATH 上的 python 可能是 Microsoft Store 的 0 字节转发桩(exit 9009)。
    // 优先级: 环境变量 DSH_PYTHON > 顶部常量 PYTHON_EXE(硬编码 Anaconda 路径)
    const PY = PYTHON_EXE;
    if (!fs.existsSync(PY)) {
      updating = false;
      res.statusCode = 500;
      return res.end(JSON.stringify({
        code: -1, report: null, log: '',
        err: `[看板] 找不到 Python 解释器: ${PY}\n请设置环境变量 DSH_PYTHON 指向你的 python.exe，或修改 dashboard/server.js 顶部的 PYTHON_EXE 常量。`,
      }));
    }
    const py = spawn(PY, ['live/db_update.py', '--fetch'], {
      cwd: ROOT, env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
    });
    let out = '', err = '', responded = false;
    const done = (code) => {
      if (responded) return;                       // spawn error 与 close 都会触发, 只回一次
      responded = true;
      clearTimeout(killer); updating = false;
      let rep = null;
      const m = out.match(/__REPORT__([^\n]*)/);
      if (m) { try { rep = JSON.parse(m[1]); } catch { rep = null; } }
      res.statusCode = code === 0 ? 200 : 500;
      res.end(JSON.stringify({ code, report: rep, log: out.slice(-4000), err: err.slice(-2000) }));
    };
    const killer = setTimeout(() => { err += '\n[看板] 超时(10分钟)强制终止'; try { py.kill(); } catch {} }, 10 * 60 * 1000);
    py.stdout.on('data', d => { out += d; });
    py.stderr.on('data', d => { err += d; });
    py.on('close', code => done(code));
    py.on('error', e => {
      err += `\n[看板] 无法启动 ${PY}: ` + String(e.message || e);
      done(-1);
    });
    return;
  }

  if (u.pathname === '/' || u.pathname === '/index.html') {
    if (!fs.existsSync(INDEX_HTML)) { res.statusCode = 404; return res.end(JSON.stringify({ error: 'index.html missing' })); }
    res.setHeader('Content-Type', 'text/html; charset=utf-8');
    return res.end(fs.readFileSync(INDEX_HTML));
  }

  if (u.pathname === '/api/tables') {
    const tabs = q(`SELECT name, type FROM sqlite_master
                    WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'
                    ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END, name`) || [];
    for (const t of tabs) {
      const c = q(`SELECT COUNT(*) AS c FROM "${t.name}"`);
      t.rows = (c && c[0]) ? c[0].c : null;
      const dc = (t.name in DATE_COL) ? DATE_COL[t.name] : 'date';
      if (dc) {
        const r = q(`SELECT MIN(${dc}) AS mn, MAX(${dc}) AS mx FROM "${t.name}"`);
        if (r && r[0]) { t.min = r[0].mn; t.max = r[0].mx; }
      }
    }
    return res.end(JSON.stringify({
      db: path.basename(DB),
      db_size_mb: Math.round(fs.statSync(DB).size / 1e6 * 10) / 10,
      tables: tabs,
    }));
  }

  if (u.pathname === '/api/rows') {
    const name = u.searchParams.get('name') || '';
    const n = Math.max(1, Math.min(200, parseInt(u.searchParams.get('n') || '10', 10) || 10));
    const allTabs = q(`SELECT name, type FROM sqlite_master
                       WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%'`);
    const found = (allTabs || []).find(t => t.name === name);
    if (!found) { res.statusCode = 404; return res.end(JSON.stringify({ error: 'table not found: ' + name })); }
    const order = found.type === 'table' ? 'rowid DESC' : '1 DESC';
    const rows = q(`SELECT * FROM "${name}" ORDER BY ${order} LIMIT ${n}`);
    if (rows && rows.error) { res.statusCode = 500; return res.end(JSON.stringify({ error: rows.error })); }
    let colNames;
    if (rows && rows.length) {
      colNames = Object.keys(rows[0]);
    } else {
      const info = q(`PRAGMA table_info("${name}")`) || [];
      colNames = info.map(c => c.name);
    }
    return res.end(JSON.stringify({ name, cols: colNames, rows: rows || [] }));
  }

  res.statusCode = 404;
  res.end(JSON.stringify({ error: 'not found' }));
});

server.listen(PORT, HOST, () => {
  console.log(`dashboard: http://${HOST}:${PORT}  db=${DB}`);
});

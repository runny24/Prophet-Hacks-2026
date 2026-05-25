"""Lightweight dashboard for the Prophet Arena trade benchmark.

Serves HTML at / and proxies /api/* to the core API so CORS and the
API key are handled in-process (the key never reaches the browser).
The slug filter is baked into the HTML at generation time.

Standalone usage (requires only ai-prophet-core)::

    prophet-dashboard --slug my_experiment

    # or equivalently
    python -m ai_prophet_core.dashboard --slug my_experiment

Via the CLI package (re-exported as ``prophet trade dashboard``)::

    prophet trade dashboard --slug my_experiment
    prophet trade eval run -m openai:gpt-4o --slug test --dashboard

The dashboard reads ``PA_SERVER_URL`` and ``PA_SERVER_API_KEY`` from the
environment when CLI args aren't supplied.
"""

import argparse
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

import httpx

from .client import DEFAULT_API_URL

# Read-only reporting API for PnL history, positions, trades, and the
# leaderboard. Authoritative state (auth, writes, owner scoping) still
# goes through core_api. The default Cloud Run URL is a hosting detail;
# override with `--reporting-url` or `PA_REPORTING_API_URL` when the
# trade-api DNS name lands.
DEFAULT_REPORTING_API_URL = "https://trade-ui-api-998105805337.us-central1.run.app"

_API_URL = ""
_API_KEY = ""
_REPORTING_URL = ""
_SLUG = ""
_HTML_BYTES = b""


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._proxy(parsed.path[4:], parsed.query,
                        base_url=_API_URL, api_key=_API_KEY, scope_slug=True)
        elif parsed.path.startswith("/report/"):
            self._proxy(parsed.path[len("/report"):], parsed.query,
                        base_url=_REPORTING_URL, api_key="", scope_slug=False)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_HTML_BYTES)

    def _proxy(self, path, query, *, base_url, api_key, scope_slug):
        url = f"{base_url}{path}"
        if query:
            url += f"?{query}"
        try:
            headers = {"X-API-Key": api_key} if api_key else None
            resp = httpx.get(url, timeout=15, headers=headers)
            data = resp.content
            # Scope /experiments to configured slug
            if scope_slug and _SLUG and path == "/experiments" and resp.status_code == 200:
                items = resp.json()
                if isinstance(items, list):
                    data = json.dumps([e for e in items if e.get("experiment_slug") == _SLUG]).encode()
                elif isinstance(items, dict) and isinstance(items.get("experiments"), list):
                    scoped = {
                        **items,
                        "experiments": [
                            e for e in items["experiments"] if e.get("experiment_slug") == _SLUG
                        ],
                    }
                    data = json.dumps(scoped).encode()
            self.send_response(resp.status_code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def log_message(self, fmt, *args):
        pass


def open_dashboard(
    api_url: str,
    slug: str = "",
    port: int = 8501,
    api_key: str | None = None,
    reporting_url: str | None = None,
    *,
    block: bool = False,
):
    """Serve the dashboard and open it in the browser.

    ``block=True`` is used by the standalone dashboard command so the local
    HTTP server stays alive until the user stops it. ``block=False`` keeps the
    dashboard as a sidecar during ``prophet trade eval run --dashboard``.
    """
    global _API_URL, _API_KEY, _REPORTING_URL, _SLUG, _HTML_BYTES
    _API_URL = api_url.rstrip("/")
    _API_KEY = api_key or ""
    _REPORTING_URL = (
        reporting_url
        or os.environ.get("PA_REPORTING_API_URL")
        or DEFAULT_REPORTING_API_URL
    ).rstrip("/")
    _SLUG = slug
    _HTML_BYTES = _HTML.replace("__REQUESTED_SLUG__", json.dumps(slug)).encode()

    server = HTTPServer(("127.0.0.1", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    webbrowser.open(f"http://localhost:{port}")
    print(f"  Dashboard: http://localhost:{port}")
    print(f"  Core API:  {_API_URL}")
    if slug:
        print(f"  Experiment: {slug}")

    if block:
        try:
            print("  Press Ctrl+C to stop the dashboard")
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard stopped")
        finally:
            server.server_close()


def main(argv: list[str] | None = None) -> int:
    """Standalone entry point: ``prophet-dashboard`` and ``python -m ai_prophet_core.dashboard``."""
    parser = argparse.ArgumentParser(
        prog="prophet-dashboard",
        description="Live dashboard for the Prophet Arena trade benchmark.",
    )
    parser.add_argument(
        "--api-url",
        default=os.environ.get("PA_SERVER_URL") or DEFAULT_API_URL,
        help="Core API base URL (env: PA_SERVER_URL).",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("PA_SERVER_API_KEY"),
        help="Prophet Arena API key (env: PA_SERVER_API_KEY).",
    )
    parser.add_argument(
        "--slug",
        default="",
        help="Filter the dashboard to a single experiment slug.",
    )
    parser.add_argument(
        "--reporting-url",
        default=os.environ.get("PA_REPORTING_API_URL"),
        help=(
            "Read-only reporting API for PnL history and leaderboard "
            "(env: PA_REPORTING_API_URL). Defaults to the hosted "
            "trade-ui-api service."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Local port to bind (default: 8501).",
    )
    args = parser.parse_args(argv)

    if not args.api_key:
        parser.error("missing API key: pass --api-key or set PA_SERVER_API_KEY")

    open_dashboard(
        api_url=args.api_url,
        slug=args.slug,
        port=args.port,
        api_key=args.api_key,
        reporting_url=args.reporting_url,
        block=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# Self-contained HTML -- uses the local /api proxy so CORS/API keys stay hidden
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#09090b;--fg:#f4f4f5;--sub:#a1a1aa;--muted:#71717a;--border:#27272a;
  --green:#22c55e;--red:#ef4444;--blue:#60a5fa
}
html,body{min-height:100%;background:var(--bg);color:var(--fg);font-family:'JetBrains Mono','SF Mono',ui-monospace,monospace;font-size:13px;line-height:1.45}
main{max-width:1280px;margin:0 auto;padding:36px 24px 48px}
.header{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin-bottom:22px;padding-bottom:20px;border-bottom:1px solid var(--border)}
.header h1{font-size:22px;line-height:1.1;font-weight:700;color:#fafafa}
.meta{font-size:11px;color:var(--muted);text-align:right;white-space:nowrap}
.mono{font-variant-numeric:tabular-nums}
.green{color:var(--green)!important}.red{color:var(--red)!important}.blue{color:var(--blue)!important}.muted{color:var(--muted)!important}
.hide{display:none!important}

.panel{background:rgba(17,17,19,.76);border:1px solid var(--border);border-radius:6px;overflow:hidden;margin-bottom:22px}
.panel-hd{display:flex;align-items:baseline;justify-content:space-between;gap:12px;padding:14px 16px 10px;border-bottom:1px solid var(--border)}
.panel-hd h2{font-size:13px;font-weight:500;color:#e4e4e7}
.panel-hd span{font-size:11px;color:var(--muted)}

.kv{margin:0;width:100%;font-size:13px;line-height:1.5}
.kv td{padding:11px 18px;border-bottom:1px solid var(--border);vertical-align:top}
.kv td:first-child{color:var(--sub);width:180px}
.kv td:last-child{color:var(--fg);text-align:right;font-variant-numeric:tabular-nums}
.kv .v{font-weight:600;white-space:nowrap}
.kv .s{color:var(--muted);font-size:11px;font-weight:400;margin-top:3px}
.kv tr:last-child td{border-bottom:none}

.chart{height:330px;padding:12px 14px 16px;position:relative}
.chart svg{display:block;width:100%;height:100%}
.chart svg circle{cursor:pointer}
.chart svg circle:hover{stroke-width:2.5;r:5}
.chart-tooltip{position:absolute;transform:translate(-50%,calc(-100% - 10px));background:#18181b;border:1px solid var(--border);border-radius:6px;padding:8px 11px;font-size:11px;pointer-events:none;white-space:nowrap;z-index:10;box-shadow:0 6px 18px rgba(0,0,0,.55);min-width:170px}
.chart-tooltip .tt-row{display:flex;justify-content:space-between;gap:18px;line-height:1.5}
.chart-tooltip .tt-row .k{color:var(--muted)}
.chart-tooltip .tt-row .v{color:var(--fg);font-variant-numeric:tabular-nums;font-weight:600}

table{width:100%;border-collapse:collapse;font-size:12px}
th{background:#141416;color:var(--muted);font-size:10px;font-weight:500;text-transform:uppercase;letter-spacing:.11em;text-align:left;padding:9px 10px;border-bottom:1px solid var(--border);white-space:nowrap}
td{padding:9px 10px;border-bottom:1px solid var(--border);vertical-align:top}
th.r,td.r{text-align:right}
tbody tr:hover{background:rgba(39,39,42,.45)}
.clip{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:360px}
.scroll{max-height:430px;overflow:auto}
.empty{color:var(--muted);padding:32px 16px;text-align:center;font-size:12px}
.warn{border:1px solid rgba(245,158,11,.35);background:rgba(245,158,11,.08);color:#fcd34d;border-radius:5px;padding:9px 11px;font-size:11px;margin-bottom:14px}

@media (max-width:720px){main{padding:22px 14px}.header{align-items:flex-start;flex-direction:column}.meta{text-align:left}.chart{height:260px}.clip{max-width:210px}}
</style>
</head>
<body>
<main>
  <div class="header">
    <h1 id="runSlug">Run</h1>
    <div class="meta mono">
      <div id="runStatus">connecting...</div>
      <div id="runStarted" class="muted"></div>
    </div>
  </div>

  <div id="err" class="hide warn"></div>

  <section class="panel">
    <table class="kv">
      <tbody id="metrics"></tbody>
    </table>
  </section>

  <section class="panel">
    <div class="panel-hd">
      <h2>Equity over ticks</h2>
      <span id="chartMeta">waiting for data</span>
    </div>
    <div class="chart" id="chart">
      <div id="chartTooltip" class="chart-tooltip hide"></div>
    </div>
  </section>

  <section class="panel">
    <div class="panel-hd">
      <h2>Positions</h2>
      <span id="posMeta">0 open</span>
    </div>
    <div id="positions"></div>
  </section>

  <section class="panel">
    <div class="panel-hd">
      <h2>Fills</h2>
      <span id="fillsMeta">0 trades</span>
    </div>
    <div id="fills"></div>
  </section>
</main>
<script>
const API = '/api';
const REQUESTED_SLUG = __REQUESTED_SLUG__;
const POLL_MS = 10000;
const PARTICIPANT_IDX = 0;

const MIN_DAILY_OBS = 3;
const MIN_WIN_RATE_TRADES = 10;

let exp = null;
let participant = null;
let portfolio = null;
let pnl = [];
let fills = [];

const $ = id => document.getElementById(id);
const esc = s => { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; };
const num = v => { const n = typeof v === 'string' ? Number(v) : v; return Number.isFinite(n) ? n : null; };
const fmt = (v, d=2) => { const n=num(v); return n==null?'—':n.toLocaleString(undefined,{maximumFractionDigits:d,minimumFractionDigits:d}); };
const fmtInt = v => { const n=num(v); return n==null?'—':Math.round(n).toLocaleString(); };
const usd = v => { const n=num(v); return n==null?'—':n.toLocaleString(undefined,{style:'currency',currency:'USD',minimumFractionDigits:2,maximumFractionDigits:2}); };
const signedUsd = v => { const n=num(v); return n==null?'—':`${n>=0?'+':'-'}${usd(Math.abs(n))}`; };
const pct = (v, scale=100, d=2) => { const n=num(v); return n==null?'—':`${n>=0?'+':''}${(n*scale).toFixed(d)}%`; };
const time = iso => { if(!iso)return '—'; const d=new Date(iso); return isFinite(d)?d.toLocaleString():'—'; };

async function get(path, opts={}) {
  const r = await fetch(API + path);
  if (r.ok) return r.json();
  if (!opts.optional) throw new Error(`${path}: HTTP ${r.status}`);
  return null;
}

const REPORT = '/report';
async function reportGet(path, opts={}) {
  const r = await fetch(REPORT + path);
  if (r.ok) return r.json();
  if (!opts.optional) throw new Error(`${path}: HTTP ${r.status}`);
  return null;
}

function asArray(payload, keys) {
  if (Array.isArray(payload)) return payload;
  for (const k of keys) if (Array.isArray(payload?.[k])) return payload[k];
  return [];
}

async function load() {
  const list = await get('/experiments', {optional:true});
  let exps = asArray(list, ['experiments']);
  exps.sort((a,b) => (a.status==='RUNNING'?0:1)-(b.status==='RUNNING'?0:1) || (b.last_activity_at||'').localeCompare(a.last_activity_at||''));
  if (!exps.length) { exp = null; $('runStatus').textContent = 'no experiments'; return; }
  const id = exps[0].experiment_id;

  const [detail, prog, parts, port, pnlPayload, tradesPayload] = await Promise.all([
    get('/experiments/'+id, {optional:true}),
    get('/experiments/'+id+'/progress', {optional:true}),
    get('/experiments/'+id+'/participants', {optional:true}),
    get('/experiments/'+id+'/participants/'+PARTICIPANT_IDX+'/portfolio', {optional:true}),
    reportGet('/experiments/'+id+'/pnl?participant_idx='+PARTICIPANT_IDX, {optional:true}),
    get('/experiments/'+id+'/trades?limit=1000', {optional:true}),
  ]);

  exp = detail || exps[0];
  if (prog) { exp._done = prog.completed||0; exp._total = prog.n_ticks||exp.n_ticks; }
  if (!prog) { exp._done = exp.completed_ticks ?? exp.completed; exp._total = exp.n_ticks; }

  const partList = asArray(parts, ['participants']);
  participant = partList.find(p => (p.participant_idx ?? p.idx ?? 0) === PARTICIPANT_IDX) || partList[0] || null;
  portfolio = port;
  pnl = asArray(pnlPayload, ['pnl']).filter(r => Number(r.participant_idx ?? 0) === PARTICIPANT_IDX);
  pnl.sort((a,b) => new Date(rowTimestamp(a)) - new Date(rowTimestamp(b)));
  fills = asArray(tradesPayload, ['fills','trades']).filter(f => Number(f.participant_idx ?? 0) === PARTICIPANT_IDX);
  fills.sort((a,b) => new Date(b.filled_at||b.timestamp) - new Date(a.filled_at||a.timestamp));

}

function elapsed(iso) {
  if (!iso) return '';
  const t = new Date(iso);
  if (!isFinite(t)) return '';
  const seconds = Math.max(0, (Date.now() - t.getTime()) / 1000);
  if (seconds < 60) return `${Math.floor(seconds)}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds/60)}m ago`;
  if (seconds < 86400) return `${(seconds/3600).toFixed(1)}h ago`;
  return `${(seconds/86400).toFixed(1)}d ago`;
}

function rowTimestamp(row) {
  return row?.tick_ts || row?.timestamp || row?.created_at || row?.updated_at || row?.as_of || row?.date;
}

function rowEquity(row) {
  return num(row?.equity ?? row?.portfolio_value ?? row?.total_equity);
}

function portfolioEquity() {
  return num(portfolio?.equity ?? portfolio?.total_equity);
}

function startingCash() {
  return num(participant?.starting_cash ?? participant?.startingCash) ?? 10000;
}

function tickPoints() {
  const out = [];
  for (const row of pnl) {
    const t = new Date(rowTimestamp(row));
    const e = rowEquity(row);
    if (isFinite(t) && e != null) {
      out.push({
        t,
        equity: e,
        cash: num(row.cash),
        totalPnl: num(row.total_pnl ?? row.totalPnl),
        positions: num(row.num_positions ?? row.numPositions),
      });
    }
  }
  return out;
}

function dailyPoints() {
  const byDay = new Map();
  for (const {t, equity} of tickPoints()) {
    const k = t.toISOString().slice(0,10);
    const prev = byDay.get(k);
    if (!prev || t > prev.t) byDay.set(k, {t, equity});
  }
  return [...byDay.values()].sort((a,b)=>a.t-b.t);
}

function metrics() {
  const ticks = tickPoints();
  const daily = dailyPoints();
  const start = startingCash();
  const latest = ticks.length ? ticks[ticks.length-1].equity : portfolioEquity();
  const totalPnl = latest != null ? latest - start : (num(portfolio?.total_pnl) ?? null);
  const ret = latest != null && start ? (latest - start) / start : null;

  const equities = daily.map(d => d.equity);
  const returns = [];
  for (let i=1; i<equities.length; i++) if (equities[i-1] !== 0) returns.push(equities[i] / equities[i-1] - 1);
  const eligible = daily.length >= MIN_DAILY_OBS;
  const mean = returns.length ? returns.reduce((s,x)=>s+x,0) / returns.length : null;
  const variance = returns.length >= 2 && mean != null ? returns.reduce((s,x)=>s+(x-mean)**2,0)/(returns.length-1) : null;
  const std = variance != null ? Math.sqrt(variance) : null;
  const sharpe = eligible && std != null ? (std === 0 ? 0 : (mean / std) * Math.sqrt(365)) : null;

  let cagr = null;
  if (daily.length >= 2 && daily[0].equity > 0) {
    const days = Math.max(0, (daily[daily.length-1].t - daily[0].t) / 86400000);
    if (days > 0) cagr = (daily[daily.length-1].equity / daily[0].equity) ** (365 / days) - 1;
  }
  let peak = equities[0] ?? null, maxDd = null;
  if (equities.length >= 2 && peak != null) {
    maxDd = 0;
    for (const e of equities) { if (e > peak) peak = e; if (peak > 0) maxDd = Math.max(maxDd, (peak - e) / peak); }
  }
  return {
    starting: start,
    equity: latest,
    totalPnl,
    returnPct: ret,
    cash: num(portfolio?.cash),
    sharpe: eligible ? sharpe : null,
    maxDrawdown: eligible ? maxDd : null,
    cagr: eligible ? cagr : null,
    nObs: daily.length,
    eligible,
    ticks,
    daily,
  };
}

function fillCost(f) {
  const notional = num(f.notional ?? f.cost);
  if (notional != null) return Math.abs(notional);
  const shares = num(f.shares), price = num(f.price);
  return shares != null && price != null ? Math.abs(shares * price) : 0;
}

function winRate() {
  const groups = new Map();
  let hasOutcome = false;
  for (const f of fills) {
    const outcome = f.market_outcome ?? f.outcome ?? null;
    if (outcome == null) continue;
    hasOutcome = true;
    const side = String(f.side || '').toUpperCase();
    const action = String(f.action || '').toUpperCase();
    const key = `${f.market_id}:${side}`;
    const g = groups.get(key) || {cost:0, sell:0, bought:0, sold:0, outcome, side};
    const shares = num(f.shares) ?? 0, cost = fillCost(f);
    if (action === 'BUY') { g.cost += cost; g.bought += shares; }
    if (action === 'SELL') { g.sell += cost; g.sold += shares; }
    g.outcome = outcome;
    groups.set(key, g);
  }
  if (!hasOutcome) return {nTrades: 0, rate: null};
  let wins = 0, total = 0;
  for (const g of groups.values()) {
    const out = String(g.outcome).toUpperCase();
    const yes = out === 'YES' || out === '1' || out === 'TRUE' || g.outcome === 1;
    const payout = g.side === 'YES' ? (yes ? 1 : 0) : (yes ? 0 : 1);
    const pnl = g.sell + (g.bought - g.sold) * payout - g.cost;
    total += 1;
    if (pnl > 0) wins += 1;
  }
  return {nTrades: total, rate: total >= MIN_WIN_RATE_TRADES ? wins / total : null};
}

function dailyGate(m) {
  if (m.eligible) return {pending: false, sub: `${m.nObs} daily P&L observations`};
  if (m.nObs === 0) {
    return {pending: true, sub: `Needs ${MIN_DAILY_OBS} days of P&L history. Run hasn't crossed a UTC day boundary yet.`};
  }
  if (m.nObs === 1) {
    return {pending: true, sub: `Have 1 of ${MIN_DAILY_OBS} days. Two more UTC midnights to go.`};
  }
  return {pending: true, sub: `Have ${m.nObs} of ${MIN_DAILY_OBS} days.`};
}

function tradesGate(wr) {
  if (wr.nTrades >= MIN_WIN_RATE_TRADES) return {pending: false, sub: `${wr.nTrades} resolved trades`};
  if (wr.nTrades === 0) {
    return {pending: true, sub: `Needs ${MIN_WIN_RATE_TRADES} resolved markets. None of your trades have settled yet.`};
  }
  return {pending: true, sub: `Have ${wr.nTrades} of ${MIN_WIN_RATE_TRADES} resolved markets.`};
}

function render() {
  if (!exp) {
    $('metrics').innerHTML = '<tr><td colspan="2" class="empty">No experiment found.</td></tr>';
    $('chartMeta').textContent = '—';
    const chartSvg = $('chart').querySelector('.chart-svg');
    if (chartSvg) chartSvg.innerHTML = '<div class="empty">No data.</div>';
    $('positions').innerHTML = '';
    $('fills').innerHTML = '';
    return;
  }
  const m = metrics();
  const wr = winRate();
  const statusColor = exp.status === 'RUNNING' ? 'green' : exp.status === 'COMPLETED' ? 'blue' : exp.status === 'ABORTED' ? 'red' : '';
  $('runSlug').textContent = exp.experiment_slug || 'Run';
  $('runStatus').innerHTML = `<span class="${statusColor}">${esc(exp.status || '—')}</span> · ${exp._done??0} / ${exp._total??'—'} ticks`;
  const started = exp.started_at || exp.created_at;
  $('runStarted').textContent = started ? `started ${elapsed(started)}` : '';

  const pnlCls = m.totalPnl == null ? '' : m.totalPnl >= 0 ? 'green' : 'red';
  const dg = dailyGate(m);
  const tg = tradesGate(wr);

  function row(label, value, sub, cls='') {
    return `<tr><td>${label}</td><td><div class="v ${cls}">${value}</div>${sub?`<div class="s">${sub}</div>`:''}</td></tr>`;
  }
  function gatedRow(label, value, gate) {
    return gate.pending
      ? row(label, `<span class="muted">Pending</span>`, gate.sub)
      : row(label, value, gate.sub);
  }

  $('metrics').innerHTML = [
    row('Equity',       usd(m.equity),         `start ${usd(m.starting)}`),
    row('Total P&L',    signedUsd(m.totalPnl), pct(m.returnPct,100,2), pnlCls),
    row('Cash',         usd(m.cash),           `${fmtInt(portfolio?.positions?.length||0)} open positions`),
    gatedRow('Sharpe',       fmt(m.sharpe,2),                              dg),
    gatedRow('Max Drawdown', pct(m.maxDrawdown,100,2).replace('+',''),     dg),
    gatedRow('CAGR',         pct(m.cagr,100,2),                            dg),
    gatedRow('Win Rate',     pct(wr.rate,100,1).replace('+',''),           tg),
  ].join('');

  renderChart(m);
  renderPositions();
  renderFills();
}

function renderChart(m) {
  const pts = m.ticks;
  const tooltip = $('chartTooltip');
  tooltip.classList.add('hide');

  // Find or recreate the SVG container, leaving the tooltip div in place.
  const chartEl = $('chart');
  let svgHost = chartEl.querySelector('.chart-svg');
  if (!svgHost) {
    svgHost = document.createElement('div');
    svgHost.className = 'chart-svg';
    svgHost.style.cssText = 'width:100%;height:100%';
    chartEl.insertBefore(svgHost, chartEl.firstChild);
  }

  if (!pts.length) {
    $('chartMeta').textContent = 'P&L history unavailable';
    svgHost.innerHTML = '<div class="empty">P&L history unavailable.</div>';
    return;
  }

  const firstT = pts[0].t, lastT = pts[pts.length-1].t;
  const sameDay = firstT.toDateString() === lastT.toDateString();
  const fmtT = t => sameDay
    ? t.toLocaleTimeString(undefined, {hour:'2-digit', minute:'2-digit'})
    : t.toLocaleDateString(undefined, {month:'short', day:'numeric'}) + ' ' +
      t.toLocaleTimeString(undefined, {hour:'2-digit', minute:'2-digit'});
  $('chartMeta').textContent = `${pts.length} tick${pts.length===1?'':'s'} · ${fmtT(firstT)} → ${fmtT(lastT)}`;

  // Plot points evenly spaced by tick index. Each circle is a discrete
  // PnL observation; the line just connects them visually.
  const W = 1000, H = 300, L = 58, R = 20, T = 16, B = 36;
  let minY = Math.min(...pts.map(p=>p.equity));
  let maxY = Math.max(...pts.map(p=>p.equity));
  if (minY === maxY) { minY -= Math.max(1, minY*.02); maxY += Math.max(1, maxY*.02); }
  const padY = (maxY-minY)*.15; minY -= padY; maxY += padY;

  const x = i => pts.length === 1 ? L + (W-L-R)/2 : L + (i/(pts.length-1))*(W-L-R);
  const y = v => T + (1-(v-minY)/(maxY-minY))*(H-T-B);

  const path = pts.map((p,i)=>`${i?'L':'M'}${x(i).toFixed(1)},${y(p.equity).toFixed(1)}`).join(' ');
  const dots = pts.map((p,i)=>{
    const pnlStr = p.totalPnl == null ? '' : signedUsd(p.totalPnl);
    const titleText = `Tick #${i+1} · ${fmtT(p.t)}\nEquity: ${usd(p.equity)}` +
      (p.cash != null ? `\nCash: ${usd(p.cash)}` : '') +
      (pnlStr ? `\nP&L: ${pnlStr}` : '');
    return `<circle data-i="${i}" cx="${x(i).toFixed(1)}" cy="${y(p.equity).toFixed(1)}" r="4" fill="#0c0c0e" stroke="#60a5fa" stroke-width="1.6"><title>${esc(titleText)}</title></circle>`;
  }).join('');

  const yTicks = [minY, (minY+maxY)/2, maxY];
  const grid = yTicks.map(v=>`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="#27272a"/><text x="8" y="${y(v)+4}" fill="#a1a1aa" font-size="11">${usd(v)}</text>`).join('');

  const desiredLabels = Math.min(5, pts.length);
  const labelIdxs = [];
  for (let k=0; k<desiredLabels; k++) {
    labelIdxs.push(Math.round(k * (pts.length-1) / (desiredLabels-1 || 1)));
  }
  const xLabels = [...new Set(labelIdxs)].map(i => {
    const cx = x(i).toFixed(1);
    const anchor = i === 0 ? 'start' : i === pts.length-1 ? 'end' : 'middle';
    return `<text x="${cx}" y="${H-18}" fill="#a1a1aa" font-size="11" text-anchor="${anchor}">#${i+1}</text>
            <text x="${cx}" y="${H-6}" fill="#71717a" font-size="10" text-anchor="${anchor}">${fmtT(pts[i].t)}</text>`;
  }).join('');

  svgHost.innerHTML = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
    ${grid}
    <line x1="${L}" x2="${W-R}" y1="${H-B}" y2="${H-B}" stroke="#3f3f46"/>
    <path d="${path}" fill="none" stroke="#60a5fa" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    ${dots}
    ${xLabels}
  </svg>`;

  // Custom hover tooltip (richer than the native <title> fallback).
  svgHost.querySelectorAll('circle[data-i]').forEach(c => {
    c.addEventListener('mouseenter', () => {
      const i = parseInt(c.dataset.i);
      const p = pts[i];
      const rows = [
        ['Tick',   `#${i+1}`],
        ['Time',   fmtT(p.t)],
        ['Equity', usd(p.equity)],
      ];
      if (p.cash != null) rows.push(['Cash', usd(p.cash)]);
      if (p.totalPnl != null) rows.push(['P&L', `<span class="${p.totalPnl>=0?'green':'red'}">${signedUsd(p.totalPnl)}</span>`]);
      if (p.positions != null) rows.push(['Positions', fmtInt(p.positions)]);
      tooltip.innerHTML = rows.map(([k,v]) => `<div class="tt-row"><span class="k">${k}</span><span class="v">${v}</span></div>`).join('');
      const circleRect = c.getBoundingClientRect();
      const chartRect = chartEl.getBoundingClientRect();
      tooltip.style.left = (circleRect.left + circleRect.width/2 - chartRect.left) + 'px';
      tooltip.style.top = (circleRect.top - chartRect.top) + 'px';
      tooltip.classList.remove('hide');
    });
    c.addEventListener('mouseleave', () => tooltip.classList.add('hide'));
  });
}

function renderPositions() {
  const positions = portfolio?.positions || [];
  $('posMeta').textContent = `${positions.length} open`;
  if (!positions.length) { $('positions').innerHTML = '<div class="empty">No open positions.</div>'; return; }
  $('positions').innerHTML = `<div class="scroll"><table>
    <thead><tr><th>Market</th><th>Side</th><th class="r">Shares</th><th class="r">Avg Entry</th><th class="r">Mark</th><th class="r">Unrealized P&L</th></tr></thead>
    <tbody>${positions.map(p => {
      const upnl = num(p.unrealized_pnl);
      const side = String(p.side || '').toUpperCase();
      const sideCls = side === 'YES' ? 'green' : side === 'NO' ? 'red' : 'muted';
      return `<tr>
        <td><div class="clip" title="${esc(p.market_id)}">${esc(p.market_id)}</div></td>
        <td class="${sideCls}">${esc(side)}</td>
        <td class="r mono">${fmt(p.shares,2)}</td>
        <td class="r mono">${fmt(p.avg_entry_price,4)}</td>
        <td class="r mono">${fmt(p.current_price,4)}</td>
        <td class="r mono ${upnl>=0?'green':'red'}">${signedUsd(upnl)}</td>
      </tr>`;
    }).join('')}</tbody>
  </table></div>`;
}

function renderFills() {
  $('fillsMeta').textContent = `${fills.length} trade${fills.length===1?'':'s'}`;
  if (!fills.length) { $('fills').innerHTML = '<div class="empty">No fills yet.</div>'; return; }
  $('fills').innerHTML = `<div class="scroll"><table>
    <thead><tr><th>Time</th><th>Action</th><th>Side</th><th>Market</th><th class="r">Shares</th><th class="r">Price</th><th class="r">Notional</th></tr></thead>
    <tbody>${fills.map(f => {
      const action = String(f.action || '').toUpperCase();
      const side = String(f.side || '').toUpperCase();
      const actionCls = action === 'BUY' ? 'green' : 'red';
      return `<tr>
        <td class="mono muted">${time(f.filled_at || f.timestamp)}</td>
        <td class="${actionCls}">${esc(action)}</td>
        <td>${esc(side)}</td>
        <td><div class="clip" title="${esc(f.market_id)}">${esc(f.market_id)}</div></td>
        <td class="r mono">${fmt(f.shares,2)}</td>
        <td class="r mono">${fmt(f.price,4)}</td>
        <td class="r mono">${usd(fillCost(f))}</td>
      </tr>`;
    }).join('')}</tbody>
  </table></div>`;
}

(async () => {
  try { await load(); } catch(e) { $('err').textContent='Error: '+e.message; $('err').classList.remove('hide'); }
  render();
  setInterval(async () => { try { await load(); render(); } catch(e) {} }, POLL_MS);
})();
</script>
</body>
</html>
"""

"""Self-contained HTML reports: one file, no external scripts, light and dark mode.

Charts follow one fixed encoding, validated for colour-vision deficiency in both modes:
the player is blue, the comparison group is orange (median line plus a 10% wash for its
middle 50%). Every value in a tooltip is also in the table view, and labels are inserted
with textContent, never innerHTML.
"""

from __future__ import annotations

import html
import json
from typing import Any

from riftwatch.analysis.score import GameScore
from riftwatch.coach.evidence import clock
from riftwatch.coach.pipeline import CoachResult
from riftwatch.features.metrics import AREAS, CURVE_METRICS

# Curves shown as charts, in reading order.
CHART_CURVES = ("cs", "gold_diff", "cs_diff", "gold", "xp_diff", "wards_placed")

_CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10);
  --you: #2a78d6; --group: #eb6834; --wash: rgba(235,104,52,0.10); --typical: rgba(11,11,11,0.05);
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
    --you: #3987e5; --group: #d95926; --wash: rgba(217,89,38,0.14); --typical: rgba(255,255,255,0.06);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
  --you: #3987e5; --group: #d95926; --wash: rgba(217,89,38,0.14); --typical: rgba(255,255,255,0.06);
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1120px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 24px; margin: 0 0 4px; font-weight: 650; }
h2 { font-size: 17px; margin: 32px 0 12px; font-weight: 650; }
.sub { color: var(--ink-2); margin: 0; }
.card { background: var(--surface); border: 1px solid var(--ring); border-radius: 12px; padding: 16px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-top: 20px; }
.tile .label { color: var(--ink-2); font-size: 13px; }
.tile .value { font-size: 26px; font-weight: 650; }
.tile .pct { color: var(--muted); font-size: 13px; }
.legend { display: flex; flex-wrap: wrap; gap: 18px; color: var(--ink-2); font-size: 13px; margin-bottom: 10px; }
.key { display: inline-flex; align-items: center; gap: 6px; }
.key i { display: inline-block; width: 18px; height: 2px; border-radius: 1px; }
.key b { display: inline-block; width: 14px; height: 10px; border-radius: 2px; box-shadow: inset 0 0 0 1px var(--group); }
.charts { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
@media (max-width: 760px) { .charts { grid-template-columns: 1fr; } }
.chart h3 { font-size: 14px; margin: 0 0 6px; font-weight: 600; }
.chart svg { width: 100%; height: auto; display: block; overflow: visible; }
.chart svg:focus { outline: 2px solid var(--you); outline-offset: 4px; border-radius: 4px; }
.tick { fill: var(--muted); font-size: 11px; }
.endlabel { fill: var(--ink); font-size: 12px; font-weight: 600; }
.tip { position: fixed; pointer-events: none; background: var(--surface); color: var(--ink);
  border: 1px solid var(--ring); border-radius: 8px; padding: 8px 10px; font-size: 12px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.12); display: none; z-index: 10; min-width: 150px; }
.tip .t { color: var(--ink-2); margin-bottom: 4px; }
.tip .row { display: flex; align-items: center; gap: 6px; }
.tip .row i { width: 12px; height: 2px; display: inline-block; }
.tip .row strong { min-width: 48px; }
.tip .row span { color: var(--ink-2); }
table.score { width: 100%; border-collapse: collapse; font-size: 13px; }
table.score td { padding: 5px 6px; border-top: 1px solid var(--grid); vertical-align: middle; }
table.score tr.area td { border-top: none; padding-top: 14px; color: var(--ink-2); font-weight: 600;
  text-transform: capitalize; }
table.score .num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
table.score .small { color: var(--muted); font-size: 12px; }
table.score td.bar { width: 40%; }
.scroll { overflow-x: auto; }
.track { position: relative; height: 16px; min-width: 64px; }
.track .line { position: absolute; left: 0; right: 0; top: 7px; height: 1px; background: var(--axis); }
.track .typ { position: absolute; left: 25%; width: 50%; top: 3px; height: 9px; background: var(--typical); border-radius: 2px; }
.track .dot { position: absolute; top: 3px; width: 10px; height: 10px; margin-left: -5px; border-radius: 50%;
  background: var(--you); box-shadow: 0 0 0 2px var(--surface); }
.deaths { position: relative; height: 54px; margin: 8px 6px 0; }
.deaths .axis { position: absolute; left: 0; right: 0; top: 22px; height: 1px; background: var(--axis); }
.deaths .mark { position: absolute; top: 14px; width: 18px; height: 18px; margin-left: -9px; border: 0;
  background: transparent; padding: 0; cursor: default; color: var(--ink); font: 700 14px/18px system-ui; }
.deaths .mark:focus { outline: 2px solid var(--you); border-radius: 4px; }
.deaths .t { position: absolute; top: 34px; color: var(--muted); font-size: 11px; transform: translateX(-50%); }
ol.deathlist { margin: 10px 0 0; padding-left: 20px; color: var(--ink-2); font-size: 13px; }
.coach .headline { font-size: 16px; font-weight: 600; margin: 0 0 12px; }
.point { border-top: 1px solid var(--grid); padding: 12px 0; }
.point .kind { color: var(--ink-2); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
.point h4 { margin: 2px 0 6px; font-size: 15px; }
.point p { margin: 4px 0; }
.point .advice { font-weight: 500; }
.point .ev { color: var(--muted); font-size: 12px; margin-top: 6px; }
.meta { color: var(--muted); font-size: 12px; margin-top: 8px; }
nav.top { display: flex; gap: 16px; align-items: center; margin-bottom: 18px; font-size: 14px; }
nav.top a, a.plain { color: var(--you); text-decoration: none; }
nav.top a:hover, a.plain:hover { text-decoration: underline; }
button.action { font: inherit; font-weight: 600; color: #fff; background: var(--you); border: 0;
  border-radius: 8px; padding: 8px 14px; cursor: pointer; }
button.action:disabled { opacity: .6; cursor: progress; }
.status { color: var(--ink-2); font-size: 13px; margin-left: 10px; }
details { margin-top: 24px; }
details table { border-collapse: collapse; font-size: 12px; margin: 10px 0 18px; }
details th, details td { border-top: 1px solid var(--grid); padding: 3px 10px; text-align: right;
  font-variant-numeric: tabular-nums; }
details th { color: var(--ink-2); font-weight: 600; }
"""

_JS = r"""
const D = JSON.parse(document.getElementById('data').textContent);
const NS = 'http://www.w3.org/2000/svg';
const el = (tag, attrs = {}, parent) => {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
};
const h = (tag, cls, text, parent) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  if (parent) parent.appendChild(n);
  return n;
};
const fmt = (v, kind) => {
  if (v == null) return '–';
  const r = Math.round(v);
  if (kind === 'signed') return (r > 0 ? '+' : '') + r.toLocaleString();
  return r.toLocaleString();
};
const niceStep = (span) => {
  const raw = span / 4, mag = Math.pow(10, Math.floor(Math.log10(raw || 1)));
  for (const m of [1, 2, 2.5, 5, 10]) if (raw <= m * mag) return m * mag;
  return 10 * mag;
};
const tip = h('div', 'tip', null, document.body);
tip.setAttribute('role', 'status');

function chart(c, host) {
  const W = 520, H = 200, L = 46, R = 46, T = 10, B = 24;
  const pts = c.points;
  const xs = pts.map(p => p.m);
  const all = pts.flatMap(p => [p.v, p.p25, p.p75, p.p50]).filter(v => v != null);
  let lo = Math.min(0, ...all), hi = Math.max(...all);
  if (hi === lo) hi = lo + 1;
  const step = niceStep(hi - lo);
  lo = Math.floor(lo / step) * step; hi = Math.ceil(hi / step) * step;
  const x = m => L + (m - xs[0]) / Math.max(1, xs[xs.length - 1] - xs[0]) * (W - L - R);
  const y = v => T + (hi - v) / (hi - lo) * (H - T - B);
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img', tabindex: 0,
                         'aria-label': `${c.label} by minute; you versus ${D.group} (table below)`}, host);
  for (let v = lo; v <= hi + 1e-9; v += step) {
    el('line', {x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: 'var(--grid)', 'stroke-width': 1}, svg);
    const t = el('text', {x: L - 6, y: y(v) + 4, 'text-anchor': 'end', class: 'tick'}, svg);
    t.textContent = fmt(v, c.kind);
  }
  if (lo < 0 && hi > 0) el('line', {x1: L, x2: W - R, y1: y(0), y2: y(0), stroke: 'var(--axis)', 'stroke-width': 1}, svg);
  for (const m of xs) if (m % 5 === 0) {
    const t = el('text', {x: x(m), y: H - 6, 'text-anchor': 'middle', class: 'tick'}, svg);
    t.textContent = m;
  }
  const band = pts.filter(p => p.p25 != null);
  if (band.length) {
    const d = band.map((p, i) => `${i ? 'L' : 'M'}${x(p.m)},${y(p.p75)}`).join('') +
              band.slice().reverse().map(p => `L${x(p.m)},${y(p.p25)}`).join('') + 'Z';
    el('path', {d, fill: 'var(--wash)'}, svg);
    el('path', {d: band.map((p, i) => `${i ? 'L' : 'M'}${x(p.m)},${y(p.p50)}`).join(''), fill: 'none',
                stroke: 'var(--group)', 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round'}, svg);
  }
  const mine = pts.filter(p => p.v != null);
  el('path', {d: mine.map((p, i) => `${i ? 'L' : 'M'}${x(p.m)},${y(p.v)}`).join(''), fill: 'none',
              stroke: 'var(--you)', 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round'}, svg);
  const last = mine[mine.length - 1];
  if (last) {
    el('circle', {cx: x(last.m), cy: y(last.v), r: 4, fill: 'var(--you)', stroke: 'var(--surface)', 'stroke-width': 2}, svg);
    const t = el('text', {x: x(last.m) + 8, y: y(last.v) + 4, class: 'endlabel'}, svg);
    t.textContent = fmt(last.v, c.kind);
  }
  const cross = el('line', {y1: T, y2: H - B, stroke: 'var(--axis)', 'stroke-width': 1, visibility: 'hidden'}, svg);
  const dot = el('circle', {r: 4, fill: 'var(--you)', stroke: 'var(--surface)', 'stroke-width': 2, visibility: 'hidden'}, svg);
  let idx = pts.length - 1;
  const show = (i, cx, cy) => {
    idx = Math.max(0, Math.min(pts.length - 1, i));
    const p = pts[idx];
    cross.setAttribute('x1', x(p.m)); cross.setAttribute('x2', x(p.m)); cross.setAttribute('visibility', 'visible');
    if (p.v != null) { dot.setAttribute('cx', x(p.m)); dot.setAttribute('cy', y(p.v)); dot.setAttribute('visibility', 'visible'); }
    tip.replaceChildren();
    h('div', 't', `${c.label} · minute ${p.m}`, tip);
    const row = (color, value, name) => {
      const r = h('div', 'row', null, tip);
      const k = h('i', null, null, r); k.style.background = color;
      h('strong', null, value, r); h('span', null, name, r);
    };
    row('var(--you)', fmt(p.v, c.kind), 'you');
    if (p.p50 != null) {
      row('var(--group)', fmt(p.p50, c.kind), `${D.group} median`);
      h('div', 't', `middle 50%: ${fmt(p.p25, c.kind)} to ${fmt(p.p75, c.kind)}` + (p.pct != null ? ` · you: better than ${p.pct}` : ''), tip);
    }
    tip.style.display = 'block';
    const box = svg.getBoundingClientRect();
    const px = cx ?? box.left + x(p.m) / W * box.width, py = cy ?? box.top + 20;
    const tw = tip.offsetWidth;
    tip.style.left = Math.min(window.innerWidth - tw - 8, px + 14) + 'px';
    tip.style.top = Math.max(8, py - 10) + 'px';
  };
  const hide = () => { tip.style.display = 'none'; cross.setAttribute('visibility', 'hidden'); dot.setAttribute('visibility', 'hidden'); };
  svg.addEventListener('pointermove', e => {
    const box = svg.getBoundingClientRect();
    const vx = (e.clientX - box.left) / box.width * W;
    let best = 0;
    pts.forEach((p, i) => { if (Math.abs(x(p.m) - vx) < Math.abs(x(pts[best].m) - vx)) best = i; });
    show(best, e.clientX, e.clientY);
  });
  svg.addEventListener('pointerleave', hide);
  svg.addEventListener('focus', () => show(idx));
  svg.addEventListener('blur', hide);
  svg.addEventListener('keydown', e => {
    if (e.key === 'ArrowLeft') { show(idx - 1); e.preventDefault(); }
    if (e.key === 'ArrowRight') { show(idx + 1); e.preventDefault(); }
  });
}

const charts = document.getElementById('charts');
for (const c of D.curves) {
  const card = h('div', 'card chart', null, charts);
  h('h3', null, c.label, card);
  chart(c, card);
}

function healthChart(host) {
  const hd = D.health;
  if (!host || !hd || !hd.series.length) return;
  const W = 560, H = 160, L = 40, R = 16, T = 12, B = 24;
  const tMax = hd.series[hd.series.length - 1][0];
  const x = t => L + t / tMax * (W - L - R);
  const y = v => T + (1 - v) * (H - T - B);
  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img',
                         'aria-label': 'Your health over the game (events listed below)'}, host);
  for (const v of [0, 0.5, 1]) {
    el('line', {x1: L, x2: W - R, y1: y(v), y2: y(v), stroke: 'var(--grid)', 'stroke-width': 1}, svg);
    const t = el('text', {x: L - 6, y: y(v) + 4, 'text-anchor': 'end', class: 'tick'}, svg);
    t.textContent = Math.round(v * 100) + '%';
  }
  for (let m = 0; m * 60 <= tMax; m += 5) {
    const t = el('text', {x: x(m * 60), y: H - 6, 'text-anchor': 'middle', class: 'tick'}, svg);
    t.textContent = m;
  }
  const d = hd.series.map(([t, v], i) => `${i ? 'L' : 'M'}${x(t)},${y(v)}`).join('');
  el('path', {d, fill: 'none', stroke: 'var(--you)', 'stroke-width': 2, 'stroke-linejoin': 'round'}, svg);
  const glyph = {death: '×', recall: '↩', loss: '▾'};
  for (const mk of hd.marks) {
    const g = el('text', {x: x(mk.t), y: T + 10, 'text-anchor': 'middle', class: 'endlabel'}, svg);
    g.textContent = glyph[mk.kind] || '•';
    const title = el('title', {}, g); title.textContent = mk.text;
  }
}
healthChart(document.getElementById('health'));

const strip = document.getElementById('deaths');
if (strip && D.deaths.length) {
  h('div', 'axis', null, strip);
  for (const d of D.deaths) {
    const left = (d.minute / D.duration_min * 100) + '%';
    const b = h('button', 'mark', '×', strip);
    b.style.left = left;
    b.setAttribute('aria-label', d.text);
    const on = (e) => {
      tip.replaceChildren(); h('div', null, d.text, tip); tip.style.display = 'block';
      const r = b.getBoundingClientRect();
      tip.style.left = Math.min(window.innerWidth - tip.offsetWidth - 8, r.left) + 'px';
      tip.style.top = (r.bottom + 6) + 'px';
    };
    b.addEventListener('pointerenter', on); b.addEventListener('focus', on);
    b.addEventListener('pointerleave', () => tip.style.display = 'none');
    b.addEventListener('blur', () => tip.style.display = 'none');
    const t = h('div', 't', d.clock, strip); t.style.left = left;
  }
}
"""


def _esc(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _curve_kind(name: str) -> str:
    return "signed" if name.endswith("_diff") else "int"


def curve_series(score: GameScore) -> list[dict[str, Any]]:
    out = []
    p = score.participant
    for name in CHART_CURVES:
        metric = CURVE_METRICS[name]
        if not metric.applies_to(p.role):
            continue
        by_minute = {s.minute: s for s in score.curves.get(name, [])}
        points = []
        for row in p.minutes[1:]:
            value = getattr(row, name)
            s = by_minute.get(row.minute)
            points.append({
                "m": row.minute, "v": value,
                "p25": s.baseline.p25 if s else None, "p50": s.baseline.p50 if s else None,
                "p75": s.baseline.p75 if s else None,
                "pct": f"{round(s.goodness)}%" if s else None,
            })
        if any(pt["v"] is not None for pt in points):
            out.append({"name": name, "label": metric.label[0].upper() + metric.label[1:],
                        "kind": _curve_kind(name), "points": points})
    return out


def _scorecard_html(score: GameScore) -> str:
    rows = []
    by_area: dict[str, list] = {}
    for s in score.game.values():
        if s.metric.coachable:
            by_area.setdefault(s.metric.area, []).append(s)
    for area in AREAS:
        if area not in by_area:
            continue
        rows.append(f'<tr class="area"><td colspan="4">{_esc(area)}</td></tr>')
        for s in sorted(by_area[area], key=lambda s: s.goodness):
            flag = "work on" if s.goodness <= 25 else "strength" if s.goodness >= 75 else ""
            rows.append(
                f"<tr><td>{_esc(s.metric.label)}<div class=\"small\">median "
                f"{_esc(s.metric.show(s.baseline.p50))} · n={s.baseline.n}"
                f"{' · Master+ ' + _esc(s.metric.show(score.reference[s.metric.name].p50)) if s.metric.name in score.reference else ''}"
                "</div></td>"
                f'<td class="num">{_esc(s.metric.show(s.value))}</td>'
                f'<td class="bar"><div class="track" role="img" aria-label="better than {round(s.goodness)}%">'
                f'<div class="line"></div><div class="typ"></div>'
                f'<div class="dot" style="left:{s.goodness:.1f}%"></div></div></td>'
                f'<td class="num">better than {round(s.goodness)}%<div class="small">{flag}</div></td></tr>'
            )
    if not rows:
        return '<p class="sub">No baselines for this tier and role yet.</p>'
    return f'<div class="scroll"><table class="score">{"".join(rows)}</table></div>'


def coach_html(result: CoachResult, coach_url: str | None = None) -> str:
    by_id = result.evidence.by_id()
    source = ("offline template coach" if result.model == "offline"
              else f"{result.model}{' · cached' if result.cached else ''}")
    parts = [f'<p class="headline">{_esc(result.output.headline)}</p>']
    for point in result.output.points:
        cited = " ".join(f"[{e}] {by_id[e].text}" for e in point.evidence_ids if e in by_id)
        parts.append(
            f'<div class="point"><div class="kind">{"work on" if point.kind == "weakness" else "strength"}'
            f" · {_esc(point.area)}</div><h4>{_esc(point.title)}</h4>"
            f"<p>{_esc(point.explanation)}</p><p class=\"advice\">{_esc(point.advice)}</p>"
            f'<p class="ev">Evidence: {_esc(cited)}</p></div>'
        )
    if result.dropped:
        parts.append(f'<p class="meta">{len(result.dropped)} point(s) removed by the grounding check.</p>')
    parts.append(f'<p class="meta">Coach: {_esc(source)}. Every number above appears in the cited '
                 "evidence; points that failed that check are not shown.</p>")
    if result.coach_pending and coach_url:
        attr = "data-stream" if coach_url.endswith("/stream") else "data-post"
        parts.append(f'<p><button class="action" {attr}="{_esc(coach_url)}">Get AI coaching</button>'
                     '<span class="status"></span></p>')
    return "".join(parts)


def _table_view(curves: list[dict[str, Any]], group: str) -> str:
    blocks = []
    for c in curves:
        rows = "".join(
            f"<tr><td>{p['m']}</td><td>{'' if p['v'] is None else round(p['v'])}</td>"
            f"<td>{'' if p['p50'] is None else round(p['p50'])}</td>"
            f"<td>{'' if p['p25'] is None else round(p['p25'])}–{'' if p['p75'] is None else round(p['p75'])}</td>"
            f"<td>{p['pct'] or ''}</td></tr>"
            for p in c["points"]
        )
        blocks.append(f"<h3>{_esc(c['label'])}</h3><div class=\"scroll\"><table><tr><th>minute</th>"
                      f"<th>you</th><th>{_esc(group)} median</th><th>middle 50%</th>"
                      f"<th>you: better than</th></tr>{rows}</table></div>")
    return "".join(blocks)


# Buttons with data-post="url" POST there and reload the page when it answers. Used for
# "Get AI coaching"; sync has its own progress loop on the player page.
ACTION_JS = r"""
// Streaming coaching: points appear as soon as they're written and pass the grounding
// check; the final answer replaces them via a reload (it's cached by then).
for (const b of document.querySelectorAll('button[data-stream]')) {
  b.addEventListener('click', async () => {
    const status = b.parentElement.querySelector('.status');
    const card = b.closest('.coach');
    const live = document.createElement('div');
    card.insertBefore(live, b.parentElement);
    b.disabled = true; status.textContent = 'Writing coaching...';
    try {
      const r = await fetch(b.dataset.stream, {method: 'POST'});
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      const reader = r.body.getReader(), dec = new TextDecoder();
      let buf = '';
      for (;;) {
        const {value, done} = await reader.read();
        if (done) break;
        buf += dec.decode(value, {stream: true});
        let cut;
        while ((cut = buf.indexOf('\n\n')) >= 0) {
          const line = buf.slice(0, cut); buf = buf.slice(cut + 2);
          if (!line.startsWith('data: ')) continue;
          const ev = JSON.parse(line.slice(6));
          if (ev.type === 'point') {
            const p = document.createElement('div'); p.className = 'point';
            const k = document.createElement('div'); k.className = 'kind';
            k.textContent = (ev.point.kind === 'weakness' ? 'work on' : 'strength') + ' · ' + ev.point.area;
            const t = document.createElement('h4'); t.textContent = ev.point.title;
            const e = document.createElement('p'); e.textContent = ev.point.explanation;
            const a = document.createElement('p'); a.className = 'advice'; a.textContent = ev.point.advice;
            p.append(k, t, e, a); live.append(p);
          } else if (ev.type === 'retrying') {
            status.textContent = 'Checking every number against the evidence...';
          } else if (ev.type === 'error') {
            throw new Error(ev.message);
          } else if (ev.type === 'final') {
            location.reload();
          }
        }
      }
    } catch (e) { b.disabled = false; status.textContent = 'Failed: ' + e.message; }
  });
}
for (const b of document.querySelectorAll('button[data-post]')) {
  b.addEventListener('click', async () => {
    const status = b.parentElement.querySelector('.status');
    b.disabled = true; if (status) status.textContent = 'Working... this takes a few seconds.';
    try {
      const r = await fetch(b.dataset.post, {method: 'POST'});
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      location.reload();
    } catch (e) { b.disabled = false; if (status) status.textContent = 'Failed: ' + e.message; }
  });
}
"""


def nav(links: list[tuple[str, str]]) -> str:
    return ('<nav class="top">'
            + "".join(f'<a href="{_esc(u)}">{_esc(t)}</a>' for t, u in links) + "</nav>")


def page(title: str, description: str, body: str, data: dict[str, Any] | None = None,
         extra_js: str = "") -> str:
    payload = json.dumps(data or {"group": "", "curves": [], "deaths": [], "duration_min": 1})
    payload = payload.replace("</", "<\\/")
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        # Inline icon: saves the browser a /favicon.ico request on every page.
        '<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 '
        'viewBox=%220 0 16 16%22%3E%3Crect width=%2216%22 height=%2216%22 rx=%224%22 '
        'fill=%22%232a78d6%22/%3E%3C/svg%3E">'
        f"<title>{_esc(title)}</title><meta name=\"description\" content=\"{_esc(description)}\">"
        f"<style>{_CSS}</style></head><body><main>{body}</main>"
        f'<script type="application/json" id="data">{payload}</script>'
        f"<script>{_JS}{ACTION_JS}{extra_js}</script></body></html>"
    )



def _moments_html(result: CoachResult) -> str:
    """Key moments from the high-elo comparison, in the same words the coach was given."""
    items = [e for e in result.evidence.items
             if e.kind == "decision" or (e.kind == "context" and e.area == "macro")]
    if not items:
        return ""
    lines = []
    for e in items:
        tag = {"weakness": "Different from high elo", "strength": "Matched high elo"}.get(e.polarity, "")
        label = f'<div class="kind">{_esc(tag)}</div>' if tag else ""
        lines.append(f'<div class="point">{label}<p>{_esc(e.text)}</p></div>')
    return ('<h2>Compared with high-elo play</h2><div class="card coach">' + "".join(lines)
            + '<p class="meta">From models trained on Grandmaster and Challenger games in your role. '
              "They describe what tended to follow each choice in those games, not certainties.</p></div>")


_LED_TO = {"death": "a death", "recall": "a recall", "stayed": "you staying"}


def _health_events(live: dict[str, Any]) -> list[tuple[float, str, str]]:
    summary = live["summary"]
    out = []
    for d in summary.get("drops", []):
        out.append((d["start"], "loss", f"{clock(d['start'] / 60)}: lost {round(d['lost'] * 100)}% "
                    f"health in {round(d['end'] - d['start'])} s, then {_LED_TO[d['led_to']]}"))
    for r in summary.get("recalls", []):
        out.append((r["t"], "recall", f"{clock(r['t'] / 60)}: recalled at {round(r['hp'] * 100)}% "
                    f"health with {r['gold']} gold"))
    for d in summary.get("deaths", []):
        how = "burst" if d["burst"] else "died"
        out.append((d["t"], "death", f"{clock(d['t'] / 60)}: {how}, {round(d['hp_10s_before'] * 100)}% "
                    "health 10 s earlier"))
    return sorted(out)


def _health_data(result: CoachResult) -> dict[str, Any] | None:
    if not result.live:
        return None
    return {"series": result.live["hp_series"],
            "marks": [{"t": t, "kind": k, "text": text} for t, k, text in _health_events(result.live)]}


def _health_html(result: CoachResult) -> str:
    if not result.live:
        return ""
    items = "".join(f"<li>{_esc(text)}</li>" for _t, _k, text in _health_events(result.live))
    return ('<h2>Health (live recording)</h2><div class="card"><div id="health"></div>'
            '<p class="meta">▾ big health loss · ↩ recall · × death. Recorded every second '
            "from the game client.</p>"
            f'<ol class="deathlist">{items or "<li>No big health losses.</li>"}</ol></div>')


def game_html(result: CoachResult, *, coach_url: str | None = None,
              links: list[tuple[str, str]] | None = None) -> str:
    game, p = result.games[0]
    score = result.scores[0]
    first = next(iter(score.game.values()), None)
    group = (first.baseline.scope.split(",")[0] if first else
             result.tier_bucket.replace("_", " ").title())
    curves = curve_series(score)
    m = p.metrics

    def tile(label: str, key: str) -> str:
        s = score.game.get(key)
        if key not in m:
            return ""
        value = s.metric.show(m[key]) if s else f"{m[key]:.1f}"
        pct = f"better than {round(s.goodness)}% of comparable players" if s else "no baseline"
        return (f'<div class="card tile"><div class="label">{_esc(label)}</div>'
                f'<div class="value">{_esc(value)}</div><div class="pct">{_esc(pct)}</div></div>')

    kda = f"{int(m.get('kills', 0))}/{int(m.get('deaths', 0))}/{int(m.get('assists', 0))}"
    tiles = (f'<div class="card tile"><div class="label">K / D / A</div><div class="value">{kda}</div>'
             f'<div class="pct">{"win" if p.win else "loss"} · {clock(game.duration_min)}</div></div>'
             + tile("CS per minute", "cs_per_min") + tile("Gold lead at 15", "gold_diff_at_15")
             + tile("Kill participation", "kill_participation") + tile("Vision per minute", "vision_per_min"))

    deaths = [{"minute": d.minute, "clock": clock(d.minute),
               "text": next((e.text for e in result.evidence.items
                             if e.kind == "death" and e.data.get("minute") == d.minute),
                            f"Death at {clock(d.minute)} in the {d.where}.")}
              for d in p.deaths]
    death_list = "".join(f"<li>{_esc(d['text'])}</li>" for d in deaths)
    title = f"{p.champion_name} {p.role.lower()} review"
    legend = (f'<div class="legend"><span class="key"><i style="background:var(--you)"></i>You</span>'
              f'<span class="key"><i style="background:var(--group)"></i>{_esc(group)} median</span>'
              f'<span class="key"><b style="background:var(--wash)"></b>Middle 50% of {_esc(group)}</span></div>')
    body = (
        (nav(links) if links else "")
        + f"<h1>{_esc(p.champion_name)} {_esc(p.role.lower())} · {'Win' if p.win else 'Loss'}</h1>"
        f'<p class="sub">{_esc(game.match_id)} · patch {_esc(game.patch)} · compared against '
        f"{_esc(group)} players{', patches ' + _esc(first.baseline.patch_window) if first else ''}</p>"
        f'<div class="tiles">{tiles}</div>'
        f'<h2>Coaching</h2><div class="card coach">{coach_html(result, coach_url)}</div>'
        + _moments_html(result)
        + f'<h2>Minute by minute</h2>{legend}<div class="charts" id="charts"></div>'
        + _health_html(result) +
        f'<h2>Deaths</h2><div class="card"><div class="deaths" id="deaths"></div>'
        f'<ol class="deathlist">{death_list or "<li>No deaths.</li>"}</ol></div>'
        f'<h2>How you compare</h2><div class="card">{_scorecard_html(score)}</div>'
        f"<details><summary>Table view of every chart</summary>{_table_view(curves, group)}</details>"
    )
    data = {"group": group, "curves": curves, "deaths": deaths, "duration_min": game.duration_min,
            "health": _health_data(result)}
    return page(title, f"RiftWatch review of {game.match_id}", body, data)


def recent_html(result: CoachResult) -> str:
    group = result.tier_bucket.replace("_", " ").title()
    rows = []
    for game, p in result.games:
        m = p.metrics
        rows.append(
            f"<tr><td>{_esc(p.champion_name)}</td><td>{_esc(p.role.lower())}</td>"
            f"<td>{'win' if p.win else 'loss'}</td>"
            f'<td class="num">{int(m.get("kills", 0))}/{int(m.get("deaths", 0))}/{int(m.get("assists", 0))}</td>'
            f'<td class="num">{m.get("cs_per_min", 0):.1f}</td>'
            f'<td class="num">{_esc(clock(game.duration_min))}</td><td>{_esc(game.match_id)}</td></tr>'
        )
    wins = sum(p.win for _, p in result.games)
    body = (
        f"<h1>Last {len(result.games)} ranked games</h1>"
        f'<p class="sub">{wins} wins, {len(result.games) - wins} losses · each game compared against '
        f"{_esc(group)} players in the role played</p>"
        f'<h2>Coaching</h2><div class="card coach">{coach_html(result)}</div>'
        f'<h2>Games</h2><div class="card scroll"><div><table class="score"><tr class="area"><td>champion</td>'
        f"<td>role</td><td>result</td><td>K/D/A</td><td>CS/min</td><td>length</td><td>match</td></tr>"
        f'{"".join(rows)}</table></div></div>'
    )
    data = {"group": group, "curves": [], "deaths": [], "duration_min": 1}
    return page("Recent games review", "RiftWatch review of recent ranked games", body, data)

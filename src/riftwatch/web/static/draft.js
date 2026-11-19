// Draft assistant (/draft): champion inputs with a datalist, the role, an optional Riot ID;
// every change asks /api/draft/suggest and redraws the suggestions. Loaded only on that page.
(() => {
  'use strict';
  const root = document.getElementById('draft');
  if (!root) return;
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
  const store = {
    get(k, d) { try { const v = sessionStorage.getItem(k); return v ? JSON.parse(v) : d; } catch { return d; } },
    set(k, v) { try { sessionStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
  };
  const norm = (s) => String(s || '').toLowerCase().replace(/[^a-z0-9]/g, '');
  const champs = JSON.parse(root.dataset.champs || '[]');
  const byName = new Map(champs.map((c) => [norm(c.name), c]));
  const list = document.getElementById('draft-champs');
  list.innerHTML = champs.map((c) => `<option value="${esc(c.name)}"></option>`).join('');

  const form = document.getElementById('draft-form');
  const out = document.getElementById('draft-out');
  const meInput = document.getElementById('draft-me');
  let role = root.dataset.role || store.get('rw-draft-role', '');
  const roleButtons = [...document.querySelectorAll('#draft-role button')];
  const showRole = () => roleButtons.forEach((b) => b.classList.toggle('on', b.dataset.role === role));
  showRole();

  // The Riot ID: from the page, else the last profile this browser opened.
  if (!meInput.value) {
    try {
      const recent = JSON.parse(localStorage.getItem('rw-recent') || '[]');
      if (recent[0] && recent[0].name && recent[0].tag) meInput.value = `${recent[0].name}#${recent[0].tag}`;
    } catch { /* storage blocked */ }
  }

  const slots = [...form.querySelectorAll('.draft-slot')];
  const champOf = (input) => byName.get(norm(input.value)) || null;
  function paint(slot) {
    const input = slot.querySelector('input'), img = slot.querySelector('img'), c = champOf(input);
    img.hidden = !c;
    if (c) { img.src = c.icon; img.alt = c.name; }
    slot.classList.toggle('bad', !!input.value.trim() && !c);
  }

  function state() {
    const picks = { ally: [], enemy: [], ban: [] };
    for (const slot of slots) {
      const input = slot.querySelector('input'), c = champOf(input);
      if (!c) continue;
      const sel = slot.querySelector('select');
      picks[input.dataset.side].push(sel && sel.value ? `${c.id}:${sel.value}` : String(c.id));
    }
    return picks;
  }

  function save() {
    store.set('rw-draft-role', role);
    store.set('rw-draft-slots', slots.map((s) => [s.querySelector('input').value, s.querySelector('select')?.value || '']));
  }
  // Restore an in-progress draft after a reload.
  store.get('rw-draft-slots', []).forEach(([v, r], i) => {
    if (!slots[i]) return;
    slots[i].querySelector('input').value = v || '';
    const sel = slots[i].querySelector('select');
    if (sel) sel.value = r || '';
    paint(slots[i]);
  });

  const sign = (x) => (x > 0 ? '+' : '') + x.toFixed(1);
  const cls = (x) => (x > 0.05 ? 'up' : x < -0.05 ? 'down' : '');

  function card(p, i) {
    const parts = p.parts.map((x) => `<span class="dp ${cls(x.pts)}" data-tip="${esc(x.label)}&#10;${esc(x.detail)}">${esc(x.label)} <b>${sign(x.pts)}</b></span>`).join('');
    const reasons = p.reasons.map((x) => `<li class="${cls(x.pts)}">${esc(x.detail)} <span class="muted">(${esc(x.label.toLowerCase())}, ${sign(x.pts)})</span></li>`).join('');
    return `<div class="draft-pick">
      <span class="n">${i + 1}</span>
      <a class="who" href="/champions/${esc(p.slug)}?role=${esc(role.toLowerCase())}"><img src="${esc(p.icon)}" alt="" loading="lazy"><b>${esc(p.name)}</b>${p.pool_games ? `<small>${p.pool_games} of your games</small>` : ''}</a>
      <div class="win"><b>${p.win.toFixed(1)}%</b><span class="bar"><i style="width:${Math.max(0, Math.min(100, (p.win - 35) * (100 / 30)))}%"></i></span></div>
      <div class="parts">${parts}</div>
      ${reasons ? `<ul class="reasons">${reasons}</ul>` : ''}
    </div>`;
  }

  function mini(p) {
    return `<a class="draft-mini" href="/champions/${esc(p.slug)}?role=${esc(role.toLowerCase())}" data-tip="${esc(p.name)}&#10;${p.win.toFixed(1)}% expected${p.parts.filter((x) => x.key === 'lane').map((x) => `&#10;${esc(x.detail)}`).join('')}"><img src="${esc(p.icon)}" alt="" loading="lazy"><span>${esc(p.name)}</span><b>${p.win.toFixed(1)}%</b></a>`;
  }

  function render(data) {
    let html = '';
    if (data.me_missing) html += `<p class="muted small">RiftWatch hasn't stored that Riot ID yet: search it first to include your pool and playstyle.</p>`;
    if (data.needs.length) html += `<div class="panel draft-needs"><div class="panel-head"><span>What the teams look like</span></div><ul>${data.needs.map((n) => `<li class="${n.kind}">${esc(n.text)}</li>`).join('')}</ul></div>`;
    html += `<div class="panel"><div class="panel-head"><span>Best picks for ${esc(data.role_name.toLowerCase())}</span><span class="muted small">${esc(data.rank)}${data.lane_opponent ? ` &middot; against ${esc(data.lane_opponent)}` : ''}${data.me ? ` &middot; for ${esc(data.me)}` : ''} &middot; ${data.candidates} champions considered</span></div>`;
    html += data.picks.length ? `<div class="draft-picks">${data.picks.map(card).join('')}</div>` : '<div class="empty"><p>No champion has enough games in this role yet.</p></div>';
    html += '</div>';
    if (data.yours.length) html += `<div class="panel"><div class="panel-head"><span>From your pool</span><span class="muted small">champions you've played 3+ times in this role</span></div><div class="panel-body draft-row-list">${data.yours.map(mini).join('')}</div></div>`;
    if (data.avoid.length) html += `<div class="panel"><div class="panel-head"><span>Careful against ${esc(data.lane_opponent || 'their laner')}</span><span class="muted small">the worst lane matchups among the candidates</span></div><div class="panel-body draft-row-list">${data.avoid.map(mini).join('')}</div></div>`;
    out.innerHTML = html;
  }

  let timer = null, seq = 0;
  function refresh() {
    save();
    clearTimeout(timer);
    if (!role) return;
    timer = setTimeout(async () => {
      const s = state(), mine = ++seq;
      const q = new URLSearchParams({ role, bans: s.ban.join(','), allies: s.ally.join(','), enemies: s.enemy.join(','),
                                      me: meInput.value.trim() });
      out.classList.add('loading');
      try {
        const r = await fetch(`/api/draft/suggest?${q}`);
        const data = await r.json();
        if (mine !== seq) return;              // a newer request is on its way
        if (!r.ok) throw new Error(data.detail || data.error || 'failed');
        render(data);
      } catch (err) {
        if (mine === seq) out.innerHTML = `<div class="panel empty"><p>Couldn't get suggestions: ${esc(err.message)}</p></div>`;
      } finally {
        if (mine === seq) out.classList.remove('loading');
      }
    }, 250);
  }

  roleButtons.forEach((b) => b.addEventListener('click', () => { role = b.dataset.role; showRole(); refresh(); }));
  form.addEventListener('submit', (e) => { e.preventDefault(); refresh(); });
  form.addEventListener('input', (e) => {
    const slot = e.target.closest('.draft-slot');
    if (slot) paint(slot);
    refresh();
  });
  form.addEventListener('change', refresh);
  document.getElementById('draft-clear').addEventListener('click', () => {
    slots.forEach((s) => { s.querySelector('input').value = ''; const sel = s.querySelector('select'); if (sel) sel.value = ''; paint(s); });
    refresh();
  });
  if (role) refresh();
  root.dataset.ready = '1';
})();

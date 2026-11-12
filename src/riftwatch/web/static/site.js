// RiftWatch website behaviour. Pages arrive fully rendered; this only adds interaction.
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const store = {
    get(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
  };
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));

  // -- theme -------------------------------------------------------------------------------
  const themeBtn = $('#theme-toggle');
  if (themeBtn) themeBtn.addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'light' ? 'dark' : 'light';
    document.documentElement.dataset.theme = next;
    store.set('rw-theme', next);
  });

  // -- command palette (find a player from any page) ---------------------------------------
  const palette = $('#palette');
  function openPalette() {
    if (!palette) return;
    palette.classList.add('open');
    const input = $('input', palette); input.focus(); input.select();
    input.dispatchEvent(new Event('focus'));
  }
  function closePalette() { palette && palette.classList.remove('open'); }
  for (const b of $$('[data-palette]')) b.addEventListener('click', openPalette);
  if (palette) palette.addEventListener('mousedown', (e) => { if (e.target === palette) closePalette(); });
  document.addEventListener('keydown', (e) => {
    const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName || '');
    if ((e.key === 'k' && (e.ctrlKey || e.metaKey)) || (e.key === '/' && !typing)) {
      if (!$('#q-big')) { e.preventDefault(); openPalette(); }
      else if (e.key === '/' || e.ctrlKey || e.metaKey) { e.preventDefault(); $('#q-big').focus(); }
    } else if (e.key === 'Escape') closePalette();
  });

  // -- recent searches and favourites ------------------------------------------------------
  const RECENT = 'rw-recent', FAVS = 'rw-favs';
  const key = (p) => `${p.region}/${p.name}#${p.tag}`.toLowerCase();
  function remember(p) {
    const list = store.get(RECENT, []).filter((x) => key(x) !== key(p));
    list.unshift(p); store.set(RECENT, list.slice(0, 8));
  }
  const profile = $('[data-profile]');
  if (profile) {
    const p = JSON.parse(profile.dataset.profile);
    if (p.synced) remember(p);
    const star = $('#fav');
    if (star) {
      const isFav = () => store.get(FAVS, []).some((x) => key(x) === key(p));
      star.classList.toggle('on', isFav());
      star.addEventListener('click', () => {
        const favs = store.get(FAVS, []).filter((x) => key(x) !== key(p));
        if (!isFav()) favs.unshift(p);
        store.set(FAVS, favs.slice(0, 12)); star.classList.toggle('on', isFav());
      });
    }
  }
  const profileUrl = (p) => `/players/${p.region}/${encodeURIComponent(p.name + '-' + p.tag)}`;
  function chip(p, removable) {
    return `<a class="chip" href="${profileUrl(p)}">${p.icon ? `<img src="${esc(p.icon)}" alt="">` : ''}` +
      `${esc(p.name)}<span class="muted">#${esc(p.tag)}</span>` +
      (removable ? `<span class="x" data-forget="${esc(key(p))}" title="Remove">&times;</span>` : '') + '</a>';
  }
  const chips = $('#recent-chips');
  function drawChips() {
    if (!chips) return;
    const favs = store.get(FAVS, []), recent = store.get(RECENT, []);
    chips.innerHTML = favs.map((p) => chip(p, false)).join('') +
      recent.filter((p) => !favs.some((f) => key(f) === key(p))).map((p) => chip(p, true)).join('');
  }
  drawChips();
  document.addEventListener('click', (e) => {
    const x = e.target.closest('[data-forget]');
    if (!x) return;
    e.preventDefault();
    store.set(RECENT, store.get(RECENT, []).filter((p) => key(p) !== x.dataset.forget));
    drawChips();
  });

  // -- search with autocomplete ------------------------------------------------------------
  for (const form of $$('form.search')) {
    const input = $('input', form), region = $('select', form), box = $('.suggest', form);
    const saved = store.get('rw-region', null);
    if (saved && region && !form.dataset.fixed) region.value = saved;
    region && region.addEventListener('change', () => store.set('rw-region', region.value));
    let items = [], active = -1, timer = null;
    function render(list, title) {
      items = list; active = -1;
      if (!list.length) { box.classList.remove('open'); return; }
      box.innerHTML = `<div class="head">${title}</div>` + list.map((p, i) =>
        `<a href="${profileUrl(p)}" data-i="${i}"><img src="${esc(p.icon || '')}" alt="" loading="lazy">` +
        `<span class="who"><b>${esc(p.name)}</b> <span>#${esc(p.tag)}</span></span>` +
        `<span class="reg">${esc(p.region.toUpperCase())}</span></a>`).join('');
      box.classList.add('open');
    }
    function recentList() {
      const favs = store.get(FAVS, []), recent = store.get(RECENT, []);
      return [...favs, ...recent.filter((p) => !favs.some((f) => key(f) === key(p)))].slice(0, 8);
    }
    input.addEventListener('focus', () => { if (!input.value.trim()) render(recentList(), 'Recent searches'); });
    input.addEventListener('input', () => {
      clearTimeout(timer);
      const q = input.value.trim();
      if (q.length < 2) { render(q ? [] : recentList(), 'Recent searches'); return; }
      timer = setTimeout(async () => {
        try {
          const r = await fetch(`/api/autocomplete?q=${encodeURIComponent(q)}&region=${region ? region.value : ''}`);
          if (r.ok) render((await r.json()).players, 'Players');
        } catch {}
      }, 140);
    });
    input.addEventListener('keydown', (e) => {
      if (!box.classList.contains('open') || !items.length) return;
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        active = (active + (e.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length;
        $$('a', box).forEach((a, i) => a.classList.toggle('on', i === active));
      } else if (e.key === 'Enter' && active >= 0) {
        e.preventDefault(); location.href = profileUrl(items[active]);
      } else if (e.key === 'Escape') { box.classList.remove('open'); }
    });
    document.addEventListener('click', (e) => { if (!form.contains(e.target)) box.classList.remove('open'); });
    form.addEventListener('submit', (e) => {
      if (!input.value.includes('#')) {
        e.preventDefault();
        input.setCustomValidity('Use your Riot ID: Name#TAG (the tag is shown under your name in the client)');
        input.reportValidity();
        setTimeout(() => input.setCustomValidity(''), 2500);
      }
    });
  }

  // -- background jobs (update, live game) -------------------------------------------------
  function friendly(message) {
    const m = String(message || '');
    if (/401|403|key was rejected|forbidden/i.test(m)) return "RiftWatch can't reach Riot right now (the site's API key needs renewing). Please try again later.";
    if (/429|rate.?limit|too many/i.test(m)) return 'Too many lookups at once. Please wait a minute and try again.';
    if (/isn.t in a game|not in a game|no active game/i.test(m)) return "This player isn't in a game right now.";
    if (/not ?found|404|no account|unknown riot id/i.test(m)) return 'No player with that Riot ID in this region. Check the name, the #TAG and the region.';
    if (/503|not configured/i.test(m)) return "Looking up new players isn't switched on for this site.";
    return 'Something went wrong talking to Riot. Please try again.';
  }
  async function runJob(url, status) {
    const r = await fetch(url, {method: 'POST'});
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.detail || body.error || r.statusText);
    let job = body.job;
    while (job.status === 'queued' || job.status === 'running') {
      if (status) status.textContent = job.progress && job.progress.length ? job.progress[job.progress.length - 1] : 'Waiting for Riot...';
      await new Promise((res) => setTimeout(res, 700));
      job = await (await fetch('/api/jobs/' + job.id)).json();
    }
    if (job.status === 'failed') throw new Error(job.error || 'failed');
    return job;
  }
  for (const btn of $$('[data-sync]')) {
    btn.addEventListener('click', async () => {
      const status = $(btn.dataset.status || '#sync-status');
      btn.disabled = true;
      const label = btn.innerHTML;
      btn.innerHTML = '<span class="spinner"></span> Updating';
      try {
        const job = await runJob(btn.dataset.sync, status);
        if (job.result && job.result.redirect) { location.href = job.result.redirect; return; }
        status.textContent = 'Done'; location.reload();
      }
      catch (e) { btn.disabled = false; btn.innerHTML = label; status.textContent = friendly(e.message); status.title = e.message; }
    });
  }
  const auto = $('[data-autosync]');
  if (auto) {
    const status = $('#auto-status');
    runJob(auto.dataset.autosync, status)
      .then((job) => {
        if (job.result && job.result.redirect) { status.textContent = 'Found them in another region...'; location.href = job.result.redirect; return; }
        status.textContent = 'Done! Loading the profile...'; location.reload();
      })
      .catch((e) => {
        $('.spinner', auto)?.remove();
        $('#auto-title').textContent = /not ?found|404|no account/i.test(e.message) ? 'Player not found' : 'Could not load this player';
        status.textContent = friendly(e.message); status.title = e.message;
        $('#auto-retry').hidden = false;
      });
  }
  for (const btn of $$('[data-scout]')) {
    btn.addEventListener('click', async () => {
      const status = $('#scout-status');
      btn.disabled = true;
      try { const job = await runJob(btn.dataset.scout, status); location.search = '?tab=live&job=' + job.id; }
      catch (e) { btn.disabled = false; status.textContent = friendly(e.message); status.title = e.message; }
    });
  }

  // -- match rows: expand, inner tabs, more ------------------------------------------------
  document.addEventListener('click', async (e) => {
    const exp = e.target.closest('.m-expand');
    if (exp) {
      const match = exp.closest('.match'), detail = $('.m-detail', match);
      match.classList.toggle('open');
      if (match.classList.contains('open') && !detail.dataset.loaded) {
        detail.innerHTML = '<div class="empty"><span class="spinner"></span></div>';
        try {
          const r = await fetch(exp.dataset.detail);
          detail.innerHTML = r.ok ? await r.text() : '<div class="empty">Could not load this game.</div>';
          detail.dataset.loaded = '1';
        } catch { detail.innerHTML = '<div class="empty">Could not load this game.</div>'; }
      }
      return;
    }
    const tab = e.target.closest('.d-tabs button');
    if (tab) {
      const root = tab.closest('.m-detail, .detail-root');
      $$('.d-tabs button', root).forEach((b) => b.classList.toggle('on', b === tab));
      $$('.d-pane', root).forEach((p) => p.classList.toggle('on', p.dataset.pane === tab.dataset.pane));
      const pane = $$('.d-pane', root).find((p) => p.dataset.pane === tab.dataset.pane);
      if (pane && pane.dataset.mapUrl && !pane.dataset.loaded) loadMap(pane);
      return;
    }
    const more = e.target.closest('[data-more]');
    if (more) {
      more.disabled = true;
      const offset = Number(more.dataset.offset);
      try {
        const days = $$('#match-list .day-group');
        const lastDay = days.length ? days[days.length - 1].dataset.day : '';
        const r = await fetch(`${more.dataset.more}&offset=${offset}&after_day=${encodeURIComponent(lastDay)}`);
        const html = await r.text();
        $('#match-list').insertAdjacentHTML('beforeend', html);
        const added = (html.match(/class="match /g) || []).length;
        more.dataset.offset = offset + added;
        if (added < 20) more.remove(); else more.disabled = false;
      } catch { more.disabled = false; }
    }
  });

  // -- tooltips ----------------------------------------------------------------------------
  const tip = document.createElement('div'); tip.id = 'tip'; document.body.append(tip);
  document.addEventListener('mouseover', (e) => {
    const t = e.target.closest('[data-tip]');
    if (!t) { tip.style.display = 'none'; return; }
    const [title, ...rest] = t.dataset.tip.split('\n');
    tip.innerHTML = `<b>${esc(title)}</b>${rest.length ? `<span>${esc(rest.join(' '))}</span>` : ''}`;
    tip.style.display = 'block';
  });
  document.addEventListener('mousemove', (e) => {
    if (tip.style.display !== 'block') return;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let x = e.clientX + 14, y = e.clientY + 16;
    if (x + w > innerWidth - 8) x = e.clientX - w - 14;
    if (y + h > innerHeight - 8) y = e.clientY - h - 14;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  });

  // -- sortable tables and quick filter ----------------------------------------------------
  for (const th of $$('table.data th[data-sort]')) {
    th.addEventListener('click', () => {
      const table = th.closest('table'), body = $('tbody', table);
      const i = [...th.parentElement.children].indexOf(th);
      const asc = th.classList.contains('sorted') && !th.classList.contains('asc');
      $$('th', table).forEach((h) => h.classList.remove('sorted', 'asc'));
      th.classList.add('sorted'); if (asc) th.classList.add('asc');
      const val = (tr) => {
        const c = tr.children[i]; const v = c.dataset.v ?? c.textContent.trim();
        const n = parseFloat(v); return isNaN(n) ? v.toLowerCase() : n;
      };
      const rows = [...body.rows].sort((a, b) => {
        const x = val(a), y = val(b);
        return (x < y ? -1 : x > y ? 1 : 0) * (asc ? 1 : -1);
      });
      body.append(...rows);
      if (table.dataset.renumber !== undefined) rows.forEach((r, k) => { if (r.cells[0]) r.cells[0].textContent = k + 1; });
    });
  }
  for (const input of $$('input[data-filter]')) {
    input.addEventListener('input', () => {
      const q = input.value.trim().toLowerCase();
      for (const tr of $$(input.dataset.filter)) tr.hidden = q && !tr.dataset.name.includes(q);
      for (const row of $$('.tier-row')) row.hidden = !$$('.tile', row).some((t) => !t.hidden);
    });
  }
  for (const sel of $$('select[data-nav]')) {
    sel.addEventListener('change', () => {
      const url = new URL(location.href); url.searchParams.set(sel.name, sel.value);
      if (!sel.value) url.searchParams.delete(sel.name);
      location.href = url.toString();
    });
  }

  // -- high-elo map: the game minute by minute on the minimap -------------------------------
  async function loadMap(pane) {
    pane.dataset.loaded = '1';
    pane.innerHTML = '<div class="empty"><span class="spinner"></span><p>Replaying the game against high-elo play...</p></div>';
    let data;
    try {
      const r = await fetch(pane.dataset.mapUrl);
      if (!r.ok) throw new Error();
      data = await r.json();
    } catch { pane.innerHTML = '<div class="empty">Could not load the map for this game.</div>'; delete pane.dataset.loaded; return; }
    const last = data.minutes.length - 1;
    const start = data.first ?? Math.min(3, last);
    const ticks = data.minutes.map((m) => m.advice && ['moment', 'good'].includes(m.advice.kind)
      ? `<i class="${m.advice.kind}" style="left:${(100 * m.minute) / Math.max(1, last)}%" title="Minute ${m.minute}"></i>` : '').join('');
    pane.innerHTML = `
      <div class="hmap">
        <div class="hmap-stage">
          <img class="bg" src="${esc(data.map)}" alt="Summoner's Rift">
          <svg class="arrows" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
            <defs>
              <marker id="ah-you" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 10 5 0 10z" class="you"/></marker>
              <marker id="ah-pro" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 10 5 0 10z" class="pro"/></marker>
            </defs>
            <line class="you" marker-end="url(#ah-you)"/><line class="pro" marker-end="url(#ah-pro)"/>
          </svg>
          <div class="events"></div>
          <div class="dots">${data.players.map((p) => `<div class="dot t${p.team}${p.me ? ' me' : ''}" data-pid="${p.pid}" data-tip="${esc(p.name)}&#10;${esc(p.champion)}${p.role ? ' · ' + esc(p.role) : ''}"><img src="${esc(p.icon)}" alt=""></div>`).join('')}</div>
        </div>
        <div class="hmap-side">
          <div class="clock"><b>Minute <span class="min"></span></b><span class="kind"></span></div>
          <div class="said"></div>
          <div class="opts"></div>
          <div class="hmap-sum">${data.compared
            ? `<span><b>${data.agreement}%</b> of minutes ${data.first}-${data.last} matched the most common high-elo choice</span><span><b>${data.moments}</b> key moment${data.moments === 1 ? '' : 's'} where you went against a strong consensus</span><span class="muted">Compared with ${data.games ? Number(data.games).toLocaleString() + ' ' : ''}Grandmaster and Challenger games in the ${esc(data.role.toLowerCase())} role. Shares are how often those players chose each option in similar spots, not a guarantee it would have worked.</span>`
            : '<span class="muted">No high-elo model covers this role yet, so this is a replay of positions, kills and objectives only.</span>'}</div>
          <div class="legend"><span><i class="l-you"></i>your move to the next minute</span><span><i class="l-pro"></i>most common high-elo choice</span></div>
        </div>
        <div class="hmap-controls">
          <button class="btn play" type="button" aria-label="Play">&#9654;</button>
          <button class="btn step" data-d="-1" type="button" aria-label="Previous minute">&lsaquo;</button>
          <div class="track"><div class="ticks">${ticks}</div><input type="range" min="0" max="${last}" value="${start}" aria-label="Minute"></div>
          <button class="btn step" data-d="1" type="button" aria-label="Next minute">&rsaquo;</button>
          <span class="keys"><button class="btn kprev" type="button" aria-label="Previous key moment">&lsaquo; Key moment</button><span class="kcount"></span><button class="btn knext" type="button" aria-label="Next key moment">Key moment &rsaquo;</button></span>
        </div>
      </div>`;
    const range = $('input[type=range]', pane);
    const you = $('line.you', pane), pro = $('line.pro', pane);
    const kindText = { moment: 'Key moment', good: 'Matched high elo', agree: 'Most common choice', differ: 'Less common choice', neutral: '' };
    // Minutes where the player went against a strong high-elo consensus, in order. Previous
    // and Next stop at the ends rather than wrapping, so two moments don't flip back and forth.
    const moments = data.minutes.map((m, i) => (m.advice && m.advice.kind === 'moment' ? i : -1)).filter((i) => i >= 0);
    const kprev = $('.kprev', pane), knext = $('.knext', pane), kcount = $('.kcount', pane);
    function showKeys(i) {
      const at = moments.indexOf(i);
      kcount.textContent = at >= 0 ? `${at + 1} of ${moments.length}` : `${moments.length} key moment${moments.length === 1 ? '' : 's'}`;
      kprev.disabled = !moments.some((k) => k < i);
      knext.disabled = !moments.some((k) => k > i);
    }
    function show(i) {
      const m = data.minutes[i];
      range.value = i;
      showKeys(i);
      $('.min', pane).textContent = m.minute;
      for (const dot of $$('.dot', pane)) {
        const p = m.pos[dot.dataset.pid];
        dot.hidden = !p;
        if (p) { dot.style.left = p.x + '%'; dot.style.top = p.y + '%'; }
      }
      const me = m.pos[String(data.me)];
      const line = (el, to) => {
        if (!me || !to) { el.style.display = 'none'; return; }
        el.style.display = '';
        el.setAttribute('x1', me.x); el.setAttribute('y1', me.y); el.setAttribute('x2', to.x); el.setAttribute('y2', to.y);
      };
      line(you, m.next);
      line(pro, m.advice && m.advice.target);
      $('.events', pane).innerHTML = m.events.map((e) => `<i class="ev ${e.kind} t${e.team}${e.mine ? ' mine' : ''}" style="left:${e.x}%;top:${e.y}%" data-tip="${e.kind === 'kill' ? (e.mine ? 'Your fight' : 'Kill') : esc(e.name)}&#10;minute ${m.minute}"></i>`).join('');
      const a = m.advice;
      const kind = $('.kind', pane);
      kind.className = 'kind ' + (a ? a.kind : '');
      kind.textContent = a ? kindText[a.kind] : '';
      if (!a) {
        $('.said', pane).innerHTML = data.compared && (m.minute < data.first || m.minute > data.last)
          ? `<p class="muted">High-elo comparisons cover minutes ${data.first} to ${data.last}, the laning phase. Later minutes show positions, kills and objectives.</p>` : '';
        $('.opts', pane).innerHTML = '';
        return;
      }
      $('.said', pane).innerHTML = `<p>You <b>${esc(a.did)}</b>${a.did_share != null ? ` &middot; <span class="share">${a.did_share}%</span> of high-elo players did this here` : ''}.</p>`
        + (a.kind === 'moment' || a.kind === 'differ' ? `<p>Most chose <b class="pro-text">${esc(a.best.toLowerCase())}</b> (${a.best_share}%).</p>` : '');
      $('.opts', pane).innerHTML = a.options.map((o) => `<div class="opt${o.mine ? ' mine' : ''}"><span>${esc(o.label)}${o.mine ? ' <em>you</em>' : ''}</span><div class="bar"><i style="width:${o.share}%"></i></div><b>${o.share}%</b></div>`).join('');
    }
    let timer = null;
    const stop = () => { clearInterval(timer); timer = null; $('.play', pane).innerHTML = '&#9654;'; };
    range.addEventListener('input', () => { stop(); show(Number(range.value)); });
    $('.play', pane).addEventListener('click', () => {
      if (timer) return stop();
      $('.play', pane).innerHTML = '&#10074;&#10074;';
      if (Number(range.value) >= last) show(0);
      timer = setInterval(() => { const i = Number(range.value) + 1; if (i > last) stop(); else show(i); }, 1100);
    });
    for (const b of $$('.step', pane)) b.addEventListener('click', () => { stop(); show(Math.min(last, Math.max(0, Number(range.value) + Number(b.dataset.d)))); });
    knext.addEventListener('click', () => {
      stop();
      const now = Number(range.value);
      const next = moments.find((k) => k > now);
      if (next !== undefined) show(next);
    });
    kprev.addEventListener('click', () => {
      stop();
      const now = Number(range.value);
      const prev = moments.filter((k) => k < now).pop();
      if (prev !== undefined) show(prev);
    });
    if (!moments.length) $('.keys', pane).hidden = true;
    show(start);
  }

  // -- why a matchup is good or bad ------------------------------------------------------
  let whyBox = null;
  function whyModal() {
    if (whyBox) return whyBox;
    whyBox = document.createElement('div');
    whyBox.className = 'why-modal';
    whyBox.setAttribute('role', 'dialog');
    whyBox.setAttribute('aria-modal', 'true');
    whyBox.innerHTML = '<div class="box"><button class="x" type="button" aria-label="Close">&times;</button><div class="body"></div></div>';
    document.body.append(whyBox);
    const close = () => whyBox.classList.remove('open');
    whyBox.addEventListener('click', (e) => { if (e.target === whyBox || e.target.closest('.x')) close(); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') close(); });
    return whyBox;
  }
  document.addEventListener('click', async (e) => {
    const btn = e.target.closest('[data-why]');
    if (!btn) return;
    const box = whyModal(), body = $('.body', box);
    const [i1, i2] = (btn.dataset.icons || '').split(' ');
    const head = `<div class="head"><img src="${esc(i1)}" alt=""><span class="vs-word">vs</span><img src="${esc(i2)}" alt=""><h3>${esc(btn.dataset.title)}</h3></div>`;
    body.innerHTML = head + '<div class="empty"><span class="spinner"></span><p>Reading both kits and the lane numbers... a new matchup takes about 20 seconds the first time.</p></div>';
    box.classList.add('open');
    try {
      const r = await fetch(btn.dataset.why, { method: 'POST' });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || 'failed');
      const ev = data.evidence || [];
      const evText = Object.fromEntries(ev.map((x) => [x.id, x.text]));
      const out = data.output;
      let html = head;
      if (out) {
        if (out.verdict) html += `<p class="verdict">${esc(out.verdict)}</p>`;
        html += (out.points || []).map((p) => `<div class="reason"><b>${esc(p.title)}</b><p>${esc(p.explanation)}</p>
          <div class="cites">${(p.evidence_ids || []).map((id) => `<span data-tip="${esc(id)}&#10;${esc(evText[id] || '')}">${esc(id)}</span>`).join('')}</div></div>`).join('');
        if ((out.tips || []).length) html += `<h4>How to play it</h4><ul class="tips">${out.tips.map((t) => `<li>${esc(t)}</li>`).join('')}</ul>`;
      } else {
        html += `<p class="muted">${esc(data.note || 'No explanation available.')}</p>`;
      }
      html += `<details${out ? '' : ' open'}><summary>The evidence (${ev.length} items: lane numbers from ranked games, then both kits)</summary><ol class="ev">${ev.map((x) => `<li><b>${esc(x.id)}</b>${esc(x.text)}</li>`).join('')}</ol></details>`;
      if (out) html += `<div class="foot">Written by AI from the evidence above; every number it uses is checked against it. ${data.cached ? 'Saved explanation.' : 'Just written.'}</div>`;
      body.innerHTML = html;
    } catch (err) {
      body.innerHTML = head + `<div class="empty">${esc(friendly(String(err.message || err)))}</div>`;
    }
  });

  // -- copy link ---------------------------------------------------------------------------
  for (const b of $$('[data-copy]')) {
    b.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(location.href); const t = b.innerHTML; b.textContent = 'Copied!'; setTimeout(() => (b.innerHTML = t), 1400); } catch {}
    });
  }
})();

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
      try { await runJob(btn.dataset.sync, status); status.textContent = 'Done'; location.reload(); }
      catch (e) { btn.disabled = false; btn.innerHTML = label; status.textContent = friendly(e.message); status.title = e.message; }
    });
  }
  const auto = $('[data-autosync]');
  if (auto) {
    const status = $('#auto-status');
    runJob(auto.dataset.autosync, status)
      .then(() => { status.textContent = 'Done! Loading the profile...'; location.reload(); })
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
      return;
    }
    const more = e.target.closest('[data-more]');
    if (more) {
      more.disabled = true;
      const offset = Number(more.dataset.offset);
      try {
        const r = await fetch(`${more.dataset.more}&offset=${offset}`);
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
    });
  }
  for (const sel of $$('select[data-nav]')) {
    sel.addEventListener('change', () => {
      const url = new URL(location.href); url.searchParams.set(sel.name, sel.value);
      if (!sel.value) url.searchParams.delete(sel.name);
      location.href = url.toString();
    });
  }

  // -- copy link ---------------------------------------------------------------------------
  for (const b of $$('[data-copy]')) {
    b.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(location.href); const t = b.innerHTML; b.textContent = 'Copied!'; setTimeout(() => (b.innerHTML = t), 1400); } catch {}
    });
  }
})();

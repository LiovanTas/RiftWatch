# RiftWatch roadmap

One commit a day. Each entry below is sized to be a single day's work that leaves `main`
green (tests pass, CLI runs). Days are a suggested order, not a contract — if a piece runs
long it spills into the next day rather than landing half-done.

## What we borrow from existing stat sites

Looked at op.gg and lolalytics (Sept 2026, patch 16.19) to decide what "rank-matched" and
"per-minute" should mean here.

| Site | What it does | What RiftWatch takes from it |
|---|---|---|
| **op.gg** summoner page | Search by Riot ID + region, an explicit *Update* button, per-champion CS/min / KDA / win rate, role share, and **"recent 20 games vs season"** deltas | Riot ID + region as the only user input; on-demand sync instead of polling; recent-vs-season comparison for the coach |
| **lolalytics** champion page | Every number is partitioned by **tier bucket** ("Emerald+"), **patch**, queue and role, and printed **next to its sample size** ("164,698 games"); win rate shown as a **delta vs the average** | Baselines keyed on (tier bucket, role, champion?, patch window); every baseline carries `n`; feedback phrased as deltas/percentiles, not raw numbers |
| **Mobalytics** (from memory, not re-checked) | Grades players across skill areas (farming, vision, aggression, survivability...) | Group findings into a few coaching areas rather than dumping 40 metrics |

What none of them do, and RiftWatch does: judge a player **minute by minute** against people
at their own rank in the same role, and hand an LLM *only* those computed comparisons so its
advice can't invent numbers.

## Architecture

```
Riot API ──► rate limiter ──► client ──► Postgres cache ──► feature extraction ──► scoring vs baselines ──► evidence ──► Claude ──► grounding check ──► report
                (token bucket,           (immutable match +      (per-minute frames,     (tier/role/patch
                 429 Retry-After)         timeline, cached         lane-opponent diffs,    percentiles, n)
                                          forever)                 events)
                                                                                  ▲
                              baseline crawler (league-exp-v4 → sampled players ──┘
                              → their ranked matches, same cache)

Win32 watchdog (separate process): hang detection on the game window + global hotkey kill switch
```

## Milestones

### M0 — Groundwork ✅ Day 1
- [x] Project layout, `pyproject.toml`, CLI entry point (`riftwatch`)
- [x] Settings from env / `.env`
- [x] Platform ↔ regional routing table, region aliases (`na`, `euw`, `kr`...), Riot ID parsing
- [x] Postgres via docker-compose, SQL migration runner, initial schema
      (accounts, matches, match_timelines, match_participants, rank_snapshots)
- [x] `riftwatch doctor`, `riftwatch db migrate|status`
- [x] CI (pytest + a real Postgres service), commit-identity hook

### M1 — Riot API layer
- [ ] **Day 2 — token-bucket rate limiter.** Parse `X-App-Rate-Limit` / `X-Method-Rate-Limit`
      (`"20:1,100:120"` = several windows at once), one limiter per routing value (each region
      has its own quota), per-method buckets. Update limits from response headers instead of
      hardcoding dev-key numbers. Fake-clock unit tests.
- [ ] **Day 3 — HTTP client.** httpx; on 429 read `Retry-After` and `X-Rate-Limit-Type`
      (application / method / service — a *service* 429 has no Retry-After, so back off
      exponentially); retry 5xx with jitter; 404 → `None`; key never logged. Tests on
      `httpx.MockTransport`.
- [ ] **Day 4 — endpoints.** account-v1 by Riot ID, match-v5 ids / match / timeline,
      league-v4 entries by PUUID. Thin typed wrappers that keep the raw JSON.
- [ ] **Day 5 — Data Dragon.** Current version, champion id → name, item names; cached on disk
      per patch.

### M2 — Cache and history
- [ ] **Day 6 — repository layer.** `get_match` / `get_timeline` read Postgres first and only hit
      the API on a miss (finished matches never change, so a hit is final). Populate
      `match_participants`. Count API calls vs cache hits.
- [ ] **Day 7 — `riftwatch sync <Name#TAG>`.** Resolve account, store a rank snapshot, page
      match ids newest-first and stop at the first id already cached.
- [ ] **Day 8 — resumable backfill.** Job table with a cursor; `riftwatch backfill` walks the
      whole season (`startTime`, `count=100` paging), survives Ctrl-C/crash, reports calls made,
      cache hits and 429s. Goal for the resume claim: full history, **zero** 429s.

### M3 — Per-minute features
- [ ] **Day 9 — timeline parser.** Per participant per minute: gold, XP, level, CS
      (lane + jungle), damage to champions, position.
- [ ] **Day 10 — lane-opponent diffs.** Opponent = enemy with the same `teamPosition`; gold/XP/CS
      diff at every minute, CS@10/@15, gold@15.
- [ ] **Day 11 — event features.** Deaths (minute, map zone, gold state when it happened),
      wards placed/cleared per minute, objective participation (dragons, grubs, herald,
      baron, towers/plates), kill participation, solo kills.
- [ ] **Day 12 — persist features.** Migration for `participant_minute_features` +
      `participant_game_summary`; golden test against a recorded fixture match.

### M4 — Rank-matched baselines
- [ ] **Day 13 — baseline crawler.** league-exp-v4 entries per tier/division → sample N players
      per bucket → their recent ranked solo matches, through the same cache and limiter.
- [ ] **Day 14 — aggregation.** `baselines(tier_bucket, role, champion_id NULL, patch_window,
      metric, minute, n, mean, sd, p10…p90)` via `percentile_cont`. Minimum-`n` threshold with
      fallback champion → role → tier (lolalytics' "low sample" idea).
- [ ] **Day 15 — scoring.** Player value → percentile/z against the baseline at each minute;
      per-game and recent-20-vs-season views (op.gg's idea).

### M5 — Grounded coach
- [ ] **Day 16 — evidence builder.** Turn scores into a short list of findings, each with an id,
      metric, minute range, player value, baseline percentile and `n`.
- [ ] **Day 17 — Claude call.** The model sees *only* the evidence list; structured output where
      every point cites evidence ids. (Load the `claude-api` reference before writing this.)
- [ ] **Day 18 — grounding validator.** Reject output that cites unknown ids or states a number
      not present in the cited evidence; retry once, else drop the point. Unit tests with
      hand-written bad outputs.
- [ ] **Day 19 — `riftwatch coach <Name#TAG> [--match ID]`.** Terminal report for one game and
      for the recent-games trend.

### M6 — Win32 watchdog
Constraint: it only looks at **window state and process handles** — no reading or writing game
memory, no input injection into the game. That keeps it clear of Vanguard.
- [ ] **Day 20 — detection.** Find the game window/process (`League of Legends.exe`); detect
      "hung fullscreen" with `IsHungAppWindow` / `SendMessageTimeout(WM_NULL)` plus
      foreground + fullscreen-rect checks. ctypes only.
- [ ] **Day 21 — kill switch.** `RegisterHotKey` with a user-chosen combo (config), on press:
      `TerminateProcess` the game, then restore the desktop (release foreground, reset display
      mode if it changed).
- [ ] **Day 22 — run mode.** `riftwatch watchdog` background loop that arms when the game starts
      and disarms when it exits; logs every trigger.

### M7 — Polish
- [ ] **Day 23 — HTML match report.** Per-minute charts of the player vs the baseline band.
- [ ] **Day 24 — metrics for the README.** Real numbers: API calls saved by the cache, backfill
      stats, 429 count.
- [ ] **Day 25 — packaging + demo.** Install instructions, sample report, screenshots.

### Later / maybe
- Live-game scouting via spectator-v5 using cached history
- Matchup-specific baselines (champion vs champion) once the crawl is big enough
- Arena / ARAM support (different timelines, different metrics)

## Riot API facts we're designing around
- **Routing:** account-v1 and match-v5 use *regional* hosts (americas / europe / asia / sea);
  league-v4 and summoner-v4 use *platform* hosts (na1, euw1, kr...).
- **Dev keys expire every 24 hours** — regenerate at developer.riotgames.com before a session.
- **Dev-key limits:** 20 requests / 1 s and 100 / 2 min per region, plus per-method limits.
  We read the real values from response headers.
- A match's `/timeline` has one frame per minute (`frameInterval` 60000 ms) with
  `participantFrames` and an `events` list.

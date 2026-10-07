# RiftWatch roadmap

## What we borrow from existing stat sites

Looked at op.gg and lolalytics (Sept 2026, patch 16.19) to decide what "rank-matched" and
"per-minute" should mean here.

| Site | What it does | What RiftWatch takes from it |
|---|---|---|
| **op.gg** summoner page | Search by Riot ID + region, an explicit *Update* button, per-champion CS/min / KDA / win rate, role share, and **"recent 20 games vs season"** deltas | Riot ID + region as the only user input; on-demand sync instead of polling; recent-vs-older comparison for the coach |
| **lolalytics** champion page | Every number is partitioned by **tier bucket**, **patch**, queue and role, and printed **next to its sample size**; win rate shown as a **delta vs the average** | Baselines keyed on tier bucket, role, champion and patch window; every baseline carries `n`; feedback phrased as percentiles and gaps to the median |
| **Mobalytics** (from memory) | Grades players across skill areas | Findings grouped into a few coaching areas instead of 40 raw metrics |

What none of them do, and RiftWatch does: judge a player **minute by minute** against people at
their own rank in the same role, and hand an LLM *only* those computed comparisons so its advice
can't invent numbers.

## Built

- [x] Riot API layer: per-region and per-endpoint token-bucket limiter that learns limits from
      headers, HTTP client that waits out 429s and retries 5xx, endpoint wrappers, Data Dragon
- [x] Postgres cache: matches and timelines downloaded once, rank snapshots, parallel downloads
- [x] `sync`, resumable `backfill`, cache hit-rate reporting
- [x] Per-minute features and ~40 per-game metrics, validated against real timelines
- [x] Rank-matched baselines from a ladder crawler (recent games only), with fallbacks
- [x] Percentile scoring and recent-vs-older trends
- [x] Grounded coach on Claude Sonnet 5.5 with prompt caching, a grounding check, and answers
      cached in Postgres; offline template coach
- [x] Terminal and self-contained HTML reports (per-minute charts, scorecard, deaths timeline)
- [x] Windows black-screen watchdog with a global kill-switch hotkey
- [x] Speed pass on both read and write paths (numbers in the README)
- [x] Website: JSON API, search, player and match pages, Update as a background job with
      progress, coaching on request (numbers in the README)

## Next: the website

The CLI is the engine; the website is a thin layer over it. The read path is already built
for this: a 20-game report is 47 ms from Postgres, and repeat coaching is a cache read.

- [x] **JSON API** over the existing pipeline, pooled connections, no raw Riot JSON on a request
- [x] **Background sync jobs** with progress, one shared rate limiter, and an update cooldown
- [x] **Coaching on request**, cached after the first time
- [x] **Pages**: search, player page, match page
- [x] **Fair queue across users**: workers take the next job from whoever has the least running
- [x] **Coaching streams to the page**: each point appears once it is written and passes the
      grounding check; the validated final answer replaces the preview
- [x] **Jobs in Postgres** (claimed with `FOR UPDATE SKIP LOCKED`, retried if a worker dies), so
      several server processes can share the work
- [x] **Baseline refresh** (`riftwatch refresh`): tops up every rank bucket on the live patch and
      rebuilds baselines; schedule it daily
- [x] **Sessions**: late-session and after-loss patterns, compared within sessions so day-to-day
      form can't fake a fatigue effect; CLI, page, API, coach evidence when clear
- [x] **Progress over time**: weekly standing and areas against a fixed yardstick, a trend
      called only past two standard errors, rank snapshots; CLI, page chart, API
- [x] **Champion pool**: per champion and role, average standing against the player's rank,
      areas, and a stronger/weaker call only when the gap beats the noise; CLI, page, coach
- [x] **Coach evaluation** (`eval-coach`): grounding, faithfulness, coverage, cost and latency
      on a seeded sample of games across ranks and roles; led to the death-totals evidence item
- [x] **Deploy prep**: Docker image (migrations on start, several workers), per-visitor
      limits in Postgres with `429`/`Retry-After`, a daily LLM coaching budget, Riot's legal
      notice, gzip and security headers, a deploy guide
- [ ] **Production key.** A public site needs a registered Riot product and a production API
      key; development keys are for personal use and expire daily.

## High-elo models

- [x] High-elo crawl: Challenger, Grandmaster and Master games from NA, EUW and KR at once
- [x] Situation / decision / outcome examples for all five roles, with player-known features only
- [x] Decision and outcome models per role, evaluated by game against baselines
- [x] Advisor: key moments in the player's games, fed to the coach and shown in the review
- [x] Objective timers as features (dragon, grubs, herald; spawn rules measured from 16.19 games).
      Measured gain was tiny (jungle 53.0% -> 53.4%): with one snapshot a minute the decision
      models are at their ceiling, so further gains need finer-grained data, not more features
- [x] Champion-specific comparisons: role baselines shifted by a champion effect pooled
      across tiers, tested against held-out tiers

## In-game data

- [x] **Live recorder** on Riot's Live Client Data API: own health, gold, items and events every
      second, linked to the match afterwards; trades, recalls and deaths in the review and coach
- [ ] **Enemy health from the screen** (computer vision on health bars) to see both sides of a
      trade. Built: bar detector, video scanner, alignment to game time by matching your own
      health curve, accuracy check against the recorder (`riftwatch vision`). Calibrated on a
      real 720p replay: 71% coverage, median error 0, 90% within 4.6 health points against the
      HUD panel. Next: cut false bars (structures, effects), tie bars to champions, align
      replays by reading the game clock, then feed enemy health into trades and the coach

## Later

- [x] Build timing: first and second legendary items and boots, against players at your rank
- [x] Live-game scouting via spectator-v5: rank, recent form and champion experience for
      all ten players, from the shared match cache (CLI and a website page)
- [x] Matchup-specific lane comparisons: lane strength per champion and role, leads judged by
      strength(you) - strength(opponent); exact-pair tables tested and found to add nothing yet
- [x] Normal draft and ranked flex games: synced and coached alongside solo/duo, scored against
      the solo/duo baselines for the player's rank, flex rank as fallback (draft and ranked only)

## Riot API facts we design around

- **Routing:** account-v1 and match-v5 use *regional* hosts (americas / europe / asia / sea);
  league-v4 and summoner-v4 use *platform* hosts (na1, euw1, kr...). SEA accounts resolve
  through `asia`.
- **Development keys expire every 24 hours** and allow 20 requests / 1 s and 100 / 2 min per
  region, plus per-endpoint limits. RiftWatch reads the real values from response headers.
- A timeline has one frame per minute plus a final frame at game end, which can land just
  before a minute mark. Events with no assists omit `assistingParticipantIds` entirely.
- `championName` in match data is an internal id, not what players see (`MonkeyKing` is
  Wukong, `Kaisa` is Kai'Sa); display names come from Data Dragon's `champion.json`, while
  the Live Client Data API already uses display names.
- `WARD_PLACED` also fires for Teemo mushrooms and an `UNDEFINED` type some champions emit by
  the hundred; only trinkets, sight and control wards are wards.

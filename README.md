# RiftWatch

League of Legends coaching that doesn't make things up.

RiftWatch pulls your match timelines from the Riot API, measures how you played minute by
minute (CS, gold, XP and the gaps to your lane opponent, deaths, vision, objectives), and
compares every measurement with players **at your rank, in your role, on recent patches**.
Only those comparisons go to Claude, and its answer is checked number by number against
them before you see it.

It also ships a Windows watchdog for the game's black-screen hang: one hotkey kills the
frozen game and gives you your desktop back.

```
$ riftwatch coach "Liovan#G2EU" --last

Warwick jungle -- LOSS -- NA1_5649639519 (patch 16.19)

  vs Platinum jungle, same role, patches 16.17-16.19
  farming
   ! CS at 15 min                         72  [....................]   1st  (median 100, n=110)
   ! CS per minute                       4.5  [....................]   1st  (median 6.4, n=110)
  ...
  fighting
   + kill participation                  67%  [##################..]  89th  (median 50%, n=110)

COACH  [claude-sonnet-5-5/adaptive/low]
  1. Farm stayed far below your rank  (work on, farming)
     You had 72 CS at 15 minutes against a median of 100, and your CS per minute was 4.5
     against a median of 6.4. From minute 9 to minute 19 you stayed below the 25th
     percentile, ending at 87 CS.
     -> Between ganks, always go back to clear your camps rather than waiting around.
     evidence E2, E3, E10: ...
```

`--html report.html` writes the same review as a self-contained page with per-minute charts
of you against the comparison group, and `riftwatch serve` runs it as a website.

## Quick start

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"            # ".[web]" for just the website extras
copy .env.example .env          # add RIOT_API_KEY (and ANTHROPIC_API_KEY for the LLM coach)
docker compose up -d db
riftwatch db migrate
riftwatch doctor
```

Riot **development keys expire every 24 hours**; regenerate one at
<https://developer.riotgames.com> before each session.

Then:

```bash
riftwatch sync "Name#TAG"               # your newest 20 ranked games
riftwatch crawl --tiers GOLD,PLATINUM   # games from the ladder to compare against
riftwatch baselines                     # build the rank-matched comparisons
riftwatch coach "Name#TAG" --last --html last-game.html
```

## Commands

| Command | What it does |
|---|---|
| `lookup Name#TAG` | Resolves a Riot ID and shows its current rank |
| `sync Name#TAG` | Downloads the newest games, stopping at the first one already cached |
| `backfill Name#TAG --since 2026-01-01` | Downloads a whole match history; rerun the same command to resume |
| `crawl --tiers ...` | Samples recent ranked games from each tier's ladder for baselines |
| `baselines` | Rebuilds the per-tier, per-role, per-champion comparison tables |
| `features` | Extracts per-minute features from any cached games that still need it |
| `coach Name#TAG` | Coaching on the last 20 games; `--last` or `--match ID` for one game |
| `cache` | Cache size and hit rate |
| `watchdog` | Arms the black-screen kill switch (Windows); `--record` also records each game |
| `record` | Records games second by second from the game client; `--import` links recordings to matches |
| `serve` | Runs the website and JSON API on http://127.0.0.1:8000 |
| `doctor`, `db migrate`, `db status` | Setup checks and schema migrations |

`coach` options: `--html FILE`, `--evidence` (print every fact the coach was given),
`--offline` (template coach, no LLM), `--tier gold` (compare against another tier),
`--refresh` (ignore the cached answer).

## How it works

```
Riot API -> rate limiter -> client -> Postgres cache -> features -> scores vs baselines -> evidence -> Claude -> grounding check -> report
```

**Riot API.** A token-bucket limiter per region and per endpoint learns the real limits from
response headers and adopts Riot's own request count when it's ahead of ours. Each token comes
back one full window after it was spent (a continuously refilling bucket would send ~199
requests in the first 120 s of a 100-per-120 s limit). 429s are waited out using `Retry-After`;
5xx and network errors back off with jitter. Downloads run on 8 threads behind the shared
limiter.

**Cache.** Finished matches never change, so a match or timeline is downloaded once and served
from Postgres forever. Ranks are stored as snapshots.

**Features.** Per player per minute: gold, XP, CS, damage, kills/deaths/assists, real wards
(Riot also logs mushrooms and hundreds of "UNDEFINED" wards for some champions), and the gap
to the lane opponent. Per game: about 40 metrics -- including when you finished your first and
second legendary items and boots -- plus every death with its time, map zone,
killer, helpers and gold state.

**Baselines.** `crawl` samples players from each tier's ladder and keeps only their games from
the last 14 days. Postgres computes n, mean, sd and the 10th-90th percentiles of every metric
per tier x role x champion, and per minute for each curve. Your own games are left out of the
yardstick. Lookups fall back from same champion to same role to the neighbouring tier.
Every stat also shows the Master+ median for the role -- from about 10,000 high-elo players per
role -- as a reference for where high elo sits.

**Coach.** Scores become numbered evidence items. Claude (Sonnet 5.5) answers in a fixed JSON
schema, citing evidence ids. The grounding check rejects any number that isn't in the evidence
a point cites; Claude gets one retry with the exact violations, then failing points are
dropped. Finished answers are cached in Postgres, so a repeat view costs nothing.

## Measured on real data

All numbers from a Windows laptop with Postgres in Docker, a Riot development key, and the
author's Platinum II NA account.

| | |
|---|---|
| Sync 20 new games (43 requests) | 4.9 s, 0 rate-limited |
| Full-season backfill (515 games) | 20 min, 999 requests, 1 rate-limited and retried |
| Coach one game (Sonnet 5.5) | ~8 s, ~$0.012, 0 grounding retries in 13 calls |
| Coach a game again | 1.2 s, $0 (cached answer) |
| Report for 20 games, from cache | 47 ms |
| Report for one game, from cache | 10 ms |
| Save one game's features | 22 ms |

Things measurement changed along the way:

- `localhost` resolved to IPv6, which Docker Desktop on Windows forwards with a ~50 ms stall on
  mid-sized writes. Using `127.0.0.1` cut saving a game's features from 134 ms to 22 ms.
- Importing the Anthropic SDK took ~3 s; it now loads only when the LLM is actually called.
- Adaptive thinking at low effort beat both higher effort and no thinking on cost and speed.
  The 853-token system prompt clears Sonnet 5.5's 512-token caching minimum and is read from
  cache on every call. There is no per-game cache breakpoint: writing each game's evidence to
  cache only pays off above a ~28% retry rate, and retries measured 0 of 13.

## Website

`riftwatch serve` starts the site. Search a Riot ID, press **Update** to pull the newest games
(a background job; the page shows its progress), open any game for the full review, and ask
for AI coaching on a game or on your recent games.

Pages never wait on Riot or the LLM: they read Postgres only. Downloads are background jobs
behind one shared rate limiter, repeat Update presses within a minute reuse the same job, and
coaching is generated only when asked for, then served from cache. Jobs live in Postgres, so
several server processes can share them; workers claim them with `FOR UPDATE SKIP LOCKED`,
take the next one from whoever has the least work running, and retry a job whose worker died.

| Endpoint | Median on real data |
|---|---|
| `GET /api/players/{region}/{Name-TAG}` | 9 ms |
| `GET /api/players/.../matches?limit=100` | 7 ms |
| `GET /api/players/.../matches/{match_id}` (scores, curves, evidence) | 18 ms |
| `GET /api/players/.../recent` (trends over 40 games) | 85 ms |
| `POST /api/players/.../sync` | returns a job; `GET /api/jobs/{id}` for progress |
| `POST /api/players/.../matches/{match_id}/coach` | generates and caches coaching |
| `POST /api/players/.../matches/{match_id}/coach/stream` | the same, as server-sent events: each point once it passes the grounding check, then the final answer |

Interactive API docs are at `/docs`.

## Black-screen watchdog (Windows)

```bash
riftwatch watchdog                      # kill switch on Ctrl+Alt+K
riftwatch watchdog --hotkey ctrl+shift+f12 --auto-kill-after 20
riftwatch watchdog --status             # what it sees right now
```

It watches the game window with Windows' own "is this window responding" checks, warns when
the game has held the screen without responding for 5 seconds, and kills it on the hotkey (or
automatically, if you ask). It never reads or writes game memory or injects input.

## High-elo models

```bash
riftwatch crawl --high-elo --regions na,euw,kr --players 60 --matches 10
riftwatch ml dataset          # per-role training tables from Grandmaster/Challenger games
riftwatch ml train            # trains and evaluates the models for each role
```

Every Grandmaster and Challenger game is turned into minute-by-minute examples for all five
roles: the *situation* a player was in, the *decision* they made over the next minute, and the
*outcome* over the three minutes after that. Each role has its own decisions -- a jungler ganks,
farms, invades or takes an objective; a mid laner stays, roams top or bot, or pushes; a support
stays with the ADC, roams or goes warding.

Situations only use what the player could know: their own state, the scoreboard, teammates'
positions, objective history, and where the enemy jungler was last *seen* in a fight. Hidden
enemy positions are left out, so the models can't learn to see through fog of war.

Two kinds of gradient-boosted models are trained per role:

- **Decision model:** how often high-elo players chose each option in a situation like this.
- **Outcome models:** what tended to follow each option -- the team taking an objective, the
  player dying, the team's gold swing, and for laners their CS gap.

Evaluation splits by game, never by row, and every model is compared with simple baselines.
The honest picture: predicting the *exact* next move beats the best baseline by only a few
points, since one snapshot a minute doesn't pin it down and good players differ. The outcome
models are the useful part, and they're what the coach leans on.

Measured on games the models never saw (3,922 Grandmaster/Challenger games, about 109,000
examples per role):

| Role | Decision accuracy | Best simple baseline | Top-2 | Objective within 3 min (AUC) | Player dies within 3 min (AUC) |
|---|---|---|---|---|---|
| Top | 66.2% | 64.2% | 81.2% | 0.77 | 0.62 |
| Jungle | 53.0% | 48.4% | 68.9% | 0.79 | 0.63 |
| Mid | 57.1% | 54.2% | 73.0% | 0.78 | 0.61 |
| ADC | 65.1% | 61.7% | 80.2% | 0.78 | 0.63 |
| Support | 37.9% | 35.5% | 60.8% | 0.78 | 0.61 |

Doubling the data from 1,830 games moved these by a point or two at most, so the limit now is
the inputs (camp timers and lane states aren't modelled yet), not the amount of data.

The coach uses them through the **advisor**: for each minute of your game it compares your move
with the options high-elo players actually chose in similar spots. A *key moment* is when your
choice was uncommon and a common alternative was followed by clearly better outcomes. The
review shows those moments, and the coach explains them as "in similar Grandmaster/Challenger
situations, most players..." -- never "you should have".

## Live recording (Windows)

```bash
riftwatch watchdog --record     # kill switch and recorder together while you play
riftwatch record                # or just the recorder
```

While a match runs, the game client serves its own state on your PC through Riot's Live Client
Data API. The recorder reads it once a second -- your health, mana, gold and abilities, every
player's score, items and respawn timers, and the event feed -- and writes each game to a local
file. `sync` imports new recordings and links them to their matches; the match review then
gains a health chart and the coach gets facts like "before 14:00 you lost a fifth of your health
in a short window 4 times; 2 were followed by a recall within a minute".

It shows your side of trades only: the client exposes your health, not your opponents', and no
positions. Everything is for after the game; nothing gives advice during a match, which Riot's
policy doesn't allow.

## Configuration

`.env` (see `.env.example`):

| Variable | Default |
|---|---|
| `RIOT_API_KEY` | required for anything that calls Riot |
| `RIFTWATCH_DATABASE_URL` | `postgresql://riftwatch:riftwatch@127.0.0.1:5432/riftwatch` |
| `RIFTWATCH_DEFAULT_REGION` | `na` |
| `ANTHROPIC_API_KEY` | optional; without it `coach` uses the offline template coach |
| `RIFTWATCH_COACH_MODEL` | `claude-sonnet-5-5` |
| `RIFTWATCH_COACH_EFFORT` / `RIFTWATCH_COACH_THINKING` | `low` / `adaptive` |
| `RIFTWATCH_KILL_HOTKEY` | `ctrl+alt+k` |

## Tests

```bash
pytest
```

Database tests need a disposable Postgres (they wipe it) and are skipped unless
`RIFTWATCH_TEST_DATABASE_URL` is set; CI always runs them. Watchdog tests that use real
windows run on Windows only.

Commits are checked by `.githooks/pre-commit`, which refuses a commit whose author or
committer doesn't match the repo's `user.email`. Enable it once per clone with
`git config core.hooksPath .githooks`.

## Legal

RiftWatch isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot
Games or anyone officially involved in producing or managing Riot Games properties. Riot
Games, and all associated properties are trademarks or registered trademarks of Riot Games,
Inc.

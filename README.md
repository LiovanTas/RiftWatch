# RiftWatch

League of Legends coaching that doesn't make things up.

RiftWatch pulls your match timelines from the Riot API, computes per-minute metrics (gold,
CS, XP and their diffs against your lane opponent, deaths, vision, objectives), and scores
each one against players **at your rank, in your role, on the current patch**. Only those
computed comparisons go to an LLM, and its answer is checked against them, so every piece of
feedback traces back to a number that was actually measured.

It also ships a small Windows watchdog for the client's black-screen hang: when the game
keeps fullscreen focus and swallows Alt-Tab, a hotkey you choose kills the game and gives you
your desktop back.

> **Status:** early development, one feature a day. See [ROADMAP.md](ROADMAP.md) for the full
> plan and what's done.

## Stack
Python 3.11+ · PostgreSQL · Riot API (account-v1, match-v5, league-v4) · Anthropic API · Win32 via ctypes

## Setup

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env        # then paste your Riot dev key into .env
docker compose up -d db
riftwatch db migrate
riftwatch doctor
```

Riot **development keys expire every 24 hours** — regenerate one at
<https://developer.riotgames.com> before each session.

## Tests

```bash
pytest
```

Database tests need a disposable Postgres (they wipe it) and are skipped unless
`RIFTWATCH_TEST_DATABASE_URL` is set. CI always runs them.

## Repository hygiene
Commits are checked by `.githooks/pre-commit`, which refuses a commit whose author or
committer doesn't match the repo's configured `user.email`. Enable it once per clone:

```bash
git config core.hooksPath .githooks
```

## Legal
RiftWatch isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot
Games or anyone officially involved in producing or managing Riot Games properties. Riot
Games, and all associated properties are trademarks or registered trademarks of Riot Games,
Inc.

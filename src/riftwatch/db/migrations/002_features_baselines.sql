-- Everything derived from the cached Riot data: per-minute features, per-game summaries,
-- the rank-bucketed baselines built from them, and the bookkeeping for crawls/backfills.

-- Which rank bucket a match's players belong to. Crawled matches take the tier of the
-- ladder page they were found on; a user's own matches take the user's rank at sync time.
-- Ranked matchmaking keeps all ten players close in rank, so every participant counts.
CREATE TABLE match_samples (
    match_id    text PRIMARY KEY REFERENCES matches ON DELETE CASCADE,
    tier_bucket text NOT NULL,         -- IRON ... DIAMOND, MASTER_PLUS
    source      text NOT NULL,         -- 'crawl' | 'player'
    added_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX match_samples_bucket ON match_samples (tier_bucket);

-- Players already sampled by the baseline crawler, so reruns pick new ones.
CREATE TABLE crawl_players (
    puuid         text PRIMARY KEY,
    platform      text NOT NULL,
    tier          text NOT NULL,
    division      text NOT NULL,
    tier_bucket   text NOT NULL,
    matches_found integer NOT NULL DEFAULT 0,
    crawled_at    timestamptz NOT NULL DEFAULT now()
);

-- Resumable match-history backfills. Paging is by offset into the newest-first id list:
-- games played during a backfill push older ids down, which only causes re-reads (deduped
-- by the cache), never skips.
CREATE TABLE backfill_jobs (
    id            bigserial PRIMARY KEY,
    puuid         text NOT NULL,
    platform      text NOT NULL,
    queue_id      integer,             -- NULL = every queue
    start_time    timestamptz,         -- oldest game to include, NULL = as far as Riot goes
    with_timeline boolean NOT NULL DEFAULT true,
    next_offset   integer NOT NULL DEFAULT 0,
    status        text NOT NULL DEFAULT 'running',  -- running | done
    ids_seen      integer NOT NULL DEFAULT 0,
    fetched       integer NOT NULL DEFAULT 0,       -- downloaded from Riot
    cached        integer NOT NULL DEFAULT 0,       -- already in Postgres
    created_at    timestamptz NOT NULL DEFAULT now(),
    updated_at    timestamptz NOT NULL DEFAULT now(),
    finished_at   timestamptz
);
CREATE INDEX backfill_jobs_open ON backfill_jobs (puuid, status);

-- Cache effectiveness: every get-or-fetch bumps one of these.
CREATE TABLE cache_counters (
    kind   text PRIMARY KEY,           -- 'match' | 'timeline'
    hits   bigint NOT NULL DEFAULT 0,
    misses bigint NOT NULL DEFAULT 0
);

-- One row per player per minute of a game (from the timeline's 60 s frames).
-- Counters (kills, wards...) are cumulative up to that minute. *_diff columns are
-- against the lane opponent and NULL when there isn't one.
CREATE TABLE participant_minute_features (
    match_id            text NOT NULL,
    participant_id      smallint NOT NULL,
    minute              smallint NOT NULL,
    gold                integer NOT NULL,   -- total gold earned
    current_gold        integer NOT NULL,   -- unspent
    xp                  integer NOT NULL,
    level               smallint NOT NULL,
    cs                  integer NOT NULL,   -- lane minions + jungle monsters
    lane_cs             integer NOT NULL,
    jungle_cs           integer NOT NULL,
    damage_to_champions integer NOT NULL,
    kills               smallint NOT NULL,
    deaths              smallint NOT NULL,
    assists             smallint NOT NULL,
    wards_placed        smallint NOT NULL,
    wards_killed        smallint NOT NULL,
    gold_diff           integer,
    xp_diff             integer,
    cs_diff             integer,
    x                   integer,
    y                   integer,
    PRIMARY KEY (match_id, participant_id, minute),
    FOREIGN KEY (match_id, participant_id)
        REFERENCES match_participants (match_id, participant_id) ON DELETE CASCADE
);

-- Whole-game metrics per player. `metrics` is a flat {name: number} object so new metrics
-- need no migration; `deaths_detail` lists each death with time, place and gold state.
CREATE TABLE participant_game_summary (
    match_id                text NOT NULL,
    participant_id          smallint NOT NULL,
    role                    text NOT NULL,
    champion_id             integer NOT NULL,
    opponent_participant_id smallint,
    metrics                 jsonb NOT NULL,
    deaths_detail           jsonb NOT NULL,
    extractor_version       integer NOT NULL,
    extracted_at            timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (match_id, participant_id),
    FOREIGN KEY (match_id, participant_id)
        REFERENCES match_participants (match_id, participant_id) ON DELETE CASCADE
);

-- Distribution of each metric for a (tier bucket, role, champion) group. champion_id 0
-- means "every champion in the role". minute NULL = a whole-game metric, otherwise the
-- value of a per-minute curve at that minute. Rebuilt wholesale by `build-baselines`.
CREATE TABLE baselines (
    tier_bucket  text NOT NULL,
    role         text NOT NULL,
    champion_id  integer NOT NULL,
    patch_window text NOT NULL,          -- e.g. '16.17-16.19'
    metric       text NOT NULL,
    minute       smallint,
    n            integer NOT NULL,
    mean         double precision NOT NULL,
    sd           double precision NOT NULL,
    p10          double precision NOT NULL,
    p25          double precision NOT NULL,
    p50          double precision NOT NULL,
    p75          double precision NOT NULL,
    p90          double precision NOT NULL,
    built_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX baselines_lookup ON baselines (tier_bucket, role, champion_id, metric, minute);

-- Read-path speed: everything a page view needs comes from small indexed tables, never
-- from the raw Riot JSON, and expensive results (LLM coaching) are computed once.

-- Champion name next to the features, so reading a game never touches matches.raw.
ALTER TABLE participant_game_summary ADD COLUMN champion_name text NOT NULL DEFAULT '';

-- One row per baseline rebuild. Readers cache baselines in memory keyed on the latest id,
-- so checking for staleness is a primary-key lookup.
CREATE TABLE baseline_builds (
    id           bigserial PRIMARY KEY,
    built_at     timestamptz NOT NULL DEFAULT now(),
    patch_window text NOT NULL,
    games        integer NOT NULL,
    rows         integer NOT NULL
);

-- Finished coaching, keyed on exactly what produced it: the same evidence and model give
-- the same answer, so a repeat view is a single indexed read instead of an LLM call.
CREATE TABLE coach_reports (
    id                   bigserial PRIMARY KEY,
    puuid                text NOT NULL,
    scope                text NOT NULL,          -- 'game' | 'recent'
    match_id             text,                   -- for scope 'game'
    evidence_fingerprint text NOT NULL,
    model                text NOT NULL,
    evidence             jsonb NOT NULL,
    output               jsonb NOT NULL,         -- validated coach answer
    dropped              jsonb NOT NULL,         -- points removed by the grounding check
    usage                jsonb NOT NULL,
    created_at           timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX coach_reports_key
    ON coach_reports (puuid, scope, coalesce(match_id, ''), evidence_fingerprint, model);

-- Match history pages sort a player's games by start time.
CREATE INDEX matches_game_start ON matches (game_start DESC);

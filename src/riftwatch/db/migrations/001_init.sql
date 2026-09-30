-- Initial schema: the Riot API cache.
--
-- Finished matches and their timelines never change, so once stored they are served from
-- here forever and never re-requested. Ranks do change, so they are kept as snapshots.

-- Riot IDs can be renamed; the PUUID is the stable key.
CREATE TABLE accounts (
    puuid      text PRIMARY KEY,
    game_name  text NOT NULL,
    tag_line   text NOT NULL,
    platform   text NOT NULL,          -- na1, euw1, kr, ...
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX accounts_riot_id ON accounts (lower(game_name), lower(tag_line));

CREATE TABLE matches (
    match_id     text PRIMARY KEY,     -- "NA1_5123456789"
    platform     text NOT NULL,
    queue_id     integer NOT NULL,     -- 420 ranked solo, 440 flex, ...
    game_version text NOT NULL,        -- "16.19.712.3456"
    patch        text NOT NULL,        -- "16.19", derived from game_version
    game_start   timestamptz NOT NULL,
    duration_s   integer NOT NULL,
    raw          jsonb NOT NULL,       -- full match-v5 response
    fetched_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX matches_queue_patch ON matches (queue_id, patch);

CREATE TABLE match_timelines (
    match_id   text PRIMARY KEY REFERENCES matches ON DELETE CASCADE,
    raw        jsonb NOT NULL,         -- full match-v5 /timeline response
    fetched_at timestamptz NOT NULL DEFAULT now()
);

-- One row per player per match, pulled out of matches.raw so history queries
-- ("this player's last 20 ranked games on Ezreal") don't have to scan JSON.
CREATE TABLE match_participants (
    match_id       text NOT NULL REFERENCES matches ON DELETE CASCADE,
    participant_id smallint NOT NULL,  -- 1-10, the key timeline frames use
    puuid          text NOT NULL,
    team_id        smallint NOT NULL,  -- 100 blue, 200 red
    team_position  text NOT NULL,      -- TOP JUNGLE MIDDLE BOTTOM UTILITY, '' if unknown
    champion_id    integer NOT NULL,
    win            boolean NOT NULL,
    kills          smallint NOT NULL,
    deaths         smallint NOT NULL,
    assists        smallint NOT NULL,
    PRIMARY KEY (match_id, participant_id)
);
CREATE INDEX match_participants_puuid ON match_participants (puuid);
CREATE INDEX match_participants_champ_role ON match_participants (champion_id, team_position);

CREATE TABLE rank_snapshots (
    id          bigserial PRIMARY KEY,
    puuid       text NOT NULL,
    queue       text NOT NULL,         -- RANKED_SOLO_5x5, RANKED_FLEX_SR
    tier        text NOT NULL,         -- IRON ... CHALLENGER
    division    text,                  -- I-IV; NULL for Master+
    lp          integer NOT NULL,
    wins        integer NOT NULL,
    losses      integer NOT NULL,
    captured_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX rank_snapshots_latest ON rank_snapshots (puuid, queue, captured_at DESC);

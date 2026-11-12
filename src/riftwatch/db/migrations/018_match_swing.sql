-- Swing for each game: the win-probability curve and every player's share of it (winprob).
-- Computed once per game and model version, on first view, after a sync, or by
-- `riftwatch winprob backfill`.
CREATE TABLE match_swing (
    match_id    text PRIMARY KEY REFERENCES matches ON DELETE CASCADE,
    version     text NOT NULL,
    data        jsonb NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- Each player's Swing on its own row, for the pages that add it up over thousands of games
-- (champion averages, roles, the biggest games): reading each game's whole data for every
-- player took over ten seconds.
CREATE TABLE player_swing (
    match_id        text NOT NULL REFERENCES match_swing ON DELETE CASCADE,
    participant_id  smallint NOT NULL,
    version         text NOT NULL,
    total           real NOT NULL,
    fights          real NOT NULL,
    objectives      real NOT NULL,
    structures      real NOT NULL,
    economy         real NOT NULL,
    PRIMARY KEY (match_id, participant_id)
);
CREATE INDEX player_swing_version_total ON player_swing (version, total DESC);

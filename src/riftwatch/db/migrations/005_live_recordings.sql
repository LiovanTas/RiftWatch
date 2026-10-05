-- Games recorded second by second from Riot's Live Client Data API on the player's own PC.
-- The local API has no match id, so match_id is filled in later by matching the Riot ID,
-- champion and start time against synced matches.
CREATE TABLE live_recordings (
    id           bigserial PRIMARY KEY,
    file         text NOT NULL UNIQUE,
    riot_id      text NOT NULL,
    champion     text NOT NULL,
    position     text NOT NULL,
    game_start   timestamptz NOT NULL,      -- wall clock at game time 0, estimated
    duration_s   real NOT NULL,
    match_id     text REFERENCES matches ON DELETE SET NULL,
    summary      jsonb NOT NULL,            -- trades, recalls, deaths (live.analyze)
    hp_series    jsonb NOT NULL,            -- [[t, hp share], ...] every 2 s, for charts
    imported_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX live_recordings_match ON live_recordings (match_id);
CREATE INDEX live_recordings_riot_id ON live_recordings (lower(riot_id));

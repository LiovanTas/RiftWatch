-- Gameplay videos as training data: each video once, its per-moment laning readings and its
-- trades. Processing is versioned (analyzer_version), so an improved analyser re-processes
-- older videos; the files themselves stay where the user keeps them.
CREATE TABLE videos (
    id               bigserial PRIMARY KEY,
    path             text NOT NULL,
    fingerprint      text NOT NULL,            -- size + hashes of the start and end: moved file = same video
    title            text NOT NULL,
    champion         text,                     -- the followed champion (from the file name or given)
    opponent         text,
    role             text,                     -- TOP JUNGLE MIDDLE BOTTOM UTILITY
    region           text,
    tier             text,                     -- IRON .. CHALLENGER
    patch            text,
    view             text NOT NULL DEFAULT 'spectator',   -- spectator (replays) | player (your own)
    match_id         text,                     -- the Riot match, when known (your own games)
    width            integer,
    height           integer,
    fps              real,
    duration_s       real,
    game_offset_s    real,                     -- game time = video time + this
    clock_agreement  real,                     -- share of clock readings that agreed on it
    status           text NOT NULL DEFAULT 'pending',     -- pending | done | failed
    error            text,
    analyzer_version integer,
    added_at         timestamptz NOT NULL DEFAULT now(),
    processed_at     timestamptz
);
CREATE UNIQUE INDEX videos_fingerprint ON videos (fingerprint);
CREATE INDEX videos_role_champion ON videos (role, champion);

-- Four readings a second through the laning phase (see vision.lane.Sample).
CREATE TABLE video_samples (
    video_id      bigint NOT NULL REFERENCES videos ON DELETE CASCADE,
    t             real NOT NULL,               -- game time, seconds
    me            real,
    opponent      real,
    distance      real,
    others        smallint NOT NULL,
    my_minions    smallint,
    their_minions smallint,
    PRIMARY KEY (video_id, t)
);

CREATE TABLE video_trades (
    video_id      bigint NOT NULL REFERENCES videos ON DELETE CASCADE,
    start_s       real NOT NULL,
    end_s         real NOT NULL,
    me_lost       real NOT NULL,
    opponent_lost real NOT NULL,
    started_by    text NOT NULL,
    skirmish      boolean NOT NULL,
    minion_edge   smallint,
    result        text NOT NULL,
    PRIMARY KEY (video_id, start_s)
);

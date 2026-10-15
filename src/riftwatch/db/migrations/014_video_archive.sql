-- Everything read from a video, so the file itself can be deleted (vision.archive): every
-- reader's output four times a second over the whole game, and snapshots of HUD regions no
-- reader understands yet. Re-analysis and retraining run from these alone.
CREATE TABLE video_readings (
    video_id         bigint PRIMARY KEY REFERENCES videos ON DELETE CASCADE,
    detector_version integer NOT NULL,
    frames           integer NOT NULL,
    first_s          real,
    last_s           real,
    duration_s       real,
    readings         bytea NOT NULL,           -- gzipped JSON lines, one per frame
    saved_at         timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE video_snapshots (
    video_id  bigint NOT NULL REFERENCES videos ON DELETE CASCADE,
    t         real NOT NULL,                   -- video seconds
    region    text NOT NULL,                   -- minimap | scoreboard | topbar
    image     bytea NOT NULL,                  -- JPEG
    PRIMARY KEY (video_id, region, t)
);

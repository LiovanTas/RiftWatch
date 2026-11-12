-- AI explanations of single minutes in the High Elo Comparison tab: why most high-elo
-- players made another choice at that moment of a player's game. Written once on request.
CREATE TABLE moment_notes (
    match_id    text NOT NULL,
    puuid       text NOT NULL,
    minute      smallint NOT NULL,
    version     text NOT NULL,          -- prompt version, so a better prompt rewrites old notes
    model       text NOT NULL,
    evidence    jsonb NOT NULL,
    output      jsonb NOT NULL,
    usage       jsonb NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (match_id, puuid, minute, version)
);

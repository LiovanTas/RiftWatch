-- AI explanations of why a lane matchup is good or bad, one per champion pair, role and
-- game version. They're written once on request and shown to everyone after that.
CREATE TABLE matchup_notes (
    champion_id integer NOT NULL,
    opponent_id integer NOT NULL,
    position    text NOT NULL,
    version     text NOT NULL,          -- Data Dragon version the kits were read from
    model       text NOT NULL,
    evidence    jsonb NOT NULL,
    output      jsonb NOT NULL,         -- validated answer
    dropped     jsonb NOT NULL,         -- points removed by the grounding check
    usage       jsonb NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (champion_id, opponent_id, position, version)
);

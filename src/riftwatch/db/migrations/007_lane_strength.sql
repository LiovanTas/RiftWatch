-- How far ahead of its lane opponent each champion tends to be, per role and lane-lead
-- metric, pooled over every tier and shrunk toward zero. A matchup is judged by
-- strength(you) - strength(them). Rebuilt with the baselines.
CREATE TABLE lane_strength (
    role        text NOT NULL,
    champion_id integer NOT NULL,
    metric      text NOT NULL,
    minute      smallint NOT NULL,      -- 0 = whole-game metric
    n           integer NOT NULL,
    strength    double precision NOT NULL,
    PRIMARY KEY (role, metric, minute, champion_id)
);

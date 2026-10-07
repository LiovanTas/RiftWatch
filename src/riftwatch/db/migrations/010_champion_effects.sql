-- Champion adjustment, computed once per baseline build instead of on every server's first
-- look at a champion: how far the champion sits from its role, pooled over tiers, gated and
-- shrunk (see baselines.build). Only effects that pass the gate are stored.
CREATE TABLE champion_effects (
    role        text NOT NULL,
    champion_id integer NOT NULL,
    metric      text NOT NULL,
    minute      smallint NOT NULL,      -- 0 = whole-game metric
    effect      double precision NOT NULL,   -- in role standard deviations, after shrinking
    n           integer NOT NULL,       -- pooled champion games
    PRIMARY KEY (role, champion_id, metric, minute)
);

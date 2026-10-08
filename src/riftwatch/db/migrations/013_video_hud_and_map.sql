-- What the replay HUD and minimap add to each laning moment and trade: mana, which abilities
-- and summoner spells are ready (bits Q W E R D F), level, map position and lane depth, being
-- in base or dead -- and, per trade, the state it started in and what followed it.
ALTER TABLE video_samples
    ADD COLUMN mana    real,
    ADD COLUMN ready   smallint,
    ADD COLUMN level   smallint,
    ADD COLUMN map_x   real,
    ADD COLUMN map_y   real,
    ADD COLUMN depth   real,
    ADD COLUMN in_base boolean,
    ADD COLUMN dead    boolean NOT NULL DEFAULT false;

ALTER TABLE video_trades
    ADD COLUMN mana              real,
    ADD COLUMN ready             smallint,
    ADD COLUMN level             smallint,
    ADD COLUMN depth             real,
    ADD COLUMN died              boolean,
    ADD COLUMN back_after_s      real,
    ADD COLUMN opponent_left_low boolean;

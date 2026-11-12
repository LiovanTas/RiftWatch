-- Everything a match-history row and the champion statistics show, one row per player per
-- game, pulled out of matches.raw (and match_timelines.raw for skill and purchase order) by
-- riftwatch.stats.extract so the website never parses match JSON to list games.
CREATE TABLE participant_builds (
    match_id             text NOT NULL REFERENCES matches ON DELETE CASCADE,
    participant_id       smallint NOT NULL,
    puuid                text NOT NULL,
    riot_name            text,
    riot_tag             text,
    team_id              smallint NOT NULL,
    position             text NOT NULL,          -- TOP JUNGLE MIDDLE BOTTOM UTILITY, '' unknown
    champion_id          integer NOT NULL,
    opponent_champion_id integer,                -- the same position on the other team
    win                  boolean NOT NULL,
    remake               boolean NOT NULL,
    queue_id             integer NOT NULL,
    patch                text NOT NULL,
    game_start           timestamptz NOT NULL,
    duration_s           integer NOT NULL,
    champ_level          smallint,
    kills                smallint NOT NULL,
    deaths               smallint NOT NULL,
    assists              smallint NOT NULL,
    team_kills           smallint NOT NULL,
    cs                   smallint NOT NULL,
    gold                 integer NOT NULL,
    damage               integer NOT NULL,       -- to champions
    damage_taken         integer NOT NULL,
    vision_score         smallint NOT NULL,
    wards_placed         smallint NOT NULL,
    wards_killed         smallint NOT NULL,
    control_wards        smallint NOT NULL,
    largest_multikill    smallint NOT NULL,
    double_kills         smallint NOT NULL,
    triple_kills         smallint NOT NULL,
    quadra_kills         smallint NOT NULL,
    penta_kills          smallint NOT NULL,
    first_blood          boolean NOT NULL,
    items                integer[] NOT NULL,     -- item0..item6 (item6: trinket), 0 = empty
    spells               integer[] NOT NULL,     -- summoner spell ids
    keystone             integer,
    primary_style        integer,
    sub_style            integer,
    perks                integer[],              -- every rune and stat shard
    profile_icon         integer,
    summoner_level       integer,
    -- From the timeline; NULL until it is fetched.
    skill_order          text,                   -- one of Q W E R per level-up
    purchases            integer[],              -- items bought in order, undone ones removed
    purchase_s           integer[],              -- game second of each purchase
    PRIMARY KEY (match_id, participant_id)
);
CREATE INDEX participant_builds_puuid ON participant_builds (puuid, game_start DESC);
CREATE INDEX participant_builds_champion ON participant_builds (champion_id, position);
CREATE INDEX participant_builds_name ON participant_builds (lower(riot_name), lower(riot_tag));

CREATE TABLE match_bans (
    match_id    text NOT NULL REFERENCES matches ON DELETE CASCADE,
    team_id     smallint NOT NULL,
    pick_turn   smallint NOT NULL,
    champion_id integer NOT NULL,
    PRIMARY KEY (match_id, team_id, pick_turn)
);
CREATE INDEX match_bans_champion ON match_bans (champion_id);

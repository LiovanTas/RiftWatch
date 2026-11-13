-- Swing split further (winprob/swing.py VERSION sw4): the draft, kills and deaths apart (fights
-- stays as their sum), and finished items.
ALTER TABLE player_swing
    ADD COLUMN draft  real NOT NULL DEFAULT 0,
    ADD COLUMN kills  real NOT NULL DEFAULT 0,
    ADD COLUMN deaths real NOT NULL DEFAULT 0,
    ADD COLUMN items  real NOT NULL DEFAULT 0;

-- What each choice (an item at each slot, the first ability maxed, the keystone, the summoner
-- spells) did beyond the game state when it was made: players' results minus the win chance
-- the network gave their team at that moment, averaged over stored games and pulled towards
-- zero for small samples (winprob/choices.py). Rebuilt whole by `riftwatch winprob choices`.
CREATE TABLE choice_values (
    champion_id  integer NOT NULL,
    kind         text NOT NULL,          -- item, skill, keystone, spells
    slot         smallint NOT NULL,      -- items: 1st, 2nd, 3rd finished item; 0 otherwise
    choice       text NOT NULL,          -- item id, Q/W/E, keystone id, "4+14" spell ids
    context      text NOT NULL,          -- all, ap_heavy, ad_heavy, tanks (the enemy team)
    games        integer NOT NULL,
    raw          real NOT NULL,          -- mean result minus expected, percentage points
    value        real NOT NULL,          -- the same, pulled towards zero for small samples
    sd           real NOT NULL,          -- its uncertainty, percentage points
    expected     real NOT NULL,          -- the average win chance when it was chosen
    win_rate     real NOT NULL,
    version      text NOT NULL,
    PRIMARY KEY (champion_id, kind, slot, choice, context)
);

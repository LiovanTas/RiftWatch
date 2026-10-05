-- Which sampled ladder player each crawled game came from, and their tier at the time.
-- match_samples only keeps a coarse bucket (MASTER_PLUS); training data for the high-elo
-- models needs to tell Challenger and Grandmaster games apart from Master ones.
CREATE TABLE crawl_player_matches (
    puuid    text NOT NULL,
    match_id text NOT NULL REFERENCES matches ON DELETE CASCADE,
    platform text NOT NULL,
    tier     text NOT NULL,          -- IRON ... CHALLENGER, as listed on the ladder
    PRIMARY KEY (puuid, match_id)
);
CREATE INDEX crawl_player_matches_match ON crawl_player_matches (match_id);
CREATE INDEX crawl_player_matches_tier ON crawl_player_matches (tier);

-- Jobs are fetched by an unguessable token, never by their sequential id: with ids in URLs
-- anyone could walk through every lookup other visitors made (scouting results list a whole
-- live-game lobby). gen_random_uuid() is built into Postgres 13+; 122 random bits.
ALTER TABLE jobs ADD COLUMN token text NOT NULL DEFAULT replace(gen_random_uuid()::text, '-', '');
CREATE UNIQUE INDEX jobs_token ON jobs (token);

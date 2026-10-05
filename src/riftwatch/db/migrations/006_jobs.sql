-- Background jobs (syncing from Riot), shared by every server process. Workers claim jobs
-- with FOR UPDATE SKIP LOCKED, so two workers never take the same one, and pick next from
-- whichever owner (visitor) has the least work running, so one busy visitor can't make
-- everyone else wait.
CREATE TABLE jobs (
    id         bigserial PRIMARY KEY,
    key        text NOT NULL,            -- dedupe key, e.g. sync:na1:name#tag
    kind       text NOT NULL,            -- handler name, e.g. 'sync'
    params     jsonb NOT NULL,
    owner      text NOT NULL,            -- who asked (for fair scheduling)
    status     text NOT NULL DEFAULT 'queued',   -- queued | running | done | failed
    progress   jsonb NOT NULL DEFAULT '[]',      -- last few progress lines
    result     jsonb,
    error      text,
    worker     text,
    created    timestamptz NOT NULL DEFAULT now(),
    started    timestamptz,
    heartbeat  timestamptz,              -- a running job not heard from for a while is retried
    finished   timestamptz
);
CREATE INDEX jobs_queued ON jobs (created) WHERE status = 'queued';
CREATE INDEX jobs_running_owner ON jobs (owner) WHERE status = 'running';
CREATE INDEX jobs_key ON jobs (key, created DESC);

-- Per-visitor request counters for the public site, one row per key per fixed window, so
-- every server process enforces the same limits.
CREATE TABLE rate_limits (
    key          text NOT NULL,          -- e.g. 'coach:203.0.113.7'
    window_start timestamptz NOT NULL,
    count        integer NOT NULL,
    PRIMARY KEY (key, window_start)
);

-- Daily LLM spend is summed over today's coaching.
CREATE INDEX coach_reports_created ON coach_reports (created_at);

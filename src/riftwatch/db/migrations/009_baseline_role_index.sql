-- Baseline sets load a role's rows across every tier (shared by its champions) and then one
-- champion's rows across every tier. The lookup index leads with the tier, so both were
-- full-table scans (~25 ms each, once per champion on a cold player page).
CREATE INDEX baselines_role_champion ON baselines (role, champion_id);

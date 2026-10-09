-- Step 3.6: store every trace event as a step row. Additive only:
-- kind = event type (plan, worker_code, execute, verify, finish, ...),
-- payload = the full event, so a run can be rebuilt from its rows.
ALTER TABLE steps ADD COLUMN IF NOT EXISTS kind TEXT;
ALTER TABLE steps ADD COLUMN IF NOT EXISTS payload JSONB;

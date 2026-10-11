-- Step 5.2: eval batches. One batch = one run of the eval harness;
-- eval_runs links each (task, repeat) to the run it produced, so the
-- results table can be rebuilt later from Postgres alone.
CREATE TABLE IF NOT EXISTS eval_batches (
    batch_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    label        TEXT,
    command      TEXT NOT NULL,
    git_commit   TEXT,
    model        TEXT,
    repeats      INTEGER NOT NULL,
    task_ids     JSONB NOT NULL,
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS eval_runs (
    batch_id     UUID NOT NULL REFERENCES eval_batches(batch_id) ON DELETE CASCADE,
    task_id      TEXT NOT NULL,
    repeat       INTEGER NOT NULL,
    run_id       UUID REFERENCES runs(run_id) ON DELETE CASCADE,
    error        TEXT,          -- set when the run couldn't be submitted or followed
    PRIMARY KEY (batch_id, task_id, repeat)
);

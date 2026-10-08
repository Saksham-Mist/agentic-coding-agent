-- Step 1.3: core schema — runs, steps, rate_limits.
-- See the design notes §5/§9 and the design notes for the fields these back.

CREATE TABLE IF NOT EXISTS runs (
    run_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id         TEXT NOT NULL,
    status          TEXT NOT NULL
                        CHECK (status IN ('queued', 'running', 'passed', 'incomplete', 'error')),
    tokens_used     INTEGER NOT NULL DEFAULT 0,
    steps_taken     INTEGER NOT NULL DEFAULT 0,
    result          JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS steps (
    id                  BIGSERIAL PRIMARY KEY,
    run_id              UUID NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    step_number         INTEGER NOT NULL,
    subtask             TEXT,
    code                TEXT,
    execution_result    JSONB,
    verifier_decision   TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, step_number)
);

CREATE INDEX IF NOT EXISTS idx_steps_run_id ON steps(run_id);

CREATE TABLE IF NOT EXISTS rate_limits (
    ip      TEXT NOT NULL,
    day     DATE NOT NULL,
    count   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (ip, day)
);

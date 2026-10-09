-- Step 4.1: where a run came from, so GET /stats can report real
-- traffic separately from development and test runs.
-- api = public POST /runs, cli = local CLI, eval = eval harness (5.2),
-- test = automated tests.
ALTER TABLE runs ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'cli';

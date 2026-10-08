"""Apply SQL migrations in migrations/ (in filename order) against DATABASE_URL.

Usage: python scripts/migrate.py
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import psycopg

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def main() -> None:
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)

    migrations_dir = ROOT / "migrations"
    sql_files = sorted(migrations_dir.glob("*.sql"))
    if not sql_files:
        print("No migrations found in", migrations_dir)
        return

    with psycopg.connect(url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    filename TEXT PRIMARY KEY,
                    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute("SELECT filename FROM schema_migrations")
            applied = {row[0] for row in cur.fetchall()}

            for path in sql_files:
                if path.name in applied:
                    print(f"skip  {path.name} (already applied)")
                    continue
                print(f"apply {path.name}")
                cur.execute(path.read_text(encoding="utf-8"))
                cur.execute(
                    "INSERT INTO schema_migrations (filename) VALUES (%s)",
                    (path.name,),
                )

    print("done")


if __name__ == "__main__":
    main()

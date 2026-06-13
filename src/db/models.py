"""Database connection + helpers. Plain sqlite3, no ORM — keep it simple."""
from __future__ import annotations

import sqlite3
from pathlib import Path

# Project root = two levels up from this file (src/db/models.py -> project root)
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "worldcup.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def get_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with sensible defaults."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Path | str = DB_PATH) -> None:
    """Create all tables from schema.sql. Safe to call repeatedly."""
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA_PATH.read_text())
        conn.commit()
    finally:
        conn.close()


def get_or_create_team(conn: sqlite3.Connection, name: str) -> int:
    """Return team_id for a team name, inserting it if new."""
    row = conn.execute("SELECT team_id FROM teams WHERE name = ?", (name,)).fetchone()
    if row:
        return row["team_id"]
    cur = conn.execute("INSERT INTO teams(name) VALUES (?)", (name,))
    return cur.lastrowid


if __name__ == "__main__":
    init_db()
    print(f"Initialized database at {DB_PATH}")

import sqlite3
from pathlib import Path
from typing import Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    inside_house INTEGER NOT NULL DEFAULT 0,
    speed REAL NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_positions_timestamp ON positions(timestamp);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event TEXT NOT NULL,
    details TEXT
);
"""


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    def insert_position(self, timestamp: str, x: float, y: float, inside_house: bool, speed: float, confidence: float) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                "INSERT INTO positions (timestamp, x, y, inside_house, speed, confidence) VALUES (?, ?, ?, ?, ?, ?)",
                (timestamp, x, y, int(inside_house), speed, confidence),
            )
            return int(cursor.lastrowid)

    def positions(self, limit: int = 1000) -> list[sqlite3.Row]:
        with self.connect() as connection:
            return list(connection.execute("SELECT * FROM positions ORDER BY timestamp DESC LIMIT ?", (limit,)))

    def latest_position(self) -> sqlite3.Row | None:
        rows = self.positions(1)
        return rows[0] if rows else None

    def insert_event(self, timestamp: str, event: str, details: str | None = None) -> None:
        with self.connect() as connection:
            connection.execute("INSERT INTO events (timestamp, event, details) VALUES (?, ?, ?)", (timestamp, event, details))


def row_to_dict(row: sqlite3.Row) -> dict:
    return dict(row)

"""SQLite persistence for subscriptions and the facts already sent to each one.

The API process and the delivery worker share this single database file (see
docker-compose.yml), so connections are opened in WAL mode with a busy timeout to
avoid "database is locked" errors under concurrent access, and every helper
closes its connection in a ``finally`` block so a mid-query error can't leak a
file handle.
"""

import datetime
import json
import os
import sqlite3
from typing import Any, Dict, List

# Location of the SQLite file. Overridable via DB_PATH so the API and the
# delivery worker can share one database on a mounted volume. Tests monkeypatch
# this attribute directly.
DB_PATH = os.getenv("DB_PATH", "subscriptions.db")

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS subscriptions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL,
    topic TEXT NOT NULL,
    cadence TEXT NOT NULL,
    max_facts INTEGER NOT NULL DEFAULT 3,
    next_send TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL,
    last_send TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_subscriptions_next_send
    ON subscriptions (next_send);

-- Records every fact delivered to a subscription together with its embedding so
-- the dedup / spaced-repetition layer can avoid resending near-duplicate facts.
CREATE TABLE IF NOT EXISTS sent_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subscription_id INTEGER NOT NULL,
    fact TEXT NOT NULL,
    embedding TEXT NOT NULL,
    sent_at TIMESTAMP NOT NULL,
    FOREIGN KEY (subscription_id) REFERENCES subscriptions (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sent_facts_subscription
    ON sent_facts (subscription_id);
"""


def get_conn():
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    # WAL + a busy timeout let the API and worker read/write the shared file
    # concurrently without tripping "database is locked".
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_db():
    """Create the tables and indexes if they don't already exist."""
    # Ensure the parent directory exists (e.g. DB_PATH=/data/subscriptions.db on
    # a host that doesn't pre-create the mount). No-op for a bare filename.
    parent = os.path.dirname(DB_PATH)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = get_conn()
    try:
        with conn:
            conn.executescript(CREATE_SQL)
            _migrate_schema(conn)
    finally:
        conn.close()


def _migrate_schema(conn):
    """Add columns introduced after the first release so existing databases
    upgrade in place (CREATE TABLE IF NOT EXISTS never alters an existing table).
    """
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(subscriptions)")}
    if "max_facts" not in cols:
        conn.execute(
            "ALTER TABLE subscriptions ADD COLUMN max_facts INTEGER NOT NULL DEFAULT 3"
        )


def add_subscription(
    email: str,
    topic: str,
    cadence: str,
    next_send: datetime.datetime,
    max_facts: int = 3,
):
    """Insert a new subscription due for its first send at ``next_send``.

    ``max_facts`` is how many grounded facts each digest should aim to include
    (the subscriber picks this on the form); it defaults to 3 for older callers.
    """
    conn = get_conn()
    now = datetime.datetime.utcnow()
    try:
        with conn:
            conn.execute(
                "INSERT INTO subscriptions (email, topic, cadence, max_facts, next_send, created_at) VALUES (?,?,?,?,?,?)",
                (email, topic, cadence, max_facts, next_send, now),
            )
    finally:
        conn.close()


def list_subscriptions() -> List[Dict[str, Any]]:
    """Return all subscriptions, newest first."""
    conn = get_conn()
    try:
        cur = conn.execute("SELECT * FROM subscriptions ORDER BY id DESC")
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def delete_subscription(sub_id: int):
    """Delete a subscription (its sent_facts cascade away with it)."""
    conn = get_conn()
    try:
        with conn:
            conn.execute("DELETE FROM subscriptions WHERE id = ?", (sub_id,))
    finally:
        conn.close()


def get_due_subscriptions(now: datetime.datetime) -> List[Dict[str, Any]]:
    """Return subscriptions whose ``next_send`` has arrived (<= ``now``)."""
    conn = get_conn()
    try:
        cur = conn.execute("SELECT * FROM subscriptions WHERE next_send <= ?", (now,))
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def claim_due_subscription(
    sub_id: int, next_send: datetime.datetime, now: datetime.datetime
) -> bool:
    """Atomically claim a due subscription by advancing its ``next_send``.

    The single UPDATE only fires while the row is still due (``next_send <= now``),
    so if the worker's poll loop and the API's ``/run-due`` race on the shared
    database, exactly one caller wins the claim (``rowcount == 1``) and the other
    sees ``rowcount == 0`` and skips it — preventing duplicate emails.
    """
    conn = get_conn()
    try:
        with conn:
            cur = conn.execute(
                "UPDATE subscriptions SET next_send = ?, last_send = ? "
                "WHERE id = ? AND next_send <= ?",
                (next_send, now, sub_id, now),
            )
        return cur.rowcount == 1
    finally:
        conn.close()


def update_next_send(
    sub_id: int, next_send: datetime.datetime, last_send: datetime.datetime = None
):
    """Reschedule a subscription's next send (and record its last send)."""
    conn = get_conn()
    try:
        with conn:
            conn.execute(
                "UPDATE subscriptions SET next_send = ?, last_send = ? WHERE id = ?",
                (next_send, last_send, sub_id),
            )
    finally:
        conn.close()


def add_sent_fact(subscription_id: int, fact: str, embedding: List[float]):
    """Persist a delivered fact and its embedding for future dedup checks."""
    conn = get_conn()
    now = datetime.datetime.utcnow()
    try:
        with conn:
            conn.execute(
                "INSERT INTO sent_facts (subscription_id, fact, embedding, sent_at) VALUES (?,?,?,?)",
                (subscription_id, fact, json.dumps(list(embedding)), now),
            )
    finally:
        conn.close()


def get_sent_fact_embeddings(subscription_id: int) -> List[List[float]]:
    """Return the stored embeddings of facts already sent to a subscription."""
    conn = get_conn()
    try:
        cur = conn.execute(
            "SELECT embedding FROM sent_facts WHERE subscription_id = ?",
            (subscription_id,),
        )
        return [json.loads(r["embedding"]) for r in cur.fetchall()]
    finally:
        conn.close()

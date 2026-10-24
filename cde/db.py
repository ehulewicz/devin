"""SQLite storage for the Cognitive Dynamics Engine.

Schema mirrors the Postgres design 1:1; embeddings stored as float32 BLOBs
(brute-force numpy cosine — a few thousand vectors is milliseconds).
Migration path: swap this module for psycopg + pgvector, same tables.
"""
import os
import sqlite3

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(ROOT, "data", "cde.db")

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS participants(
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE,
    wearer_code TEXT UNIQUE,          -- A1..A6
    is_subject INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS utterances(
    id INTEGER PRIMARY KEY,
    wearer TEXT,                      -- A1_JAKE
    file TEXT,                        -- source srt filename
    day INTEGER,
    t_abs REAL,                       -- (day-1)*86400 + file_offset + srt_offset
    t_end REAL,
    hms TEXT,                         -- wall-clock HH:MM:SS
    speaker TEXT,                     -- canonical speaker
    text_en TEXT,
    text_zh TEXT
);
CREATE INDEX IF NOT EXISTS ix_utt_wearer_time ON utterances(wearer, t_abs);
CREATE INDEX IF NOT EXISTS ix_utt_day ON utterances(day);

CREATE TABLE IF NOT EXISTS events(
    id INTEGER PRIMARY KEY,
    subject TEXT,                     -- A1_JAKE
    day INTEGER,
    t_start REAL,                     -- absolute: when it was recorded
    t_end REAL,
    occurred_at REAL,                 -- differs from t_start when recounting the past
    recorded_at REAL,
    time_uncertain INTEGER DEFAULT 0,
    summary TEXT,
    mood TEXT,
    embedding BLOB
);
CREATE INDEX IF NOT EXISTS ix_events_time ON events(subject, t_start);

CREATE TABLE IF NOT EXISTS event_utterances(
    event_id INTEGER REFERENCES events(id),
    utterance_id INTEGER REFERENCES utterances(id),
    PRIMARY KEY(event_id, utterance_id)
);

CREATE TABLE IF NOT EXISTS event_participants(
    event_id INTEGER REFERENCES events(id),
    entity_id INTEGER REFERENCES entities(id),
    role TEXT,
    PRIMARY KEY(event_id, entity_id)
);

CREATE TABLE IF NOT EXISTS entities(
    id INTEGER PRIMARY KEY,
    type TEXT,                        -- TOPIC | GOAL | PERSON | BEHAVIOR_KIND
    canonical TEXT,
    embedding BLOB,
    UNIQUE(type, canonical)
);

CREATE TABLE IF NOT EXISTS entity_aliases(
    entity_id INTEGER REFERENCES entities(id),
    alias TEXT,
    PRIMARY KEY(entity_id, alias)
);

CREATE TABLE IF NOT EXISTS event_entities(
    event_id INTEGER REFERENCES events(id),
    entity_id INTEGER REFERENCES entities(id),
    relation TEXT,                    -- topic | goal | person
    PRIMARY KEY(event_id, entity_id, relation)
);

CREATE TABLE IF NOT EXISTS mentions(
    id INTEGER PRIMARY KEY,
    type TEXT,                        -- THOUGHT | EMOTION | GOAL | BEHAVIOR | CAUSAL_CLAIM
    event_id INTEGER REFERENCES events(id),
    speaker TEXT,
    text TEXT,
    payload_json TEXT
);

CREATE TABLE IF NOT EXISTS mention_evidence(
    mention_id INTEGER REFERENCES mentions(id),
    utterance_id INTEGER REFERENCES utterances(id),
    PRIMARY KEY(mention_id, utterance_id)
);

CREATE TABLE IF NOT EXISTS edges(
    id INTEGER PRIMARY KEY,
    src_type TEXT,                    -- PERSON | EVENT | UTTERANCE | GOAL | MENTION | TOPIC
    src_id INTEGER,
    dst_type TEXT,
    dst_id INTEGER,
    relation TEXT,                    -- SPOKE PARTICIPATED_IN EXPRESSES PRECEDES REFERS_TO
                                      -- SIMILAR_TO SAME_THREAD ADVANCES_GOAL INFLUENCES
                                      -- INTERACTION_PATTERN
    confidence REAL,
    status TEXT,                      -- observed | hypothesis | rejected
    evidence_type TEXT,               -- self_report | transcript | temporal | semantic | extraction | statistical
    rationale TEXT,
    alternatives_json TEXT,
    retrieval_score REAL
);
CREATE INDEX IF NOT EXISTS ix_edges_dst ON edges(dst_type, dst_id);
CREATE INDEX IF NOT EXISTS ix_edges_src ON edges(src_type, src_id);
CREATE INDEX IF NOT EXISTS ix_edges_rel ON edges(relation, status);

CREATE TABLE IF NOT EXISTS edge_evidence(
    edge_id INTEGER REFERENCES edges(id),
    utterance_id INTEGER REFERENCES utterances(id),
    PRIMARY KEY(edge_id, utterance_id)
);

CREATE TABLE IF NOT EXISTS goal_state(
    goal_entity_id INTEGER REFERENCES entities(id),
    day INTEGER,
    status TEXT,                      -- open | in-progress | done
    event_id INTEGER REFERENCES events(id),
    PRIMARY KEY(goal_entity_id, day)
);

CREATE TABLE IF NOT EXISTS pair_behavior(
    edge_id INTEGER REFERENCES edges(id),
    src_entity_id INTEGER REFERENCES entities(id),   -- person expressing the behavior
    dst_entity_id INTEGER REFERENCES entities(id),   -- other person present
    behavior_entity_id INTEGER REFERENCES entities(id),
    day INTEGER,
    shared INTEGER,                   -- events with both present that day
    behaved INTEGER,                  -- of those, events where src expressed the behavior
    events_json TEXT,                 -- event ids contributing to `behaved`
    PRIMARY KEY(src_entity_id, dst_entity_id, behavior_entity_id, day)
);

CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v BLOB);

-- stretch
CREATE TABLE IF NOT EXISTS patterns(
    id INTEGER PRIMARY KEY,
    name TEXT,
    description TEXT,
    embedding BLOB
);
CREATE TABLE IF NOT EXISTS pattern_occurrences(
    pattern_id INTEGER REFERENCES patterns(id),
    event_id INTEGER REFERENCES events(id),
    PRIMARY KEY(pattern_id, event_id)
);
"""


def connect(path=None):
    os.makedirs(os.path.dirname(path or DB_PATH), exist_ok=True)
    con = sqlite3.connect(path or DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init_db(con=None):
    close = con is None
    con = con or connect()
    con.executescript(SCHEMA)
    con.commit()
    if close:
        con.close()
    return con

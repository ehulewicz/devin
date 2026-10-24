"""Stage 2 — extract: one LLM call per hour of transcript (Jake only).

Utterances are listed as `[u<id>] HH:MM:SS Speaker: english | 中文`.
Every extracted item must cite >=1 utterance id that exists in the window —
anything else is dropped (enforced in validate + prompts).

Raw JSON cached in data/extracted/<file>.json so reruns are free.
Resumable: files already in the DB (any events) are skipped unless --force.

Usage: python -m cde.extract [A1_JAKE] [--limit N] [--workers 8] [--force]
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

from cde import db, llm

ROOT = db.ROOT
CACHE = os.path.join(ROOT, "data", "extracted")

SCHEMA_DESC = """{
 "events": [{"start_utt": int, "end_utt": int, "summary": str,
             "participants": [str], "topics": [str], "mood": str|null,
             "recounts_past": bool, "when_hint": str|null}],
 "thoughts":     [{"utterance_id": int, "speaker": str, "text": str}],
 "emotions":     [{"utterance_id": int, "speaker": str, "emotion": str, "text": str}],
 "goals":        [{"utterance_id": int, "speaker": str, "goal": str,
                   "status": "open"|"in-progress"|"done"}],
 "behaviors":    [{"utterance_id": int, "speaker": str, "behavior": str}],
 "causal_claims":[{"utterance_id": int, "speaker": str, "claim": str,
                   "earlier_desc": str, "later_desc": str,
                   "refers_to_past": bool, "when_hint": str|null}]
}"""

PROMPT = """You are analyzing one hour of first-person lifelog transcript (wearer: {subject}, day {day}, file {file}).
Each line is `[u<utterance_id>] HH:MM:SS Speaker: english_text | chinese_text`.

Extract the cognitive structure of this hour. Rules:
- Group consecutive utterances into EVENTS: coherent activities/conversations (a meal, planning a party, an argument, commuting). An event usually spans a few minutes. Cover the interesting parts of the hour; skip pure filler. start_utt/end_utt are the first/last utterance ids of the span.
- Only extract thoughts/emotions/goals/behaviors that are EXPRESSED in the text — never infer hidden mental states.
- causal_claims: when a speaker explicitly says earlier X led to / caused / is why later Y (e.g. "that's why we…", "because of yesterday's…"). earlier_desc/later_desc are short phrases describing the two sides. refers_to_past=true when the earlier side happened before this hour.
- recounts_past=true on an event means the conversation is ABOUT something that already happened (e.g. telling a story about yesterday). Give when_hint ("yesterday", "two days ago", "day 3 evening") if stated or implied.
- Every item MUST cite utterance ids that appear in the list below. Events cite via start_utt/end_utt; all other items via utterance_id.
- Keep summaries concrete and short (<=25 words). topics: 1-4 lowercase noun phrases. participants: first names only.
- Aim for roughly 4-10 events; return empty lists rather than inventing content.

Reply with ONLY a JSON object matching this shape (no markdown, no commentary):
{schema}

TRANSCRIPT:
{transcript}"""


def load_window(con, subject, file):
    rows = con.execute(
        "SELECT id, hms, speaker, text_en, text_zh, t_abs, t_end FROM utterances "
        "WHERE wearer=? AND file=? ORDER BY t_abs",
        (subject, file),
    ).fetchall()
    lines = []
    for r in rows:
        en = r["text_en"] or ""
        zh = r["text_zh"] or ""
        body = en if not zh else (f"{en} | {zh}" if en else zh)
        lines.append(f"[u{r['id']}] {r['hms']} {r['speaker']}: {body}")
    return rows, "\n".join(lines)


def parse_when_hint(hint, day, t_start):
    """Return (occurred_at, uncertain). Understands yesterday/N days ago/day K."""
    if not hint:
        return None, 1
    h = hint.lower()
    m = re.search(r"day\s*(\d)", h)
    if m:
        k = int(m.group(1))
        if 1 <= k <= 7:
            return (k - 1) * 86400 + t_start % 86400, 1
    if "yesterday" in h or "last night" in h:
        return t_start - 86400, 1
    m = re.search(r"(\d+|a few|couple)\s*days?\s*ago", h)
    if m:
        n = {"a few": 3, "couple": 2}.get(m.group(1), int(m.group(1)) if m.group(1).isdigit() else 0)
        return t_start - n * 86400, 1
    if "earlier today" in h or "this morning" in h:
        return t_start - 4 * 3600, 1
    return None, 1


def person_entity(cur, name):
    name = name.strip()
    if not name:
        return None
    cur.execute(
        "SELECT e.id FROM entities e LEFT JOIN entity_aliases a ON a.entity_id=e.id "
        "WHERE e.type='PERSON' AND (e.canonical=? OR a.alias=?)",
        (name, name),
    )
    row = cur.fetchone()
    if row:
        return row["id"]
    cur.execute("INSERT OR IGNORE INTO entities(type, canonical) VALUES('PERSON', ?)", (name,))
    cur.execute("SELECT id FROM entities WHERE type='PERSON' AND canonical=?", (name,))
    eid = cur.fetchone()["id"]
    cur.execute("INSERT OR IGNORE INTO entity_aliases(entity_id, alias) VALUES(?, ?)", (eid, name))
    return eid


def store(con, subject, day, file, rows, data):
    """Insert one hour's extraction. Utterance ids validated against the window."""
    if not rows:
        return 0
    valid = {r["id"] for r in rows}
    by_id = {r["id"]: r for r in rows}
    cur = con.cursor()
    # idempotent per file: wipe prior extraction rows for this window's events
    cur.execute(
        "SELECT id FROM events WHERE subject=? AND day=? AND t_start>=? AND t_start<?",
        (subject, day, rows[0]["t_abs"], rows[-1]["t_abs"] + 1),
    )
    old = [r["id"] for r in cur.fetchall()]
    if old:
        q = ",".join("?" * len(old))
        # These derived rows reference events directly and may already exist when
        # a cached hour is replayed after embedding/linking. Remove them before
        # replacing the source events so a resume remains idempotent with foreign
        # keys enabled. The derived stages are rebuilt after extraction finishes.
        cur.execute(f"DELETE FROM goal_state WHERE event_id IN ({q})", old)
        cur.execute(f"DELETE FROM pattern_occurrences WHERE event_id IN ({q})", old)
        cur.execute(
            f"DELETE FROM edge_evidence WHERE edge_id IN ("
            f"SELECT id FROM edges WHERE (src_type='EVENT' AND src_id IN ({q})) "
            f"OR (dst_type='EVENT' AND dst_id IN ({q})))",
            old + old,
        )
        cur.execute(
            f"DELETE FROM edges WHERE (src_type='EVENT' AND src_id IN ({q})) "
            f"OR (dst_type='EVENT' AND dst_id IN ({q}))",
            old + old,
        )
        for t in ("event_utterances", "event_participants", "event_entities"):
            cur.execute(f"DELETE FROM {t} WHERE event_id IN ({q})", old)
        cur.execute(
            f"DELETE FROM mention_evidence WHERE mention_id IN (SELECT id FROM mentions WHERE event_id IN ({q}))",
            old,
        )
        cur.execute(f"DELETE FROM mentions WHERE event_id IN ({q})", old)
        cur.execute(f"DELETE FROM events WHERE id IN ({q})", old)

    utterance_ids = sorted(valid)
    for ev in data.get("events", []):
        s, e = ev.get("start_utt"), ev.get("end_utt")
        if s not in valid or e not in valid or s > e:
            continue
        t_start, t_end = by_id[s]["t_abs"], by_id[e]["t_end"]
        occurred, uncertain = (t_start, 0)
        if ev.get("recounts_past"):
            occurred, uncertain = parse_when_hint(ev.get("when_hint"), day, t_start)
            occurred = occurred if occurred is not None else t_start
        cur.execute(
            """INSERT INTO events(subject,day,t_start,t_end,occurred_at,recorded_at,
                                  time_uncertain,summary,mood)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (subject, day, t_start, t_end, occurred, t_start, uncertain,
             (ev.get("summary") or "").strip(), ev.get("mood")),
        )
        eid = cur.lastrowid
        span = [u for u in utterance_ids if s <= u <= e]
        cur.executemany(
            "INSERT OR IGNORE INTO event_utterances(event_id,utterance_id) VALUES(?,?)",
            [(eid, u) for u in span],
        )
        for p in ev.get("participants", []):
            pid = person_entity(cur, p)
            if pid:
                cur.execute(
                    "INSERT OR IGNORE INTO event_participants(event_id,entity_id,role) VALUES(?,?,?)",
                    (eid, pid, "participant"),
                )
        for t in ev.get("topics", []):
            t = str(t).strip().lower()
            if not t:
                continue
            cur.execute(
                "INSERT OR IGNORE INTO entities(type,canonical) VALUES('TOPIC',?)", (t,)
            )
            tid = cur.execute(
                "SELECT id FROM entities WHERE type='TOPIC' AND canonical=?", (t,)
            ).fetchone()["id"]
            cur.execute(
                "INSERT OR IGNORE INTO event_entities(event_id,entity_id,relation) VALUES(?,?,?)",
                (eid, tid, "topic"),
            )

    def add_mentions(kind, items, text_key):
        for m in items:
            uid = m.get("utterance_id")
            if uid not in valid:
                continue
            # attach to the containing event if one exists
            cur.execute(
                """SELECT e.id FROM events e JOIN event_utterances eu ON eu.event_id=e.id
                   WHERE eu.utterance_id=? LIMIT 1""",
                (uid,),
            )
            row = cur.fetchone()
            payload = {k: v for k, v in m.items() if k not in ("utterance_id", text_key)}
            cur.execute(
                "INSERT INTO mentions(type,event_id,speaker,text,payload_json) VALUES(?,?,?,?,?)",
                (kind, row["id"] if row else None, m.get("speaker"),
                 str(m.get(text_key) or m.get("claim") or ""), json.dumps(payload, ensure_ascii=False)),
            )
            mid = cur.lastrowid
            cur.execute(
                "INSERT OR IGNORE INTO mention_evidence(mention_id,utterance_id) VALUES(?,?)",
                (mid, uid),
            )

    add_mentions("THOUGHT", data.get("thoughts", []), "text")
    add_mentions("EMOTION", data.get("emotions", []), "emotion")
    add_mentions("GOAL", data.get("goals", []), "goal")
    add_mentions("BEHAVIOR", data.get("behaviors", []), "behavior")
    add_mentions("CAUSAL_CLAIM", data.get("causal_claims", []), "claim")
    con.commit()
    return len(data.get("events", []))


def extract_file(subject, day, file, force=False):
    cache_path = os.path.join(CACHE, f"{file}.json")
    if os.path.exists(cache_path) and not force:
        return file, json.load(open(cache_path))
    con = db.connect()
    rows, transcript = load_window(con, subject, file)
    con.close()
    if not rows:
        return file, None
    prompt = PROMPT.format(subject=subject, day=day, file=file,
                           schema=SCHEMA_DESC, transcript=transcript)
    data = llm.chat_json(prompt)
    os.makedirs(CACHE, exist_ok=True)
    json.dump(data, open(cache_path, "w"), ensure_ascii=False)
    return file, data


def main():
    subject = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "A1_JAKE"
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 8
    force = "--force" in sys.argv

    con = db.connect()
    files = [r["file"] for r in con.execute(
        "SELECT DISTINCT file, day FROM utterances WHERE wearer=? ORDER BY day, t_abs",
        (subject,))]
    day_of = {r["file"]: r["day"] for r in con.execute(
        "SELECT DISTINCT file, day FROM utterances WHERE wearer=?", (subject,))}
    if limit:
        files = files[:limit]
    print(f"{len(files)} hour-files for {subject} (backend={llm.BACKEND})", flush=True)

    def work(f):
        try:
            _, data = extract_file(subject, day_of[f], f, force=force)
            return f, data, None
        except Exception as e:
            return f, None, str(e)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for f, data, err in pool.map(work, files):
            done += 1
            if err:
                print(f"[{done}/{len(files)}] {f} ERROR {err[:160]}", flush=True)
                continue
            con2 = db.connect()
            rows, _ = load_window(con2, subject, f)
            n = store(con2, subject, day_of[f], f, rows, data) if data else 0
            con2.close()
            print(f"[{done}/{len(files)}] {f} events={n}", flush=True)
    con.close()


if __name__ == "__main__":
    main()

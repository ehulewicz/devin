"""Stage 3 — embed + entity resolution.

Embeddings: OpenAI text-embedding-3-small when a key is present; otherwise a
hashed TF-IDF (2048-dim) over English text — deterministic, no dependencies.
Chinese text is intentionally excluded from the vector space (kept for
evidence display only).

Entity resolution: greedy cosine clustering (>= 0.82) of goal strings into
canonical GOAL entities; topics already canonicalized lowercase at insert;
the 6 wearers + observed speakers become PERSON entities with aliases.

Usage: python -m cde.embed
"""
import hashlib
import json
import sys
import re

import numpy as np

from cde import db, llm

DIM = 2048
TOKEN = re.compile(r"[a-z0-9']+")


def _hash_token(tok):
    return int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little") % DIM


def tfidf_fit(texts):
    """Fit idf over corpus; return (vectors, idf). Vectors L2-normalized."""
    docs = []
    df = np.zeros(DIM, dtype=np.float64)
    for t in texts:
        toks = TOKEN.findall((t or "").lower())
        tf = {}
        for tok in toks:
            h = _hash_token(tok)
            tf[h] = tf.get(h, 0) + 1.0
        docs.append(tf)
        for h in tf:
            df[h] += 1
    n = max(len(texts), 1)
    idf = np.log((1 + n) / (1 + df)) + 1.0
    return tfidf_transform(texts, idf), idf


def tfidf_transform(texts, idf):
    out = []
    for t in texts:
        v = np.zeros(DIM, dtype=np.float32)
        tf = {}
        for tok in TOKEN.findall((t or "").lower()):
            h = _hash_token(tok)
            tf[h] = tf.get(h, 0) + 1.0
        for h, c in tf.items():
            v[h] = (1 + np.log(c)) * idf[h]
        norm = np.linalg.norm(v)
        if norm:
            v /= norm
        out.append(v)
    return out


def _save_idf(con, idf):
    con.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v BLOB)")
    con.execute("INSERT OR REPLACE INTO meta(k,v) VALUES('tfidf_idf',?)",
                (idf.astype(np.float32).tobytes(),))
    con.commit()


def load_idf(con):
    try:
        r = con.execute("SELECT v FROM meta WHERE k='tfidf_idf'").fetchone()
    except Exception:
        return None
    return np.frombuffer(r["v"], dtype=np.float32) if r else None


def embed_texts(texts, con=None):
    """OpenAI embeddings if available, else hashed TF-IDF (idf persisted in meta)."""
    vecs = llm.embed(texts)
    if vecs is not None:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return [v / n for v, n in zip(vecs, norms)]
    if con is not None:
        idf = load_idf(con)
        if idf is not None:
            return tfidf_transform(texts, idf)
    vs, idf = tfidf_fit(texts)
    if con is not None:
        _save_idf(con, idf)
    return vs


def embed_query(texts, con):
    """Query-time embedding: reuse stored idf so vectors share the corpus space."""
    vecs = llm.embed(texts)
    if vecs is not None:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return [v / n for v, n in zip(vecs, norms)]
    idf = load_idf(con)
    if idf is None:
        vs, _ = tfidf_fit(texts)
        return vs
    return tfidf_transform(texts, idf)


def blob(v):
    return np.asarray(v, dtype=np.float32).tobytes()


def unblob(b):
    return np.frombuffer(b, dtype=np.float32) if b else None


def event_text(ev, topics, people):
    parts = [ev["summary"] or "", ev["mood"] or ""]
    parts += topics
    parts += people
    return " | ".join(p for p in parts if p)


GOAL_CLUSTER_PROMPT = """Below is a numbered list of goal statements extracted from one week of one person's lifelog. Group them into canonical objectives: merge statements that describe the same underlying goal even when phrased differently (e.g. "buy skewers for the BBQ", "get barbecue supplies", "order charcoal" -> "shop for the barbecue"). Keep genuinely distinct goals separate. Favor a handful of recurring threads (e.g. the final-day party/event planning) over hundreds of one-off micro-goals.

Name each canonical goal as a SPECIFIC, RECOGNIZABLE thread of this person's week — e.g. "plan and host the final-day party", "bake the mango cream cake", "recoup shared expenses", "win the auctioned items". NEVER use generic category labels like "shopping", "social engagement", "task resolution", "food preparation" — always include the concrete object/outcome.

GOALS:
@@GOALS@@

Reply with ONLY a JSON object: {"<canonical goal name>": [<indices into the list>]} — every index must appear exactly once."""

BEHAVIOR_CLUSTER_PROMPT = """Below is a numbered list of observed behavior statements extracted from one week of one person's lifelog. Group them into canonical behavior categories: merge statements describing the same kind of interpersonal action even when phrased differently or targeting different specifics (e.g. "asks Tasha for help with the cable", "requests help fixing the app" -> "asking others for help"; "teases Shure about his hair", "jokes with the group" -> "teasing/joking"). Keep genuinely distinct behaviors separate. Prefer a compact set of recurring behavior kinds over hundreds of one-off actions.

BEHAVIORS:
@@GOALS@@

Reply with ONLY a JSON object: {"<canonical behavior name>": [<indices into the list>]} — every index must appear exactly once."""


def cluster_goal_strings(items, thresh=0.82, prompt=GOAL_CLUSTER_PROMPT):
    """Mention texts -> canonical entity clusters.
    LLM canonicalization in batches (semantic merging beats cosine on varied
    phrasings); falls back to greedy cosine clustering if no LLM is available."""
    texts = [t for _, _, t in items]
    clusters = []  # (rep_text, member_idx[])
    ok = True
    B = 90
    for off in range(0, len(items), B):
        chunk = items[off : off + B]
        listing = "\n".join(f"{i}. {t}" for i, (_, _, t) in enumerate(chunk))
        try:
            out = llm.chat_json(prompt.replace("@@GOALS@@", listing))
        except Exception as e:
            print(f"cluster batch {off} LLM failed ({str(e)[:80]}); cosine fallback", flush=True)
            ok = False
            break
        if not isinstance(out, dict):
            ok = False
            break
        assigned = set()
        for canon, idxs in out.items():
            members = [off + int(i) for i in idxs
                       if isinstance(i, (int, float)) and 0 <= int(i) < len(chunk)]
            members = [m for m in members if m not in assigned]
            if members:
                assigned.update(members)
                clusters.append((canon.strip(), members))
        for i in range(len(chunk)):  # unassigned -> singleton clusters
            if off + i not in assigned:
                clusters.append((chunk[i][2], [off + i]))
    if ok and len(clusters) > 60:
        # merge pass: canonical names were produced per-batch; unify across batches
        names = [c[0] for c in clusters]
        listing = "\n".join(f"{i}. {n}" for i, n in enumerate(names))
        try:
            out = llm.chat_json(prompt.replace("@@GOALS@@", listing))
            if isinstance(out, dict):
                assigned = set()
                merged = []
                for canon, idxs in out.items():
                    members = []
                    for i in idxs:
                        if isinstance(i, (int, float)) and 0 <= int(i) < len(names):
                            members.extend(clusters[int(i)][1])
                            assigned.add(int(i))
                    if members:
                        merged.append((canon.strip(), sorted(set(members))))
                for i, c in enumerate(clusters):
                    if i not in assigned:
                        merged.append(c)
                clusters = merged
        except Exception as e:
            print(f"goal merge pass failed ({str(e)[:80]}); keeping per-batch clusters", flush=True)
    if not ok:
        vecs = embed_texts(texts)
        clusters = []
        for i, v in enumerate(vecs):
            best, best_s = None, thresh
            for c in clusters:
                s = float(np.dot(v, c[0]))
                if s >= best_s:
                    best, best_s = c, s
            if best is None:
                clusters.append((v, texts[i], [i]))
            else:
                best[2].append(i)
        return [(rep, members) for _, rep, members in clusters]
    return clusters


def main():
    goals_only = "--goals-only" in sys.argv
    con = db.connect()
    cur = con.cursor()

    # 1) event embeddings
    events = cur.execute("SELECT * FROM events ORDER BY id").fetchall()
    if not goals_only:
        topics = {}
        for r in cur.execute(
            "SELECT ee.event_id, e.canonical FROM event_entities ee "
            "JOIN entities e ON e.id=ee.entity_id WHERE e.type='TOPIC'"
        ):
            topics.setdefault(r["event_id"], []).append(r["canonical"])
        people = {}
        for r in cur.execute(
            "SELECT ep.event_id, e.canonical FROM event_participants ep "
            "JOIN entities e ON e.id=ep.entity_id"
        ):
            people.setdefault(r["event_id"], []).append(r["canonical"])
        texts = [event_text(ev, topics.get(ev["id"], []), people.get(ev["id"], [])) for ev in events]
        oa = llm.embed(texts)
        if oa is not None:
            norms = np.linalg.norm(oa, axis=1, keepdims=True)
            norms[norms == 0] = 1
            vecs = [v / n for v, n in zip(oa, norms)]
        else:
            vecs, idf = tfidf_fit(texts)
            _save_idf(con, idf)
        for ev, v in zip(events, vecs):
            cur.execute("UPDATE events SET embedding=? WHERE id=?", (blob(v), ev["id"]))
        con.commit()
        print(f"embedded {len(events)} events")

    # 2) goal resolution: cluster GOAL mentions into canonical GOAL entities
    cur.execute("DELETE FROM event_entities WHERE relation='goal'")
    cur.execute("DELETE FROM entity_aliases WHERE entity_id IN (SELECT id FROM entities WHERE type='GOAL')")
    cur.execute("DELETE FROM goal_state")
    cur.execute("DELETE FROM entities WHERE type='GOAL'")
    cur.execute("UPDATE mentions SET payload_json=json_remove(payload_json,'$.goal_entity_id') WHERE type='GOAL'")
    goal_mentions = cur.execute(
        "SELECT id, event_id, text FROM mentions WHERE type='GOAL'"
    ).fetchall()
    items = [(m["id"], m["event_id"], m["text"]) for m in goal_mentions]
    clusters = cluster_goal_strings(items)
    # embed canonical goal strings for later matching
    canon_vecs = embed_texts([c[0] for c in clusters])
    for (rep, members), v in zip(clusters, canon_vecs):
        cur.execute(
            "INSERT OR IGNORE INTO entities(type,canonical,embedding) VALUES('GOAL',?,?)",
            (rep[:160], blob(v)),
        )
        gid = cur.execute(
            "SELECT id FROM entities WHERE type='GOAL' AND canonical=?", (rep[:160],)
        ).fetchone()["id"]
        for mi in members:
            mid, eid, _ = items[mi]
            cur.execute(
                "INSERT OR IGNORE INTO event_entities(event_id,entity_id,relation) VALUES(?,?,'goal')",
                (eid, gid),
            )
            cur.execute(
                "UPDATE mentions SET payload_json=json_set(COALESCE(payload_json,'{}'),'$.goal_entity_id',?) WHERE id=?",
                (gid, mid),
            )
    con.commit()
    print(f"goals: {len(items)} mentions -> {len(clusters)} canonical entities")

    # 3) embed TOPIC + PERSON entity canonical names
    for typ in ("TOPIC", "PERSON"):
        rows = cur.execute(
            "SELECT id, canonical FROM entities WHERE type=? AND embedding IS NULL", (typ,)
        ).fetchall()
        for r, v in zip(rows, embed_texts([r["canonical"] for r in rows])):
            cur.execute("UPDATE entities SET embedding=? WHERE id=?", (blob(v), r["id"]))
    con.commit()
    n = {t: cur.execute("SELECT COUNT(*) c FROM entities WHERE type=?", (t,)).fetchone()["c"]
         for t in ("PERSON", "TOPIC", "GOAL")}
    print("entities:", n)
    con.close()


if __name__ == "__main__":
    main()

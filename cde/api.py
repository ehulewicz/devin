"""Stage 5 — FastAPI serving the graph UI (web/).

Run: uvicorn cde.api:app --port 8000
"""
import json
import math
import os

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from cde import db

app = FastAPI(title="Cognitive Dynamics Engine")
WEB = os.path.join(db.ROOT, "web")


def con():
    return db.connect()


def day_hour(t):
    return int(t // 86400) + 1, (t % 86400) / 3600


@app.get("/api/stats")
def stats():
    c = con()
    out = {"nodes": {}, "edges": {}, "grounding": {}}
    out["nodes"]["utterances"] = c.execute("SELECT COUNT(*) c FROM utterances").fetchone()["c"]
    out["nodes"]["events"] = c.execute("SELECT COUNT(*) c FROM events").fetchone()["c"]
    for r in c.execute("SELECT type, COUNT(*) c FROM entities GROUP BY type"):
        out["nodes"][r["type"].lower() + "s"] = r["c"]
    for r in c.execute("SELECT type, COUNT(*) c FROM mentions GROUP BY type"):
        out["nodes"]["mentions_" + r["type"].lower()] = r["c"]
    for r in c.execute("SELECT relation, status, COUNT(*) c FROM edges GROUP BY relation, status"):
        out["edges"][f"{r['relation']}:{r['status']}"] = r["c"]
    acc = c.execute(
        """SELECT COUNT(*) c FROM edges e WHERE e.status IN ('observed','hypothesis')
           AND e.relation IN ('INFLUENCES','REFERS_TO','SAME_THREAD')"""
    ).fetchone()["c"]
    with_ev = c.execute(
        """SELECT COUNT(DISTINCT e.id) c FROM edges e JOIN edge_evidence ee ON ee.edge_id=e.id
           WHERE e.status IN ('observed','hypothesis')
           AND e.relation IN ('INFLUENCES','REFERS_TO','SAME_THREAD')"""
    ).fetchone()["c"]
    out["grounding"] = {"judged_edges": acc, "with_evidence": with_ev,
                        "pct": round(100 * with_ev / acc, 1) if acc else None}
    out["llm_backend"] = __import__("cde.llm", fromlist=["BACKEND"]).BACKEND
    c.close()
    return out


@app.get("/api/graph")
def graph(subject: str = "A1_JAKE", days: str = "1-7",
          level: str = "event", min_conf: float = 0.0, show_rejected: bool = False):
    c = con()
    lo, hi = 1, 7
    if "-" in days:
        lo, hi = [int(x) for x in days.split("-", 1)]
    elif days:
        lo = hi = int(days)

    evs = c.execute(
        "SELECT * FROM events WHERE subject=? AND day BETWEEN ? AND ? ORDER BY t_start",
        (subject, lo, hi),
    ).fetchall()
    ev_ids = {e["id"] for e in evs}
    nodes = [
        {"id": f"e{e['id']}", "type": "EVENT", "label": e["summary"][:60],
         "day": e["day"], "t": e["t_start"], "hour": (e["t_start"] % 86400) / 3600,
         "mood": e["mood"], "recount": e["time_uncertain"]}
        for e in evs
    ]


    edges = []
    # event-event edges
    q = """SELECT * FROM edges WHERE src_type='EVENT' AND dst_type='EVENT'"""
    for r in c.execute(q):
        if r["src_id"] not in ev_ids or r["dst_id"] not in ev_ids:
            continue
        if r["status"] == "rejected" and not show_rejected:
            continue
        if r["confidence"] is not None and r["confidence"] < min_conf and r["relation"] == "INFLUENCES":
            continue
        edges.append({"id": f"edge-{r['id']}", "src": f"e{r['src_id']}", "dst": f"e{r['dst_id']}",
                      "relation": r["relation"], "status": r["status"],
                      "confidence": r["confidence"], "evidence_type": r["evidence_type"],
                      "rationale": r["rationale"], "score": r["retrieval_score"]})
    # goal nodes + ADVANCES_GOAL
    goal_rows = c.execute(
        """SELECT DISTINCT e.id, e.canonical FROM entities e
           JOIN event_entities ee ON ee.entity_id=e.id AND ee.relation='goal'
           JOIN events ev ON ev.id=ee.event_id
           WHERE e.type='GOAL' AND ev.subject=? AND ev.day BETWEEN ? AND ?""",
        (subject, lo, hi),
    ).fetchall()
    gids = set()
    for g in goal_rows:
        gids.add(g["id"])
        days_ev = c.execute(
            """SELECT ev.day, ev.t_start FROM event_entities ee JOIN events ev ON ev.id=ee.event_id
               WHERE ee.entity_id=? AND ee.relation='goal' ORDER BY ev.t_start""",
            (g["id"],),
        ).fetchall()
        states = c.execute(
            "SELECT day, status FROM goal_state WHERE goal_entity_id=?", (g["id"],)
        ).fetchall()
        nodes.append({"id": f"g{g['id']}", "type": "GOAL", "label": g["canonical"][:60],
                      "days": [d["day"] for d in days_ev],
                      "t": days_ev[0]["t_start"] if days_ev else 0,
                      "hour": 25.5, "states": {s["day"]: s["status"] for s in states}})
    for r in c.execute(
        """SELECT e.* FROM edges e JOIN events s ON s.id=e.src_id
           WHERE e.src_type='EVENT' AND e.dst_type='GOAL' AND s.subject=? AND s.day BETWEEN ? AND ?""",
        (subject, lo, hi),
    ):
        if r["src_id"] in ev_ids and r["dst_id"] in gids:
            edges.append({"id": f"edge-{r['id']}", "src": f"e{r['src_id']}", "dst": f"g{r['dst_id']}",
                          "relation": "ADVANCES_GOAL", "status": r["status"],
                          "confidence": r["confidence"], "evidence_type": r["evidence_type"],
                          "rationale": r["rationale"], "score": r["retrieval_score"]})
    c.close()
    return {"nodes": nodes, "edges": edges}


@app.get("/api/event/{eid}")
def event(eid: int):
    c = con()
    e = c.execute("SELECT * FROM events WHERE id=?", (eid,)).fetchone()
    if not e:
        return {"error": "not found"}
    utts = [dict(r) for r in c.execute(
        """SELECT u.id, u.hms, u.speaker, u.text_en, u.text_zh FROM utterances u
           JOIN event_utterances eu ON eu.utterance_id=u.id
           WHERE eu.event_id=? ORDER BY u.t_abs""", (eid,))]
    mentions = []
    for m in c.execute("SELECT * FROM mentions WHERE event_id=?", (eid,)):
        ev = [r["utterance_id"] for r in c.execute(
            "SELECT utterance_id FROM mention_evidence WHERE mention_id=?", (m["id"],))]
        mentions.append({"id": m["id"], "type": m["type"], "speaker": m["speaker"],
                         "text": m["text"], "payload": json.loads(m["payload_json"] or "{}"),
                         "evidence": ev})
    def edge_view(r):
        ev_q = [u["utterance_id"] for u in c.execute(
            "SELECT utterance_id FROM edge_evidence WHERE edge_id=?", (r["id"],))]
        return {"id": r["id"], "relation": r["relation"], "status": r["status"],
                "other_id": r["dst_id"] if r["src_id"] == eid else r["src_id"],
                "direction": "out" if r["src_id"] == eid else "in",
                "confidence": r["confidence"], "evidence_type": r["evidence_type"],
                "rationale": r["rationale"], "alternatives": json.loads(r["alternatives_json"] or "[]"),
                "retrieval_score": r["retrieval_score"], "evidence_utterances": ev_q}
    out_edges = [edge_view(r) for r in c.execute(
        "SELECT * FROM edges WHERE src_type='EVENT' AND src_id=?", (eid,))]
    in_edges = [edge_view(r) for r in c.execute(
        "SELECT * FROM edges WHERE dst_type='EVENT' AND dst_id=?", (eid,))]
    parts = [r["canonical"] for r in c.execute(
        "SELECT e.canonical FROM event_participants ep JOIN entities e ON e.id=ep.entity_id "
        "WHERE ep.event_id=?", (eid,))]
    topics = [r["canonical"] for r in c.execute(
        "SELECT e.canonical FROM event_entities ee JOIN entities e ON e.id=ee.entity_id "
        "WHERE ee.event_id=? AND ee.relation='topic'", (eid,))]
    goals = [{"id": r["id"], "canonical": r["canonical"]} for r in c.execute(
        "SELECT e.id, e.canonical FROM event_entities ee JOIN entities e ON e.id=ee.entity_id "
        "WHERE ee.event_id=? AND ee.relation='goal'", (eid,))]
    c.close()
    return {"id": eid, "day": e["day"], "t_start": e["t_start"], "t_end": e["t_end"],
            "hms": f"{int(e['t_start']%86400//3600):02d}:{int(e['t_start']%3600//60):02d}",
            "summary": e["summary"], "mood": e["mood"],
            "occurred_at": e["occurred_at"], "recorded_at": e["recorded_at"],
            "time_uncertain": e["time_uncertain"], "participants": parts,
            "topics": topics, "goals": goals, "utterances": utts,
            "mentions": mentions, "edges_out": out_edges, "edges_in": in_edges}


@app.get("/api/goal/{gid}/thread")
def goal_thread(gid: int):
    c = con()
    g = c.execute("SELECT * FROM entities WHERE id=?", (gid,)).fetchone()
    if not g:
        return {"error": "not found"}
    evs = [dict(r) for r in c.execute(
        """SELECT ev.id, ev.day, ev.t_start, ev.summary, ev.mood FROM events ev
           JOIN event_entities ee ON ee.event_id=ev.id AND ee.relation='goal'
           WHERE ee.entity_id=? ORDER BY ev.t_start""", (gid,))]
    states = [dict(r) for r in c.execute(
        "SELECT day, status, event_id FROM goal_state WHERE goal_entity_id=? ORDER BY day",
        (gid,))]
    c.close()
    return {"id": gid, "canonical": g["canonical"], "events": evs, "states": states}


@app.get("/api/chain")
def chain(event_id: int = Query(...), max_hops: int = 30):
    """Backward traversal over INFLUENCES/REFERS_TO/SAME_THREAD to earliest root."""
    c = con()
    hops = []
    seen = set()
    cur_id = event_id
    for _ in range(max_hops):
        r = c.execute(
            """SELECT e.*, s.t_start st, s.summary ss, s.day sd,
                      (SELECT COUNT(*) FROM edge_evidence x WHERE x.edge_id=e.id) nev
               FROM edges e
               JOIN events s ON s.id=e.src_id
               WHERE e.dst_type='EVENT' AND e.dst_id=? AND e.src_type='EVENT'
               AND e.relation IN ('INFLUENCES','REFERS_TO','SAME_THREAD')
               AND e.status!='rejected'
               ORDER BY e.relation IN ('INFLUENCES','REFERS_TO') DESC,
                        nev > 0 DESC, e.confidence DESC, s.t_start
               LIMIT 1""",
            (cur_id,),
        ).fetchone()
        if not r or r["src_id"] in seen:
            break
        seen.add(r["src_id"])
        ev = [u["utterance_id"] for u in c.execute(
            "SELECT utterance_id FROM edge_evidence WHERE edge_id=?", (r["id"],))]
        quotes = [dict(u) for u in c.execute(
            f"SELECT id, hms, speaker, text_en, text_zh FROM utterances WHERE id IN "
            f"({','.join('?'*len(ev))})", ev)] if ev else []
        hops.append({"edge_id": r["id"], "relation": r["relation"], "status": r["status"],
                     "confidence": r["confidence"], "rationale": r["rationale"],
                     "event_id": r["src_id"], "day": r["sd"], "t_start": r["st"],
                     "summary": r["ss"], "quotes": quotes})
        cur_id = r["src_id"]
    end = c.execute("SELECT id, day, t_start, summary FROM events WHERE id=?",
                    (event_id,)).fetchone()
    c.close()
    return {"target": dict(end) if end else None, "chain": hops}


@app.get("/api/search")
def search(q: str, subject: str = "A1_JAKE", k: int = 12):
    import numpy as np
    from cde.embed import embed_query, unblob
    c = con()
    evs = [dict(r) for r in c.execute(
        "SELECT id, day, t_start, summary, mood, embedding FROM events WHERE subject=?",
        (subject,)) if r["embedding"]]
    v = embed_query([q], c)[0]
    scored = sorted(
        ((float(np.dot(v, unblob(e["embedding"]))), e) for e in evs),
        key=lambda x: -x[0])[:k]
    c.close()
    return [{"score": round(s, 3), "id": e["id"], "day": e["day"],
             "t_start": e["t_start"], "summary": e["summary"], "mood": e["mood"]}
            for s, e in scored]


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"))


app.mount("/web", StaticFiles(directory=WEB), name="web")

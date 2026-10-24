"""Stage 5 — FastAPI serving the graph UI (web/).

Run: uvicorn cde.api:app --port 8000
"""
import json
import math
import os
import sqlite3

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
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
    try:
        ev_theme = {r["event_id"]: r["theme_id"] for r in
                    c.execute("SELECT event_id, theme_id FROM event_themes")}
    except sqlite3.OperationalError:
        ev_theme = {}
    nodes = [
        {"id": f"e{e['id']}", "type": "EVENT", "label": e["summary"][:60],
         "day": e["day"], "t": e["t_start"], "hour": (e["t_start"] % 86400) / 3600,
         "mood": e["mood"], "recount": e["time_uncertain"],
         "theme": ev_theme.get(e["id"])}
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


@app.get("/api/people")
def people(subject: str = "A1_JAKE"):
    """Person entities with event counts + flagged pattern counts."""
    c = con()
    rows = c.execute(
        """SELECT e.id, e.canonical,
                  (SELECT COUNT(*) FROM event_participants ep
                   WHERE ep.entity_id=e.id) n_events,
                  (SELECT COUNT(*) FROM edges x WHERE x.relation='INTERACTION_PATTERN'
                   AND x.status!='rejected' AND x.src_type='PERSON' AND x.src_id=e.id) out_p,
                  (SELECT COUNT(*) FROM edges x WHERE x.relation='INTERACTION_PATTERN'
                   AND x.status!='rejected' AND x.dst_type='PERSON' AND x.dst_id=e.id) in_p
           FROM entities e WHERE e.type='PERSON' ORDER BY n_events DESC""",
    ).fetchall()
    c.close()
    return [{"id": r["id"], "name": r["canonical"], "events": r["n_events"],
             "patterns_out": r["out_p"], "patterns_in": r["in_p"]} for r in rows]


@app.get("/api/person/{pid}/patterns")
def person_patterns(pid: int):
    """Flagged INTERACTION_PATTERN edges in/out of one person, with per-day stats."""
    c = con()
    p = c.execute("SELECT id, canonical FROM entities WHERE id=? AND type='PERSON'",
                  (pid,)).fetchone()
    if not p:
        return {"error": "not found"}
    names = {r["id"]: r["canonical"] for r in c.execute(
        "SELECT id, canonical FROM entities")}
    out = []
    for r in c.execute(
        """SELECT * FROM edges WHERE relation='INTERACTION_PATTERN'
           AND status!='rejected' AND src_type='PERSON' AND dst_type='PERSON'
           AND (src_id=? OR dst_id=?) ORDER BY retrieval_score DESC""",
        (pid, pid),
    ):
        days = [{"day": d["day"], "shared": d["shared"], "behaved": d["behaved"],
                 "events": json.loads(d["events_json"] or "[]")}
                for d in c.execute(
                    "SELECT day, shared, behaved, events_json FROM pair_behavior "
                    "WHERE edge_id=? ORDER BY day", (r["id"],))]
        ev_ids = [u["utterance_id"] for u in c.execute(
            "SELECT utterance_id FROM edge_evidence WHERE edge_id=?", (r["id"],))]
        quotes = [dict(u) for u in c.execute(
            "SELECT id, hms, speaker, text_en, text_zh FROM utterances "
            f"WHERE id IN ({','.join('?'*len(ev_ids))}) ORDER BY t_abs", ev_ids)] if ev_ids else []
        out.append({
            "edge_id": r["id"], "direction": "out" if r["src_id"] == pid else "in",
            "other": names.get(r["dst_id"] if r["src_id"] == pid else r["src_id"]),
            "other_id": r["dst_id"] if r["src_id"] == pid else r["src_id"],
            "status": r["status"], "evidence_type": r["evidence_type"],
            "z": r["retrieval_score"], "rationale": r["rationale"],
            "days": days, "quotes": quotes})
    c.close()
    return {"id": pid, "name": p["canonical"], "patterns": out}


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


# ---------- semantic zoom: themes -> causal -> full graph ----------

def _theme_maps(c):
    """event_id -> theme_id, theme_id -> row ({} if themes not built yet)."""
    try:
        et = {r["event_id"]: r["theme_id"]
              for r in c.execute("SELECT event_id, theme_id FROM event_themes")}
        th = {r["id"]: dict(r) for r in c.execute("SELECT * FROM themes")}
    except sqlite3.OperationalError:
        et, th = {}, {}
    return et, th


def _judged_edges(c):
    return c.execute(
        """SELECT * FROM edges WHERE src_type='EVENT' AND dst_type='EVENT'
           AND relation IN ('INFLUENCES','REFERS_TO','SAME_THREAD')
           AND status!='rejected'""").fetchall()


def _theme_card(c, t, evs, jedges):
    ids = [e["id"] for e in evs]
    idset = set(ids)
    ts = sorted(e["t_start"] for e in evs)
    n = len(ts)
    card = {"id": t["id"], "name": t["name"], "description": t["description"],
            "kind": t.get("kind"),
            "n_events": n, "days": sorted({e["day"] for e in evs}),
            "t_first": ts[0] if n else None, "t_last": ts[-1] if n else None,
            "t_median": ((ts[n // 2] if n % 2 else (ts[n // 2 - 1] + ts[n // 2]) / 2)
                         if n else None)}
    ph = ",".join("?" * len(ids))
    card["top_people"] = [[r["canonical"], r["n"]] for r in c.execute(
        f"""SELECT e.canonical, COUNT(*) n FROM event_participants ep
            JOIN entities e ON e.id=ep.entity_id
            WHERE ep.event_id IN ({ph})
            GROUP BY e.id ORDER BY n DESC LIMIT 3""", ids)] if ids else []
    card["top_goals"] = [[r["id"], r["canonical"], r["n"]] for r in c.execute(
        f"""SELECT e.id, e.canonical, COUNT(*) n FROM event_entities ee
            JOIN entities e ON e.id=ee.entity_id
            WHERE ee.relation='goal' AND ee.event_id IN ({ph})
            GROUP BY e.id ORDER BY n DESC LIMIT 3""", ids)] if ids else []
    card["moods"] = [[r["mood"], r["n"]] for r in c.execute(
        f"""SELECT mood, COUNT(*) n FROM events
            WHERE id IN ({ph}) AND mood IS NOT NULL
            GROUP BY mood ORDER BY n DESC LIMIT 2""", ids)] if ids else []
    card["n_causal"] = sum(
        1 for r in jedges
        if r["relation"] in ("INFLUENCES", "REFERS_TO")
        and r["src_id"] in idset and r["dst_id"] in idset)
    return card


@app.get("/api/zoom/themes")
def zoom_themes(subject: str = "A1_JAKE"):
    c = con()
    et, themes = _theme_maps(c)
    evs = {e["id"]: e for e in c.execute(
        "SELECT * FROM events WHERE subject=?", (subject,))}
    by_theme = {}
    for eid, tid in et.items():
        if eid in evs:
            by_theme.setdefault(tid, []).append(evs[eid])
    jedges = _judged_edges(c)
    out = [_theme_card(c, t, by_theme.get(tid, []), jedges)
           for tid, t in sorted(
               themes.items(),
               key=lambda kv: kv[1]["rank"] if kv[1]["rank"] is not None else 1e9)]
    links = {}
    for r in jedges:
        s, d = et.get(r["src_id"]), et.get(r["dst_id"])
        if s is None or d is None or s == d:
            continue
        l = links.setdefault((s, d), {
            "src": s, "dst": d, "n": 0,
            "by_relation": {"INFLUENCES": 0, "REFERS_TO": 0, "SAME_THREAD": 0}})
        l["n"] += 1
        l["by_relation"][r["relation"]] += 1
    c.close()
    return {"themes": out, "links": list(links.values())}


@app.get("/api/zoom/theme/{tid}")
def zoom_theme(tid: int, subject: str = "A1_JAKE"):
    c = con()
    et, themes = _theme_maps(c)
    t = themes.get(tid)
    if not t:
        c.close()
        return {"error": "not found"}
    evs = [e for e in c.execute(
        "SELECT * FROM events WHERE subject=? ORDER BY t_start", (subject,))
        if et.get(e["id"]) == tid]
    ev_ids = {e["id"] for e in evs}
    jedges = _judged_edges(c)
    card = _theme_card(c, t, evs, jedges)
    counts = {}
    for t2 in et.values():
        counts[t2] = counts.get(t2, 0) + 1
    nodes = [{"id": f"e{e['id']}", "type": "EVENT",
              "label": (e["summary"] or "")[:60], "day": e["day"],
              "t": e["t_start"], "hour": (e["t_start"] % 86400) / 3600,
              "mood": e["mood"], "recount": e["time_uncertain"], "theme": tid}
             for e in evs]
    edges = []
    cross = {}
    connected = set()
    for r in jedges:
        s_in, d_in = r["src_id"] in ev_ids, r["dst_id"] in ev_ids
        if s_in and d_in:
            edges.append({"id": f"edge-{r['id']}", "src": f"e{r['src_id']}",
                          "dst": f"e{r['dst_id']}", "relation": r["relation"],
                          "status": r["status"], "confidence": r["confidence"],
                          "evidence_type": r["evidence_type"],
                          "rationale": r["rationale"], "score": r["retrieval_score"]})
        elif s_in != d_in:
            other = et.get(r["dst_id"] if s_in else r["src_id"])
            if other is None:
                continue
            connected.add(other)
            key = (r["src_id"] if s_in else r["dst_id"], other, s_in)
            cross[key] = cross.get(key, 0) + 1
    for (eid, ot, out), k in sorted(cross.items()):
        edges.append({"id": f"b-e{eid}-t{ot}" + ("" if out else "-in"),
                      "src": f"e{eid}" if out else f"t{ot}",
                      "dst": f"t{ot}" if out else f"e{eid}",
                      "relation": "CROSS_THEME", "n": k})
    for ot in sorted(connected):
        nodes.append({"id": f"t{ot}", "type": "THEME",
                      "label": themes[ot]["name"] if ot in themes else str(ot),
                      "n_events": counts.get(ot, 0)})
    conn = {}
    for (eid, ot, out), k in cross.items():
        cc = conn.setdefault(ot, {"n_out": 0, "n_in": 0})
        cc["n_out" if out else "n_in"] += k
    connected_list = sorted(
        ({"id": ot,
          "name": themes[ot]["name"] if ot in themes else str(ot),
          "kind": themes[ot].get("kind") if ot in themes else None,
          "n_out": v["n_out"], "n_in": v["n_in"]}
         for ot, v in conn.items()),
        key=lambda x: -(x["n_out"] + x["n_in"]))
    # insights that mention any of this theme's top non-Jake people
    related = []
    try:
        names = [p[0] for p in card["top_people"] if p[0] != "Jake"]
        if names:
            for r in c.execute(
                """SELECT id, title, lens, body FROM insights
                   WHERE lens IN ('around_you','you_around','portrait')
                   ORDER BY rank"""):
                txt = f"{r['title'] or ''} {r['body'] or ''}".lower()
                if any(nm.lower() in txt for nm in names):
                    related.append({"id": r["id"], "title": r["title"],
                                    "lens": r["lens"]})
                    if len(related) >= 6:
                        break
    except sqlite3.OperationalError:
        pass
    # spine: longest intra-theme path over judged edges, DP in t_start order
    pos = {e["id"]: i for i, e in enumerate(evs)}
    preds = {}
    for r in jedges:
        if (r["src_id"] in pos and r["dst_id"] in pos
                and pos[r["src_id"]] < pos[r["dst_id"]]):
            preds.setdefault(r["dst_id"], []).append(r["src_id"])
    dp, par = {}, {}
    for e in evs:
        i = e["id"]
        best, bp = 0, None
        for p in preds.get(i, []):
            if dp[p] > best:
                best, bp = dp[p], p
        dp[i] = best + 1
        par[i] = bp
    spine = []
    if dp:
        i = max(dp, key=dp.get)
        while i is not None:
            spine.append(i)
            i = par[i]
        spine.reverse()
    c.close()
    return {"theme": card, "nodes": nodes, "edges": edges, "spine": spine,
            "connected": connected_list, "related_insights": related}


# ---------- insights ----------

@app.get("/api/insights")
def insights():
    c = con()
    try:
        rows = c.execute("SELECT * FROM insights ORDER BY rank").fetchall()
    except sqlite3.OperationalError:
        c.close()
        return []
    out = [{"id": r["id"], "rank": r["rank"], "lens": r["lens"],
            "title": r["title"], "body": r["body"], "status": r["status"],
            "stat": json.loads(r["stat_json"] or "{}"),
            "evidence": json.loads(r["evidence_json"] or "{}")} for r in rows]
    c.close()
    return out


@app.get("/api/insight/{iid}")
def insight(iid: int):
    c = con()
    try:
        r = c.execute("SELECT * FROM insights WHERE id=?", (iid,)).fetchone()
    except sqlite3.OperationalError:
        r = None
    if not r:
        c.close()
        return {"error": "not found"}
    ev = json.loads(r["evidence_json"] or "{}")
    ph = lambda xs: ",".join("?" * len(xs))  # noqa: E731
    uids = ev.get("utterance_ids") or []
    quotes = [dict(x) for x in c.execute(
        f"""SELECT id, day, hms, speaker, text_en, text_zh FROM utterances
            WHERE id IN ({ph(uids)}) ORDER BY t_abs LIMIT 12""", uids)] if uids else []
    eids = ev.get("event_ids") or []
    events = [{"id": x["id"], "day": x["day"],
               "hms": f"{int(x['t_start'] % 86400 // 3600):02d}:"
                      f"{int(x['t_start'] % 3600 // 60):02d}",
               "summary": x["summary"]}
              for x in c.execute(
                  f"SELECT id, day, t_start, summary FROM events "
                  f"WHERE id IN ({ph(eids)})", eids)] if eids else []
    kedge = ev.get("edge_ids") or []
    edges = []
    esum = {}
    rows = c.execute(f"SELECT * FROM edges WHERE id IN ({ph(kedge)})",
                     kedge).fetchall() if kedge else []
    for x in rows:
        for i in (x["src_id"], x["dst_id"]):
            if i not in esum:
                s = c.execute("SELECT summary FROM events WHERE id=?", (i,)).fetchone()
                esum[i] = s["summary"] if s else None
        edges.append({"id": x["id"], "relation": x["relation"],
                      "confidence": x["confidence"],
                      "src": {"id": x["src_id"], "summary": esum[x["src_id"]]},
                      "dst": {"id": x["dst_id"], "summary": esum[x["dst_id"]]}})
    gids = ev.get("goal_ids") or []
    goals = [dict(x) for x in c.execute(
        f"SELECT id, canonical FROM entities WHERE id IN ({ph(gids)})",
        gids)] if gids else []
    c.close()
    return {"id": r["id"], "rank": r["rank"], "lens": r["lens"],
            "title": r["title"], "body": r["body"], "status": r["status"],
            "stat": json.loads(r["stat_json"] or "{}"), "evidence": ev,
            "quotes": quotes, "events": events, "edges": edges, "goals": goals}


@app.get("/api/audio/status")
def audio_status():
    from cde import audio
    return audio.status()


@app.get("/api/audio/event/{eid}")
def audio_event(eid: int):
    """Playlist for sequential playback of an event's utterance clips."""
    c = con()
    rows = [dict(r) for r in c.execute(
        """SELECT u.id, u.hms, u.speaker, u.text_en, u.t_abs, u.t_end
           FROM utterances u JOIN event_utterances eu ON eu.utterance_id=u.id
           WHERE eu.event_id=? ORDER BY u.t_abs""", (eid,))]
    c.close()
    return {"event_id": eid,
            "clips": [{"uid": r["id"], "speaker": r["speaker"], "hms": r["hms"],
                       "text": r["text_en"],
                       "dur": round((r["t_end"] or r["t_abs"]) - r["t_abs"] + 0.8, 2)}
                      for r in rows]}


@app.get("/api/audio/{uid}")
def audio_clip(uid: int):
    from cde import audio
    c = con()
    path = audio.clip_for_utterance(c, uid)
    c.close()
    if not path:
        return JSONResponse({"error": "no media for this utterance"}, status_code=404)
    return FileResponse(path, media_type="audio/aac")


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"))


app.mount("/web", StaticFiles(directory=WEB), name="web")

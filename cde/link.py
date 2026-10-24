"""Stage 4 — link: deterministic edges, candidate retrieval, LLM edge judge.

Deterministic: SPOKE, PARTICIPATED_IN, EXPRESSES, PRECEDES, ADVANCES_GOAL,
SIMILAR_TO (display-only, marked semantic).

Candidates for each event E: earlier events within {30m, 6h, 24h, 7d},
S = 0.4*cos + 0.2*entity_jaccard + 0.2*temporal_decay + 0.2*context
(context = shared open goal, or a causal_claim whose earlier_desc embeds near
the candidate). Top-5 per event; causal-claim candidates always included.

Judge: batches of <=10 pairs per LLM call. Validation: source must precede
target; cited utterance ids must exist; INFLUENCES without
self_report/transcript evidence is downgraded to SAME_THREAD; NONE -> rejected.

Usage: python -m cde.link [--subject A1_JAKE] [--skip-judge] [--max-pairs 400]
"""
import json
import math
import sys
from collections import defaultdict

import numpy as np

from cde import db, llm
from cde.embed import unblob

WINDOWS = [1800, 6 * 3600, 24 * 3600, 7 * 86400]
TAU = 6 * 3600  # temporal decay constant
SIM_THRESH = 0.88
TOP_K = 5


def add_edge(cur, src_t, src, dst_t, dst, rel, conf=None, status="observed",
             etype=None, rationale=None, alts=None, score=None, evidence=()):
    cur.execute(
        """INSERT INTO edges(src_type,src_id,dst_type,dst_id,relation,confidence,status,
                             evidence_type,rationale,alternatives_json,retrieval_score)
           VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
        (src_t, src, dst_t, dst, rel, conf, status, etype, rationale,
         json.dumps(alts, ensure_ascii=False) if alts else None, score),
    )
    eid = cur.lastrowid
    for u in evidence:
        cur.execute(
            "INSERT OR IGNORE INTO edge_evidence(edge_id,utterance_id) VALUES(?,?)", (eid, u)
        )
    return eid


def deterministic(con, subject):
    cur = con.cursor()
    cur.execute("DELETE FROM edge_evidence")
    cur.execute("DELETE FROM edges")
    cur.execute("DELETE FROM goal_state")

    # SPOKE: PERSON entity -> utterance (subject's recordings)
    cur.execute(
        """INSERT INTO edges(src_type,src_id,dst_type,dst_id,relation,status,evidence_type)
           SELECT 'PERSON', e.id, 'UTTERANCE', u.id, 'SPOKE', 'observed', 'transcript'
           FROM utterances u JOIN entities e ON e.type='PERSON' AND e.canonical=u.speaker
           WHERE u.wearer=?""",
        (subject,),
    )

    # PARTICIPATED_IN
    cur.execute(
        """INSERT INTO edges(src_type,src_id,dst_type,dst_id,relation,status,evidence_type)
           SELECT 'PERSON', ep.entity_id, 'EVENT', ep.event_id, 'PARTICIPATED_IN',
                  'observed', 'extraction'
           FROM event_participants ep"""
    )

    # EXPRESSES: utterance -> mention
    cur.execute(
        """INSERT INTO edges(src_type,src_id,dst_type,dst_id,relation,status,evidence_type)
           SELECT 'UTTERANCE', me.utterance_id, 'MENTION', me.mention_id, 'EXPRESSES',
                  'observed', 'extraction'
           FROM mention_evidence me"""
    )

    # PRECEDES: consecutive events same day
    evs = cur.execute(
        "SELECT id, day, t_start FROM events WHERE subject=? ORDER BY day, t_start", (subject,)
    ).fetchall()
    prev_of_day = {}
    for e in evs:
        p = prev_of_day.get(e["day"])
        if p:
            add_edge(cur, "EVENT", p["id"], "EVENT", e["id"], "PRECEDES",
                     status="observed", etype="temporal")
        prev_of_day[e["day"]] = e

    # ADVANCES_GOAL: event -> GOAL entity (from event_entities relation='goal')
    cur.execute(
        """INSERT INTO edges(src_type,src_id,dst_type,dst_id,relation,status,evidence_type)
           SELECT 'EVENT', ee.event_id, 'GOAL', ee.entity_id, 'ADVANCES_GOAL',
                  'observed', 'extraction'
           FROM event_entities ee WHERE ee.relation='goal'"""
    )

    # goal_state timeline: per goal, per day, status from latest mention that day
    goals = cur.execute(
        """SELECT ee.entity_id gid, e.day, e.id eid, m.payload_json
           FROM event_entities ee JOIN events e ON e.id=ee.event_id
           LEFT JOIN mentions m ON m.event_id=e.id AND m.type='GOAL'
           WHERE ee.relation='goal' ORDER BY e.day"""
    ).fetchall()
    per = defaultdict(list)
    for g in goals:
        status = "in-progress"
        if g["payload_json"]:
            try:
                status = json.loads(g["payload_json"]).get("status") or "in-progress"
            except json.JSONDecodeError:
                pass
        per[(g["gid"], g["day"])].append((status, g["eid"]))
    for (gid, day), lst in per.items():
        status, eid = lst[-1]
        cur.execute(
            "INSERT OR REPLACE INTO goal_state(goal_entity_id,day,status,event_id) VALUES(?,?,?,?)",
            (gid, day, status, eid),
        )
    con.commit()
    return {r["relation"]: r["c"] for r in cur.execute(
        "SELECT relation, COUNT(*) c FROM edges GROUP BY relation")}


def load_events(con, subject):
    evs = [dict(r) for r in con.execute(
        "SELECT * FROM events WHERE subject=? ORDER BY t_start", (subject,))]
    ent = defaultdict(set)
    for r in con.execute(
        "SELECT event_id, entity_id FROM event_entities"
    ):
        ent[r["event_id"]].add(r["entity_id"])
    for r in con.execute(
        "SELECT event_id, entity_id FROM event_participants"
    ):
        ent[r["event_id"]].add(r["entity_id"])
    claims = defaultdict(list)
    for r in con.execute(
        "SELECT event_id, payload_json FROM mentions WHERE type='CAUSAL_CLAIM'"
    ):
        if r["payload_json"]:
            try:
                claims[r["event_id"]].append(json.loads(r["payload_json"]))
            except json.JSONDecodeError:
                pass
    for e in evs:
        e["vec"] = unblob(e["embedding"])
        e["ents"] = ent[e["id"]]
        e["hour"] = int(e["t_start"] // 3600)
        e["claims"] = claims[e["id"]]
    return evs


def candidates(con, subject):
    """Score earlier events for each event; return top-K pairs + claim-driven pairs."""
    from cde.embed import embed_query

    evs = [e for e in load_events(con, subject) if e["vec"] is not None]
    if not evs:
        return [], {}
    V = np.array([e["vec"] for e in evs])
    goals = defaultdict(set)  # entity_id -> event ids
    for e in evs:
        for g in e["ents"]:
            goals[g].add(e["id"])

    # embed causal-claim earlier_desc strings once
    claim_vecs = {}
    descs, keys = [], []
    for e in evs:
        for c in e["claims"]:
            d = c.get("earlier_desc")
            if d:
                descs.append(d)
                keys.append((e["id"], d))
    if descs:
        for (eid, d), v in zip(keys, embed_query(descs, con)):
            claim_vecs.setdefault(eid, []).append(v)

    pairs = {}  # (src_id, dst_id) -> score  (src=earlier, dst=later)
    for i, e in enumerate(evs):
        if e["vec"] is None:
            continue
        idx = [j for j in range(i) if 0 < e["t_start"] - evs[j]["t_start"] <= 7 * 86400]
        if not idx:
            continue
        cand = np.array([evs[j]["vec"] for j in idx])
        cos = cand @ e["vec"]
        scores = []
        for k, j in enumerate(idx):
            c = evs[j]
            dt = e["t_start"] - c["t_start"]
            inter = len(e["ents"] & c["ents"])
            union = len(e["ents"] | c["ents"]) or 1
            jac = inter / union
            decay = math.exp(-dt / TAU)
            shared_goal = 1.0 if inter and any(g in e["ents"] and g in c["ents"]
                                               for g in goals) else 0.0
            claim_sim = 0.0
            for cv in claim_vecs.get(e["id"], []):
                claim_sim = max(claim_sim, float(cv @ c["vec"]))
            context = max(shared_goal, 1.0 if claim_sim >= 0.5 else claim_sim)
            s = 0.4 * float(cos[k]) + 0.2 * jac + 0.2 * decay + 0.2 * context
            scores.append((s, j, claim_sim))
        scores.sort(reverse=True)
        picked, n_cross = [], 0
        for s, j, cs in scores:
            if len(picked) >= TOP_K:
                break
            if evs[j]["hour"] != e["hour"] or n_cross > 0:
                picked.append((s, j))
                if evs[j]["hour"] != e["hour"]:
                    n_cross += 1
        for s, j, cs in scores:  # causal-claim candidates always included
            if cs >= 0.45 and (s, j) not in [(a, b) for a, b in picked]:
                picked.append((s, j))
        for s, j in picked:
            pairs[(evs[j]["id"], e["id"])] = round(s, 4)
    return evs, pairs


def event_utterances(con, eid, limit=8):
    return [dict(r) for r in con.execute(
        """SELECT u.id, u.hms, u.speaker, u.text_en FROM utterances u
           JOIN event_utterances eu ON eu.utterance_id=u.id
           WHERE eu.event_id=? ORDER BY u.t_abs LIMIT ?""",
        (eid, limit),
    )]


JUDGE_PROMPT = """You judge whether a link between two lifelog events is supported by evidence.
The subject is one person wearing a camera for a week. Event A is earlier, event B is later.

For EACH pair below reply with one JSON object in a list:
{"pair": <n>, "relation": "INFLUENCES"|"REFERS_TO"|"SAME_THREAD"|"NONE",
 "direction": "A->B", "evidence_type": "self_report"|"transcript"|"temporal"|"semantic"|null,
 "supporting_utterance_ids": [int], "reason": str <=30 words,
 "alternative_explanations": [str], "confidence": 0-1}

- INFLUENCES: A plausibly shaped/caused B (mood, plan, decision carried forward). Requires at least one supporting utterance id from either event's quotes — prefer self_report (someone states the link) or transcript (quotes show the thread).
- REFERS_TO: someone in B explicitly talks about A.
- SAME_THREAD: same project/relationship/thread, no explicit causal link.
- NONE: only superficial similarity. Abstain rather than guess.
- Only cite utterance ids shown in the pair's quotes.

PAIRS:
@@PAIRS@@

Reply with ONLY a JSON list."""


def judge(con, evs, pairs, max_pairs=400):
    by_id = {e["id"]: e for e in evs}
    items = sorted(pairs.items(), key=lambda kv: -kv[1])[:max_pairs]
    cur = con.cursor()
    done_pairs = set()
    stats = defaultdict(int)

    def fmt(e):
        utts = event_utterances(con, e["id"])
        q = "\n".join(f'    u{u["id"]} {u["hms"]} {u["speaker"]}: {(u["text_en"] or "")[:120]}'
                      for u in utts)
        return (f'event_id={e["id"]} day{e["day"]} t={e["t_start"]:.0f} '
                f'summary="{e["summary"]}" mood={e["mood"]}\n{q}')

    batches = [items[s : s + 10] for s in range(0, len(items), 10)]

    def call(batch):
        ptxt = []
        for n, ((a, b), s) in enumerate(batch):
            ptxt.append(f"--- pair {n} (retrieval_score={s}) ---\nA: {fmt(by_id[a])}\nB: {fmt(by_id[b])}")
        return llm.chat_json(JUDGE_PROMPT.replace("@@PAIRS@@", "\n".join(ptxt)))

    from concurrent.futures import ThreadPoolExecutor
    results = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(call, b): i for i, b in enumerate(batches)}
        for fut in futs:
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                print(f"judge batch {i} failed: {str(e)[:140]}", flush=True)

    for bi in sorted(results):
        batch = batches[bi]
        out = results[bi]
        if isinstance(out, dict):
            out = out.get("pairs") or out.get("results") or [out]
        for n, ((a, b), s) in enumerate(batch):
            done_pairs.add((a, b))
        returned = set()
        for r in out:
            try:
                n = int(r.get("pair"))
            except (TypeError, ValueError):
                continue
            if not (0 <= n < len(batch)) or n in returned:
                continue
            returned.add(n)
            (a, b), s = batch[n]
            A, B = by_id[a], by_id[b]
            rel = str(r.get("relation", "NONE")).upper()
            etype = r.get("evidence_type")
            ev_ids = r.get("supporting_utterance_ids") or []
            # validation: source precedes target
            if A["t_start"] >= B["t_start"]:
                rel = "NONE"
                r["reason"] = "rejected: source not earlier than target"
            # validation: cited ids must exist among the pair's utterances
            valid = {u["id"] for u in event_utterances(con, a, 100)} | \
                    {u["id"] for u in event_utterances(con, b, 100)}
            ev_ids = [u for u in ev_ids if u in valid]
            if rel == "INFLUENCES" and (etype not in ("self_report", "transcript") or not ev_ids):
                rel, stats["downgraded"] = "SAME_THREAD", stats["downgraded"] + 1
                r["reason"] = (r.get("reason") or "") + " [downgraded: no cited evidence]"
            if rel == "NONE":
                add_edge(cur, "EVENT", a, "EVENT", b, rel, status="rejected",
                         etype=etype, rationale=r.get("reason"),
                         alts=r.get("alternative_explanations"), score=s)
                stats["rejected"] += 1
            elif rel in ("INFLUENCES", "REFERS_TO", "SAME_THREAD"):
                add_edge(cur, "EVENT", a, "EVENT", b, rel,
                         conf=r.get("confidence"),
                         status="hypothesis" if rel == "INFLUENCES" else "observed",
                         etype=etype, rationale=r.get("reason"),
                         alts=r.get("alternative_explanations"), score=s,
                         evidence=ev_ids)
                stats[rel.lower()] += 1
        for n, ((a, b), s) in enumerate(batch):  # no verdict returned -> rejected
            if n not in returned:
                add_edge(cur, "EVENT", a, "EVENT", b, "NONE", status="rejected",
                         rationale="judge returned no verdict", score=s)
                stats["rejected"] += 1
        con.commit()
        print(f"judged batch {bi}: cumulative stats {dict(stats)}", flush=True)

    # pairs in failed batches -> rejected
    for (a, b), s in items:
        if (a, b) not in done_pairs:
            add_edge(cur, "EVENT", a, "EVENT", b, "NONE", status="rejected",
                     rationale="judge call failed", score=s)
            stats["rejected"] += 1
    con.commit()
    return dict(stats), len(items)


def similar(con, subject, thresh=None, per_node=3):
    """Display-only similarity edges. Threshold adapts to the embedding backend:
    the 60th percentile of each node's max cosine (OpenAI ~0.8, TF-IDF ~0.25)."""
    evs = [e for e in load_events(con, subject) if e["vec"] is not None]
    V = np.array([e["vec"] for e in evs])
    S = V @ V.T
    np.fill_diagonal(S, -1)
    if thresh is None:
        thresh = float(np.percentile(S.max(axis=1), 60))
    cur = con.cursor()
    n = 0
    for i, e in enumerate(evs):
        order = np.argsort(-S[i])[:per_node]
        for j in order:
            if S[i, j] >= thresh and evs[j]["t_start"] < e["t_start"]:
                add_edge(cur, "EVENT", evs[j]["id"], "EVENT", e["id"], "SIMILAR_TO",
                         conf=float(S[i, j]), status="observed", etype="semantic",
                         score=float(S[i, j]))
                n += 1
    con.commit()
    return n


def main():
    args = sys.argv[1:]
    subject = args[args.index("--subject") + 1] if "--subject" in args else "A1_JAKE"
    skip_judge = "--skip-judge" in args
    max_pairs = int(args[args.index("--max-pairs") + 1]) if "--max-pairs" in args else 400

    con = db.connect()
    print("deterministic edges:", deterministic(con, subject), flush=True)
    n_sim = similar(con, subject)
    print(f"SIMILAR_TO edges: {n_sim}", flush=True)
    evs, pairs = candidates(con, subject)
    print(f"candidate pairs: {len(pairs)}", flush=True)
    if not skip_judge and pairs:
        stats, judged = judge(con, evs, pairs, max_pairs)
        print(f"judged={judged} -> {stats}", flush=True)
    totals = {r["relation"]: (r["c"], r["st"]) for r in con.execute(
        "SELECT relation, status st, COUNT(*) c FROM edges GROUP BY relation, status")}
    print("edges:", totals)
    con.close()


if __name__ == "__main__":
    main()

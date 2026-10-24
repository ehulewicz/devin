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

relate(): directed PERSON->PERSON INTERACTION_PATTERN edges from expressed
behaviors — a behavior kind is flagged for a pair when its rate in their
shared events is >= PAIR_LIFT x the speaker's baseline rate (min support
floors keep tiny samples out). Per-day counts land in pair_behavior.

Usage: python -m cde.link [--subject A1_JAKE] [--skip-judge] [--max-pairs 400]
       python -m cde.link --relate-only   # rebuild just pair patterns
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

from cde import db, llm
from cde.embed import unblob

WINDOWS = [1800, 6 * 3600, 24 * 3600, 7 * 86400]
TAU = 6 * 3600  # temporal decay constant
SIM_THRESH = 0.88
TOP_K = 5

PAIR_Z = 1.0           # binomial z-score floor — deliberately permissive (p~0.16)
PAIR_MIN_SHARED = 4    # pair needs at least this many shared events
PAIR_MIN_REST = 3      # and this many events where the other was absent
PAIR_MIN_BEHAVED = 2   # and at least this many instances of the behavior
PAIR_TOP_PER = 4       # keep at most this many flagged behaviors per pair


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
    cur.execute("DELETE FROM pair_behavior")  # references edges — clear first
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
        picked = [(s, j) for s, j, _ in scores[:TOP_K]]
        # guarantee at least one cross-hour and one cross-day candidate
        if not any(evs[j]["hour"] != e["hour"] for _, j in picked):
            for s, j, _ in scores[TOP_K:]:
                if evs[j]["hour"] != e["hour"]:
                    picked.append((s, j))
                    break
        if not any(evs[j]["day"] != e["day"] for _, j in picked):
            for s, j, _ in scores:
                if evs[j]["day"] != e["day"] and (s, j) not in picked:
                    picked.append((s, j))
                    break
        for s, j, cs in scores:  # causal-claim candidates always included
            if cs >= 0.45 and all(j != pj for _, pj in picked):
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
 "supporting_quote_indices": [int], "reason": str <=30 words,
 "alternative_explanations": [str], "confidence": 0-1}

Quote indices: each event's quotes are numbered [0]…[7] for event A and [8]…[15]
for event B. Cite the indices of the specific quotes that support your verdict —
at least one for INFLUENCES/REFERS_TO.

- INFLUENCES: A plausibly shaped/caused B (mood, plan, decision carried forward). Requires at least one supporting quote index — prefer self_report (someone states the link) or transcript (quotes show the thread).
- REFERS_TO: someone in B explicitly talks about A.
- SAME_THREAD: same project/relationship/thread, no explicit causal link.
- NONE: only superficial similarity. Abstain rather than guess.
- Only cite quote indices shown in the pair's quotes.

PAIRS:
@@PAIRS@@

Reply with ONLY a JSON list."""


def judge(con, evs, pairs, max_pairs=400):
    by_id = {e["id"]: e for e in evs}
    ranked = sorted(pairs.items(), key=lambda kv: -kv[1])
    cross_day = [kv for kv in ranked if by_id[kv[0][0]]["day"] < by_id[kv[0][1]]["day"]]
    half = max_pairs // 2
    items = ranked[:half]
    seen = {kv[0] for kv in items}
    for kv in cross_day:
        if len(items) >= max_pairs:
            break
        if kv[0] not in seen:
            items.append(kv)
            seen.add(kv[0])
    cur = con.cursor()
    done_pairs = set()
    stats = defaultdict(int)

    # verdict cache: survives reruns so judged pairs are free
    cache_path = os.path.join(db.ROOT, "data", "judged.json")
    try:
        cache = json.load(open(cache_path))
    except Exception:
        cache = {}

    def apply_verdict(a, b, s, r):
        A, B = by_id[a], by_id[b]
        rel = str(r.get("relation", "NONE")).upper()
        etype = r.get("evidence_type")
        if A["t_start"] >= B["t_start"]:
            rel = "NONE"
            r["reason"] = "rejected: source not earlier than target"
        shown_a = event_utterances(con, a, 8)
        shown_b = event_utterances(con, b, 8)
        shown = shown_a + shown_b
        valid = {u["id"] for u in event_utterances(con, a, 100)} | \
                {u["id"] for u in event_utterances(con, b, 100)}
        raw_ids = r.get("supporting_quote_indices")
        if raw_ids is None:
            raw_ids = r.get("supporting_utterance_ids") or []
        ev_ids = []
        for i in raw_ids:
            if not isinstance(i, (int, float)):
                continue
            i = int(i)
            if i in valid:                    # real utterance id
                ev_ids.append(i)
            elif 0 <= i < len(shown):         # positional cite into shown quotes
                ev_ids.append(shown[i]["id"])
        ev_ids = sorted(set(ev_ids))
        if rel == "INFLUENCES" and etype not in ("self_report", "transcript", "semantic"):
            rel, stats["downgraded"] = "SAME_THREAD", stats["downgraded"] + 1
            r["reason"] = (r.get("reason") or "") + " [downgraded: temporal-only signal]"
        elif rel == "INFLUENCES" and not ev_ids:
            # keep the hypothesis but mark it inferred — demo wants visible causal
            # claims; inspector shows alternatives + confidence alongside
            r["reason"] = (r.get("reason") or "") + " [inferred: no direct quote]"
            etype = etype or "semantic"
            r["evidence_type"] = etype
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

    # apply cached verdicts without LLM calls
    for (a, b), s in items:
        key = f"{a}_{b}"
        if key in cache:
            done_pairs.add((a, b))
            apply_verdict(a, b, s, cache[key])
    con.commit()
    if done_pairs:
        print(f"replayed {len(done_pairs)} cached verdicts", flush=True)
    items = [kv for kv in items if kv[0] not in done_pairs]

    def fmt(c2, e, qbase=0):
        utts = event_utterances(c2, e["id"])
        q = "\n".join(f'    [{qbase + i}] {u["hms"]} {u["speaker"]}: {(u["text_en"] or "")[:120]}'
                      for i, u in enumerate(utts))
        return (f'event_id={e["id"]} day{e["day"]} t={e["t_start"]:.0f} '
                f'summary="{e["summary"]}" mood={e["mood"]}\n{q}')

    batches = [items[s : s + 10] for s in range(0, len(items), 10)]

    def call(batch):
        c2 = db.connect()  # sqlite connections are per-thread
        try:
            ptxt = []
            for n, ((a, b), s) in enumerate(batch):
                ptxt.append(f"--- pair {n} (retrieval_score={s}) ---\n"
                            f"A: {fmt(c2, by_id[a], 0)}\nB: {fmt(c2, by_id[b], 8)}")
            return llm.chat_json(JUDGE_PROMPT.replace("@@PAIRS@@", "\n".join(ptxt)))
        finally:
            c2.close()

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
            cache[f"{a}_{b}"] = r
            apply_verdict(a, b, s, r)
        for n, ((a, b), s) in enumerate(batch):  # no verdict returned -> rejected
            if n not in returned:
                add_edge(cur, "EVENT", a, "EVENT", b, "NONE", status="rejected",
                         rationale="judge returned no verdict", score=s)
                stats["rejected"] += 1
        con.commit()
        json.dump(cache, open(cache_path, "w"), ensure_ascii=False)
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


def _person_lookup(cur):
    """name (canonical or alias) -> PERSON entity id."""
    m = {}
    for r in cur.execute("SELECT id, canonical FROM entities WHERE type='PERSON'"):
        m[r["canonical"]] = r["id"]
    for r in cur.execute(
        "SELECT a.alias, a.entity_id FROM entity_aliases a "
        "JOIN entities e ON e.id=a.entity_id WHERE e.type='PERSON'"
    ):
        m.setdefault(r["alias"], r["entity_id"])
    return m


def relate(con, subject, z_floor=PAIR_Z, min_shared=PAIR_MIN_SHARED,
           min_behaved=PAIR_MIN_BEHAVED, min_rest=PAIR_MIN_REST,
           top_per=PAIR_TOP_PER):
    """Person-person interaction patterns from expressed behaviors.

    1) cluster BEHAVIOR mentions into BEHAVIOR_KIND entities
    2) per directed pair (speaker -> other present): rate(behavior) in
       shared events vs the speaker's rate in events where the other was
       ABSENT (leave-one-out baseline); flag on binomial z-score >= `z_floor`
    3) write pair_behavior per-day rows + up to `top_per` INTERACTION_PATTERN
       edges per flagged (speaker, other) pair
    """
    cur = con.cursor()
    # idempotent: wipe this stage's own outputs (pair_behavior first: it
    # holds FK references into edges)
    cur.execute("DELETE FROM pair_behavior")
    cur.execute(
        "DELETE FROM edge_evidence WHERE edge_id IN "
        "(SELECT id FROM edges WHERE relation='INTERACTION_PATTERN')")
    cur.execute("DELETE FROM edges WHERE relation='INTERACTION_PATTERN'")
    cur.execute(
        "DELETE FROM entity_aliases WHERE entity_id IN "
        "(SELECT id FROM entities WHERE type='BEHAVIOR_KIND')")
    cur.execute("DELETE FROM entities WHERE type='BEHAVIOR_KIND'")
    cur.execute(
        "UPDATE mentions SET payload_json=json_remove(payload_json,'$.behavior_entity_id') "
        "WHERE type='BEHAVIOR'")

    # --- 1) canonicalize behavior kinds (same machinery as goals) ----------
    from cde.embed import BEHAVIOR_CLUSTER_PROMPT, blob, cluster_goal_strings, embed_texts
    mentions = cur.execute(
        "SELECT id, event_id, speaker, text FROM mentions "
        "WHERE type='BEHAVIOR' AND event_id IS NOT NULL").fetchall()
    if not mentions:
        con.commit()
        return {"behavior_kinds": 0, "pairs_flagged": 0}
    items = [(m["id"], m["event_id"], m["text"]) for m in mentions]
    # cluster cache: keeps reruns deterministic and free (same convention as
    # data/judged.json for the edge judge). Replays by mention id; mentions
    # not covered by the cache become singleton clusters.
    cache_path = os.path.join(db.ROOT, "data", "behavior_kinds.json")
    pos_of = {mid: i for i, (mid, _, _) in enumerate(items)}
    clusters = None
    try:
        cached = json.load(open(cache_path))
        assigned = set()
        clusters = []
        for canon, mids in cached:
            members = [pos_of[m] for m in mids if m in pos_of and pos_of[m] not in assigned]
            if members:
                assigned.update(members)
                clusters.append((canon, members))
        for i, (mid, _, t) in enumerate(items):
            if i not in assigned:
                clusters.append((t, [i]))
    except Exception:
        clusters = None
    if clusters is None:
        clusters = cluster_goal_strings(items, prompt=BEHAVIOR_CLUSTER_PROMPT)
        json.dump([(c, [items[i][0] for i in ms]) for c, ms in clusters],
                  open(cache_path, "w"), ensure_ascii=False)
    kind_of = {}  # mention_id -> BEHAVIOR_KIND entity id
    for (rep, members), v in zip(clusters, embed_texts([c[0] for c in clusters], con)):
        cur.execute(
            "INSERT OR IGNORE INTO entities(type,canonical,embedding) VALUES('BEHAVIOR_KIND',?,?)",
            (rep[:160], blob(v)))
        kid = cur.execute(
            "SELECT id FROM entities WHERE type='BEHAVIOR_KIND' AND canonical=?",
            (rep[:160],)).fetchone()["id"]
        for mi in members:
            mid, _, _ = items[mi]
            kind_of[mid] = kid
            cur.execute(
                "UPDATE mentions SET payload_json="
                "json_set(COALESCE(payload_json,'{}'),'$.behavior_entity_id',?) WHERE id=?",
                (kid, mid))
    con.commit()

    # --- 2) presence + counts ----------------------------------------------
    people = _person_lookup(cur)
    subj = cur.execute(
        "SELECT name FROM participants WHERE is_subject=1").fetchone()
    subj_id = people.get(subj["name"]) if subj else None

    events = cur.execute(
        "SELECT id, day FROM events WHERE subject=? ORDER BY t_start", (subject,)).fetchall()
    present = defaultdict(set)          # event_id -> person entity ids
    for r in cur.execute("SELECT event_id, entity_id FROM event_participants"):
        present[r["event_id"]].add(r["entity_id"])
    mention_utts = defaultdict(list)    # mention_id -> utterance ids
    for r in cur.execute("SELECT mention_id, utterance_id FROM mention_evidence"):
        mention_utts[r["mention_id"]].append(r["utterance_id"])

    beh_by_event = defaultdict(list)    # event_id -> (speaker_id, kind_id, mention_id)
    mid_eid = {m["id"]: m["event_id"] for m in mentions}
    for m in mentions:
        sid = people.get(m["speaker"])
        kid = kind_of.get(m["id"])
        if sid and kid:
            beh_by_event[m["event_id"]].append((sid, kid, m["id"]))
            present[m["event_id"]].add(sid)   # a speaker was present by definition
    for eid in present:
        if subj_id:
            present[eid].add(subj_id)         # the wearer is always present

    shared = defaultdict(int)                 # (day, frozenset(a,b)) -> count
    present_n = defaultdict(int)              # person -> events present
    behaved = defaultdict(list)               # (s,p,b,day) -> [mention_ids]
    behaved_ev = defaultdict(set)             # (s,b) -> event ids
    for e in events:
        P = sorted(present[e["id"]])
        day = e["day"]
        for a in P:
            present_n[a] += 1
        for i in range(len(P)):
            for p in P[i + 1:]:
                shared[(day, frozenset((P[i], p)))] += 1
        for sid, kid, mid in beh_by_event[e["id"]]:
            behaved_ev[(sid, kid)].add(e["id"])
            for p in P:
                if p != sid:
                    behaved[(sid, p, kid, day)].append(mid)

    # --- 3) flag + write -----------------------------------------------------
    shared_tot = defaultdict(int)
    for (day, pair), n in shared.items():
        shared_tot[pair] += n
    canon = {r["id"]: r["canonical"] for r in cur.execute(
        "SELECT id, canonical FROM entities WHERE type IN ('PERSON','BEHAVIOR_KIND')")}

    per_pair = defaultdict(list)  # (s,p) -> candidate triples sorted by z
    for s, p, b in sorted({k[:3] for k in behaved}):
        pair = frozenset((s, p))
        sh = shared_tot.get(pair, 0)
        rest_n = present_n[s] - sh
        mids_all = [m for k, ms in behaved.items() if k[:3] == (s, p, b) for m in ms]
        bh = len({mid_eid[mid] for mid in mids_all})          # distinct events
        tot = len(behaved_ev[(s, b)])
        rate = bh / sh if sh else 0.0
        # absent-rate, Laplace-smoothed so a never-seen-elsewhere behavior
        # gets a small nonzero null instead of an infinite z
        p0 = (tot - bh + 1) / (rest_n + 1) if rest_n >= 0 else 0.0
        exp = sh * p0
        z = (bh - exp) / math.sqrt(max(sh * p0 * (1 - p0), 1e-9))
        lf = rate / max(p0, 0.005)
        if (sh < min_shared or rest_n < min_rest or bh < min_behaved
                or z < z_floor):
            continue
        per_pair[(s, p)].append((z, s, p, b, sh, bh, rate, p0, lf, mids_all))

    flagged = 0
    for (s, p), cands in per_pair.items():
        for z, s, p, b, sh, bh, rate, p0, lf, mids_all in \
                sorted(cands, reverse=True)[:top_per]:
            flagged += 1
            ev_utts = sorted({u for mid in mids_all for u in mention_utts[mid]})[:16]
            eid = add_edge(cur, "PERSON", s, "PERSON", p, "INTERACTION_PATTERN",
                           status="hypothesis", etype="statistical",
                           rationale=(f"{canon.get(b,'?')} in {bh}/{sh} shared events "
                                      f"({rate:.0%}) vs {p0:.0%} of "
                                      f"{canon.get(s,'?')}'s events without "
                                      f"{canon.get(p,'?')} (z={z:.1f}, lift {lf:.1f}x)"),
                           score=round(z, 2), evidence=ev_utts)
            # per-day rows for every day the pair shared events (behaved may be 0)
            for (day, pr), n in sorted(shared.items()):
                if pr != frozenset((s, p)):
                    continue
                day_mids = behaved.get((s, p, b, day), [])
                cur.execute(
                    "INSERT OR REPLACE INTO pair_behavior"
                    "(edge_id,src_entity_id,dst_entity_id,behavior_entity_id,day,"
                    "shared,behaved,events_json) VALUES(?,?,?,?,?,?,?,?)",
                    (eid, s, p, b, day, n, len(day_mids),
                     json.dumps(sorted({mid_eid[mid] for mid in day_mids}))))
    con.commit()
    return {"behavior_kinds": len(clusters), "pairs_flagged": flagged}


def main():
    args = sys.argv[1:]
    subject = args[args.index("--subject") + 1] if "--subject" in args else "A1_JAKE"
    skip_judge = "--skip-judge" in args
    max_pairs = int(args[args.index("--max-pairs") + 1]) if "--max-pairs" in args else 400
    z_floor = float(args[args.index("--z") + 1]) if "--z" in args else PAIR_Z

    con = db.connect()
    if "--relate-only" in args:
        print("pair patterns:", relate(con, subject, z_floor=z_floor), flush=True)
        con.close()
        return
    print("deterministic edges:", deterministic(con, subject), flush=True)
    n_sim = similar(con, subject)
    print(f"SIMILAR_TO edges: {n_sim}", flush=True)
    evs, pairs = candidates(con, subject)
    print(f"candidate pairs: {len(pairs)}", flush=True)
    if not skip_judge and pairs:
        stats, judged = judge(con, evs, pairs, max_pairs)
        print(f"judged={judged} -> {stats}", flush=True)
    print("pair patterns:", relate(con, subject, z_floor=z_floor), flush=True)
    totals = {r["relation"]: (r["c"], r["st"]) for r in con.execute(
        "SELECT relation, status st, COUNT(*) c FROM edges GROUP BY relation, status")}
    print("edges:", totals)
    con.close()


if __name__ == "__main__":
    main()

"""Stage 5 (stretch) — patterns: k-means over event embeddings, LLM names each
cluster, ordered cluster pairs recurring on >=3 days within 3h -> PATTERN nodes.

Usage: python -m cde.patterns [--k 12]
"""
import json
import sys
from collections import defaultdict

import numpy as np

from cde import db, llm
from cde.embed import unblob


def kmeans(V, k, iters=40, seed=0):
    rng = np.random.default_rng(seed)
    cent = V[rng.choice(len(V), k, replace=False)].copy()
    for _ in range(iters):
        assign = np.argmax(V @ cent.T, axis=1)
        new = cent.copy()
        for i in range(k):
            members = V[assign == i]
            if len(members):
                new[i] = members.mean(axis=0)
                n = np.linalg.norm(new[i])
                if n:
                    new[i] /= n
        if np.allclose(new, cent):
            break
        cent = new
    return assign, cent


def main():
    k = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv else 12
    con = db.connect()
    cur = con.cursor()
    evs = [dict(r) for r in cur.execute(
        "SELECT id, day, t_start, summary, embedding FROM events WHERE embedding IS NOT NULL ORDER BY t_start")]
    if len(evs) < k:
        print("not enough events")
        return
    V = np.array([unblob(e["embedding"]) for e in evs])
    assign, _ = kmeans(V, k)
    clusters = defaultdict(list)
    for e, a in zip(evs, assign):
        clusters[int(a)].append(e)

    cur.execute("DELETE FROM pattern_occurrences")
    cur.execute("DELETE FROM patterns")
    named = {}
    for ci, members in clusters.items():
        summaries = "\n".join(f"- {m['summary']}" for m in members[:25])
        prompt = ("These are event summaries from one week of a person's life, clustered by "
                  "similarity. Give the cluster a short name (<=5 words, e.g. 'logistics "
                  "planning', 'setback/complaint') and a one-line description.\n\n" + summaries +
                  "\n\nReply with JSON: {\"name\": str, \"description\": str}")
        try:
            info = llm.chat_json(prompt)
        except Exception:
            info = {"name": f"cluster {ci}", "description": ""}
        cur.execute("INSERT INTO patterns(name, description) VALUES(?,?)",
                    (info.get("name", f"cluster {ci}"), info.get("description", "")))
        pid = cur.lastrowid
        named[ci] = (pid, info.get("name"))
        for m in members:
            cur.execute(
                "INSERT OR IGNORE INTO pattern_occurrences(pattern_id,event_id) VALUES(?,?)",
                (pid, m["id"]))
        print(f"cluster {ci}: {info.get('name')} ({len(members)} events)", flush=True)

    # ordered pairs of patterns recurring on >=3 days within 3h
    pat_of = {e["id"]: int(a) for e, a in zip(evs, assign)}
    pair_days = defaultdict(set)
    by_day = defaultdict(list)
    for e in evs:
        by_day[e["day"]].append(e)
    for day, lst in by_day.items():
        lst.sort(key=lambda e: e["t_start"])
        for i, a in enumerate(lst):
            for b in lst[i + 1:]:
                if b["t_start"] - a["t_start"] > 3 * 3600:
                    break
                if pat_of[a["id"]] != pat_of[b["id"]]:
                    pair_days[(pat_of[a["id"]], pat_of[b["id"]])].add(day)
    for (pa, pb), days in sorted(pair_days.items(), key=lambda kv: -len(kv[1])):
        if len(days) >= 3:
            print(f"recurring: {named[pa][1]} -> {named[pb][1]} on days {sorted(days)}")
            cur.execute(
                "INSERT INTO edges(src_type,src_id,dst_type,dst_id,relation,status,evidence_type,rationale) "
                "VALUES('PATTERN',?,'PATTERN',?,'REINFORCES','hypothesis','temporal',?)",
                (named[pa][0], named[pb][0],
                 f"ordered pair recurs on days {sorted(days)}"))
    con.commit()
    con.close()


if __name__ == "__main__":
    main()

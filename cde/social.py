"""Stage 7 — week-level social metrics for the people panel.

Same source and speaker attribution as cde/insights.py:
EgoLife/clean/utterances/*.jsonl, one moment per moment_id, heard_by = which
recorders captured it. Generalises Ethan's day-1 trends (scripts/
compute_trends.py, review CSVs) to all 7 days, with two corrections:
  - replies / talk-overs are found in each recorder's own stream (so they are
    real conversational adjacency), then de-duplicated on the moment pair —
    the day-1 counts were amplified ~3-4x because every nearby recorder
    re-counted the same exchange;
  - co-presence = distinct minutes in which both recorders captured the same
    moment (heard_by), not text matching within +-6 s.

Tables (rebuilt each run):
  social_person(person, day, lines, airtime_s)
  social_pair(a, b, day, copresence_min, b_replies_a, a_replies_b,
              a_over_b, b_over_a, b_hears_a, a_hears_b)
    a precedes b in wearer order; b_replies_a = b answered a (b's line began
    <= 4 s after a's); a_over_b = a started before b's line ended;
    b_hears_a = a's lines captured on b's recorder.

Run: python -m cde.social
"""
import json
import os
from collections import Counter, defaultdict

from cde import db

ORDER = ["Jake", "Alice", "Tasha", "Lucia", "Katrina", "Shure"]
WEARERS = {"Jake": "A1_JAKE", "Alice": "A2_ALICE", "Tasha": "A3_TASHA",
           "Lucia": "A4_LUCIA", "Katrina": "A5_KATRINA", "Shure": "A6_SHURE"}
REC2NAME = {v: k for k, v in WEARERS.items()}
CLEAN = os.path.join(db.ROOT, "EgoLife", "clean", "utterances")
REPLY_GAP, OVERLAP = 4.0, 0.05


def load():
    rows = defaultdict(list)                     # moment_id -> rows
    for rec in WEARERS.values():
        with open(os.path.join(CLEAN, f"{rec}.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                r["t"] = (r["day"] - 1) * 86400 + r["start"]
                r["e"] = (r["day"] - 1) * 86400 + r["end"]
                rows[r["moment_id"]].append(r)
    speaker, moment = {}, {}
    for mid, rs in rows.items():               # attribution identical to insights.py
        selfrows = [r for r in rs if REC2NAME[r["recorder"]] == r["speaker"]]
        if len({r["speaker"] for r in selfrows}) == 1:
            r0, spk = selfrows[0], selfrows[0]["speaker"]
        else:
            r0 = rs[0]
            spk = Counter(r["moment_speaker"] for r in rs).most_common(1)[0][0]
        speaker[mid] = spk
        moment[mid] = {"t": r0["t"], "e": r0["e"], "day": r0["day"],
                       "heard": set(rs[0]["heard_by"])}
    streams = defaultdict(list)                  # recorder -> its own rows
    for rs in rows.values():
        for r in rs:
            streams[r["recorder"]].append((r["t"], r["e"], r["moment_id"]))
    for s in streams.values():
        s.sort()
    return speaker, moment, streams


def main():
    speaker, moment, streams = load()
    person = defaultdict(lambda: [0, 0.0])                       # (name, day) -> [lines, airtime]
    pair = defaultdict(lambda: defaultdict(int))                 # (a, b, day) -> col -> n
    co_min = defaultdict(set)                                    # (a, b, day) -> minutes

    def key(x, y):
        return (x, y) if ORDER.index(x) < ORDER.index(y) else (y, x)

    for mid, m in moment.items():
        s, day = speaker[mid], m["day"]
        names = sorted((REC2NAME[r] for r in m["heard"]), key=ORDER.index)
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                co_min[(x, y, day)].add(int(m["t"] // 60))
        if s not in WEARERS:
            continue
        person[(s, day)][0] += 1
        person[(s, day)][1] += max(0.0, m["e"] - m["t"])
        for y in names:
            if y != s:
                a, b = key(s, y)
                pair[(a, b, day)]["b_hears_a" if s == a else "a_hears_b"] += 1

    seen_reply, seen_over = set(), set()
    for rows in streams.values():
        for (t1, e1, m1), (t2, e2, m2) in zip(rows, rows[1:]):
            s1, s2 = speaker[m1], speaker[m2]
            if s1 == s2 or s1 not in WEARERS or s2 not in WEARERS or m1 == m2:
                continue
            a, b = key(s1, s2)
            day = moment[m1]["day"]
            if t2 - t1 <= REPLY_GAP and (m1, m2) not in seen_reply:
                seen_reply.add((m1, m2))
                pair[(a, b, day)]["b_replies_a" if s1 == a else "a_replies_b"] += 1
            if t2 < e1 - OVERLAP and (m1, m2) not in seen_over:
                seen_over.add((m1, m2))
                pair[(a, b, day)]["b_over_a" if s1 == a else "a_over_b"] += 1

    con = db.connect()
    con.executescript("""
        DROP TABLE IF EXISTS social_person;
        CREATE TABLE social_person(person TEXT, day INTEGER, lines INTEGER, airtime_s REAL);
        DROP TABLE IF EXISTS social_pair;
        CREATE TABLE social_pair(a TEXT, b TEXT, day INTEGER, copresence_min INTEGER,
            b_replies_a INTEGER, a_replies_b INTEGER, a_over_b INTEGER, b_over_a INTEGER,
            b_hears_a INTEGER, a_hears_b INTEGER);
    """)
    days = sorted({m["day"] for m in moment.values()})
    con.executemany("INSERT INTO social_person VALUES(?,?,?,?)",
                    [(p, d, person[(p, d)][0], round(person[(p, d)][1], 1))
                     for p in ORDER for d in days])
    cols = ["b_replies_a", "a_replies_b", "a_over_b", "b_over_a", "b_hears_a", "a_hears_b"]
    con.executemany(
        "INSERT INTO social_pair VALUES(?,?,?,?,?,?,?,?,?,?)",
        [(a, b, d, len(co_min[(a, b, d)]), *(pair[(a, b, d)][c] for c in cols))
         for i, a in enumerate(ORDER) for b in ORDER[i + 1:] for d in days])
    con.commit()

    tot = {p: sum(person[(p, d)][0] for d in days) for p in ORDER}
    allp = sum(tot.values())
    print("lines share:", {p: f"{tot[p] / allp:.1%}" for p in ORDER})
    for r in con.execute(
            """SELECT a, b, SUM(copresence_min) cop, SUM(b_replies_a) ba, SUM(a_replies_b) ab,
                      SUM(a_over_b) ao, SUM(b_over_a) bo, SUM(b_hears_a) bh, SUM(a_hears_b) ah
               FROM social_pair GROUP BY a, b ORDER BY cop DESC"""):
        print(f"  {r['a']:>7}–{r['b']:<7} co {r['cop']:>5} min · replies {r['b']} answers "
              f"{r['a']} {r['ba']:>5} / {r['a']} answers {r['b']} {r['ab']:>5} · over "
              f"{r['ao']}/{r['bo']} · hears {r['bh']}/{r['ah']}")
    con.close()


if __name__ == "__main__":
    main()

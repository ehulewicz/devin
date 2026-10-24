"""Semantic zoom L1 — cluster Jake's GOAL entities into human-readable themes.

One LLM call groups the goals; events inherit a theme by majority vote of
their goals, or by nearest theme embedding centroid when they have no themed
goal. Idempotent: clears the three theme tables each run. The raw LLM JSON is
cached in data/themes.json — delete that file to force a re-cluster.

Run: python -m cde.themes
"""
import json
import os

import numpy as np

from cde import db, llm
from cde.embed import unblob

SUBJECT = "A1_JAKE"
CACHE = os.path.join(db.ROOT, "data", "themes.json")

NAME_PROMPT = """Below are the recurring goals from one week of one person's (Jake's) lifelog, with how many events each appears in. Propose 8-10 THEMES that together cover the week: human-readable areas someone would recognize at a glance.

Good names (concrete, 2-4 words): "Final-day party", "Cooking & baking", "Shopping runs", "Recording & data duty", "Games & icebreakers", "Dance rehearsal", "Money & reimbursements", "Room decoration", "Auction & gifts", "Barbecue night".
Banned words (too generic): activities, engagement, tasks, errands, management, hobbies, projects, wellness, daily, social, personal, miscellaneous, general.
Split the biggest area into concrete sub-themes rather than one giant bucket — no theme should plausibly hold more than a quarter of the events.

GOALS (goal [events]):
@@GOALS@@

Reply with ONLY JSON: {"themes": [{"name": str, "description": str (one sentence, second person, e.g. "You and the group ...")}]}"""

ASSIGN_PROMPT = """Assign each goal below to exactly one of these themes, using the theme number.

THEMES:
@@THEMES@@

GOALS:
@@GOALS@@

Reply with ONLY JSON mapping every goal number to a theme number: {"<goal number>": <theme number>, ...}"""


def _cluster(goals):
    """Two-step clustering: name themes, then batch-assign every goal to one.
    Returns {"themes": [{"name", "description", "goals": [1-based indices]}]}."""
    listing = "\n".join(f"{g['canonical']} [{g['n']} events]" for g in goals)
    data = llm.chat_json(NAME_PROMPT.replace("@@GOALS@@", listing))
    themes = data.get("themes") or []
    if not themes:
        raise RuntimeError("name step returned no themes")
    tlist = "\n".join(f"{i}. {t.get('name')} — {t.get('description', '')}"
                      for i, t in enumerate(themes, 1))
    assign = {}  # 1-based goal index -> 0-based theme index
    for off in range(0, len(goals), 60):
        glist = "\n".join(f"{i}. {goals[i - 1]['canonical']}"
                          for i in range(off + 1, min(off + 60, len(goals)) + 1))
        m = llm.chat_json(ASSIGN_PROMPT.replace("@@THEMES@@", tlist)
                          .replace("@@GOALS@@", glist))
        for k, v in (m or {}).items():
            try:
                gi, ti = int(k), int(v)
            except (TypeError, ValueError):
                continue
            if 1 <= gi <= len(goals) and 1 <= ti <= len(themes):
                assign[gi] = ti - 1
    return {"themes": [{"name": t.get("name"), "description": t.get("description", ""),
                        "goals": sorted(gi for gi, x in assign.items() if x == ti)}
                       for ti, t in enumerate(themes)]}


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def _centroid(vectors):
    vs = [v for v in vectors if v is not None]
    return _norm(np.mean(vs, axis=0)) if vs else None


def main():
    con = db.init_db(db.connect())
    for t in ("event_themes", "goal_themes", "themes"):
        con.execute(f"DELETE FROM {t}")
    con.commit()

    goals = con.execute(
        """SELECT e.id, e.canonical, e.embedding,
                  (SELECT COUNT(*) FROM event_entities ee
                   JOIN events ev ON ev.id=ee.event_id
                   WHERE ee.entity_id=e.id AND ee.relation='goal'
                     AND ev.subject=?) n
           FROM entities e WHERE e.type='GOAL' ORDER BY n DESC, e.id""",
        (SUBJECT,),
    ).fetchall()

    cached = os.path.exists(CACHE)
    if cached:
        data = json.load(open(CACHE))
    else:
        data = _cluster(goals)
        with open(CACHE, "w") as f:
            json.dump(data, f, indent=1)
    themes = data.get("themes") or []
    if not themes:
        raise SystemExit("no themes in LLM response / cache")

    # goal -> theme index (first occurrence wins if the LLM double-assigns)
    gtheme = {}
    for ti, th in enumerate(themes):
        for gi in th.get("goals") or []:
            try:
                gi = int(gi)
            except (TypeError, ValueError):
                continue
            if 1 <= gi <= len(goals):
                gtheme.setdefault(goals[gi - 1]["id"], ti)

    # goals the LLM left out -> nearest theme by goal-embedding centroid
    cents = {ti: _centroid(unblob(g["embedding"]) for g in goals
                           if gtheme.get(g["id"]) == ti)
             for ti in range(len(themes))}
    cents = {ti: v for ti, v in cents.items() if v is not None}
    fallback = max(range(len(themes)),
                   key=lambda t: sum(1 for x in gtheme.values() if x == t))
    stray = 0
    for g in goals:
        if g["id"] in gtheme:
            continue
        v = unblob(g["embedding"])
        gtheme[g["id"]] = (max(cents, key=lambda t: float(np.dot(_norm(v), cents[t])))
                           if v is not None and cents else fallback)
        stray += 1

    evs = con.execute(
        "SELECT id, day, t_start, embedding FROM events WHERE subject=? ORDER BY t_start",
        (SUBJECT,),
    ).fetchall()
    evgoals = {}
    for r in con.execute(
        """SELECT ee.event_id, ee.entity_id FROM event_entities ee
           JOIN events ev ON ev.id=ee.event_id
           WHERE ee.relation='goal' AND ev.subject=?""",
        (SUBJECT,),
    ):
        if r["entity_id"] in gtheme:
            evgoals.setdefault(r["event_id"], []).append(gtheme[r["entity_id"]])

    pop = {}
    for ts in evgoals.values():
        for t in ts:
            pop[t] = pop.get(t, 0) + 1

    assigned = {}  # event_id -> (theme_idx, source)
    for e in evs:
        ts = evgoals.get(e["id"])
        if not ts:
            continue
        cnt = {}
        for t in ts:
            cnt[t] = cnt.get(t, 0) + 1
        assigned[e["id"]] = (max(cnt, key=lambda t: (cnt[t], pop.get(t, 0))), "goal")

    # events without a themed goal -> nearest theme centroid (over that theme's
    # goal-assigned events; goal-embedding centroid if the theme has none yet)
    ecents = {ti: _centroid(unblob(e["embedding"]) for e in evs
                            if assigned.get(e["id"], (None,))[0] == ti)
              for ti in range(len(themes))}
    for ti in range(len(themes)):
        if ecents.get(ti) is None:
            ecents[ti] = cents.get(ti)
    ecents = {ti: v for ti, v in ecents.items() if v is not None}
    big = max(pop, key=pop.get) if pop else fallback
    for e in evs:
        if e["id"] in assigned:
            continue
        v = unblob(e["embedding"])
        assigned[e["id"]] = (
            (max(ecents, key=lambda t: float(np.dot(_norm(v), ecents[t])))
             if v is not None and ecents else big),
            "embedding",
        )

    counts = {ti: 0 for ti in range(len(themes))}
    days = {ti: set() for ti in range(len(themes))}
    eday = {e["id"]: e["day"] for e in evs}
    for eid, (ti, _) in assigned.items():
        counts[ti] += 1
        days[ti].add(eday[eid])
    order = sorted(range(len(themes)), key=lambda t: -counts[t])
    keep = [ti for ti in order if counts[ti] > 0]

    tid = {}
    for ti in keep:
        th = themes[ti]
        cur = con.execute(
            "INSERT INTO themes(name, description) VALUES(?,?)",
            (th.get("name") or f"theme {ti + 1}", th.get("description") or ""),
        )
        tid[ti] = cur.lastrowid
    for gid, ti in gtheme.items():
        if ti not in tid:
            continue
        con.execute(
            "INSERT INTO goal_themes(goal_entity_id, theme_id) VALUES(?,?)",
            (gid, tid[ti]),
        )
    for eid, (ti, src) in assigned.items():
        if ti not in tid:
            continue
        con.execute(
            "INSERT INTO event_themes(event_id, theme_id, source) VALUES(?,?,?)",
            (eid, tid[ti], src),
        )
    for rank, ti in enumerate(keep):
        con.execute("UPDATE themes SET rank=? WHERE id=?", (rank, tid[ti]))
    con.commit()

    nsrc = {}
    for _, s in assigned.values():
        nsrc[s] = nsrc.get(s, 0) + 1
    print(f"{'cache' if cached else 'llm'}: {len(keep)}/{len(themes)} themes kept, "
          f"{len(gtheme)} goals themed ({stray} stragglers), "
          f"{len(assigned)} events assigned {nsrc}")
    for rank, ti in enumerate(keep):
        th = themes[ti]
        print(f"#{rank} [{tid[ti]}] {th.get('name')}: {counts[ti]} events, "
              f"days {sorted(days[ti])} — {th.get('description', '')[:80]}")
    con.close()


if __name__ == "__main__":
    main()

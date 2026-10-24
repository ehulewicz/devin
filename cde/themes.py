"""Semantic zoom L1 — cluster Jake's GOAL entities into human-readable themes.

One LLM call groups the goals; events inherit a theme by majority vote of
their goals, or by nearest theme embedding centroid when they have no themed
goal. Idempotent: clears the three theme tables each run. The raw LLM JSON is
cached in data/themes.json — delete that file to force a re-cluster.

Run: python -m cde.themes
"""
import json
import os
import re
import sqlite3

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

SOCIAL_NAME_PROMPT = """Below are events from one week of Jake's lifelog that are mainly about social interaction with his housemates (Alice, Tasha, Lucia, Katrina, Shure) and guests. Each line: day, who was involved, mood, summary.

Propose 4-7 SOCIAL THEMES that show HOW the interactions go — the kind of interaction, not the topic. Examples of good names: "Banter & teasing", "Helping & teaching", "Negotiating decisions", "Sharing stories", "Hosting guests", "Friction & disagreements", "Encouraging each other". If one person clearly dominates a kind of interaction, put them in the name (e.g. "Baking lessons with Tasha", "Teasing with Shure").
Banned words: social, interactions, activities, engagement, general, miscellaneous.

EVENTS:
@@EVENTS@@

Reply with ONLY JSON: {"themes": [{"name": str, "description": str (one sentence, second person, e.g. "You and Shure ...")}]}"""

SOCIAL_ASSIGN_PROMPT = """Assign each event below to exactly one of these social themes, using the theme number.

THEMES:
@@THEMES@@

EVENTS:
@@EVENTS@@

Reply with ONLY JSON mapping every event number to a theme number: {"<event number>": <theme number>, ...}"""


def _social_cluster(con, sevs):
    """Two-step split of one 'social' theme's events into interaction
    sub-themes. Returns {"themes": [{name, description}],
    "assign": {event_id: 0-based sub-theme index}}."""
    ids = [e["id"] for e in sevs]
    parts = {}
    if ids:
        for r in con.execute(
            f"""SELECT ep.event_id, e.canonical FROM event_participants ep
                JOIN entities e ON e.id=ep.entity_id
                WHERE ep.event_id IN ({','.join('?' * len(ids))})""",
            ids,
        ):
            if r["canonical"] != "Jake":
                parts.setdefault(r["event_id"], []).append(r["canonical"])
    lines = [
        f"{i}. D{e['day']} [with: {', '.join(parts.get(e['id'], [])) or '—'}] "
        f"({e['mood'] or '—'}) {e['summary']}"
        for i, e in enumerate(sevs, 1)
    ]
    data = llm.chat_json(SOCIAL_NAME_PROMPT.replace("@@EVENTS@@", "\n".join(lines)))
    sths = data.get("themes") or []
    if not sths:
        raise RuntimeError("social name step returned no themes")
    tlist = "\n".join(f"{i}. {t.get('name')} — {t.get('description', '')}"
                      for i, t in enumerate(sths, 1))
    assign = {}  # event_id (str) -> 0-based sub-theme index
    for off in range(0, len(sevs), 50):
        glist = "\n".join(lines[off:off + 50])
        m = llm.chat_json(SOCIAL_ASSIGN_PROMPT.replace("@@THEMES@@", tlist)
                          .replace("@@EVENTS@@", glist))
        for k, v in (m or {}).items():
            try:
                ei, ti = int(k), int(v)
            except (TypeError, ValueError):
                continue
            if 1 <= ei <= len(sevs) and 1 <= ti <= len(sths):
                assign[str(sevs[ei - 1]["id"])] = ti - 1
    return {"themes": sths, "assign": assign}


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
    try:
        con.execute("ALTER TABLE themes ADD COLUMN kind TEXT")
    except sqlite3.OperationalError:
        pass
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
        "SELECT id, day, t_start, summary, mood, embedding FROM events "
        "WHERE subject=? ORDER BY t_start",
        (SUBJECT,),
    ).fetchall()
    evgoals = {}
    goalevs = {}
    for r in con.execute(
        """SELECT ee.event_id, ee.entity_id FROM event_entities ee
           JOIN events ev ON ev.id=ee.event_id
           WHERE ee.relation='goal' AND ev.subject=?""",
        (SUBJECT,),
    ):
        goalevs.setdefault(r["entity_id"], []).append(r["event_id"])
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

    # ---- split 'social/interaction' themes into interaction sub-themes ----
    kinds = {ti: "activity" for ti in range(len(themes))}
    split_tis = [ti for ti in range(len(themes))
                 if re.search(r"social|interaction",
                              str(themes[ti].get("name") or ""), re.I)]
    if split_tis:
        scache_path = os.path.join(db.ROOT, "data", "social_themes.json")
        soc = (json.load(open(scache_path))
               if os.path.exists(scache_path) else {})
        soc_changed = not os.path.exists(scache_path)
        for ti in split_tis:
            sevs = [e for e in evs if assigned[e["id"]][0] == ti]
            key = themes[ti].get("name") or str(ti)
            entry = soc.get(key)
            if entry is None:
                entry = _social_cluster(con, sevs)
                soc[key] = entry
                soc_changed = True
            sths = entry.get("themes") or []
            amap = entry.get("assign") or {}
            base = len(themes)
            for s in sths:
                themes.append({"name": s.get("name"),
                               "description": s.get("description", "")})
                kinds[len(themes) - 1] = "social"
            sub = {}
            for e in sevs:
                try:
                    v = int(amap.get(str(e["id"])))
                except (TypeError, ValueError):
                    continue
                if 0 <= v < len(sths):
                    sub[e["id"]] = v
            subcents = {k: _centroid(unblob(e["embedding"]) for e in sevs
                                     if sub.get(e["id"]) == k)
                        for k in range(len(sths))}
            subcents = {k: v for k, v in subcents.items() if v is not None}
            subcnt = {}
            for v in sub.values():
                subcnt[v] = subcnt.get(v, 0) + 1
            biggest = max(subcnt, key=subcnt.get) if subcnt else 0
            for e in sevs:
                if e["id"] in sub:
                    continue
                v = unblob(e["embedding"])
                sub[e["id"]] = (
                    max(subcents,
                        key=lambda k: float(np.dot(_norm(v), subcents[k])))
                    if v is not None and subcents else biggest)
            for e in sevs:
                assigned[e["id"]] = (base + sub[e["id"]], "social")
            # remap this theme's goals to the sub-theme holding most of each
            # goal's events (drop the goal_theme row if the goal has none)
            for gid in [g for g, t2 in gtheme.items() if t2 == ti]:
                cnt = {}
                for eid in goalevs.get(gid, []):
                    a = assigned.get(eid)
                    if a and base <= a[0] < base + len(sths):
                        cnt[a[0]] = cnt.get(a[0], 0) + 1
                if cnt:
                    gtheme[gid] = max(cnt, key=cnt.get)
                else:
                    del gtheme[gid]
        if soc_changed:
            with open(scache_path, "w") as f:
                json.dump(soc, f, indent=1)

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
            "INSERT INTO themes(name, description, kind) VALUES(?,?,?)",
            (th.get("name") or f"theme {ti + 1}", th.get("description") or "",
             kinds.get(ti, "activity")),
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
        print(f"#{rank} [{tid[ti]}] {th.get('name')} ({kinds.get(ti, 'activity')}): "
              f"{counts[ti]} events, "
              f"days {sorted(days[ti])} — {th.get('description', '')[:80]}")
    con.close()


if __name__ == "__main__":
    main()

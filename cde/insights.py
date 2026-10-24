"""Stage 6 — insights: how Jake acts, and how others act around him.

Presence comes from `heard_by` in EgoLife/clean/utterances/*.jsonl (which
recorders captured each moment; consistent across recorders for all 123,670
moments). For a speaker S and another wearer O, each of S's own-mic utterances
is labelled:
  with    — O's recorder captured this moment, or S's and O's recorders
            co-captured any moment within +-1 minute
  without — O's recorder was active within +-1 minute but shared nothing with
            S (O was recording somewhere else)
  unknown — O's recorder idle: excluded (recorder-off is not absence)
Window-level presence avoids a capture bias: short utterances ("okay", laughs)
are less often picked up by a second recorder, so per-utterance heard_by alone
would push them into "without".

Own-mic utterance = the speaker's own recorder labels the line as theirs (the
~1% of moments where recorders disagree on the speaker are not attributed).

Group-size confound: Jake is usually with the whole group, so "with Jake"
also means "with more people". Every comparison is stratified by how many
OTHER wearers were present (len(heard_by) minus S and O): rates are
standardised to the with-condition's group-size mix and z is the
Cochran–Mantel–Haenszel statistic. z still treats utterances as independent
(autocorrelation inflates it), so it ranks findings rather than acting as a
p-value.

Features are keyword heuristics on the English translation (+ a few zh cues).

Writes tables presence_effects + insights and data/insights.json.
Run: python -m cde.insights
"""
import json
import math
import os
import re
from collections import Counter, defaultdict

from cde import db, llm

WEARERS = {"Jake": "A1_JAKE", "Alice": "A2_ALICE", "Tasha": "A3_TASHA",
           "Lucia": "A4_LUCIA", "Katrina": "A5_KATRINA", "Shure": "A6_SHURE"}
REC2NAME = {v: k for k, v in WEARERS.items()}
SUBJ = "Jake"
CLEAN = os.path.join(db.ROOT, "EgoLife", "clean", "utterances")
OUT_JSON = os.path.join(db.ROOT, "data", "insights.json")

MIN_N, MIN_Z, LIFT_HI, LIFT_LO = 100, 4.0, 1.25, 0.8
SELF_LIFT_HI, SELF_LIFT_LO = 1.1, 0.9      # self-vs-group has n ~ 22k vs 67k

DIRECTIVE = re.compile(
    r"^\s*(let's|let us|go|come|put|take|bring|give|hold|wait|stop|look|try|use|"
    r"open|close|turn|move|help|grab|get|check|remember|make sure|don't|do not|"
    r"please|you (should|need to|have to|must|can|go|take|put))\b", re.I)
LAUGH = re.compile(r"\b(ha(ha)+|he(he)+|hah|lol)\b|laugh", re.I)
ASSENT = re.compile(r"^\s*(ok|okay|yes|yeah|yep|sure|right|alright|all right|"
                    r"got it|fine|uh-huh|mm-hmm|mhm|exactly|correct)\b", re.I)
PLAN = re.compile(r"\b(plan|plans|planning|schedule|tomorrow|arrange|prepare|"
                  r"organi[sz]e|budget|order|buy|purchase|deadline|timeline|"
                  r"rehears\w*|we need to|we should|should we)\b", re.I)

FEATURES = {
    "question": lambda en, zh: "?" in en or "？" in zh or zh.endswith(("吗", "呢")),
    "directive": lambda en, zh: bool(DIRECTIVE.match(en)),
    "laughter": lambda en, zh: bool(LAUGH.search(en)) or "哈哈" in zh,
    "assent": lambda en, zh: bool(ASSENT.match(en)) and len(en.split()) <= 4,
    "planning": lambda en, zh: bool(PLAN.search(en)),
}
VERB2 = {"question": "ask questions", "directive": "give directions",
         "laughter": "laugh", "assent": "say yes / okay",
         "planning": "talk about plans", "talk_share": "talk"}
VERB3 = {"question": "asks questions", "directive": "gives directions",
         "laughter": "laughs", "assent": "says yes / okay",
         "planning": "talks about plans", "talk_share": "talks"}

PORTRAIT_PROMPT = """You are helping Jake understand himself from one week of lifelog data (6 housemates wearing cameras). Below are statistical findings, each with an id. Write 3-4 BIG-PICTURE hypotheses about Jake — his role in the group and how specific people relate to him (e.g. "You're the group's hub", "Shure loosens up around you").

Rules:
- Second person ("You ..."), title <= 9 words, body <= 45 words.
- Each hypothesis must cite the ids of the findings that support it (at least 2).
- Only name a role or trait (hub, coordinator, peacemaker, ...) if the cited findings actually measure it — e.g. coordinator needs a directions/planning/airtime finding. Never claim anything the findings don't show.
- Phrase as a hypothesis ("may", "seems"), not a fact.

FINDINGS:
@@FINDINGS@@

Reply with ONLY JSON: {"portraits": [{"title": str, "body": str, "supports": [ids]}]}"""


def cmh(tabs):
    """tabs: per-stratum (k1, n1, k2, n2) -> (n1, k1, p1, n2, k2, p2_std, z_mh).
    p2 is standardised to condition-1's stratum mix; z is Cochran–Mantel–Haenszel."""
    valid = [t for t in tabs if t[1] and t[3]]
    n1 = sum(t[1] for t in valid)
    if not n1:
        return 0, 0, 0.0, 0, 0, 0.0, 0.0
    k1, n2, k2 = (sum(t[i] for t in valid) for i in (0, 3, 2))
    p2 = sum(t[1] / n1 * t[2] / t[3] for t in valid)
    num = var = 0.0
    for a, na, b, nb in valid:
        N, m = na + nb, a + b
        num += a - na * m / N
        if N > 1:
            var += na * nb * m * (N - m) / (N * N * (N - 1))
    return n1, k1, k1 / n1, n2, k2, p2, (num / math.sqrt(var) if var else 0.0)


def load_moments(con):
    idmap = {(r["wearer"], round(r["t_abs"], 2), r["speaker"]): r["id"]
             for r in con.execute("SELECT id, wearer, t_abs, speaker FROM utterances")}
    rows = defaultdict(list)
    for rec in WEARERS.values():
        with open(os.path.join(CLEAN, f"{rec}.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                rows[r["moment_id"]].append(r)
    moments = []
    for rs in rows.values():
        selfrows = [r for r in rs if REC2NAME[r["recorder"]] == r["speaker"]]
        own_mic = len({r["speaker"] for r in selfrows}) == 1
        if own_mic:
            r0, spk = selfrows[0], selfrows[0]["speaker"]
        else:
            r0 = rs[0]
            spk = Counter(r["moment_speaker"] for r in rs).most_common(1)[0][0]
        t = (r0["day"] - 1) * 86400 + r0["start"]
        en, zh = (r0.get("text_en") or "").strip(), (r0.get("text_zh") or "").strip()
        moments.append({
            "t": t, "min": int(t // 60), "day": r0["day"], "speaker": spk,
            "heard": set(rs[0]["heard_by"]), "own_mic": own_mic, "nw": len(en.split()),
            "uid": idmap.get((r0["recorder"], round(t, 2), r0["speaker"])) if own_mic else None,
            "f": {k: fn(en, zh) for k, fn in FEATURES.items()} if own_mic else {}})
    return moments


def pick(cands, k=8):
    """cands: (day, uid, n_words) -> up to k uids spread across days."""
    good = [c for c in cands if 4 <= c[2] <= 18] or cands
    byday = defaultdict(list)
    for d, u, _ in good:
        byday[d].append(u)
    out = []
    while len(out) < k and any(byday.values()):
        for d in sorted(byday):
            if byday[d] and len(out) < k:
                out.append(byday[d].pop(len(byday[d]) // 2))
    return out


def effect(subject, other, feature, tabs, ex):
    n1, k1, p1, n2, k2, p2, z = cmh(tabs)
    return {"subject": subject, "other": other, "feature": feature,
            "n_with": n1, "k_with": k1, "p_with": p1,
            "n_without": n2, "k_without": k2, "p_without": p2,
            "lift": (p1 / p2) if p2 else None, "z": z, "examples": ex}


def compute_effects(moments):
    active, co = defaultdict(set), defaultdict(set)
    for m in moments:
        hb = sorted(m["heard"])
        for i, a in enumerate(hb):
            active[a].add(m["min"])
            for b in hb[i + 1:]:
                co[(a, b)].add(m["min"])

    def near(s, u):
        return (u - 1) in s or u in s or (u + 1) in s

    def presence(m, srec, orec):
        if orec in m["heard"] or near(co[tuple(sorted((srec, orec)))], m["min"]):
            return "with"
        return "without" if near(active[orec], m["min"]) else None

    own = defaultdict(list)          # speaker -> own-mic moments
    for m in moments:
        if m["own_mic"]:
            own[m["speaker"]].append(m)
    effects = []
    for s, srec in WEARERS.items():
        captured = [m for m in moments if srec in m["heard"]]
        for o, orec in WEARERS.items():
            if o == s:
                continue
            n = defaultdict(Counter)                          # cond -> g -> n
            k = defaultdict(lambda: defaultdict(Counter))     # cond -> f -> g -> k
            ex = defaultdict(list)
            for m in own[s]:
                c = presence(m, srec, orec)
                if not c:
                    continue
                g = len(m["heard"] - {srec, orec})
                n[c][g] += 1
                for f, v in m["f"].items():
                    if v:
                        k[c][f][g] += 1
                        if c == "with" and m["uid"]:
                            ex[f].append((m["day"], m["uid"], m["nw"]))
            for f in FEATURES:
                tabs = [(k["with"][f][g], n["with"][g], k["without"][f][g], n["without"][g])
                        for g in range(5)]
                effects.append(effect(s, o, f, tabs, pick(ex[f])))
            # talk share: S's share of conversation captured on S's mic,
            # excluding O's own lines (O's speech only exists when O is present)
            tn, tk, tex = defaultdict(Counter), defaultdict(Counter), []
            for m in captured:
                if m["speaker"] == o:
                    continue
                c = presence(m, srec, orec)
                if not c:
                    continue
                g = len(m["heard"] - {srec, orec})
                tn[c][g] += 1
                if m["speaker"] == s:
                    tk[c][g] += 1
                    if c == "with" and m["uid"]:
                        tex.append((m["day"], m["uid"], m["nw"]))
            tabs = [(tk["with"][g], tn["with"][g], tk["without"][g], tn["without"][g])
                    for g in range(5)]
            effects.append(effect(s, o, "talk_share", tabs, pick(tex)))
        # profile: S vs the other five wearers pooled, stratified by others present
        rest = [m for x, ms in own.items() if x in WEARERS and x != s for m in ms]
        for f in FEATURES:
            tabs = []
            for g in range(6):
                mine = [m for m in own[s] if len(m["heard"]) - 1 == g]
                theirs = [m for m in rest if len(m["heard"]) - 1 == g]
                tabs.append((sum(m["f"][f] for m in mine), len(mine),
                             sum(m["f"][f] for m in theirs), len(theirs)))
            effects.append(effect(s, "GROUP", f, tabs, pick(
                [(m["day"], m["uid"], m["nw"]) for m in own[s] if m["f"][f] and m["uid"]])))
    return effects


def notable(e):
    hi, lo = (SELF_LIFT_HI, SELF_LIFT_LO) if e["other"] == "GROUP" else (LIFT_HI, LIFT_LO)
    return (e["n_with"] >= MIN_N and e["n_without"] >= MIN_N and e["lift"]
            and abs(e["z"]) >= MIN_Z and (e["lift"] >= hi or e["lift"] <= lo))


def stat(e):
    return {"z": round(e["z"], 1), "lift": round(e["lift"], 2),
            "with": f"{e['p_with']:.1%} of {e['n_with']:,}",
            "without (size-matched)": f"{e['p_without']:.1%} of {e['n_without']:,}"}


def card(lens, e):
    s, o, f, l = e["subject"], e["other"], e["feature"], e["lift"]
    share = f == "talk_share"
    unit = "as much" if share else "as often"
    if lens == "self":
        title = f"You {VERB2[f]} {l:.1f}× {unit} as the rest of the group"
        body = (f"{e['p_with']:.1%} of your {e['n_with']:,} lines vs "
                f"{e['p_without']:.1%} for the other five wearers, compared in "
                f"same-size groups.")
    elif lens == "around_you":
        title = f"{s} {VERB3[f]} {l:.1f}× {unit} when you're around"
        what = (f"{s}'s share of the conversation (your lines excluded)" if share
                else f"{s}'s lines")
        body = (f"{what}: {e['p_with']:.1%} with you present (n={e['n_with']:,}) vs "
                f"{e['p_without']:.1%} while you were elsewhere (n={e['n_without']:,}), "
                f"matched on how many other housemates were around.")
    else:
        title = f"You {VERB2[f]} {l:.1f}× {unit} when {o} is around"
        what = (f"Your share of the conversation ({o}'s lines excluded)" if share
                else "Your lines")
        body = (f"{what}: {e['p_with']:.1%} with {o} present (n={e['n_with']:,}) vs "
                f"{e['p_without']:.1%} while {o} was elsewhere (n={e['n_without']:,}), "
                f"matched on how many other housemates were around.")
    return {"lens": lens, "title": title, "body": body + f" z={e['z']:.1f}",
            "status": "observed", "stat": stat(e),
            "evidence": {"utterance_ids": e["examples"]}}


def airtime_card(moments, effects):
    shares = Counter(m["speaker"] for m in moments if m["speaker"] in WEARERS)
    tot = sum(shares.values())
    (top1, n1), (top2, n2) = shares.most_common(2)
    if top1 != SUBJ:
        return None
    ex = next((e["examples"] for e in effects if e["subject"] == SUBJ
               and e["other"] == "GROUP" and e["feature"] == "directive"), [])
    return {"lens": "self", "status": "observed",
            "title": f"You do the most talking: {n1 / tot:.0%} of everything the six of you said",
            "body": f"{n1:,} of {tot:,} lines across the week; next is {top2} at "
                    f"{n2 / tot:.0%}. An even split would be 17% each.",
            "stat": {"share": round(n1 / tot, 3), "next": f"{top2} {n2 / tot:.1%}"},
            "evidence": {"utterance_ids": ex}}


def top(effects, pred, k):
    out = []
    for e in sorted((e for e in effects if pred(e) and notable(e)),
                    key=lambda e: -abs(e["z"])):
        out.append(e)
        if len(out) == k:
            break
    return out


def thread_cards(con):
    cards = []
    for g in con.execute(
            """SELECT e.id, e.canonical, COUNT(DISTINCT ev.day) d, COUNT(*) n,
                      MIN(ev.day) lo, MAX(ev.day) hi
               FROM entities e JOIN event_entities ee ON ee.entity_id=e.id
                 AND ee.relation='goal'
               JOIN events ev ON ev.id=ee.event_id AND ev.subject='A1_JAKE'
               GROUP BY e.id ORDER BY d DESC, n DESC LIMIT 2"""):
        eids = [r["id"] for r in con.execute(
            """SELECT ev.id FROM events ev JOIN event_entities ee ON ee.event_id=ev.id
               WHERE ee.entity_id=? AND ee.relation='goal' ORDER BY ev.t_start LIMIT 10""",
            (g["id"],))]
        cards.append({"lens": "threads", "status": "observed",
                      "title": f"Long-running thread: {g['canonical']}",
                      "body": f"{g['n']} events across {g['d']} days "
                              f"(day {g['lo']}–{g['hi']}).",
                      "stat": {"events": g["n"], "days": g["d"]},
                      "evidence": {"goal_ids": [g["id"]], "event_ids": eids}})
    for r in con.execute(
            """SELECT e.id, e.src_id, e.dst_id, e.confidence, e.evidence_type,
                      e.rationale, s.day sd, d.day dd, s.summary ss, d.summary ds,
                      (SELECT COUNT(*) FROM edge_evidence x WHERE x.edge_id=e.id) nev
               FROM edges e JOIN events s ON s.id=e.src_id JOIN events d ON d.id=e.dst_id
               WHERE e.relation='INFLUENCES' AND s.day < d.day
               ORDER BY nev > 0 DESC, e.confidence DESC LIMIT 3"""):
        uids = [x["utterance_id"] for x in con.execute(
            "SELECT utterance_id FROM edge_evidence WHERE edge_id=?", (r["id"],))]
        cards.append({"lens": "threads", "status": "hypothesis",
                      "title": f"Day {r['sd']} → day {r['dd']}: something carried forward",
                      "body": f"“{r['ss']}” → “{r['ds']}”. {r['rationale'] or ''}",
                      "stat": {"confidence": r["confidence"],
                               "evidence": r["evidence_type"]},
                      "evidence": {"edge_ids": [r["id"]],
                                   "event_ids": [r["src_id"], r["dst_id"]],
                                   "utterance_ids": uids}})
    return cards


def portrait_cards(findings):
    listing = "\n".join(f"{i}. {c['title']} — {c['body']}" for i, c in enumerate(findings))
    try:
        out = llm.chat_json(PORTRAIT_PROMPT.replace("@@FINDINGS@@", listing))
    except Exception as e:  # portrait layer is optional
        print("portrait LLM failed:", str(e)[:100])
        return []
    cards = []
    for p in (out or {}).get("portraits", []):
        sup = sorted({int(i) for i in p.get("supports", [])
                      if isinstance(i, (int, float)) and 0 <= int(i) < len(findings)})
        if len(sup) < 2 or not p.get("title"):
            continue
        uids = [u for i in sup for u in findings[i]["evidence"].get("utterance_ids", [])[:3]]
        body = re.sub(r"\s*\(\s*\d+(\s*,\s*\d+)*\s*\)", "", p.get("body", ""))
        cards.append({"lens": "portrait", "status": "hypothesis",
                      "title": p["title"], "body": body,
                      "stat": {"based_on": [findings[i]["title"] for i in sup]},
                      "evidence": {"utterance_ids": uids[:12]}})
    return cards


def main():
    con = db.connect()
    moments = load_moments(con)
    own = sum(1 for m in moments if m["own_mic"])
    mapped = sum(1 for m in moments if m["uid"])
    print(f"moments: {len(moments):,} (own-mic attributed: {own:,}, "
          f"mapped to DB ids: {mapped:,})", flush=True)
    effects = compute_effects(moments)

    air = airtime_card(moments, effects)
    self_c = ([air] if air else []) + [card("self", e) for e in top(
        effects, lambda e: e["subject"] == SUBJ and e["other"] == "GROUP", 4)]
    around = [card("around_you", e) for e in top(
        effects, lambda e: e["other"] == SUBJ, 8)]
    you_ar = [card("you_around", e) for e in top(
        effects, lambda e: e["subject"] == SUBJ and e["other"] in WEARERS, 6)]
    findings = self_c + around + you_ar
    cards = portrait_cards(findings) + findings + thread_cards(con)

    con.executescript("""
        DROP TABLE IF EXISTS presence_effects;
        CREATE TABLE presence_effects(subject TEXT, other TEXT, feature TEXT,
            n_with INTEGER, k_with INTEGER, p_with REAL,
            n_without INTEGER, k_without INTEGER, p_without REAL,
            lift REAL, z REAL, examples_json TEXT);
        DROP TABLE IF EXISTS insights;
        CREATE TABLE insights(id INTEGER PRIMARY KEY, rank INTEGER, lens TEXT,
            title TEXT, body TEXT, stat_json TEXT, status TEXT, evidence_json TEXT);
    """)
    con.executemany(
        "INSERT INTO presence_effects VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
        [(e["subject"], e["other"], e["feature"], e["n_with"], e["k_with"],
          e["p_with"], e["n_without"], e["k_without"], e["p_without"], e["lift"],
          e["z"], json.dumps(e["examples"])) for e in effects])
    con.executemany(
        "INSERT INTO insights(rank,lens,title,body,stat_json,status,evidence_json) "
        "VALUES(?,?,?,?,?,?,?)",
        [(i, c["lens"], c["title"], c["body"], json.dumps(c["stat"]), c["status"],
          json.dumps(c["evidence"])) for i, c in enumerate(cards)])
    con.commit()
    n_notable = sum(1 for e in effects if notable(e))
    json.dump({"moments": len(moments), "own_mic": own, "mapped": mapped,
               "notable_effects": n_notable, "cards": cards},
              open(OUT_JSON, "w"), ensure_ascii=False, indent=1)
    print(f"effects: {len(effects)} ({n_notable} notable) -> {len(cards)} insight cards")
    for c in cards:
        print(f"  [{c['lens']}/{c['status']}] {c['title']}")
    con.close()


if __name__ == "__main__":
    main()

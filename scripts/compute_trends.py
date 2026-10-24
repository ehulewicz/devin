"""Compute per-user trends and pairwise relationships from review/DAY1 CSVs.

No DB needed — reads review/DAY1_<WEARER>.csv directly.

Signals computed:
- activity trend: utterances captured per hour, per wearer (all speech around them
  + their own share).
- exposure: how much of each speaker's voice each wearer captured (who they were
  around / listening to).
- co-presence: utterances captured by BOTH wearers (same normalized text within
  +-6s) -> shared-moment count + overlapped seconds = strongest physical-proximity
  evidence (this is the heard_by signal from SPEC.md section 8).
- conversation edges: speaker A immediately followed (<4s gap) by speaker B in a
  wearer transcript -> directed reply weight, aggregated across files.

Writes analysis/trends.json.
"""
import csv
import json
import os
import re
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REVIEW = os.path.join(ROOT, "review")
OUT = os.path.join(ROOT, "analysis", "trends.json")

WEARERS = ["A1_JAKE", "A2_ALICE", "A3_TASHA", "A4_LUCIA", "A5_KATRINA", "A6_SHURE"]
NAME = {w: w.split("_")[1].title() for w in WEARERS}
WSHORT = {w: w.split("_")[0] for w in WEARERS}  # A1..A6

NORM = re.compile(r"[^a-z0-9 ]+")
CO_WINDOW = 6.0      # same text within +-6s = same captured moment
REPLY_GAP = 4.0      # speaker change within 4s = conversational exchange


def hms_to_s(h):
    hh, mm, ss = h.split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def norm(t):
    return NORM.sub("", (t or "").lower()).strip()


def load(w):
    rows = []
    with open(os.path.join(REVIEW, f"DAY1_{w}.csv")) as f:
        for r in csv.DictReader(f):
            en = (r["text_en"] or "").strip()
            zh = (r["text_zh"] or "").strip()
            if not en and not zh:
                continue
            rows.append({
                "t": hms_to_s(r["start"]),
                "speaker": r["speaker"],
                "en": en,
                "key": norm(en) or ("zh:" + norm(zh)),
            })
    rows.sort(key=lambda x: x["t"])
    return rows


def main():
    data = {w: load(w) for w in WEARERS}

    # ---- per-user trends -------------------------------------------------
    users = {}
    for w in WEARERS:
        rows = data[w]
        own = NAME[w]
        hours = defaultdict(lambda: [0, 0])     # hour -> [captured, own]
        heard = defaultdict(int)                # speaker -> count in my file
        for r in rows:
            h = int(r["t"] // 3600)
            hours[h][0] += 1
            if r["speaker"] == own:
                hours[h][1] += 1
            heard[r["speaker"]] += 1
        span = (rows[0]["t"], rows[-1]["t"]) if rows else (0, 0)
        top = sorted(heard.items(), key=lambda kv: -kv[1])
        users[w] = {
            "utterances_captured": len(rows),
            "own_utterances": heard.get(own, 0),
            "active_span_hours": [round(span[0] / 3600, 2), round(span[1] / 3600, 2)],
            "hourly": {h: {"captured": c, "own": o} for h, (c, o) in sorted(hours.items())},
            "speakers_heard": dict(top),
        }

    # ---- co-presence: shared captured moments ----------------------------
    # map: normalized text -> [times]
    index = {w: defaultdict(list) for w in WEARERS}
    for w in WEARERS:
        for r in data[w]:
            if r["key"]:
                index[w][r["key"]].append(r["t"])

    copresence = {}   # (w_i, w_j) -> {shared_utts, window}
    for i, wi in enumerate(WEARERS):
        for wj in WEARERS[i + 1:]:
            shared = 0
            ts_i, ts_j = [], []
            for key, times_i in index[wi].items():
                times_j = index[wj].get(key)
                if not times_j:
                    continue
                for ti in times_i:
                    if any(abs(ti - tj) <= CO_WINDOW for tj in times_j):
                        shared += 1
                        ts_i.append(ti)
                        break
            if shared:
                copresence[f"{WSHORT[wi]}-{WSHORT[wj]}"] = {
                    "shared_utterances": shared,
                    "first": round(min(ts_i) / 3600, 2),
                    "last": round(max(ts_i) / 3600, 2),
                    "span_hours": round((max(ts_i) - min(ts_i)) / 3600, 2),
                }

    # ---- conversation edges (reply structure, per file, aggregated) ------
    reply = defaultdict(int)      # (a -> b) weight
    reply_near = defaultdict(int) # wearer replies within 4s of speaker X
    for w in WEARERS:
        own = NAME[w]
        for a, b in zip(data[w], data[w][1:]):
            if a["speaker"] == b["speaker"]:
                continue
            if b["t"] - a["t"] > REPLY_GAP:
                continue
            reply[(a["speaker"], b["speaker"])] += 1
            if b["speaker"] == own:
                reply_near[(a["speaker"], own)] += 1

    edges = [{"from": a, "to": b, "replies": n}
             for (a, b), n in sorted(reply.items(), key=lambda kv: -kv[1])
             if n >= 30]

    out = {
        "basis": "review/DAY1_*.csv — day-1 transcripts for all 6 wearers; "
                 "week-level trends extrapolated from HANDOFF.md graph stats",
        "signals": {
            "co_presence": "same utterance captured by two wearers within 6s "
                           "(physical proximity / heard_by signal)",
            "replies": "speaker A immediately followed (<4s) by speaker B",
            "hourly": "utterances captured per hour; 'own' = wearer's own speech",
        },
        "users": users,
        "co_presence": copresence,
        "reply_edges_top": edges,
        "wearer_replies_received": {
            f"{a}->{b}": n for (a, b), n in sorted(reply_near.items(), key=lambda kv: -kv[1])
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), ensure_ascii=False, indent=1)
    # self-contained JS payload for web/trends.html (works over file:// too)
    js_path = os.path.join(ROOT, "web", "trends-data.js")
    with open(js_path, "w") as f:
        f.write("window.TRENDS_DATA = ")
        json.dump(out, f, ensure_ascii=False)
        f.write(";\n")
    print(json.dumps({k: (len(v) if isinstance(v, (dict, list)) else v)
                      for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()

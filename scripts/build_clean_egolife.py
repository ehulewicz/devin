"""Build a cleaned, normalized copy of the EgoLife transcripts + dense captions.

Raw SRTs under EgoLife/EgoLifeCap are left untouched. Output (EgoLife/clean/):
  utterances/<RECORDER>.jsonl  one row per spoken utterance, absolute time, canonical speaker
  captions/<RECORDER>.jsonl    one row per dense-caption segment (first-person, Chinese)
  speakers.json                canonical speakers, the raw labels mapped to each, counts
  manifest.json                coverage, counts, drops and flag totals

Usage: python3 -I scripts/build_clean_egolife.py [EgoLife dir]
"""
import collections, difflib, json, os, re, sys

SRC_ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                           os.path.join(os.path.dirname(__file__), "..", "EgoLife"))
CAP_ROOT = os.path.join(SRC_ROOT, "EgoLifeCap")
OUT = os.path.join(SRC_ROOT, "clean")

TS = re.compile(r"^(\d+):(\d\d):(\d\d)[,.](\d+)\s*-->\s*(\d+):(\d\d):(\d\d)[,.](\d+)")
FILE_HOUR = re.compile(r"_(\d{2})(\d{2})(\d{2})\d{2}(?:\.mp4)?\.srt$")  # A1_JAKE_DAY1_11000000(.mp4).srt
SPEAKER_LINE = re.compile(r"^([^:：]{1,40})[:：]\s*(.*)$")
CJK = re.compile(r"[一-鿿]")
NORM = re.compile(r"[^\w一-鿿]+")

# The Chinese line's label is authoritative; the English line is a machine translation that
# sometimes relabels speakers (Shure的女友 -> "Shure"/"Sister", Lucia -> "Jake").
ALIASES = {
    "其他人": "Others", "其它人": "Others", "其他": "Others", "其它": "Others", "其他它": "Others",
    "其人也": "Others", "Other": "Others", "Other people": "Others", "Other person": "Others",
    "Someone": "Others", "Unidentified": "Others", "Everyone": "Others",
    "Shure的女友": "Shure's GF", "Shure's GF": "Shure's GF",
    "刘婷": "Liu Ting", "Liuting": "Liu Ting", "Ting Liu": "Liu Ting", "Liu Ting": "Liu Ting",
    "男嘉宾": "Male Guest", "Male Guest": "Male Guest", "女嘉宾": "Female Guest",
    "Female Guest": "Female Guest", "嘉宾": "Guest", "Guest": "Guest",
    "陌生人": "Stranger", "Stranger": "Stranger", "路人": "Stranger", "Bystander": "Stranger",
    "小狗老板": "Dog Boss", "Dog Boss": "Dog Boss", "Puppy Boss": "Dog Boss",
    "Puppy Owner": "Dog Boss", "小狗家长": "Dog Boss", "Small Dog Owner": "Dog Boss",
    "音乐声": "Music", "Music": "Music", "(歌词)": "Music", "(Lyrics)": "Music",
    "手机声音": "Phone Voice", "Phone Voice": "Phone Voice", "AI": "Phone Voice", "Voice": "Phone Voice",
    "邻居": "Neighbor", "Neighbor": "Neighbor", "阿姨": "Auntie", "Auntie": "Auntie",
    "老板": "Boss", "Boss": "Boss", "服务员": "Server", "Server": "Server", "司机": "Driver",
    "Driver": "Driver", "外卖员": "Delivery Person", "快递员": "Delivery Person",
    "Delivery Guy": "Delivery Person", "Delivery person": "Delivery Person", "Courier": "Delivery Person",
    "谢斌竹": "Xie Binzhu", "Xie Binzhu": "Xie Binzhu", "杨硕": "Yang Shuo", "Yang Shuo": "Yang Shuo",
    "花丛": "Huacong", "Huacong": "Huacong", "Sister": "Sister",
    # transcriber spellings / nicknames of the housemates
    "俢硕": "Shure", "修硕": "Shure", "修确": "Shure", "修厦": "Shure", "修硕能": "Shure",
    "小Shure": "Shure", "杨Shure": "Shure", "CShure": "Shure",
    "小薇": "Tasha", "薇": "Tasha", "Tahsa": "Tasha", "图纱": "Tasha", "2Tasha": "Tasha", "xiaowei": "Tasha",
    "露丝": "Lucia", "路西": "Lucia", "露西板": "Lucia", "luxi": "Lucia",
    "小Lucia": "Little Lucia", "Little Lucia": "Little Lucia",
    "小汾": "Katrina", "小芬": "Katrina", "爱丽": "Alice", "爱丽沙": "Alice", "爱丽比": "Alice",
    "小黄": "Nicous", "叉叉": "Jack",
}
HOUSEMATES = ["Jake", "Alice", "Tasha", "Lucia", "Katrina", "Shure"]
CANON = set(HOUSEMATES) | {"Nicous", "Jack", "Choiszt", "Violet"}
LOOKUP = ({k.lower(): v for k, v in ALIASES.items()} | {v.lower(): v for v in ALIASES.values()}
          | {n.lower(): n for n in CANON})
SUFFIXES = sorted(LOOKUP, key=len, reverse=True)  # longest first: "其他人" before "其他"
KNOWN = set(LOOKUP.values())


def canonical_speaker(label, fallback=None):
    """Return (speaker, repaired). Handles labels the previous line's text bled into
    ("就按一个蛋挞一个币其他人" -> Others) and trailing notes ("修硕（通话）" -> Shure)."""
    label = label.strip()
    bare = re.sub(r"\s*[（(][^)）]*[)）]$", "", label) or label
    for cand in (label, bare):
        if cand.lower() in LOOKUP:
            return LOOKUP[cand.lower()], cand != label
    for key in SUFFIXES:
        if len(bare) > len(key) and bare.lower().endswith(key):
            return LOOKUP[key], True
    if fallback:
        spk, _ = canonical_speaker(fallback)
        if spk in KNOWN:
            return spk, True
    return "Unknown", True


def split_speaker(line):
    m = SPEAKER_LINE.match(line)
    return (m.group(1).strip(), m.group(2).strip()) if m else (None, line.strip())


def srt_blocks(path):
    """Yield (start, end, body_lines). Line-based, so blocks missing their blank-line
    separator (seen in A4_LUCIA DAY6) are still split correctly."""
    lines = open(path, encoding="utf-8-sig").read().splitlines()
    cur, i = None, 0
    while i < len(lines):
        line = lines[i].strip()
        m = TS.match(lines[i + 1].strip()) if line.isdigit() and i + 1 < len(lines) else None
        if m:
            if cur:
                yield cur
            g = [int(x) for x in m.groups()]
            st = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 10 ** len(m.group(4))
            en = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 10 ** len(m.group(8))
            cur, i = (st, en, []), i + 2
            continue
        if line and cur:
            cur[2].append(line)
        i += 1
    if cur:
        yield cur


def hms(t):
    t = int(t)
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def source_files(kind):
    for rec in sorted(os.listdir(os.path.join(CAP_ROOT, kind))):
        rdir = os.path.join(CAP_ROOT, kind, rec)
        if not os.path.isdir(rdir):  # stray .DS_Store etc.
            continue
        days = [d for d in os.listdir(rdir) if re.fullmatch(r"DAY\d+", d)]
        for day in sorted(days, key=lambda d: int(d[3:])):
            for fn in sorted(os.listdir(os.path.join(rdir, day))):
                m = FILE_HOUR.search(fn)
                if m:
                    off = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
                    yield rec, int(day[3:]), os.path.join(rdir, day, fn), off, m.group(1)


def build_utterances(stats, label_map):
    rows = []
    for rec, day, path, off, hh in source_files("Transcript"):
        wearer = rec.split("_", 1)[1].capitalize()
        stats["transcript_files"] += 1
        for n, (st, en, body) in enumerate(srt_blocks(path)):
            stats["blocks"] += 1
            flags = []
            parsed = [split_speaker(l) for l in body]
            # Last line is the English translation when it is labelled and CJK-free.
            if len(parsed) >= 2 and parsed[-1][0] and (len(parsed) == 2 or not CJK.search(parsed[-1][1])):
                src, (en_label, text_en) = parsed[:-1], parsed[-1]
            elif len(parsed) == 1 and not CJK.search(parsed[0][1]):
                src, (en_label, text_en) = parsed, parsed[0]  # spoken in English, no translation needed
            else:
                src, en_label, text_en = parsed, None, None
            zh_label = next((l for l, _ in src if l), None)
            text_zh = " / ".join(t for _, t in src if t)
            if not text_zh and not text_en:
                stats["dropped_empty"] += 1
                continue
            if len(src) > 1:
                flags.append("multiline")
            if text_en is None:
                flags.append("untranslated")
            if zh_label:
                speaker, repaired = canonical_speaker(zh_label, fallback=en_label)
            elif en_label:
                speaker, repaired = canonical_speaker(en_label)
            else:
                speaker, repaired = "Unknown", False
            if repaired:
                flags.append("speaker_label_repaired")
            if en_label and canonical_speaker(en_label)[0] != speaker:
                flags.append("en_speaker_mismatch")
            if speaker == "Unknown":
                flags.append("speaker_unknown")
            if st >= 3600:
                flags.append("past_hour_end")
            if len(NORM.sub("", text_zh or text_en or "")) <= 1:
                flags.append("filler")
            label_map[speaker].update(l for l in (zh_label, en_label) if l)
            rows.append({
                "id": f"{rec}/D{day}/{hh}/{n:05d}", "recorder": rec, "day": day,
                "start": round(off + st, 3), "end": round(off + en, 3), "time": hms(off + st),
                "speaker": speaker, "is_wearer": speaker == wearer,
                "text_zh": text_zh or None, "text_en": text_en or None,
                "raw_speaker_zh": zh_label, "raw_speaker_en": en_label, "flags": flags,
            })
    return rows


def link_moments(rows, stats):
    """Group copies of the same utterance captured by several recorders' microphones into
    one moment_id, so downstream retrieval doesn't treat them as separate memories."""
    by_day = collections.defaultdict(list)
    for r in rows:
        by_day[r["day"]].append(r)
    for day, items in by_day.items():
        items.sort(key=lambda r: r["start"])
        active, seq = [], 0  # groups: dict(start, norm, speaker, recorders, members)
        for r in items:
            norm = NORM.sub("", r["text_zh"] or r["text_en"] or "").lower()
            active = [g for g in active if r["start"] - g["start"] <= 2.0]
            best, best_sim = None, 0.0
            for g in active:
                if r["recorder"] in g["recorders"]:
                    continue
                dt = abs(r["start"] - g["start"])
                if len(norm) <= 3 or len(g["norm"]) <= 3:  # 嗯/好/对: only exact, same speaker, tight timing
                    ok = norm == g["norm"] and r["speaker"] == g["speaker"] and dt <= 0.6
                    sim = 1.0 if ok else 0.0
                else:
                    sm = difflib.SequenceMatcher(None, norm, g["norm"], autojunk=False)
                    sim = sm.ratio() if sm.quick_ratio() >= 0.6 else 0.0
                    ok = sim >= 0.6
                if ok and sim > best_sim:
                    best, best_sim = g, sim
            if best is None:
                seq += 1
                best = {"id": f"D{day}-{seq:06d}", "start": r["start"], "norm": norm,
                        "speaker": r["speaker"], "recorders": set(), "members": []}
                active.append(best)
            best["recorders"].add(r["recorder"])
            best["members"].append(r)
            r["moment_id"] = best["id"]
            if len(best["members"]) > 1:
                heard = sorted(best["recorders"])
                votes = collections.Counter(m["speaker"] for m in best["members"]).most_common()
                # Majority speaker across recorders; on a tie each copy keeps its own label.
                consensus = votes[0][0] if len(votes) == 1 or votes[0][1] > votes[1][1] else None
                for m in best["members"]:
                    m["heard_by"] = heard
                    m["moment_speaker"] = consensus or m["speaker"]
                    if len(votes) > 1 and "speaker_disputed" not in m["flags"]:
                        m["flags"].append("speaker_disputed")
            else:
                r["heard_by"] = [r["recorder"]]
                r["moment_speaker"] = r["speaker"]
        stats["moments"] += seq


def build_captions(stats):
    rows = []
    for rec, day, path, off, hh in source_files("DenseCaption"):
        stats["caption_files"] += 1
        for n, (st, en, body) in enumerate(srt_blocks(path)):
            text = "；".join(body).strip()
            if not text:
                stats["captions_dropped_empty"] += 1
                continue
            rows.append({"id": f"{rec}/D{day}/{hh}/{n:05d}", "recorder": rec, "day": day,
                         "start": round(off + st, 3), "end": round(off + en, 3),
                         "time": hms(off + st), "text_zh": text})
    return rows


def write_jsonl(rows, subdir):
    os.makedirs(os.path.join(OUT, subdir), exist_ok=True)
    by_rec = collections.defaultdict(list)
    for r in rows:
        by_rec[r["recorder"]].append(r)
    for rec, rs in by_rec.items():
        rs.sort(key=lambda r: (r["day"], r["start"]))
        with open(os.path.join(OUT, subdir, f"{rec}.jsonl"), "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    stats, label_map = collections.Counter(), collections.defaultdict(collections.Counter)
    utts = build_utterances(stats, label_map)
    link_moments(utts, stats)
    caps = build_captions(stats)
    write_jsonl(utts, "utterances")
    write_jsonl(caps, "captions")

    coverage = collections.defaultdict(dict)
    for r in utts:
        coverage[r["recorder"]].setdefault(f"DAY{r['day']}", set()).add(r["time"][:2])
    flags = collections.Counter(f for r in utts for f in r["flags"])
    heard = collections.Counter(len(r["heard_by"]) for r in utts)
    spk_counts = collections.Counter(r["speaker"] for r in utts)
    manifest = {
        "source": os.path.relpath(CAP_ROOT, SRC_ROOT),
        "counts": {"utterances": len(utts), "captions": len(caps), **stats},
        "flags": dict(flags.most_common()),
        "utterances_by_recorder_count": dict(sorted(heard.items())),
        "hours_covered": {rec: {d: sorted(h) for d, h in days.items()} for rec, days in sorted(coverage.items())},
    }
    speakers = {s: {"utterances": c, "housemate": s in HOUSEMATES,
                    "raw_labels": dict(label_map[s].most_common())}
                for s, c in spk_counts.most_common()}
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), ensure_ascii=False, indent=1)
    json.dump(speakers, open(os.path.join(OUT, "speakers.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(manifest["counts"], indent=1))
    print("flags:", manifest["flags"])
    print("utterance copies by #recorders:", manifest["utterances_by_recorder_count"])
    print("speakers:", {s: v["utterances"] for s, v in list(speakers.items())[:20]})


if __name__ == "__main__":
    main()

"""Stage 1 — ingest: parse all SRT transcripts into the utterances table.

Reuses the battle-tested parser in scripts/build_clean_egolife.py (speaker
aliases, zh/en splitting, missing blank-line handling). Idempotent: truncates
and reloads. ~10s for 402 files.

Usage: python -m cde.ingest
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import build_clean_egolife as clean  # noqa: E402

from cde import db  # noqa: E402

WEARERS = {
    "A1_JAKE": ("Jake", "A1", 1),
    "A2_ALICE": ("Alice", "A2", 0),
    "A3_TASHA": ("Tasha", "A3", 0),
    "A4_LUCIA": ("Lucia", "A4", 0),
    "A5_KATRINA": ("Katrina", "A5", 0),
    "A6_SHURE": ("Shure", "A6", 0),
}


def parse_file(path):
    """Yield (srt_offset_start, srt_offset_end, speaker, text_en, text_zh)."""
    for st, en, body in clean.srt_blocks(path):
        parsed = [clean.split_speaker(l) for l in body]
        if len(parsed) >= 2 and parsed[-1][0] and (
            len(parsed) == 2 or not clean.CJK.search(parsed[-1][1])
        ):
            src, (en_label, text_en) = parsed[:-1], parsed[-1]
        elif len(parsed) == 1 and not clean.CJK.search(parsed[0][1]):
            src, (en_label, text_en) = parsed, parsed[0]
        else:
            src, en_label, text_en = parsed, None, None
        zh_label = next((l for l, _ in src if l), None)
        text_zh = " / ".join(t for _, t in src if t)
        if not text_zh and not text_en:
            continue
        if zh_label:
            speaker, _ = clean.canonical_speaker(zh_label, fallback=en_label)
        elif en_label:
            speaker, _ = clean.canonical_speaker(en_label)
        else:
            speaker = "Unknown"
        yield st, en, speaker, text_en, text_zh or None


def main():
    con = db.init_db(db.connect())
    cur = con.cursor()
    cur.executemany(
        "INSERT OR REPLACE INTO participants(name, wearer_code, is_subject) VALUES(?,?,?)",
        [(n, c, s) for n, c, s in WEARERS.values()],
    )
    cur.execute("DELETE FROM utterances")
    n_files = n_rows = 0
    for rec, day, path, off, hh in clean.source_files("Transcript"):
        n_files += 1
        for st, en, speaker, text_en, text_zh in parse_file(path):
            t_abs = (day - 1) * 86400 + off + st
            cur.execute(
                """INSERT INTO utterances(wearer,file,day,t_abs,t_end,hms,speaker,text_en,text_zh)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (rec, os.path.basename(path), day, t_abs, (day - 1) * 86400 + off + en,
                 clean.hms(off + st), speaker, text_en, text_zh),
            )
            n_rows += 1
    con.commit()
    print(f"files={n_files} utterances={n_rows}")
    for r in con.execute(
        "SELECT wearer, day, COUNT(*) c FROM utterances GROUP BY wearer, day ORDER BY wearer, day"
    ):
        print(f"  {r['wearer']:10s} day{r['day']}  {r['c']}")
    con.close()


if __name__ == "__main__":
    main()

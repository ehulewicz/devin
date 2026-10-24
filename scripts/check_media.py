"""Phase 0 — media discovery for the audio feature.

Finds playable media under EgoLife/, checks ffmpeg availability, and reports
coverage: what fraction of ingested utterance hour-files have a media sibling.

Safe to run anywhere — read-only. On the pipeline machine:

    python3 scripts/check_media.py [EgoLife dir]

Exit code 0 always; read the report.
"""
import os
import re
import shutil
import sqlite3
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EGO = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(ROOT, "EgoLife")

MEDIA_EXT = (".mp4", ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".mkv", ".mov")
FILE_HOUR = re.compile(r"_DAY(\d+)_(\d{2})(\d{2})(\d{2})\d{2}(?:\.mp4)?\.srt$")


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def main():
    print(f"EgoLife root: {EGO}  (exists={os.path.isdir(EGO)})")
    print(f"ffmpeg: {find_ffmpeg() or 'NOT FOUND — `brew install ffmpeg` or `pip install imageio-ffmpeg`'}")
    if not os.path.isdir(EGO):
        print("\nNo dataset here — run this on the pipeline machine.")
        return

    # 1) inventory media files
    media = []
    for dp, _, fns in os.walk(EGO):
        for fn in fns:
            if fn.lower().endswith(MEDIA_EXT):
                media.append(os.path.join(dp, fn))
    print(f"\nmedia files found: {len(media)}")
    by_ext = {}
    for m in media:
        by_ext.setdefault(os.path.splitext(m)[1].lower(), []).append(m)
    for ext, lst in sorted(by_ext.items()):
        print(f"  {ext}: {len(lst)}  e.g. {os.path.relpath(lst[0], EGO)}")

    if not media:
        print("\nNo media — feature falls back to synthetic TTS (say / OpenAI tts-1).")
        return

    # 2) probe audio stream of the first file
    probe = shutil.which("ffprobe")
    if probe:
        r = subprocess.run(
            [probe, "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_name,duration", "-of", "csv=p=0", media[0]],
            capture_output=True, text=True)
        print(f"\nffprobe {os.path.basename(media[0])}: {r.stdout.strip() or r.stderr.strip()[:120]}")

    # 3) coverage vs utterances table (if the DB exists)
    db = os.path.join(ROOT, "data", "cde.db")
    media_base = {os.path.splitext(os.path.basename(m))[0]: m for m in media}
    if os.path.exists(db):
        con = sqlite3.connect(db)
        files = [r[0] for r in con.execute("SELECT DISTINCT file FROM utterances")]
        hit = miss = 0
        misses = []
        for f in files:
            base = f[:-4] if f.endswith(".srt") else f          # strip .srt
            base = base[:-4] if base.endswith(".mp4") else base  # -> ..._DAY1_13000000
            if base in media_base or f in media_base:
                hit += 1
            else:
                miss += 1
                misses.append(f)
        print(f"\nutt hour-file coverage: {hit}/{hit+miss} files have media "
              f"({100*hit/max(hit+miss,1):.0f}%)")
        for m in misses[:8]:
            print(f"  missing: {m}")
        con.close()
    else:
        print(f"\nno data/cde.db — skipped utterance coverage check")
        # still show whether media names look like the SRT naming
        sample = os.path.basename(media[0])
        m = FILE_HOUR.search(sample + (".srt" if not sample.endswith(".srt") else ""))
        print(f"sample media name '{sample}': "
              f"{'matches SRT hour-file pattern' if m else 'does NOT match expected pattern — check layout'}")


if __name__ == "__main__":
    main()

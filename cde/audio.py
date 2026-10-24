"""Voice feature — slice original EgoLife audio behind utterances, and cheap
energy-VAD features for trend deepening.

Media lookup: utterances.file is the SRT name (A1_JAKE_DAY1_11000000[.mp4].srt);
the media sibling shares the stem (…_11000000.mp4|m4a|wav…). Index maps stem ->
path by walking CDE_EGOLIFE (default <repo>/EgoLife) once, lazily.

Clip serving: in-file offset = t_abs - (day-1)*86400 - file_offset, where
file_offset comes from the filename HH MM SS fields. Clips cached in
data/audio/u<id>.m4a — first play costs one ffmpeg seek, then it's a file send.

ffmpeg: system binary, else imageio-ffmpeg's bundled one (pip install
imageio-ffmpeg — pure wheel, no brew). Everything degrades to None/False when
no media or no ffmpeg, so the rest of the app never breaks.

Trends (--audio flag in scripts/compute_trends.py): silence_spans() runs
ffmpeg silencedetect over a media file -> voiced timeline used to refine
per-utterance speech seconds and cross-speaker overlap.

Usage:
    python -m cde.audio --warm 123 456       # pre-slice utterance clips
    python -m cde.audio --event 1294         # pre-slice one event's evidence
    python -m cde.audio --probe <media>      # print voiced-span stats
"""
import bisect
import os
import re
import shutil
import subprocess
import sys

from cde import db

EGO_ROOT = os.environ.get("CDE_EGOLIFE") or os.path.join(db.ROOT, "EgoLife")
AUDIO_DIR = os.path.join(db.ROOT, "data", "audio")
MEDIA_EXT = (".mp4", ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".mkv", ".mov")

# A1_JAKE_DAY1_11000000(.mp4).srt -> day=1, 11:00:00 file offset
FILE_STAMP = re.compile(r"_DAY(\d+)_(\d{2})(\d{2})(\d{2})\d{2}")
SRT_NAME = re.compile(r"(.+)_DAY(\d+)_\d{8}(?:\.mp4)?\.srt$")
# raw media chunks: DAY1_A1_JAKE_11094208.mp4 -> day=1, subject=A1_JAKE, clip start 11:09:42
MEDIA_STAMP = re.compile(r"^DAY(\d+)_(.+)_(\d{2})(\d{2})(\d{2})\d{2}$")
PAD = 0.4          # seconds of context around each clip
MAX_DUR = 25.0     # hard cap per clip
CLIP_GAP = 45.0    # assumed coverage for the last chunk of a (day, subject) run

_index = None      # media stem -> path
_clips = None      # (day, subject) -> [(start_sec, path)] sorted
_ffmpeg = None


def ffmpeg():
    global _ffmpeg
    if _ffmpeg is None:
        _ffmpeg = shutil.which("ffmpeg") or ""
        if not _ffmpeg:
            try:
                import imageio_ffmpeg
                _ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                _ffmpeg = ""
    return _ffmpeg or None


def media_index():
    """stem -> absolute media path, built lazily by walking EGO_ROOT."""
    global _index
    if _index is None:
        _index = {}
        if os.path.isdir(EGO_ROOT):
            for dp, _, fns in os.walk(EGO_ROOT):
                for fn in fns:
                    if fn.lower().endswith(MEDIA_EXT):
                        _index.setdefault(os.path.splitext(fn)[0], os.path.join(dp, fn))
    return _index


def media_for_file(srt_name):
    """utterance.file -> media path or None."""
    base = srt_name[:-4] if srt_name.endswith(".srt") else srt_name
    if base.endswith(".mp4"):
        base = base[:-4]
    idx = media_index()
    if base in idx:
        return idx[base]
    for ext in MEDIA_EXT:                       # e.g. stem.mp4.srt naming
        p = idx.get(base + ext)
        if p:
            return p
    return None


def clip_index():
    """(day, subject) -> [(in-day start sec, path)] for chunked media stems."""
    global _clips
    if _clips is None:
        _clips = {}
        for stem, path in media_index().items():
            m = MEDIA_STAMP.match(stem)
            if not m:
                continue
            sec = int(m.group(3)) * 3600 + int(m.group(4)) * 60 + int(m.group(5))
            _clips.setdefault((int(m.group(1)), m.group(2)), []).append((sec, path))
        for v in _clips.values():
            v.sort()
    return _clips


def clip_for_time(srt_name, day, t_day):
    """chunked-media lookup: clip covering in-day time t_day, else None."""
    sm = SRT_NAME.search(srt_name)
    if not sm:
        return None
    clips = clip_index().get((int(sm.group(2)), sm.group(1)))
    if not clips:
        return None
    starts = [s for s, _ in clips]
    i = bisect.bisect_right(starts, t_day) - 1
    if i < 0:
        return None
    start, path = clips[i]
    end = clips[i + 1][0] if i + 1 < len(clips) else start + CLIP_GAP
    if t_day >= end + 2:
        return None
    return path, t_day - start


def file_offset_seconds(srt_name):
    m = FILE_STAMP.search(srt_name)
    return int(m.group(2)) * 3600 + int(m.group(3)) * 60 + int(m.group(4)) if m else None


def locate(u):
    """utterance row -> (media_path, in_file_start, duration) or None."""
    dur = (u["t_end"] or u["t_abs"] + 1.5) - u["t_abs"]
    if dur <= 0:
        return None
    t_day = u["t_abs"] - (u["day"] - 1) * 86400
    media = media_for_file(u["file"])
    off = file_offset_seconds(u["file"])
    if media and off is not None:
        start = t_day - off
        if start < -2:
            return None
        return media, max(start - PAD, 0.0), min(dur + 2 * PAD, MAX_DUR)
    hit = clip_for_time(u["file"], u["day"], t_day)
    if not hit:
        return None
    path, rel = hit
    pad = min(PAD, rel)
    return path, rel - pad, min(dur + pad + PAD, MAX_DUR)


def clip_for_utterance(con, uid):
    """Return path to a cached .m4a clip for utterance uid, or None."""
    u = con.execute(
        "SELECT id, file, day, t_abs, t_end FROM utterances WHERE id=?", (uid,)
    ).fetchone()
    if not u:
        return None
    loc = locate(u)
    exe = ffmpeg()
    if not loc or not exe:
        return None
    media, start, dur = loc
    out = os.path.join(AUDIO_DIR, f"u{uid}.m4a")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out
    os.makedirs(AUDIO_DIR, exist_ok=True)
    tmp = out + ".tmp"                    # atomic: failed/concurrent writes never served
    try:
        r = subprocess.run(
            [exe, "-y", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}", "-i", media,
             "-vn", "-ac", "1", "-ar", "16000", "-c:a", "aac", "-f", "adts", tmp],
            capture_output=True, timeout=60)
        ok = r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) >= 400
    except subprocess.TimeoutExpired:
        ok = False
    if not ok:
        if os.path.exists(tmp):
            os.unlink(tmp)
        return None
    os.replace(tmp, out)
    return out


# ---------- energy VAD for trend deepening ----------

_SIL_START = re.compile(r"silence_start: ([\d.]+)")
_SIL_END = re.compile(r"silence_end: ([\d.]+) \| silence_duration: ([\d.]+)")


def silence_spans(media, noise_db=-35, min_sil=0.35):
    """ffmpeg silencedetect -> list of (silence_start, silence_end) seconds."""
    exe = ffmpeg()
    if not exe:
        return []
    r = subprocess.run(
        [exe, "-i", media, "-af", f"silencedetect=n={noise_db}dB:d={min_sil}",
         "-f", "null", "-"],
        capture_output=True, text=True, timeout=600)
    spans, cur = [], None
    for line in r.stderr.splitlines():
        m = _SIL_START.search(line)
        if m:
            cur = float(m.group(1))
            continue
        m = _SIL_END.search(line)
        if m and cur is not None:
            spans.append((cur, float(m.group(1))))
            cur = None
    if cur is not None:                   # silence running to EOF
        spans.append((cur, float("inf")))
    return spans


def voiced_seconds(media, t0, t1):
    """Speech seconds inside [t0,t1] of media = span minus intersecting silence."""
    sil = silence_spans(media)
    gap = sum(max(0.0, min(t1, se) - max(t0, ss)) for ss, se in sil if se > t0 and ss < t1)
    return max(0.0, (t1 - t0) - gap)


def status():
    return {"available": bool(ffmpeg() and media_index()),
            "ffmpeg": bool(ffmpeg()), "media_files": len(media_index()),
            "ego_root": EGO_ROOT if os.path.isdir(EGO_ROOT) else None}


def _warm(con, uids):
    ok = fail = 0
    for uid in uids:
        if clip_for_utterance(con, uid):
            ok += 1
        else:
            fail += 1
            print(f"  u{uid}: no clip", flush=True)
    print(f"warmed {ok} clips, {fail} failed")


def main():
    args = sys.argv[1:]
    con = db.connect()
    if "--probe" in args:
        media = args[args.index("--probe") + 1]
        spans = silence_spans(media)
        print(f"{len(spans)} silence spans; first/last: {spans[:3]} … {spans[-3:] if spans else ''}")
        return
    if "--event" in args:
        eid = int(args[args.index("--event") + 1])
        uids = [r["utterance_id"] for r in con.execute(
            "SELECT utterance_id FROM event_utterances WHERE event_id=?", (eid,))]
        _warm(con, uids)
        return
    if "--warm" in args:
        i = args.index("--warm") + 1
        uids = [int(x) for x in args[i:] if x.isdigit()]
        _warm(con, uids)
        return
    print(status())


if __name__ == "__main__":
    main()

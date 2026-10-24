# Cognitive Dynamics Engine — Build Spec

How each step of the pipeline works, as implemented in `cde/`.

**Demo claim:** a conversation today connects to an experience days earlier
through a traceable chain of evidence — not just because embeddings are similar.

```
SRT transcripts ──ingest──► utterances (6 wearers, en+zh, absolute time)
                               │
              LLM extract (Jake, per hour file, utterance-ID grounded)
                               │
   events · thoughts · emotions · goals · behaviors · causal_claims
                               │
              embed + entity resolution (topics / goals / people)
                               │
    deterministic edges ──► scored candidates ──► LLM edge judge
                               │
                SQLite graph ──► FastAPI ──► Cytoscape timeline UI
```

Storage is SQLite (`data/cde.db`); embeddings are float32 BLOBs with brute-force
numpy cosine. The schema mirrors the Postgres design 1:1 — the migration path
is swapping `cde/db.py` for psycopg + pgvector, same tables.

---

## 0. LLM backend — `cde/llm.py`

One wrapper, backend chosen from `.env`/environment:

| Order | Backend | How it's called |
|---|---|---|
| 1 | OpenAI | `OPENAI_API_KEY` set → `chat.completions` (`CDE_CHAT_MODEL`, default `gpt-4o-mini`); `text-embedding-3-small` for embeddings |
| 2 | Codex CLI | `codex exec --sandbox read-only -o <tmpfile> <prompt>`; the `-o` file holds the final message, which keeps stdout's session/tool noise out of the parse. Ran with `model_reasoning_effort="low"` for latency. |
| 3 | Claude CLI | `claude -p --output-format text` with the prompt on stdin |

`chat()` tries backends in order and is sticky on success — if the active
backend hits a rate/session limit mid-run, the next call falls through to the
next backend automatically. `CDE_BACKEND` in `.env` forces one.
`chat_json()` parses leniently (strips ``` fences, takes the outermost
`{…}`/`[…]`) and retries once with a "reply with ONLY JSON" nudge.

Embeddings only exist on the OpenAI path; every caller falls back to the local
hashed TF-IDF (below) when `llm.embed()` returns `None`.

---

## 1. Ingest — `cde/ingest.py` (`python -m cde.ingest`)

Parses all 402 SRTs into `utterances`. Reuses the normalization functions in
`scripts/build_clean_egolife.py` rather than re-implementing them:

- **`srt_blocks(path)`** — line-based SRT parser. Yields `(start_s, end_s,
  body_lines)` per block. Splits on "digit line followed by timestamp line", so
  blocks missing their blank-line separator still parse.
- **Bilingual blocks** — each block is `Speaker: 中文` then optionally
  `Speaker: English`. The last line is treated as the English translation only
  if it carries a speaker label and is CJK-free; a single CJK-free line means
  the utterance was spoken in English. zh-only blocks keep `text_en=NULL`.
- **`canonical_speaker()`** — normalizes speaker labels through a hand-built
  alias map (修硕→Shure, 小薇→Tasha, 其他人→Others, …), repairs labels that
  absorbed the previous line's text, strips trailing `(通话)`-style notes.
  The Chinese line's label is authoritative — the English translation
  sometimes relabels speakers.
- **Absolute time** — filename `A1_JAKE_DAY1_13000000.srt` gives day=1 and a
  13:00:00 file offset; `t_abs = (day-1)*86400 + file_offset + srt_offset`.
  `hms` keeps the wall-clock string for display.

Idempotent: truncates `utterances` and reloads. The six wearers are seeded into
`participants` (Jake `is_subject=1`). Result: **300,940 utterances** (Jake:
56,638 across 74 hour-files).

---

## 2. Extract — `cde/extract.py` (`python -m cde.extract A1_JAKE`)

One LLM call per hour-file (~6–8k input tokens). The prompt lists the hour's
utterances as `[u<id>] HH:MM:SS Speaker: english | 中文` and asks for a strict
JSON object:

```
events        {start_utt, end_utt, summary, participants[], topics[],
               mood, recounts_past, when_hint}
thoughts      {utterance_id, speaker, text}
emotions      {utterance_id, speaker, emotion, text}
goals         {utterance_id, speaker, goal, status: open|in-progress|done}
behaviors     {utterance_id, speaker, behavior}
causal_claims {utterance_id, speaker, claim, earlier_desc, later_desc,
               refers_to_past, when_hint}
```

Rules enforced in the prompt **and** in `store()` validation: only *expressed*
thoughts/emotions; every extracted item must cite ≥1 utterance id that exists
in the window — anything else is dropped.

**Storage.** `store()` per file:
- wipes prior extraction rows for that window (events → event_utterances,
  event_participants, event_entities, mentions + mention_evidence, plus any
  derived `edges`/`edge_evidence`/`goal_state`/`pattern_occurrences` that
  reference them — required for FK-safe idempotent replays),
- inserts each event with `t_start/t_end` taken from its boundary utterances'
  absolute times. `recounts_past` events get `recorded_at = t_start` and
  `occurred_at` resolved from `when_hint` ("yesterday" → −86400s, "day K" →
  that day's wall-clock, "N days ago" → −N·86400), `time_uncertain=1` —
  that's how a conversation *about* a past experience is anchored to the past
  rather than to when it was told.
- participants → PERSON entities (created on first sight, dedup by canonical
  name + `entity_aliases`); topics → TOPIC entities via `event_entities`.
- mentions carry the extracted cognitive items; `mention_evidence` links each
  to its cited utterance.

**Resumable + cached.** Raw JSON lands in `data/extracted/<file>.json`;
reruns replay cache without LLM calls. Files already cached are free, so a
crashed run resumes where it stopped. Concurrency via
`ThreadPoolExecutor(--workers)`.

---

## 3. Embed + resolve — `cde/embed.py` (`python -m cde.embed`)

**Event vectors.** Each event is embedded as `summary | mood | topics |
participants`. English only — Chinese stays in the DB for evidence display but
never enters the vector space (keeps a single embedding language).

- OpenAI backend → `text-embedding-3-small`, L2-normalized, stored as a
  float32 BLOB.
- Fallback → hashed TF-IDF: tokens hashed (md5) into 2048 dims, weighted
  `tf·idf` with `idf` fit on the event corpus and **persisted in `meta`**.
  Persisting idf matters: query vectors and causal-claim vectors must be
  transformed with the corpus idf or they land in a different space —
  `embed_query()` does exactly that for API search and claim matching.

**Goal resolution.** Every GOAL mention string is embedded and greedily
clustered (cosine ≥ 0.82, first member's text becomes the canonical name) into
canonical GOAL entities. `event_entities(relation='goal')` then links each
event to its goal entity — that's what lets "the party" become one node the
whole week's events attach to. TOPIC/PERSON canonical names get embeddings too
(entity-level similarity).

---

## 4. Link — `cde/link.py` (`python -m cde.link`)

Wipes `edges`/`edge_evidence`/`goal_state`, then rebuilds in three passes.

### 4a. Deterministic edges (SQL, no LLM)

| Edge | From → To | Source |
|---|---|---|
| SPOKE | PERSON → UTTERANCE | speaker column (subject's files) |
| PARTICIPATED_IN | PERSON → EVENT | `event_participants` |
| EXPRESSES | UTTERANCE → MENTION | `mention_evidence` |
| PRECEDES | EVENT → EVENT | consecutive events same day |
| ADVANCES_GOAL | EVENT → GOAL | `event_entities(relation='goal')` |
| SIMILAR_TO | EVENT ↔ EVENT | top-3 cosine ≥ adaptive threshold — **display only, never causal** |

`goal_state` is rebuilt too: per GOAL entity per day, the latest mention status
(open → in-progress → done) — the persistent-state tracker.

The SIMILAR_TO threshold adapts to the backend: 60th percentile of each node's
max cosine (OpenAI ≈0.8, TF-IDF ≈0.25–0.4).

### 4b. Candidate retrieval

For each event E, all earlier events within 7 days are scored:

```
S = 0.4·cosine + 0.2·entity_jaccard + 0.2·temporal_decay + 0.2·context
```

- `entity_jaccard` — shared topics/goals/participants over the union.
- `temporal_decay` — `exp(-Δt/6h)`; nearby hours score higher but nothing is
  excluded.
- `context` — max(shared open goal, similarity between E's causal-claim
  `earlier_desc` and the candidate's event vector). This is the mechanism that
  lets "seeds wouldn't sprout in 7 days" pull the seed-planting event out of
  three days of transcript.

Per event: top-5, **plus** guaranteed inclusion of the best cross-hour and best
cross-day candidate, **plus** every causal-claim candidate ≥0.45 — recall over
precision; the judge is the precision stage.

### 4c. LLM edge judge

Candidates batched ≤10 pairs/call (6 parallel workers). Each pair is shown as
both events' summaries, moods, times, and up to 8 evidence utterances with
utterance ids. Per pair the model returns:

```
{pair, relation: INFLUENCES|REFERS_TO|SAME_THREAD|NONE, direction,
 evidence_type: self_report|transcript|temporal|semantic,
 supporting_utterance_ids[], reason, alternative_explanations[], confidence}
```

**Validation (deterministic, after the model answers):**
- source event must precede target (time-travel edges → rejected);
- cited utterance ids must exist among the pair's evidence quotes;
- INFLUENCES without `self_report`/`transcript` evidence → downgraded to
  SAME_THREAD;
- NONE → stored as a `rejected` edge with the judge's reason — rejected
  candidates are data too (the UI can show them);
- pairs the judge never returns a verdict on → rejected.

INFLUENCES edges are stored `status='hypothesis'` — they're causal *claims
about cognition*, not facts; REFERS_TO/SAME_THREAD are `observed`. Accepted
edges keep `confidence`, `rationale`, `alternatives_json`, `retrieval_score`,
and `edge_evidence` rows for every cited utterance.

---

## 5. Patterns (stretch) — `cde/patterns.py`

k-means over event embeddings (k≈12) → one LLM call per cluster to name it
("logistics planning", "setback/complaint"…) → PATTERN nodes +
`pattern_occurrences`. Ordered pattern pairs recurring on ≥3 days within 3h get
a hypothesis `REINFORCES` edge between patterns.

---

## 6. API — `cde/api.py` (`uvicorn cde.api:app --port 8000`)

| Endpoint | Returns |
|---|---|
| `GET /api/stats` | node/edge counts per type+status, % judged edges carrying valid evidence (the grounding metric), active LLM backend |
| `GET /api/graph?subject=A1_JAKE&days=1-7&min_conf=` | `{nodes, edges}`: event nodes with absolute-time coords + mood, GOAL nodes with per-day states; event→event edges and ADVANCES_GOAL; rejected hidden unless `show_rejected` |
| `GET /api/event/{id}` | summary, occurred_at vs recorded_at, participants, topics, goals, the full utterance evidence (en+zh), extracted mentions with cited ids, in/out edges with rationale + alternatives + evidence utterance ids |
| `GET /api/goal/{id}/thread` | a goal's events and status timeline across days |
| `GET /api/chain?event_id=` | backward traversal over INFLUENCES/REFERS_TO/SAME_THREAD (best-confidence predecessor per hop) to the earliest root; each hop carries its quotes — **the money shot** |
| `GET /api/search?q=` | semantic search over event vectors |

Static: `/` → `web/index.html`, `/web/*` → static assets (Cytoscape vendored
locally — no CDN dependency at demo time).

---

## 7. Frontend — `web/index.html` (single file)

- **Timeline canvas.** x = absolute time (day columns), events stacked in a
  lane, GOAL diamonds on a top lane. Zoom presets week/day/hour re-scale x;
  day buttons refilter via `?days=`.
- **Certainty encoded in edges.** PRECEDES thin grey · SIMILAR_TO dotted green
  labelled "semantic only" · SAME_THREAD/REFERS_TO blue · INFLUENCES orange,
  width = confidence, dashed unless `self_report` · rejected red dashes behind
  a toggle · ADVANCES_GOAL purple.
- **Inspector.** Click a node → evidence utterances with timestamps (en+zh),
  mentions with cited `u…` ids, every in/out edge with its rationale,
  alternative explanations, and evidence references; edges are clickable to
  jump between events.
- **"Trace back".** `/api/chain` highlighted hop-by-hop to the earliest root,
  each hop listing its supporting quotes — the cross-day evidence chain.
- **Replay mode (the demo video).** `▶ replay` rebuilds the graph live: sim-time
  advances ~1 hour of life per 1.2s (scrub slider + 1×/2×/4× speed), event
  nodes pop in at their `t_start`, edges appear once both endpoints exist, and
  *weight accumulates* — event nodes grow per incident edge, GOAL diamonds grow
  per `ADVANCES_GOAL` link, and whenever a SIMILAR_TO/SAME_THREAD/INFLUENCES
  edge joins an already-connected thread component, every edge in that
  component thickens (recurring similar activity = visibly heavier
  connections). A day-boundary camera refit keeps the frame composed. Screen-
  record the playback for the submission video.
- **Goal bar.** One chip per GOAL entity → horizontal thread view with
  open/in-progress/done status per day.
- Semantic search box (debounced) → jump-to-event.

---

## 8. Unused dataset signals

What EgoLife provides that the pipeline does not consume, with the
cost/benefit call on each.

**On disk, currently ignored:**

- `clean/captions/*.jsonl` — 357k dense captions (~60k for Jake), first-person
  action descriptions in Chinese, roughly one per 1.5s. Parsed by
  `build_clean_egolife.py`, never ingested (no captions table). Potential
  uses: BEHAVIOR mentions for silent stretches, a non-LLM event-boundary
  signal, event-summary validation. **Deferred** — zh-only text conflicts with
  the English-only vector space, and the demo's evidence chains live in
  speech anyway.
- `clean/utterances/*.jsonl` fields dropped at ingest — `moment_id` (same
  utterance captured by multiple recorders), `heard_by` (which recorders
  captured it = who was physically present → deterministic WITNESSED edges;
  also fixes `event_participants` recall, which today only contains people
  the LLM named), `moment_speaker` (majority-vote speaker across recorders —
  more reliable than per-file labels for SPOKE), `is_wearer`, and `flags`
  (`speaker_disputed`/`untranslated`/`filler` → evidence weighting).
  **Cheapest credible add:** new columns on `utterances` + ingest carry.
- Other five wearers' transcripts — ingested (300k utterances) but extraction
  runs on Jake only. Their utterances about shared moments are evidence for
  Jake's events sitting unused in the DB; moment-level fusion across wearers
  would be a redesign.

**In the HF release, not downloaded:**

- `EgoLifeQA/` — per-subject QA JSONs (Jake ≈590KB) of questions designed to
  require hour/day-spanning recall: the only ground-truth eval set for
  `/api/chain`. Many items are visual-only and unanswerable from transcript —
  partial eval at best. Pull when a quantitative claim is needed.
- `EgoIT/` — instruction-tuning caption/QA sets, mostly redundant with above.
- Gaze / IMU / 15 GoPro exocentric views / 3D scans — out of scope.

**Priority:** `heard_by`/`moment_id`/`moment_speaker` first (cheap, adds real
non-LLM evidence), then EgoLifeQA for evaluation, captions last (highest
integration cost), other-wearer extraction only if demo needs it.

---

## Schema (SQLite, mirrors Postgres 1:1)

`participants(name, wearer_code, is_subject)` ·
`utterances(wearer, file, day, t_abs, t_end, hms, speaker, text_en, text_zh)` ·
`events(subject, day, t_start, t_end, occurred_at, recorded_at,
time_uncertain, summary, mood, embedding)` · `event_utterances` ·
`event_participants` · `entities(type[TOPIC|GOAL|PERSON], canonical,
embedding)` + `entity_aliases` · `event_entities(relation)` ·
`mentions(type[THOUGHT|EMOTION|GOAL|BEHAVIOR|CAUSAL_CLAIM], event_id, speaker,
text, payload_json)` + `mention_evidence` ·
`edges(src_type, src_id, dst_type, dst_id, relation, confidence,
status[observed|hypothesis|rejected], evidence_type, rationale,
alternatives_json, retrieval_score)` + `edge_evidence` ·
`goal_state(goal_entity_id, day, status, event_id)` ·
`patterns` + `pattern_occurrences` (stretch) · `meta` (persisted TF-IDF idf).

## Run order

```bash
pip install -r requirements.txt
python -m cde.ingest            # ~10s, all 402 files
python -m cde.extract A1_JAKE   # resumable, cached in data/extracted/
python -m cde.embed             # vectors + goal/topic resolution
python -m cde.link              # deterministic + candidates + judge
python -m cde.patterns          # stretch
uvicorn cde.api:app --port 8000 # http://127.0.0.1:8000
```

## Design invariants

- Only **expressed** mental content is extracted — prompts and validation both
  enforce it.
- Every extracted item cites ≥1 real utterance id or is dropped.
- No CAUSES edges: causal relations exist only as `INFLUENCES` *hypotheses*
  with evidence, confidence, and recorded alternatives.
- SIMILAR_TO is display-only and labelled "semantic only" — the demo's point is
  that the interesting edges are the judged, evidence-backed ones.

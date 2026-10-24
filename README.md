# Cognitive Dynamics Engine

Georgia Tech Devin Hackathon.

**Demo claim:** a conversation today connects to an experience days earlier
through a traceable chain of evidence — not just because embeddings are similar.

**Product framing:** Jake wore a camera for a week with five housemates. CDE
lets him understand *himself* (how he acts), *how he deals with each person*,
and *how they act differently around him* — every conclusion clickable down to
the quotes (and original audio) behind it.

We ingest one week of first-person lifelog transcripts (EgoLife, 6 wearers ×
7 days, bilingual zh/en), extract the cognitive structure of each of Jake's
hours (events, thoughts, emotions, goals, behaviors, explicit causal claims —
all grounded in utterance ids), embed and link events across days, judge
candidate links with an LLM that must cite evidence or abstain, and compute
presence statistics over all six recorders. Every accepted edge and every
insight carries the evidence that supports it.

---

## Demo script (presentation order)

1. **Open the app → Themes lens (top zoom level).** 8–10 human-readable bubbles
   ("Final-day party", "Cooking & baking", …) laid out across the week, sized
   by event count, linked by how many judged links cross between them.
2. **Click a bubble → Causal lens (middle level).** Only that theme's events on
   the time axis, only LLM-judged links (INFLUENCES / REFERS_TO / SAME_THREAD),
   stubs to the themes it feeds into, longest causal "spine" highlighted.
3. **Click an event → inspector → "Trace back".** Walks back to the earliest
   root across days; every hop shows the relation, confidence, rationale and
   the quotes that justify it — with ▶ to play the original audio.
4. **All lens (bottom level) → ▶ replay.** The full graph builds hour by hour;
   repeated similar activity thickens edges and grows goal nodes.
5. **Insights.** "Big picture" hypotheses → "You" → "How others act around you"
   → "How you act around them" → "Your threads". Click any card → the quotes,
   events and numbers behind it.
6. **People bar / social graph.** Person → flagged interaction patterns;
   `social graph →` opens the day-1 relationship view (`web/trends.html`).

Talking points that land: *"Katrina, Lucia and Alice laugh 1.4–1.9× more when
Jake is around — compared in same-size groups, so it's not just 'more people'."*
*"Jake does 24% of all the talking (an even split is 17%) and says yes/okay
2.1× as often as everyone else — a facilitator, not a dictator (directions are
only 1.2×)."* *"This day-6 event traces back to a day-1 decision, quote by
quote."*

---

## Architecture

```
SRT transcripts (402 files) ──ingest──► utterances (300,940; all 6 wearers, en+zh, absolute time)
        │                                             │
        │                               segment Jake's 74 hour-files
        │                                             │
        │                         LLM extract (utterance-id grounded JSON)
        │                                             │
        │                events · thoughts · emotions · goals · behaviors · causal claims
        │                                             │
        │                   embed (OpenAI) + entity resolution + goal canonicalization
        │                                             │
        │         deterministic edges ──► candidate retrieval ──► LLM edge judge
        │                                             │
        │              relate(): PERSON→PERSON interaction patterns (z-test)
        │                                             │
        │                    themes: goals → 8–10 named themes (zoom level 1)
        │                                             │
clean/utterances/*.jsonl (heard_by) ──► insights: presence effects (CMH z) + insight cards
                                                      │
                               SQLite graph ──► FastAPI ──► Cytoscape UI (3 zoom levels)
```

| Module | Stage |
|---|---|
| `cde/ingest.py` | parse SRTs → `utterances` |
| `cde/extract.py` | per-hour LLM extraction, grounded, cached |
| `cde/embed.py` | embeddings, entity resolution, goal canonicalization |
| `cde/link.py` | deterministic edges, candidates, judge, `relate()` |
| `cde/themes.py` | theme clustering for zoom level 1 |
| `cde/insights.py` | presence effects + insight cards |
| `cde/patterns.py` | stretch k-means clusters (superseded by themes) |
| `cde/api.py` | FastAPI |
| `cde/audio.py` | original-audio clips per utterance (Ethan) |
| `web/index.html` | main UI |
| `web/trends.html` | day-1 social graph (Ethan) |

---

## How each stage works (and why)

### 1. Ingest — `cde/ingest.py`
- 402 SRT files (`Transcript/<wearer>/DAY<n>/*.srt`) → 300,940 utterances.
- Reuses `scripts/build_clean_egolife.py` normalization: speaker aliases,
  Chinese label authoritative when the English translation label is wrong,
  missing blank lines, zh/en split.
- Absolute time = filename hour offset + SRT offset; `t_abs = (day-1)·86400 + seconds_of_day`.
- **All six wearers are ingested** even though only Jake is extracted — they
  provide speakers, quotes, and the presence signal.

### 2. Extract — `cde/extract.py` (Jake only)
- One LLM call per hour-file (74 files) → 655 events + 2,808 mentions
  (thought / emotion / goal / behavior / causal claim).
- **Grounding rule:** every item must cite utterance ids that exist in that
  hour's window, else it is dropped. Raw JSON cached in `data/extracted/`.
- Events carry `occurred_at` vs `recorded_at` so recounted events ("yesterday
  we…") aren't misplaced in time.
- *Decision:* Jake-only extraction because of time/cost (5 more wearers = ~370
  more LLM calls); the stage is subject-parameterized.

### 3. Embed + resolve — `cde/embed.py`
- OpenAI `text-embedding-3-small` (1,536-dim) on English text; hashed TF-IDF
  fallback with persisted IDF when no key is set.
- **Goal canonicalization:** LLM groups raw goal strings into named threads.
  The prompt explicitly bans generic labels ("shopping", "social engagement")
  and asks for concrete outcomes → 270 goals such as *"plan and host the
  final-day party"* (6 days / 42 events).
- `--goals-only` re-canonicalizes without re-embedding.

### 4. Link — `cde/link.py`
- **Deterministic edges:** SPOKE, PARTICIPATED_IN, EXPRESSES, PRECEDES,
  ADVANCES_GOAL, SIMILAR_TO (adaptive percentile threshold; display-only,
  never causal).
- **Candidate retrieval:** `S = 0.4·cos + 0.2·entity-jaccard + 0.2·temporal decay + 0.2·context`,
  source must precede target; every event is guaranteed a cross-hour and a
  cross-day candidate; half the judge budget is reserved for cross-day pairs
  (temporal decay would otherwise bury them).
- **LLM edge judge** (gpt-4o-mini, 10 pairs/call, 6 parallel workers, 800
  pairs): returns relation, evidence type, confidence, rationale,
  alternative explanations and **supporting quote indices**.
  - *Positional citation:* quotes are numbered [0]–[7] for event A and
    [8]–[15] for event B; the model cites indices, we map back to utterance
    ids deterministically. Models cited raw ids unreliably (grounding went
    8% → 26% after the switch).
  - *Validation:* time-travel edges rejected; INFLUENCES with only a
    temporal signal is downgraded to SAME_THREAD; NONE verdicts are stored as
    `rejected` edges (never silently dropped).
  - *Loosened causality (demo decision):* INFLUENCES is kept for
    `self_report`, `transcript` **or `semantic`** evidence; if the model cites
    no quote, the rationale is tagged `[inferred: no direct quote]` instead of
    downgrading.
  - Verdicts are cached in `data/judged.json`, so reruns replay them for free.
- **`relate()` — INTERACTION_PATTERN:** PERSON→PERSON edges ("X shows
  behavior B more when Y is present than when absent"), directed, status
  `hypothesis`, per-day rows in `pair_behavior`, quotes in `edge_evidence`.
  Filters on shared events, absent baseline and z (details in SPEC.md).

### 5. Themes — `cde/themes.py` (zoom level 1)
- Step 1: the LLM proposes 8–10 concrete theme names from the 270 goals
  (generic words banned, no catch-all buckets).
- Step 2: the LLM classifies every goal into a theme in batches of 60; goals it
  misses fall back to the nearest theme by embedding.
- Events take the majority theme of their goals; events without goals take
  the nearest theme centroid by embedding. Every event gets exactly one theme.
  Empty themes are dropped.
- Cached in `data/themes.json` (v1 single-call output kept as `data/themes_v1.json`).

### 6. Insights — `cde/insights.py` (what Jake gets out of it)
**Presence signal.** `EgoLife/clean/utterances/*.jsonl` gives every moment
a `heard_by` list (which recorders captured it = who was physically there).
300,940 rows collapse to **123,670 unique moments**; `heard_by` agrees across
recorders for all of them.

**Attribution.** A line counts as speaker S's only if S's *own* recorder
labels it as S (88,171 own-mic moments); the ~1% of moments where recorders
disagree on the speaker are left out.

**Presence labels** for S's line relative to another wearer O:
- *with* — O's recorder captured it, or S's and O's recorders co-captured
  anything within ±1 minute;
- *without* — O's recorder was active within ±1 minute but shared nothing
  with S (O was recording somewhere else);
- *unknown* (excluded) — O's recorder idle. A recorder that's off is not
  treated as absence.

Using a window rather than per-utterance presence avoids a capture bias:
short lines ("okay", laughs) are less often picked up by a second mic.

**Features** (keyword rules on the English translation, plus a few zh cues):
question, directive, laughter, assent (yes/okay), planning talk, and talk
share (S's share of the conversation, with O's own lines excluded).

**Group-size confound → stratification.** Jake is usually with the whole
group, so "with Jake" also means "with more people". Every comparison is
stratified by how many *other* wearers were present. Rates are standardised
to the same group-size mix, and z is the **Cochran–Mantel–Haenszel**
statistic. This killed most raw "talks less around Jake" effects (they were
a group-size artifact); only what survived is shown.

**Comparisons**
| Lens | Comparison |
|---|---|
| *self* | Jake vs the other five pooled, plus airtime share |
| *around_you* | each housemate with Jake present vs absent |
| *you_around* | Jake with each housemate present vs absent |

**Thresholds**
- Present and absent sides each need n ≥ 100, and |z| ≥ 4.
- Lift must be ≥ 1.25 or ≤ 0.8 (self-vs-group: 1.1 / 0.9, since n is about
  22k vs 66k).
- 210 effects computed → 31 notable.

**Cards** (22 in the `insights` table)
- *observed* — templated straight from the statistics.
- *threads* — the longest goal threads, plus the top cross-day INFLUENCES
  edges (status *hypothesis*).
- *portrait* — 3–4 "big picture" hypotheses written by one LLM pass. Each
  must cite at least 2 findings, may only name a role the findings measure,
  and is worded as "may / seems". Wording changes between runs.

Each card stores evidence ids (utterances / events / edges / goals), which
`/api/insight/{id}` resolves to quotes and events.

### 7. API + UI
- **Three zoom levels:**

  | Level | Shows | Endpoint |
  |---|---|---|
  | Themes | named theme bubbles | `/api/zoom/themes` |
  | Causal | events inside one theme, judged links only, spine, cross-theme stubs | `/api/zoom/theme/{id}` |
  | All | the full graph | `/api/graph` |

  The header has a `themes | causal | all` lens switch (default: themes) and a
  breadcrumb. The week/day/hour buttons are a separate *time-scale* zoom.
- **Insights panel:** cards grouped by lens; click a card for evidence quotes
  (playable), events (click through to the graph) and statistics.
- **Inspector:** utterances (en + zh), extracted mentions, edges in/out with
  rationale, confidence and alternatives; "Trace back"; ▶ audio.
- **Ethan Lam's surfaces:**
  - `web/trends.html` + `analysis/TRENDS.md` — day-1 social graph built from
    `review/DAY1_*.csv`: co-presence (same line within ±6 s), reply pairs,
    talk-over, airtime, monologues.
  - `cde/audio.py` + `/api/audio/*` — original EgoLife audio clips per
    utterance.

---

## Epistemics (what the labels mean)

| Status | Meaning | Example |
|---|---|---|
| `observed` | deterministic, or a straight count/statistic | PRECEDES, ADVANCES_GOAL, "You laugh 1.4×" |
| `hypothesis` | an interpretation: causal claims, behavior patterns, portraits | INFLUENCES, INTERACTION_PATTERN, portrait cards |
| `rejected` | candidate the judge declined (kept for audit; hidden by default) | NONE |

**Edge legend:**
- PRECEDES: thin grey
- SIMILAR_TO: dotted green ("semantic only", never causal)
- SAME_THREAD / REFERS_TO: blue
- INFLUENCES: orange, width = confidence; solid if self-reported, dashed if inferred from the transcript
- ADVANCES_GOAL: dashed purple
- INTERACTION_PATTERN: teal (people bar)
- Rejected: dashed red, hidden behind a toggle

---

## Decision log

| Decision | Why | Trade-off |
|---|---|---|
| Extract Jake only | ~1 h budget; one rich point of view beats six thin ones | others appear only as context |
| `heard_by` presence instead of extracting the other 5 | uses all 300k lines (thousands of utterances per pair, which fixes the 7-day thin-data problem); no LLM cost or rate limits | behavior = keyword features, not LLM-labelled behaviors |
| Stratify by group size (CMH) | "with Jake" ≈ "with everyone"; raw talk-share effects were artifacts | fewer, smaller, more defensible effects |
| ±1-minute window presence | per-utterance capture misses short lines | coarser presence |
| OpenAI gpt-4o-mini + text-embedding-3-small | fast, cheap, parallel; fallback chain OpenAI → `codex exec` → `claude -p` | judge stricter than codex (48 → 9 INFLUENCES before loosening) |
| Positional quote citation | models cite small indices reliably | prompt is slightly longer |
| Loosened INFLUENCES (semantic allowed, `[inferred]` tag) | demo needs visible causal claims | more hypotheses; marked as such |
| Specific goal names (generic labels banned) | "shopping" bubbles tell no story | names are LLM-generated |
| Two-step themes (name, then classify) | a single call dropped 210 of 270 goals | 6 LLM calls instead of 1 |
| No data doctoring | quotes and audio are clickable, so fake evidence would show instantly; real full-week data already gives strong n | — |
| Cache everything (`data/extracted`, `judged.json`, `themes.json`) | reruns are free, and survived a CLI session limit mid-run | cache invalidation is manual |
| SQLite mirroring the Postgres design | zero setup for the hackathon | single-writer |

---

## Numbers to quote (current run)

**Data**
- 402 SRTs · 300,940 utterances · 123,670 unique moments (88,171 own-mic attributed).
- Jake: 74 hour-files · 655 events · 2,808 grounded mentions · 270 goals · 1,514 topics · 14 people.

**Links**
- 800 candidate pairs judged → 22 INFLUENCES (11 cross-day) · 6 REFERS_TO · 562 SAME_THREAD · 210 rejected.
- 118 accepted links carry cited quotes.
- Deterministic edges: 53,313 SPOKE · 2,810 PARTICIPATED_IN · 2,808 EXPRESSES · 828 ADVANCES_GOAL · 648 PRECEDES · 249 SIMILAR_TO.
- 50 INTERACTION_PATTERN.

**Insights**
- Self:
  - Jake's airtime is 24.5% (next is Shure at 19%; an even split is 17%).
  - yes/okay 2.06× (z=26), laughter 1.41× (z=12), directions 1.15× (z=4).
- Around Jake:
  - Katrina laughs 1.92× (z=5.9), Lucia 1.77× (z=5.2), Alice 1.38× (z=5.0).
  - Shure, Lucia and Katrina say yes/okay 1.44×, 1.67× and 1.59×.
  - Katrina's talk share is 0.76× (z=−14).
- Jake around them: laughs 0.71× with Alice and 0.55× with Tasha; says yes/okay 1.57× with Alice.
- 210 presence effects computed → 31 notable → 22 insight cards.

**Themes (zoom level 1)**
- Event preparation 215 · Cooking & baking 122 · Social interactions 106 ·
  Documenting memories 74 · Final-day party 36 · Games & icebreakers 33 ·
  Dance rehearsal 27 · Room decoration 25 · Money & reimbursements 14 ·
  Auction & gifts 3.
- All 270 goals classified explicitly; 428 events placed by goal vote, 227 by embedding.

---

## Limitations (say these before a judge does)

- **One point of view:** only Jake's hours are extracted; the other five feed
  presence statistics and quotes.
- **Presence = a recorder captured it.** Idle recorders are excluded rather
  than treated as absent; ±1-minute resolution.
- **Behavior features are keyword rules on machine translation**: noisy, but
  applied identically in every condition.
- **z ranks findings; it isn't a p-value.** Utterances are autocorrelated.
  Group size is controlled for; activity type and time of day are not.
- **LLM judgments vary by model:** codex vs gpt-4o-mini produced different
  INFLUENCES counts on overlapping data.
- **Generated wording:** portrait cards and theme/goal names are LLM text and
  change between runs; the statistics underneath are deterministic.
- **Seven days of data.** Per-pair *behavior patterns* from extracted events
  are thin; the utterance-level presence stats are not.

## Likely questions → answers

- **"Isn't this just embedding similarity?"** Embeddings only *propose*
  candidates. A link is accepted only when the judge cites quotes for it, and
  SIMILAR_TO is display-only. Rejected candidates are kept and can be shown
  with the toggle.
- **"Do you claim causality?"** No. INFLUENCES is a *hypothesis*, carrying
  confidence, evidence type and the alternative explanations the judge gave.
- **"People laugh more around Jake — isn't that just a bigger group?"** We
  compare within same-size groups (CMH). The raw "talks less around Jake"
  effects mostly disappeared under that control, and we kept only what
  survived.
- **"Isn't 'absent' just a recorder being off?"** No. Absent requires the
  other person's recorder to be active within ±1 minute while sharing nothing
  with the speaker.
- **"Why only Jake?"** Time and cost. The pipeline is subject-parameterized,
  all six wearers are already ingested, and `heard_by` gives the cross-person
  view without more extraction.
- **"Is any data fabricated?"** No. Every card and edge links to source
  quotes, and the original audio plays in the app.
- **"How does it scale?"** Per-hour extraction and verdict caches,
  idempotent stages, a `--max-pairs` judge budget, and a SQLite schema that
  maps 1:1 to Postgres.

---

## Run

```bash
pip install -r requirements.txt
# .env: OPENAI_API_KEY=...  (else codex / claude CLI backends)
python3 -m cde.ingest            # ~10s, all 402 files
python3 -m cde.extract A1_JAKE   # LLM extraction, resumable, cached
python3 -m cde.embed             # embeddings + entity resolution (--goals-only to redo goals)
python3 -m cde.link --max-pairs 800   # deterministic + candidates + judge + relate()
python3 -m cde.themes            # zoom level 1 (delete data/themes.json to re-cluster)
python3 -m cde.insights          # presence effects + insight cards (~20s, 1 LLM call)
uvicorn cde.api:app --port 8000  # open http://127.0.0.1:8000
```

## API

**Graph**
- `GET /api/stats`
- `GET /api/graph?subject=A1_JAKE&days=1-7` (event nodes carry `theme`)
- `GET /api/event/{id}`
- `GET /api/goal/{id}/thread`
- `GET /api/chain?event_id=` (walks back to the earliest root, quotes per hop)
- `GET /api/search?q=`

**Semantic zoom**
- `GET /api/zoom/themes`
- `GET /api/zoom/theme/{id}`

**Insights**
- `GET /api/insights`
- `GET /api/insight/{id}`

**People**
- `GET /api/people`
- `GET /api/person/{id}/patterns`

**Audio**
- `GET /api/audio/status`
- `GET /api/audio/event/{id}`
- `GET /api/audio/{utterance_id}`

## Schema (SQLite `data/cde.db`, mirrors the Postgres design 1:1)

- **Source data:** `participants`, `utterances`.
- **Events:** `events` (occurred_at vs recorded_at), `event_utterances`, `event_participants`.
- **Entities:** `entities` (PERSON | TOPIC | GOAL | BEHAVIOR_KIND) + `entity_aliases`, `event_entities`.
- **Mentions:** `mentions` + `mention_evidence`.
- **Graph:** `edges` (relation, confidence, status observed | hypothesis | rejected, evidence_type, rationale, alternatives_json, retrieval_score) + `edge_evidence`.
- **State over time:** `goal_state`, `pair_behavior`.
- **Themes:** `themes`, `goal_themes`, `event_themes`.
- **Insights:** `presence_effects`, `insights`.
- **Legacy:** `patterns` + `pattern_occurrences`.

More detail on each stage's spec and the unused dataset signals: `SPEC.md`.
Pipeline state and handoff notes: `HANDOFF.md`.

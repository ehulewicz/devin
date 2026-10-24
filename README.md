# Cognitive Dynamics Engine

Georgia Tech Devin Hackathon.

**Demo claim:** a conversation today connects to an experience days earlier
through a traceable chain of evidence — not just because embeddings are similar.

We ingest one week of first-person lifelog transcripts (EgoLife, 6 wearers ×
7 days, bilingual zh/en), extract the cognitive structure of each hour
(events, thoughts, emotions, goals, behaviors, explicit causal claims — all
grounded in utterance ids), embed and link events across days, and judge
candidate edges with an LLM that must cite evidence or abstain. Every accepted
edge carries the quotes that support it.

## Architecture

```
SRT transcripts ──ingest──► utterances (all 6 wearers, en+zh, absolute time)
                               │
                     segment (Jake, per hour file)
                               │
             LLM extract (utterance-ID grounded, JSON)
                               │
   events · thoughts · emotions · goals · behaviors · causal_claims
                               │
         embed + entity resolution (topics/goals/people)
                               │
   deterministic edges ──► candidate retrieval (S = w·[sem, entity, temporal, context])
                               │
            LLM edge judge (hypothesis | rejected, evidence ids, alternatives)
                               │
                SQLite graph ──► FastAPI ──► Cytoscape timeline UI
```

- `cde/ingest.py` — parse 402 SRTs → `utterances` (reuses the normalization in
  `scripts/build_clean_egolife.py`: canonical speakers, zh/en split).
- `cde/extract.py` — one LLM call per hour-file; every extracted item must cite
  utterance ids that exist in the window, else dropped. Raw JSON cached in
  `data/extracted/` so reruns are free.
- `cde/embed.py` — OpenAI `text-embedding-3-small` when `OPENAI_API_KEY` is
  set, else hashed TF-IDF (2048-dim). English text only; zh kept for display.
  Greedy cosine clustering resolves goal strings into canonical GOAL entities.
- `cde/link.py` — deterministic edges (SPOKE, PARTICIPATED_IN, EXPRESSES,
  PRECEDES, ADVANCES_GOAL, SIMILAR_TO*), scored candidate retrieval, LLM edge
  judge. Validation: source must precede target; cited ids must exist;
  INFLUENCES without self_report/transcript evidence → downgraded to
  SAME_THREAD; NONE stored as rejected. *SIMILAR_TO is display-only, never causal.
- `cde/patterns.py` — stretch: k-means event clusters, LLM-named PATTERN nodes.
- `cde/api.py` + `web/index.html` — graph API + timeline UI with evidence
  inspector, "trace back" chain highlight, goal threads.

## Schema (SQLite `data/cde.db`, mirrors the Postgres design 1:1)

`participants`, `utterances`, `events` (occurred_at vs recorded_at for
recounted events), `event_utterances`, `event_participants`, `entities`
(PERSON|TOPIC|GOAL) + `entity_aliases`, `event_entities`, `mentions` +
`mention_evidence`, `edges` (relation, confidence, status
observed|hypothesis|rejected, evidence_type, rationale, alternatives_json,
retrieval_score) + `edge_evidence`, `goal_state`, `patterns` +
`pattern_occurrences`.

## Run

```bash
pip install -r requirements.txt
# optional: cp .env.example .env and set OPENAI_API_KEY (else `claude -p` backend)
python -m cde.ingest            # ~10s, all 402 files
python -m cde.extract A1_JAKE   # LLM extraction, resumable, cached
python -m cde.embed             # embeddings + entity resolution
python -m cde.link              # deterministic + candidates + judge
python -m cde.patterns          # stretch
uvicorn cde.api:app --port 8000 # open http://127.0.0.1:8000
```

## API

`GET /api/stats` · `GET /api/graph?subject=A1_JAKE&days=1-7` ·
`GET /api/event/{id}` · `GET /api/goal/{id}/thread` ·
`GET /api/chain?event_id=` (backward traversal to earliest root, quotes per
hop) · `GET /api/search?q=`

## Edge certainty legend (UI)

PRECEDES thin grey · SIMILAR_TO dotted green ("semantic only") ·
SAME_THREAD / REFERS_TO blue · INFLUENCES orange, width = confidence, dashed
unless self-reported · rejected candidates red, hidden behind a toggle.

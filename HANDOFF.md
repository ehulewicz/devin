# Handoff — CDE pipeline state and next steps

Context: `SPEC.md` is the build spec; §8 (new) documents dataset signals the
pipeline doesn't use and the agreed priority for adding them.

## Where the pipeline stands (data/cde.db)

| Stage | Status |
|---|---|
| ingest | done — 300,940 utterances (Jake: 56,638 / 74 hour-files) |
| extract | done — 73/74 cached files in `data/extracted/` (only DAY6_21000000 missing), 647 events, 2,930 mentions |
| embed | re-running — LLM goal-canonicalization pass in flight (raw cosine clustering produced 679 micro-goals; merging into recurring threads) |
| link | done — deterministic + 400 pairs judged; verdicts cached in `data/judged.json` so reruns replay free |
| patterns | done (stretch) — 12 named clusters + occurrences; PATTERN→PATTERN edges dropped (all pairs trivially "recur daily") |

Edge counts: SPOKE 53,313 · EXPRESSES 2,930 · PARTICIPATED_IN 2,785 ·
PRECEDES 640 · ADVANCES_GOAL 690 · SIMILAR_TO 192 · **INFLUENCES 47
(hypothesis) · REFERS_TO 19 · SAME_THREAD 224 · rejected 110**.

## Blocking next step — none; pipeline is complete end-to-end

Money shot verified: `GET /api/chain?event_id=1073` — day-6 "Jake revises the
delayed schedule" traces 4 hops: REFERS_TO→day-4 schedule review,
SAME_THREAD→day-1 timeline draft, INFLUENCES→day-1 3 PM plan,
INFLUENCES→day-1 auction proposal, each hop carrying supporting utterance ids.
Also `event_id=1077` (day-6 auction → day-1 proposal, INFLUENCES conf 0.94).

Backend note: `llm.py` order is OpenAI → `codex exec` → `claude -p`; Claude
hit its session limit ~14:20 (resets 5:30pm ET) so everything after ran on
codex. `CDE_BACKEND` in `.env` forces a backend.

UI: http://127.0.0.1:8000 — "▶ replay" animates the hour-by-hour build for the
demo video; event 1073 + Trace back is the walkthrough.

## Agreed priorities for dataset signals (SPEC.md §8)

1. **`heard_by` / `moment_id` / `moment_speaker`** — fields already exist in
   `EgoLife/clean/utterances/*.jsonl`; add columns to `utterances`, carry
   through ingest, emit deterministic WITNESSED edges (PERSON → EVENT for
   wearers present). Cheap, adds non-LLM evidence, fixes participant recall.
2. **EgoLifeQA** — download `EgoLifeQA_A1_JAKE.json` from
   `huggingface.co/datasets/lmms-lab/EgoLife`; use as eval for `/api/chain`.
   Partial only — many questions are visual-only.
3. **Dense captions** — deferred (zh-only vs English-only vector space;
   highest integration cost).
4. **Other-wearer extraction** — deferred; transcripts already ingested.

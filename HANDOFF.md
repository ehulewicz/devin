# Handoff — CDE pipeline state and next steps

Context: `SPEC.md` is the build spec; §8 documents dataset signals the
pipeline doesn't use and the agreed priority for adding them.

## Where the pipeline stands (data/cde.db)

| Stage | Status |
|---|---|
| ingest | done — 300,940 utterances (Jake: 56,638 / 74 hour-files) |
| extract | done — 74/74 cached files in `data/extracted/`, 655 events, ~2,930 mentions |
| embed | done — OpenAI `text-embedding-3-small` (1,536-dim); goals LLM-canonicalized to 270 specific threads ("plan and host the final-day party" 6d/42ev) |
| link | done — deterministic + 800 pairs judged by `gpt-4o-mini` with positional quote citation; INTERACTION_PATTERN (person→person z-test) built; verdicts cached in `data/judged.json` |
| patterns | superseded by themes (12 legacy clusters kept) |
| themes | done — `python3 -m cde.themes`: 10 named themes (two-step LLM name + classify), every event assigned; cache `data/themes.json` |
| insights | done — `python3 -m cde.insights`: heard_by presence effects over all 6 recorders, group-size stratified (CMH z), 31 notable → 22 cards |

Edge counts: SPOKE 53,313 · EXPRESSES 2,808 · PARTICIPATED_IN 2,810 ·
PRECEDES 648 · ADVANCES_GOAL 828 · SIMILAR_TO 249 · **INFLUENCES 22
(hypothesis, loosened — inferred edges kept + marked) · REFERS_TO 6 ·
SAME_THREAD 562 · INTERACTION_PATTERN 50 · rejected 210**.

Full design, decision log, numbers and Q&A prep: README.md.

## Blocking next step — none; pipeline is complete end-to-end

`/api/chain` orders predecessor hops by relation type (INFLUENCES/REFERS_TO
first), then evidence presence, then confidence — so traced chains prefer
evidence-backed causal hops.

Verified demo chains:
- `event_id=1294` — day-2 coffee/milk count ← INFLUENCES day-1 baking
  guidance (quote: "The key was your guidance and experience") ← SAME_THREAD
  day-1 cream piping.
- `event_id=1812` — day-7 expense totaling ← INFLUENCES day-5 reimbursement
  discussion (2 quotes).
- `event_id=1586` — day-4 barbecue replan ← INFLUENCES day-3 grill rehearsal.
- `event_id=1636` — day-5 steak/dough prep ← SAME_THREAD d4 ← SAME_THREAD d3,
  quotes on both hops.

Backend note: OpenAI key in `.env` — backend order OpenAI → `codex exec` →
`claude -p`; `CDE_BACKEND` forces one. gpt-4o-mini judges stricter than
codex did (fewer INFLUENCES, more rejections); the positional-quote-cite
prompt now requires cited evidence or INFLUENCES downgrades to SAME_THREAD.

UI: http://127.0.0.1:8000 — "▶ replay" animates the hour-by-hour build for
the demo video; event 1294 or 1812 + Trace back is the walkthrough.

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

# Handoff — CDE pipeline state and next steps

Context: `SPEC.md` is the build spec; §8 (new) documents dataset signals the
pipeline doesn't use and the agreed priority for adding them.

## Where the pipeline stands (data/cde.db)

| Stage | Status |
|---|---|
| ingest | done — 300,940 utterances (Jake: 56,638 / 74 hour-files) |
| extract | done — 73 cached files in `data/extracted/`, 647 events, 2,930 mentions |
| embed | done — event vectors + entity resolution (2,196 entities) |
| link | **partial** — deterministic edges built, LLM judge has NOT run |

Edge counts now: SPOKE 53,313 · EXPRESSES 2,930 · PARTICIPATED_IN 2,785 ·
PRECEDES 640 · ADVANCES_GOAL 690 · SIMILAR_TO 192 · **INFLUENCES /
REFERS_TO / SAME_THREAD / rejected: 0**.

## Blocking next step

```bash
python -m cde.link            # do NOT pass --skip-judge; default --max-pairs 400
```

Then verify judged edges landed:

```bash
sqlite3 data/cde.db \
  "SELECT relation,status,COUNT(*) FROM edges GROUP BY relation,status"
```

Then confirm the money shot works: pick a high-confidence INFLUENCES edge and
hit `GET /api/chain?event_id=<target>` — the demo needs at least one multi-hop
cross-day chain where each hop carries supporting utterance ids.

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

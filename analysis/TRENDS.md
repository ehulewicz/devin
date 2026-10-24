# Trends & Relationships — EgoLife wearers

Computed by `scripts/compute_trends.py` from `review/DAY1_*.csv` (~55.9k
utterances, day 1, all six wearers). Numbers are **observed**; week-level
extrapolations are marked **[assumed]** and lean on the HANDOFF graph stats
(Jake's extracted week: party planning, auction, BBQ replan, grill rehearsal,
schedule delays).

## Signals used

| Signal | Definition | Weight meaning |
|---|---|---|
| co-presence | same utterance captured by both wearers within ±6s | physical proximity — the `heard_by` signal from SPEC §8 |
| replies | speaker A immediately followed (<4s) by speaker B | conversational engagement per dyad |
| exposure | count of speaker X's utterances in wearer Y's recording | how much of X's voice Y's day contained |

Reply counts are amplified ~3–4× because each exchange is captured by every
nearby recorder — compare dyads relatively, not absolutely.

## Relationship graph (day 1)

Co-presence in shared utterances / span — everyone was together most of the
day; ordering still shows structure:

```
Alice — Tasha    2143 shared · 9.5h   strongest pair
Jake  — Alice    1600 · 10.9h
Alice — Lucia    1541 · 10.9h
Jake  — Tasha    1493 · 9.6h
Lucia — Katrina  1481 · 10.9h
Tasha — Lucia    1459 · 9.2h
Tasha — Katrina  1448 · 9.6h
Alice — Katrina  1437 · 10.9h
Katrina— Shure   1415 · 7.4h
Tasha — Shure    1381 · 7.4h
Alice — Shure    1360 · 7.4h
Jake  — Lucia    1265 · 10.9h
Jake  — Katrina  1194 · 10.9h
Jake  — Shure    1086 · 7.5h
Lucia — Shure    1024 · 7.5h           weakest pair
```

**Every Shure (A6) pair caps at ~7.5h, ending ~18:37.** Everyone else overlaps
into 20–22h. Shure's own file keeps recording to ~22:00 — he was present but
away from the group after early evening on day 1.

## Top conversation dyads (replies)

```
Jake  ⇄ Lucia   1862 / 1822        strongest dyad — near-symmetric
Shure ⇄ Jake    1562 / 1468
Tasha ⇄ Jake    1518 / 1465
Alice ⇄ Tasha   1438 / 1324        strongest non-Jake pair
Katrina⇄ Jake   1335 / 1326
Tasha ⇄ Lucia   1088 / 1052
Jake  ⇄ Alice   1077 / 1045
Katrina⇄ Lucia   910 /  907
Shure ⇄ Katrina  889 /  857
Lucia ⇄ Alice    847 /  830
```

All six exchange with all six — one dense group, not factions. Jake is the
hub: he is a top-3 partner for every wearer, and inbound replies to Jake
exceed outbound in 3 of 5 dyads (Shure, Tasha, Katrina address him more than
he initiates — consistent with an organizer/host role; his exchanges with
Lucia and Alice are the balanced ones). **[assumed week]** Jake's extracted
goals (final-day party, auction, schedule) are group-facing; he likely remains
the coordination hub all week — the 47→9 INFLUENCES hypotheses that survived
judging mostly radiate from his plans.

## Per-user trends

### Jake (A1) — the hub / planner
- 9,023 captured, 3,093 own (**34% self-share — highest**); active 11:10–22:05,
  latest night of anyone.
- Heard by everyone: the **most-captured other voice in all five other
  transcripts** — and in Alice's (2,309 vs her own 1,793) and Tasha's
  (2,323 vs 2,114) he is heard more than the wearer hears herself.
- Rhythm: heavy morning block (11–13 ≈ 1.4k/h), quiet 14:00, evening resume
  17–20, tapers after 21.
- Top ties: Lucia (strongest), then Shure ≈ Tasha ≈ Katrina.
- **[assumed week]** Graph extraction centers on him: recurring threads
  (party/BBQ planning, auction, schedule rework) plus the verified day-6→day-1
  INFLUENCES chains. Trend: plans seeded day 1–3 resolve into the day-7 event.

### Lucia (A4) — Jake's primary partner
- 9,480 captured, 2,482 own (26%); span 11:10–22:04.
- Second-most-heard voice group-wide (1.3–1.75k in every file); Jake⇄Lucia is
  the single busiest dyad (3,684 combined replies).
- Strong independent ties to Katrina, Tasha, Alice — not just "with Jake".
- **[assumed week]** Second node in the social core; the day-4 BBQ replan /
  day-3 grill rehearsal events are group logistics she plausibly co-ran.

### Tasha (A3) — connector, paired with Alice
- 9,617 captured, 2,114 own (22%); span 11:10–20:45, **earliest departure
  besides Shure** (nothing captured after ~20:45).
- Alice⇄Tasha is the strongest non-Jake dyad; she also bridges to Lucia and
  Katrina — mid-to-high engagement with everyone.
- **[assumed week]** Alice–Tasha pairing is a stable sub-thread inside the
  group graph; expect their shared events to cluster together.

### Alice (A2) — quiet center of the Tasha pair
- 9,571 captured but only 1,793 own (**19% self-share — quietest**); span
  11:10–22:04.
- Highest co-presence totals overall (top-2 pairs both hers): always physically
  present, speaks least. Low-key but never absent.
- **[assumed week]** Likely under-represented as a graph participant —
  extraction only names people who get mentioned; co-presence edges would
  credit her correctly (argues for the `heard_by`/WITNESSED work in SPEC §8).

### Katrina (A5) — steady mid-tier, thinnest afternoon
- 8,204 captured (**lowest total**), 1,869 own (23%); noticeable afternoon
  dip — only 511 captured at 13h and 81 at 14h vs 1.3–1.5k for others.
- Balanced ties: Lucia ⇄ Katrina and Shure ⇄ Katrina both ~900 each; engaged
  but not dominant in any dyad.
- **[assumed week]** Mid-core member; her day-1 afternoon lull reads as an
  individual off-camera stretch, not a group event.

### Shure (A6) — early-evening departures + private thread
- 9,072 captured, 2,372 own (26%); the afternoon **17:00 spike is the largest
  single hour anywhere** (2,144 captured — a big group moment).
- Splits off ~18:37: co-presence with every wearer ends there while his own
  recording continues to ~22:00.
- **Only file containing "Shure's GF" (382) and "Sister" (57)** — a private
  social thread outside the group on day 1.
- Strongest dyad is with Jake; weakest group tie is Lucia.
- **[assumed week]** Expect recurring evening splits + GF/sister events as a
  parallel thread in his part of the graph.

## Timing signals (added — `compute_trends.py`, rendered in `web/trends.html`)

| Signal | Definition |
|---|---|
| reply latency | median gap from A's utterance end to B's start (floored at 1s by SRT resolution — `--audio` VAD refines) |
| talk-over rate | B starts before A ends, as % of that member's exchanges |
| airtime | voiced seconds per wearer (transcript durations; VAD-refined with `--audio` + EgoLife media) |
| monologue | longest consecutive same-speaker streak |

Findings:
- **Alice⇄Tasha are the most patient dyad** — 25–27% talk-over vs the ~40%
  group norm; they let each other finish. Strengthens the "stable sub-pair"
  reading beyond co-presence alone.
- **Jake's dyads are the hottest** — 36–50% talk-over in every pairing;
  animated exchanges, consistent with the hub role.
- **Airtime**: Jake 78min > Lucia 68 > Shure 65 > Tasha 55 > Katrina 52 >
  Alice 43 — Alice is the quiet center by time, not just utterance count.
- **Longest monologues**: Shure 42-utterance streak — when he talks, he holds
  the floor (his isolated-evening recording likely contributes).

## Group-level trends

- **Shared rhythm**: group together 11:10→~22:00 with a universal capture gap
  14:00–17:00 (all six files thin/absent — a scheduled break or no-recording
  window, not six coincidences). Loud group block 17–19 (incl. Shure's 2,144
  spike), taper after 20.
- **Structure**: one connected component — no isolated pairs. Jake = hub;
  Lucia = co-core; Alice–Tasha = distinct sub-pair; Shure = semi-detached
  evenings.
- **Graph implication**: co-presence ordering ≈ what deterministic WITNESSED
  edges would produce — SPEC §8's `heard_by`/`moment_id` carry-over is the
  cheapest way to turn this table into real graph edges, and it would fix
  participant recall for quiet members (Alice) that extraction misses.

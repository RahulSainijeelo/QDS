# QDS Dashboard

A view onto the detection engine: the eight scenarios, their verdicts, and the
evidence behind each one. The dashboard is deliberately a *window*, not an
illustration — every number it shows is produced by running the real engine and
serialising the result, so it can never drift from what the engine actually says.

## What is here today, and what is not

This directory is being built in two layers, and only the lower one exists so far.
Stated plainly so nothing here is mistaken for runnable:

**Built — the data and the logic that reads it.**

`public/data/` holds the committed engine output: `index.json` (the catalogue —
geometry, declared spec, the eleven rule names, and one headline card per
scenario) and `scenarios/<id>.json` (the full detail for each of the eight runs),
plus `session-lab.json`. This is the whole dataset; it is diffable and
reproducible from one script.

`lib/` is the TypeScript layer that consumes it: `types.ts` is the data contract,
written against the real JSON rather than the Python dataclasses (the comment at
its head explains why that distinction caught a real bug); `data.ts` loads and
shallow-validates snapshots, static-first with an optional live path; `format.ts`,
`select.ts`, and `chart.ts` are formatting, selection, and chart-shaping helpers.

`scripts/generate_snapshots.py` is the generator that writes `public/data` from
the engine.

**Not built — the application shell.** `app/` is empty and there is no
`package.json`, `tsconfig.json`, or Next.js config, so there is no `npm run dev`
to run yet. The components under `components/` are being authored separately and
in parallel (`figures.tsx`, `nav.tsx`, `tables.tsx`, `ui.tsx` are landing), but
with no build config they are not yet wired into a runnable app — treat the
dashboard as a **specified interface** in progress, described in
`docs/ARCHITECTURE.md` §8. The data contract it will consume is already real and
populated, so what remains is presentation, not substance.

## Regenerating the data

The data is produced by the engine, so this step needs the Python side (NumPy;
see `engine/README.md`) and nothing from Node:

```bash
# from the repository root
python3 web/scripts/generate_snapshots.py
```

That runs all eight scenarios end to end and rewrites `public/data/index.json`
and `public/data/scenarios/*.json`. Useful flags: `--only <id>,<id>` regenerates
named scenarios without touching the index; `--bits` and `--L` change the geometry
(default `message_bits=16`, `L=24`, giving 768 signature slots and 384 checks).

Nothing in the output is hand-written or adjusted for presentation. If a verdict
looks surprising, the engine said it — two of the eight scenarios are weak attacks
that return *clean* because they stay below the acceptance threshold, reported in
`docs/DETECTION.md` §7.

## The data contract

`index.json` carries `geometry`, `declared_spec`, the `rules` list, and a
`scenarios` array of headline cards (verdict, pooled rate, flagged count, primary
attribution). Each `scenarios/<id>.json` carries the full `detection` block —
`verdict`, `alarm`, `combined_p_value`, `alpha`/`alpha_per_test`, the per-test
`tests` array, the `estimates` and `estimates_by_basis`, the `thresholds`, the
`attribution` list, `spec_violations`, and the protocol-level fields
(`protocol_rejects`, `authentication_failed`, `rejections`) — alongside the
`session` document and the scenario's declared spec and channel.

The authoritative description of these shapes is `lib/types.ts`, which was written
field-by-field against real output. When in doubt, read the JSON, not the Python:
`RateEstimate`, for instance, serialises its interval as `ci_low`/`ci_high`, and a
contract written from the dataclass would have guessed `low`/`high` and produced
`undefined` everywhere.

## Design: static-first

The committed snapshots are the primary deployment. A static export needs nothing
at runtime but the JSON in `public/data`, which is why that data is checked in
rather than generated on the fly. The live path in `data.ts` — re-running the
engine for a single scenario on demand — is strictly additive: if the Python
engine is not checked out beside the app, or the app was exported statically, the
live path is simply unavailable and the static snapshots stand. Build the app
shell against this assumption and the dashboard works with or without the engine
present.

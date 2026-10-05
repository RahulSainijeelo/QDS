# QDS Dashboard

A view onto the detection engine: the eight scenarios, their verdicts, and the
evidence behind each one. The dashboard is deliberately a *window*, not an
illustration — every number it shows is produced by running the real engine and
serialising the result, so it can never drift from what the engine actually says.

## Running the dashboard

The app is a Next.js project, and it is built and runnable. From `web/`:

```bash
pnpm install
pnpm dev            # http://localhost:3000, reads the committed snapshots
pnpm build          # static export to out/ — no Node runtime needed to serve it
pnpm test           # the lib-layer harness (node --experimental-strip-types)
```

`pnpm build` writes a complete static site to `out/` that any file host can serve
or that opens straight off disk: every page reads its data from the committed JSON
in `public/data` at build time, so the default build needs no server. Node ≥ 22.6
is required (the test harness uses native TypeScript type-stripping); the
toolchain is Next 16 and React 19, pinned in `pnpm-lock.yaml`.

### Live mode (optional)

Setting `QDS_LIVE=1` turns the static export into a running server and activates a
single on-demand endpoint, `GET /api/live?id=<scenario>`, that re-runs the engine
for one scenario instead of reading its committed snapshot:

```bash
QDS_LIVE=1 pnpm dev        # or: pnpm live
```

This is strictly additive. In the default build the live route file
(`app/api/live/route.live.ts`) is not a recognised route filename, so a static
export can never accidentally ship it, and a host without the Python engine beside
it simply falls back to the committed snapshots. `next.config.mjs` documents how
the two modes are kept mutually exclusive.

### What is here

`app/` is the App Router shell: the catalogue page (`page.tsx`) and the
per-scenario routes `s/[id]`, `s/[id]/statistics`, and `s/[id]/detection`.
`components/` holds the presentation (`figures.tsx`, `nav.tsx`, `tables.tsx`,
`ui.tsx`). `lib/` is the TypeScript layer that reads the data: `types.ts` is the
contract, written against the real JSON rather than the Python dataclasses (the
comment at its head explains why that distinction caught a real bug); `data.ts`
loads and shallow-validates snapshots, static-first with the optional live path;
`format.ts`, `select.ts`, and `chart.ts` are formatting, selection, and
chart-shaping helpers. `public/data/` holds the committed engine output, and
`scripts/generate_snapshots.py` regenerates it.

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
live path is simply unavailable and the static snapshots stand. The app is built
against this assumption, so the dashboard works with or without the engine
present.

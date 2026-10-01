/**
 * The lib layer, exercised under plain `node --experimental-strip-types`.
 *
 *   node --experimental-strip-types web/lib/lib.test.ts
 *
 * This is the one part of the dashboard this environment can genuinely run,
 * so it is where the verification effort goes. The React and Next layers are
 * checked by `next build` on the user's machine (see BUILD.md); everything
 * that can be a pure function lives here and is tested here, against the real
 * snapshots that `generate_snapshots.py` produced -- not fixtures, not mocks.
 *
 * There is no test framework because `npm install` is unavailable in the
 * environment this was written in. The harness below is a dozen lines and
 * exits non-zero on the first category of failure, which is all a CI step
 * needs.
 */

import { fileURLToPath } from "node:url";
import * as nodePath from "node:path";

import {
  ABSENT,
  bits,
  count,
  duration,
  hexToAscii,
  humanise,
  interval,
  pct,
  pValue,
  rate,
  ratio,
  signed,
  small,
  timestamp,
} from "./format.ts";

import {
  basisRows,
  byFamily,
  calibrationRows,
  comparisonMatrix,
  coverageGaps,
  declarationIsIndependent,
  estimateRows,
  explainVerdict,
  flaggedPremiseRules,
  ladderProblems,
  matchesExpectation,
  oracleRows,
  phaseRows,
  pooledPositionSeries,
  positionSeries,
  rankedAttributions,
  tallyTests,
  testRows,
  verifierRows,
  whyNotApplicable,
} from "./select.ts";

import {
  bandPath,
  frame,
  gridLayout,
  horizontalBars,
  horizontalErrorBar,
  linearScale,
  linePath,
  logScale,
  logTicks,
  pValueDomain,
  pValueLayout,
  rateDomain,
  series,
  stepPath,
  thresholdMarkers,
  tickDigits,
  ticks,
} from "./chart.ts";

import {
  cardById,
  loadAllFromDisk,
  loadIndexFromDisk,
  loadSnapshotFromDisk,
  parseIndex,
  parseSnapshot,
  parseSnapshotText,
  scenarioIds,
  SnapshotError,
} from "./data.ts";

import type { Snapshot, SnapshotIndex } from "./types.ts";

// ---------------------------------------------------------------------------
// harness
// ---------------------------------------------------------------------------

let passed = 0;
const failures: string[] = [];
let group = "";

function section(name: string): void {
  group = name;
}

function ok(cond: boolean, msg: string): void {
  if (cond) {
    passed++;
  } else {
    failures.push(`[${group}] ${msg}`);
  }
}

function eq<T>(got: T, want: T, msg: string): void {
  ok(got === want, `${msg}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
}

function approx(got: number, want: number, msg: string, tol = 1e-9): void {
  ok(Math.abs(got - want) <= tol, `${msg}: got ${got}, want ${want} (±${tol})`);
}

function throws(fn: () => unknown, needle: string, msg: string): void {
  try {
    fn();
    failures.push(`[${group}] ${msg}: expected a throw, got none`);
  } catch (e) {
    const m = (e as Error).message ?? "";
    ok(m.includes(needle), `${msg}: error "${m}" should mention "${needle}"`);
  }
}

// ---------------------------------------------------------------------------
// load the real data
// ---------------------------------------------------------------------------

const HERE = nodePath.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = nodePath.join(HERE, "..", "public", "data");

const index: SnapshotIndex = await loadIndexFromDisk(DATA_DIR);
const all = await loadAllFromDisk(DATA_DIR);
const snaps = all.snapshots;
const honest = snaps["honest-lab"];
const intercept = snaps["intercept-resend"];
const ebreak = snaps["entanglement-breaking"];
const degraded = snaps["degraded-link"];
const basisBiased = snaps["basis-biased"];

// ---------------------------------------------------------------------------
// format.ts
// ---------------------------------------------------------------------------

section("format");
eq(pct(null), ABSENT, "pct(null) is an em dash, not 0%");
eq(pct(undefined), ABSENT, "pct(undefined) is absent");
eq(pct(0.018644, 2), "1.86%", "pct rounds to the requested precision");
eq(pct(Number.NaN), ABSENT, "pct(NaN) is absent, not NaN%");
eq(rate(null), ABSENT, "rate(null) is absent");
eq(rate(0.0186440678, 4), "0.0186", "rate keeps four decimals by default");

eq(pValue(null), ABSENT, "a null p-value prints as a dash, not zero");
eq(pValue(0), "<1e-300", "a p-value that underflowed is not reported as certain");
eq(pValue(0.5), "0.5000", "a mid-range p-value stays in decimal form");
eq(pValue(0.99999), "1.000", "a p-value at the ceiling reads as 1.000");
ok(pValue(1e-10).includes("e"), "a tiny p-value switches to scientific notation");
eq(pValue(1e-10), "1.0e-10", "scientific p-value drops the leading-zero exponent padding");

eq(ratio(5, 0), ABSENT, "a zero denominator is a dash, never a division");
eq(ratio(5, 590), "5/590", "ratio prints k/n");
eq(count(1234567), "1,234,567", "counts get thousands separators");
eq(count(null), ABSENT, "a null count is absent");
eq(interval(null, 0.1), ABSENT, "a half-open interval is absent, not guessed");
eq(interval(0.0093, 0.0331, 4), "[0.0093, 0.0331]", "interval prints a closed range");
eq(bits(1200), "1200 bits", "large bit counts drop the decimal");
eq(bits(12.34), "12.3 bits", "small bit counts keep one decimal");
eq(duration(5e-6), "5µs", "sub-millisecond durations render in microseconds");
eq(duration(0.1), "100.0ms", "sub-second durations render in milliseconds");
eq(duration(5), "5.00s", "seconds render with two decimals");
eq(duration(75), "1m 15s", "minutes render as m and s");
eq(small(0), "0", "an exact zero small-number is just 0");
ok(small(2.18e-6).includes("e"), "a dark-count fraction renders in scientific form");
eq(humanise("pooled_rate_vs_spec"), "Pooled rate vs spec", "humanise de-snakes and capitalises");
eq(humanise(null), ABSENT, "humanise(null) is absent");
eq(signed(0.5), "+0.5000", "a positive difference carries an explicit plus");
eq(signed(-0.5), "-0.5000", "a negative difference keeps its minus");
ok(typeof timestamp(1790682493) === "string" && timestamp(1790682493) !== ABSENT, "a finite timestamp renders");
eq(timestamp(null), ABSENT, "a null timestamp is absent");

// The signed message was the two bytes b"hi"; its hex must round-trip.
eq(hexToAscii("6869"), "hi", "the declaration message decodes to its ASCII");
eq(hexToAscii(honest.session.declaration?.message_hex), "hi", "the real snapshot's message is 'hi'");
eq(hexToAscii("ff00"), null, "non-printable bytes decode to null, not mojibake");
eq(hexToAscii("abc"), null, "odd-length hex is rejected");

// ---------------------------------------------------------------------------
// select.ts -- verdict routing
// ---------------------------------------------------------------------------

section("select/verdict");
const vHonest = explainVerdict(honest.detection);
eq(vHonest.verdict, "clean", "honest link reads clean");
eq(vHonest.tone, "ok", "a clean verdict is toned ok");

const vIntercept = explainVerdict(intercept.detection);
eq(vIntercept.verdict, "compromised", "intercept-resend reads compromised");
eq(vIntercept.tone, "alarm", "a compromised verdict is toned alarm");
// intercept-resend breaches the rate (pooled rate ~0.24 >> accept ~0.04) but
// its rejections list may be empty when verifiers still technically accepted;
// the headline must be one of the two rate/premise alarm forms, never the
// bare fallback.
ok(vIntercept.headline !== "Compromised", "a real compromise explains itself rather than falling back");

// degraded-link is the discriminating case: compromised with no adversary,
// via a premise rule, at a rate *below* the accept threshold.
eq(degraded.detection.verdict, "compromised", "degraded link reads compromised");
eq(degraded.detection.authentication_failed, false, "degraded link did not fail authentication");
ok(
  !degraded.detection.rejections.some((r) => r.rate_exceeded),
  "degraded link was not refused on rate (it is under the accept threshold)",
);
eq(
  flaggedPremiseRules(degraded.detection).join(","),
  "yield_vs_declared",
  "degraded link's only flag is the premise rule yield_vs_declared",
);
eq(
  explainVerdict(degraded.detection).headline,
  "A premise of the analysis is false",
  "degraded link is explained as a false premise, not a rate breach",
);

eq(matchesExpectation(cardById(index, "honest-lab")!), true, "honest-lab met its 'clean' expectation");
eq(
  matchesExpectation(cardById(index, "basis-biased")!),
  false,
  "basis-biased expected 'suspicious' but the engine said compromised -- surfaced, not hidden",
);
eq(
  matchesExpectation(cardById(index, "degraded-link")!),
  null,
  "a non-verdict expectation string yields null, not a false match",
);

// ---------------------------------------------------------------------------
// select.ts -- tests
// ---------------------------------------------------------------------------

section("select/tests");
for (const s of Object.values(snaps)) {
  const t = tallyTests(s.detection);
  eq(t.total, s.detection.tests.length, `${s.id}: tally covers every test`);
  eq(
    t.flagged + t.passed + t.inapplicable,
    t.total,
    `${s.id}: every test is flagged, passed, or inapplicable -- nothing double-counted`,
  );
  eq(t.flagged, s.detection.n_flagged, `${s.id}: flagged tally matches the engine's n_flagged`);

  const rows = testRows(s.detection);
  // The editorial invariant: an inapplicable test is never a passed one.
  for (const r of rows) {
    if (!r.outcome.applicable) {
      eq(r.status, "inapplicable", `${s.id}/${r.outcome.name}: a test that could not run is not 'passed'`);
    }
    if (r.status === "flagged") {
      ok(r.outcome.flagged, `${s.id}/${r.outcome.name}: 'flagged' status implies the outcome flagged`);
    }
  }
}
// sequential_onset has no p-value by construction; it must still classify.
const soRow = testRows(honest.detection).find((r) => r.outcome.name === "sequential_onset")!;
eq(soRow.outcome.p_value, null, "sequential_onset carries no p-value by construction");
ok(soRow.status !== "inapplicable" || !soRow.outcome.applicable, "a null p-value alone does not make a test inapplicable");

// premise flagging is consistent with the type-level set
for (const r of testRows(degraded.detection)) {
  if (r.outcome.name === "yield_vs_declared") ok(r.premise, "yield_vs_declared is marked a premise rule");
}

eq(typeof whyNotApplicable(honest.detection.tests[0]), "object", "whyNotApplicable returns a string or null") as unknown;
ok(
  whyNotApplicable(honest.detection.tests[0]) === null || typeof whyNotApplicable(honest.detection.tests[0]) === "string",
  "whyNotApplicable is string-or-null",
);

const attrs = rankedAttributions(intercept.detection);
ok(
  attrs.every((a, i) => i === 0 || attrs[i - 1].priority <= a.priority),
  "attributions come back in non-decreasing priority order",
);

// ---------------------------------------------------------------------------
// select.ts -- estimates & bases
// ---------------------------------------------------------------------------

section("select/estimates");
const est = estimateRows(honest);
eq(est.length, 4, "the four headline estimates are present");
eq(est.map((e) => e.key).join(","), "pooled_rate,detector_yield,dark_fraction,decoy_rate", "estimates come in reading order");
approx(est[0].estimate.value!, 0.01864406779661017, "pooled_rate value is the real engine number");
eq(est[0].declared, honest.detection.thresholds.spec_rate, "pooled_rate is judged against the declared floor");
eq(est[3].declared, honest.detection.thresholds.spec_rate, "decoy_rate shares the signature floor -- that is what makes decoy_vs_signature a test");

const bRows = basisRows(honest);
eq(bRows.length, 2, "both measurement bases are present");
const zRow = bRows.find((r) => r.basis === "Z")!;
const xRow = bRows.find((r) => r.basis === "X")!;
eq(zRow.declared, honest.detection.declared.spec_rate_z, "Z basis is compared against its own declared floor");
eq(xRow.declared, honest.detection.declared.spec_rate_x, "X basis is compared against its own declared floor");
ok(zRow.declared !== xRow.declared, "the per-basis floors genuinely differ -- pooling to one number would be wrong");

// ---------------------------------------------------------------------------
// select.ts -- verifications, positions, phases, calibration
// ---------------------------------------------------------------------------

section("select/session");
const vr = verifierRows(honest);
eq(vr.length, 2, "honest-lab has one direct and one forwarded verification");
ok(vr.some((r) => r.kind === "accept"), "a direct recipient is present");
ok(vr.some((r) => r.kind === "transfer"), "a forwarded recipient is present");

const ps = positionSeries(vr[0].doc);
ok(ps.length > 0, "a verification yields a per-position series");
ok(ps.every((p, i) => i === 0 || ps[i - 1].position <= p.position), "position series is sorted");
for (const p of ps) {
  if (p.checks > 0) approx(p.rate!, p.mismatch / p.checks, `position ${p.position}: rate is mismatch/checks`);
}

const pooled = pooledPositionSeries(honest);
// Pooled checks at each position must equal the sum across verifications.
const manual = new Map<number, number>();
for (const row of vr) for (const p of positionSeries(row.doc)) manual.set(p.position, (manual.get(p.position) ?? 0) + p.checks);
for (const p of pooled) eq(p.checks, manual.get(p.position), `pooled position ${p.position}: checks are summed across verifiers`);

const phases = phaseRows(honest);
const fixedOrder = phases.filter((p) => !p.key.startsWith("verify:")).map((p) => p.key);
eq(fixedOrder.join(","), "enrol,distribute,symmetrise,calibrate,sign", "protocol phases appear in their mandatory order");
ok(phases.filter((p) => p.key.startsWith("verify:")).length >= 1, "verification steps are appended");
const shareSum = phases.reduce((a, p) => a + p.share, 0);
approx(shareSum, 1, "phase shares sum to one", 1e-6);

const cal = calibrationRows(honest);
ok(cal.length >= 1, "calibration rows are present");
for (const c of cal) {
  eq(typeof c.violation, "boolean", `${c.verifier}: violation is a boolean`);
  eq(typeof c.informative, "boolean", `${c.verifier}: informative is a boolean`);
}

// ---------------------------------------------------------------------------
// select.ts -- cross-scenario comparison
// ---------------------------------------------------------------------------

section("select/comparison");
const matrix = comparisonMatrix(index);
eq(matrix.rules.length, 11, "all eleven detectors are rows in the matrix");
// Every cell must agree with the card's own flagged list.
matrix.rules.forEach((ruleName, ri) => {
  matrix.scenarios.forEach((sc, ci) => {
    eq(matrix.flagged[ri][ci], sc.flagged.includes(ruleName), `cell ${ruleName}×${sc.id} matches the card`);
  });
});
eq(
  matrix.activeRules.length + matrix.silentRules.length,
  matrix.rules.length,
  "active and silent rules partition the detector set -- no rule is dropped",
);
ok(matrix.activeRules.includes("pooled_rate_vs_spec"), "the pooled-rate detector fired somewhere in the suite");

const fam = byFamily(index);
eq(fam.map((f) => f.family).filter((f) => ["baseline", "attack", "hardware"].includes(f)).join(","), "baseline,attack,hardware", "families read baseline, attack, hardware");

const gaps = coverageGaps(index);
// collective-depolarizing and coherent-probe are attacks the Bonferroni-
// corrected suite does not catch; a dashboard that hid them would be marketing.
ok(gaps.missed.some((s) => s.id === "coherent-probe"), "the coherent-probe evasion is surfaced as a coverage gap");
ok(gaps.missed.some((s) => s.id === "collective-depolarizing"), "the weak depolarising evasion is surfaced as a coverage gap");
for (const m of gaps.missed) eq(m.family, "attack", "a 'missed' scenario is an attack that read clean");
for (const f of gaps.falseAlarms) eq(f.family, "baseline", "a 'false alarm' is a baseline that did not read clean");

// ---------------------------------------------------------------------------
// select.ts -- oracle, independence, ladder
// ---------------------------------------------------------------------------

section("select/oracle");
const oracle = oracleRows(honest);
ok(oracle.length >= 1, "the privileged oracle is surfaced for inspection");
for (const o of oracle) {
  ok(Number.isFinite(o.meanChsh), `${o.verifier}: CHSH value is finite`);
  ok(o.meanChsh > 1.9 && o.meanChsh <= 2.9, `${o.verifier}: CHSH sits in the physical range for this channel`);
  ok(o.eveGuessProbability >= 0 && o.eveGuessProbability <= 1, `${o.verifier}: Eve's guess probability is a probability`);
}
eq(declarationIsIndependent(honest.detection), true, "the lab spec sheet was declared independently of the channel");
eq(ladderProblems(honest.detection).length, 0, "the honest-link threshold ladder is well-formed");

// ---------------------------------------------------------------------------
// chart.ts -- scales
// ---------------------------------------------------------------------------

section("chart/scales");
const lin = linearScale([0, 1], [0, 100]);
approx(lin(0.5), 50, "linear scale maps the midpoint");
approx(lin.invert(50), 0.5, "linear scale inverts");
// A collapsed domain must be widened, never divided by zero.
const flat = linearScale([5, 5], [0, 100]);
ok(Number.isFinite(flat(5)), "a degenerate domain does not produce NaN");
approx(flat(5), 50, "a single point lands centred, with honest space around it");

const lg = logScale([1e-10, 1], [0, 100]);
approx(lg(1), 100, "log scale maps the top of the domain");
approx(lg(1e-10), 0, "log scale maps the bottom of the domain");
ok(Number.isFinite(lg(0)), "log scale clamps zero to the floor rather than returning -Infinity");
ok(Number.isFinite(lg(-5)), "log scale clamps negatives to the floor");

// ---------------------------------------------------------------------------
// chart.ts -- ticks
// ---------------------------------------------------------------------------

section("chart/ticks");
const tk = ticks([0, 0.2], 5);
ok(tk.length >= 3 && tk.length <= 7, "tick count is near the request");
ok(tk.every((v) => Number.isFinite(v)), "ticks are all finite");
ok(tk.every((v) => v >= -1e-9 && v <= 0.2 + 1e-9), "ticks stay within the domain");
// 1/2/5 stepping: the gap should be a round 0.05 here.
approx(tk[1] - tk[0], 0.05, "ticks step by a round 1/2/5 amount");
const lt = logTicks([1e-6, 1]);
ok(lt.every((v) => Number.isFinite(v) && v > 0), "log ticks are positive and finite");
ok(lt.length >= 2, "log ticks span at least two decades");
eq(tickDigits([0, 0.05, 0.1]), 2, "tickDigits finds the fewest decimals that keep labels distinct");

// ---------------------------------------------------------------------------
// chart.ts -- series & paths
// ---------------------------------------------------------------------------

section("chart/series");
const se = series([
  { x: 2, y: 0.2 },
  { x: 0, y: 0.1 },
  { x: 1, y: null },
  { x: 3, y: Number.NaN },
]);
eq(se.points.length, 2, "null and NaN points are dropped from the series");
eq(se.dropped, 2, "the dropped count is reported for the caption");
eq(se.points[0].x, 0, "surviving points are sorted by x");

const sx = linearScale([0, 3], [0, 300]);
const sy = linearScale([0, 0.3], [100, 0]);
ok(linePath(se.points, sx, sy).startsWith("M"), "a line path starts with a moveto");
eq(linePath([], sx, sy), "", "an empty series renders no path");
const step = stepPath(se.points, sx, sy);
ok(step.startsWith("M") && step.includes("L"), "a step path holds each level flat then jumps");
const band = bandPath([{ x: 0, lo: 0.05, hi: 0.15 }, { x: 1, lo: 0.08, hi: 0.2 }], sx, sy);
ok(band.endsWith("Z"), "a confidence band closes its path");

// ---------------------------------------------------------------------------
// chart.ts -- error bars & thresholds
// ---------------------------------------------------------------------------

section("chart/errorbars");
const axis = linearScale([0, 0.05], [0, 500]);
// An interval wider than the axis must report clipping, not silently shrink.
const bar = horizontalErrorBar(0.02, 0.005, 0.2, axis, 10);
ok(bar.clippedHi, "an interval running off the top of the axis is flagged clipped");
ok(bar.hi <= 500 + 0.5, "the clipped end is pinned to the axis, not drawn past it");
const tight = horizontalErrorBar(0.02, 0.018, 0.022, axis, 10);
ok(!tight.clippedLo && !tight.clippedHi, "an interval inside the axis is not flagged clipped");

const markers = thresholdMarkers(honest.detection.thresholds, axis);
eq(markers.length, 4, "all four ladder levels become markers");
ok(markers.every((m, i) => i === 0 || markers[i - 1].value <= m.value), "markers are ordered up the ladder");
eq(markers[0].tone, "spec", "the lowest marker is the declared floor");
eq(markers[3].tone, "forger", "the highest marker is the forger bound");
eq(markers.map((m) => m.rank).join(","), "0,1,2,3", "ranks are assigned in ascending order for label staggering");

const rd = rateDomain([0.01, 0.02, null, 0.25]);
ok(rd[0] >= 0 && rd[1] > 0.25, "rate domain covers every value and never dips below zero");

// ---------------------------------------------------------------------------
// chart.ts -- grids, bars, p-value layout
// ---------------------------------------------------------------------------

section("chart/grid");
const grid = gridLayout(11, 8, (r, c) => matrix.flagged[r][c]);
eq(grid.cells.length, 88, "the rule×scenario grid has a cell per pair");
ok(grid.width > 0 && grid.height > 0, "the grid has a positive extent");

const barsOut = horizontalBars(phaseRows(honest).map((p) => ({ key: p.key, label: p.label, value: p.seconds })), 400);
ok(barsOut.bars.length === phaseRows(honest).length, "one bar per phase");
ok(barsOut.bars.every((b) => b.width >= 0 && b.width <= 400 + 1e-9), "bar widths stay within the plot");

const pAxis = logScale([1e-12, 1], [0, 400]);
const pl = pValueLayout(
  [
    { name: "a", title: "A", p_value: 0.5, flagged: false, applicable: true },
    { name: "b", title: "B", p_value: 0, flagged: true, applicable: true },
    { name: "c", title: "C", p_value: null, flagged: false, applicable: true },
    { name: "d", title: "D", p_value: 1e-20, flagged: true, applicable: true },
  ],
  pAxis,
);
ok(pl[0].at !== null && !pl[0].underflow, "an ordinary p-value is placed on the axis");
ok(pl[1].underflow && pl[1].at !== null, "a p-value of zero is flagged underflow and pinned, not dropped");
eq(pl[2].at, null, "a null p-value gets no position -- it has none by construction");
ok(pl[3].underflow, "a p-value below the axis floor is treated as underflow");
const pd = pValueDomain([0.5, 1e-8, null], 9.09e-4);
ok(pd[0] >= 1e-12 && pd[1] === 1, "the p-value domain is floored and topped at one");

// pValueLayout on the real report: sequential_onset (null p) must get at:null.
const realPl = pValueLayout(
  honest.detection.tests.map((t) => ({ name: t.name, title: t.title, p_value: t.p_value, flagged: t.flagged, applicable: t.applicable })),
  pAxis,
);
const realSo = realPl.find((p) => p.name === "sequential_onset")!;
eq(realSo.at, null, "the real sequential_onset lays out with no axis position");

// ---------------------------------------------------------------------------
// data.ts -- parsing & validation
// ---------------------------------------------------------------------------

section("data/validation");
eq(scenarioIds(index).length, 8, "the index lists all eight scenarios");
eq(Object.keys(snaps).length, 8, "every listed scenario loaded from disk");
ok(cardById(index, "honest-lab") !== null, "cardById finds a known scenario");
eq(cardById(index, "no-such"), null, "cardById returns null for an unknown id");

// A round-trip through the string parsers must succeed on real data.
const reSnap: Snapshot = parseSnapshot(JSON.parse(JSON.stringify(honest)), "honest-lab.json");
eq(reSnap.id, "honest-lab", "parseSnapshot accepts a real snapshot");
const reIdx = parseIndex(JSON.parse(JSON.stringify(index)));
eq(reIdx.scenarios.length, 8, "parseIndex accepts the real index");

// Validation must fail loudly, naming the offending field.
throws(() => parseSnapshotText("{not json", "broken.json"), "broken.json", "invalid JSON names its source file");
throws(
  () => parseSnapshot({ id: "x", session: {}, detection: { tests: [], flagged: [], thresholds: {}, estimates: {} } }, "x.json"),
  "detection.verdict",
  "a snapshot missing its verdict is rejected by field name",
);
throws(
  () => parseSnapshot({ id: "x", detection: { verdict: "clean", tests: [], flagged: [], thresholds: {}, estimates: {} } }, "x.json"),
  "session",
  "a snapshot missing its session is rejected",
);
throws(
  () => parseIndex({ geometry: {}, rules: [], scenarios: [] }),
  "at least one scenario",
  "an empty index is rejected",
);
throws(
  () => parseSnapshot({ id: "x", session: {}, detection: { verdict: "bogus", tests: [], flagged: [], thresholds: {}, estimates: {} } }, "x.json"),
  "clean|suspicious|compromised",
  "an illegal verdict value is rejected",
);
ok(new SnapshotError("msg", "f.json").message.includes("f.json"), "SnapshotError carries its source");

// Loading a single snapshot by id works and validates.
const one = await loadSnapshotFromDisk("entanglement-breaking", DATA_DIR);
eq(one.id, "entanglement-breaking", "loadSnapshotFromDisk returns the right snapshot");
ok(one.detection.estimates.pooled_rate.value! > 0.3, "entanglement-breaking's pooled rate is the real, severe number");

// A missing file fails with a helpful, named error.
await (async () => {
  try {
    await loadSnapshotFromDisk("does-not-exist", DATA_DIR);
    failures.push("[data/validation] a missing snapshot should throw");
  } catch (e) {
    ok((e as Error).message.includes("generate_snapshots"), "a missing snapshot points at how to produce it");
  }
})();

// ---------------------------------------------------------------------------
// report
// ---------------------------------------------------------------------------

console.log(`\n${"-".repeat(60)}`);
if (failures.length === 0) {
  console.log(`  all ${passed} assertions passed`);
  console.log(`${"-".repeat(60)}\n`);
  process.exit(0);
} else {
  console.log(`  ${passed} passed, ${failures.length} FAILED`);
  console.log(`${"-".repeat(60)}`);
  for (const f of failures) console.log(`  ✗ ${f}`);
  console.log("");
  process.exit(1);
}

/**
 * Selectors: everything the pages need to know about a snapshot, computed
 * once here rather than inline in JSX.
 *
 * These are pure functions over the snapshot documents, which is what makes
 * them testable under plain `node` (see `web/lib/lib.test.ts`). Nothing in
 * this file imports React or touches the filesystem.
 *
 * One editorial rule is enforced here rather than left to each component: a
 * test that could not run is never counted as a test that passed. The engine
 * distinguishes `applicable: false` from `flagged: false`, and collapsing the
 * two would turn an absent measurement into a reassuring one.
 */

import {
  PREMISE_RULES,
  RULE_NAMES,
  type Attribution,
  type CalibrationDoc,
  type DetectionReport,
  type RateEstimate,
  type Rejection,
  type RuleName,
  type ScenarioCard,
  type Snapshot,
  type SnapshotIndex,
  type TestOutcome,
  type VerificationDoc,
  type Verdict,
} from "./types.ts";

// ---------------------------------------------------------------------------
// verdicts
// ---------------------------------------------------------------------------

export type Tone = "ok" | "watch" | "alarm" | "muted";

export interface VerdictMeaning {
  verdict: Verdict;
  tone: Tone;
  headline: string;
  /** why the engine landed here, in the engine's own terms */
  because: string;
}

/**
 * Explain a verdict, including *which* of the three routes to `compromised`
 * was taken. The routes are not interchangeable: a MAC failure has no
 * innocent explanation, a rate breach has several, and a premise violation
 * means the thresholds were being applied to a situation they were not
 * derived for. Showing only the word "compromised" throws that away.
 */
export function explainVerdict(d: DetectionReport): VerdictMeaning {
  if (d.verdict === "compromised") {
    if (d.authentication_failed) {
      return {
        verdict: d.verdict,
        tone: "alarm",
        headline: "Authentication failed",
        because:
          "A recipient rejected the signature on the classical half: the MAC tag, " +
          "the signer's registry entry, or the anti-replay ledger did not check " +
          "out. Honest noise cannot produce this.",
      };
    }
    const breached = d.rejections.some((r) => r.rate_exceeded);
    if (breached) {
      return {
        verdict: d.verdict,
        tone: "alarm",
        headline: "Signature refused on error rate",
        because:
          "A recipient's measured error rate reached the acceptance threshold, " +
          "so the protocol refused the signature on its own terms. The threshold " +
          "was fixed in advance from the declared noise floor.",
      };
    }
    const premise = flaggedPremiseRules(d);
    if (premise.length > 0) {
      return {
        verdict: d.verdict,
        tone: "alarm",
        headline: "A premise of the analysis is false",
        because:
          `${premise.map(humanRule).join(", ")} fired. The thresholds are ` +
          "derived assuming the loss budget is as declared, the transcript is " +
          "uniform, and the recipients hold equivalent keys. With one of those " +
          "false, a rate under the threshold is not evidence of anything.",
      };
    }
    return {
      verdict: d.verdict,
      tone: "alarm",
      headline: "Compromised",
      because: "The engine refused this run.",
    };
  }

  if (d.verdict === "suspicious") {
    return {
      verdict: d.verdict,
      tone: "watch",
      headline: "Departure from specification",
      because:
        `${d.n_flagged} of ${d.n_applicable} applicable detectors fired, but ` +
        "every recipient would still accept and the premises still hold. " +
        "Measurably different from the declared floor is not the same as " +
        "operationally unsafe.",
    };
  }

  return {
    verdict: d.verdict,
    tone: "ok",
    headline: "Within specification",
    because:
      "No detector fired and every recipient accepted. At a Bonferroni-corrected " +
      `alpha of ${d.alpha_per_test?.toExponential(1) ?? "?"} per test, this is ` +
      "the expected result for a link running inside its noise floor.",
  };
}

/** Flagged rules that invalidate a premise rather than just raising the rate. */
export function flaggedPremiseRules(d: DetectionReport): string[] {
  return d.flagged.filter((n) => PREMISE_RULES.has(n));
}

/**
 * Whether the scenario did what it was built to do. This is the honest
 * comparison: a scenario whose expectation is `clean` and whose verdict is
 * `compromised` is a finding about the engine, not a rendering problem, and
 * the dashboard should say so rather than quietly agreeing with itself.
 */
export function matchesExpectation(card: ScenarioCard): boolean | null {
  const e = card.expectation.trim().toLowerCase();
  if (e === "clean" || e === "suspicious" || e === "compromised") {
    return card.verdict === e;
  }
  return null;
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

export type TestStatus = "flagged" | "passed" | "inapplicable";

export interface TestRow {
  outcome: TestOutcome;
  status: TestStatus;
  /** true when firing means a premise is false, not merely that the rate is high */
  premise: boolean;
  /** the alpha this test was actually judged against */
  alpha: number | null;
}

/** Classify every detector, preserving the engine's report order. */
export function testRows(d: DetectionReport): TestRow[] {
  return d.tests.map((outcome) => ({
    outcome,
    status: !outcome.applicable
      ? ("inapplicable" as const)
      : outcome.flagged
        ? ("flagged" as const)
        : ("passed" as const),
    premise: PREMISE_RULES.has(outcome.name),
    alpha: d.alpha_per_test,
  }));
}

export interface TestTally {
  flagged: number;
  passed: number;
  inapplicable: number;
  total: number;
}

export function tallyTests(d: DetectionReport): TestTally {
  const rows = testRows(d);
  return {
    flagged: rows.filter((r) => r.status === "flagged").length,
    passed: rows.filter((r) => r.status === "passed").length,
    inapplicable: rows.filter((r) => r.status === "inapplicable").length,
    total: rows.length,
  };
}

/** Why a test could not run, if the rule recorded a reason. */
export function whyNotApplicable(o: TestOutcome): string | null {
  const w = o.detail?.["why_not"];
  return typeof w === "string" && w.length > 0 ? w : null;
}

/** `pooled_rate_vs_spec` to `Pooled rate vs spec`, for prose. */
export function humanRule(name: string): string {
  const s = name.replace(/_/g, " ").replace(/\bvs\b/g, "vs");
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** Look one detector up by name. */
export function findTest(d: DetectionReport, name: RuleName): TestOutcome | null {
  return d.tests.find((t) => t.name === name) ?? null;
}

/**
 * Attributions in table order. The engine returns every match rather than
 * only the first, because the secondary matches are what let an operator
 * tell a story about the primary one.
 */
export function rankedAttributions(d: DetectionReport): Attribution[] {
  return [...d.attribution].sort((a, b) => a.priority - b.priority);
}

// ---------------------------------------------------------------------------
// estimates
// ---------------------------------------------------------------------------

export interface EstimateRow {
  key: string;
  estimate: RateEstimate;
  /** the declared figure this estimate is judged against, when there is one */
  declared: number | null;
  declaredLabel: string | null;
}

/**
 * The four headline estimates, each paired with the declared figure it is
 * compared against. `decoy_rate` has no separate declaration -- it is judged
 * against the same spec rate as the signature slots, which is precisely why
 * `decoy_vs_signature` is a meaningful test.
 */
export function estimateRows(s: Snapshot): EstimateRow[] {
  const e = s.detection.estimates;
  const spec = s.detection.thresholds.spec_rate;
  const out: EstimateRow[] = [];
  const add = (key: string, declared: number | null, label: string | null) => {
    const est = e[key];
    if (est) out.push({ key, estimate: est, declared, declaredLabel: label });
  };
  add("pooled_rate", spec, "declared noise floor");
  add("detector_yield", s.detection.declared.yield, "declared loss budget");
  add("dark_fraction", s.detection.declared.dark_fraction, "declared dark count");
  add("decoy_rate", spec, "declared noise floor");
  return out;
}

export interface BasisRow {
  basis: string;
  estimate: RateEstimate;
  declared: number | null;
}

/**
 * Per-basis rates against their per-basis declared floors.
 *
 * The declared floor is one number on a sheet, but the honest error rate is
 * not equal in both bases -- memory dephasing spares Z eigenstates, and an
 * X-basis check passes through more gate insertions. The engine splits the
 * pooled floor in the proportion the declared hardware model predicts, which
 * is why comparing both bases to a single number would be wrong.
 */
export function basisRows(s: Snapshot): BasisRow[] {
  const declared: Record<string, number | null> = {
    Z: s.detection.declared.spec_rate_z,
    X: s.detection.declared.spec_rate_x,
  };
  return Object.entries(s.detection.estimates_by_basis).map(([basis, estimate]) => ({
    basis,
    estimate,
    declared: declared[basis] ?? s.detection.thresholds.spec_rate,
  }));
}

// ---------------------------------------------------------------------------
// verification reports
// ---------------------------------------------------------------------------

export interface VerifierRow {
  key: string;
  doc: VerificationDoc;
  /** "accept" for a direct recipient, "transfer" for a forwarded one */
  kind: "accept" | "transfer";
  rejection: Rejection | null;
}

/**
 * Every verification this session produced, direct and forwarded, in one
 * list. The distinction is kept because the thresholds differ: a forwarded
 * signature is judged against the transfer threshold, which is looser, and
 * that is the whole content of transferability.
 */
export function verifierRows(s: Snapshot): VerifierRow[] {
  const rows: VerifierRow[] = [];
  const rejectionFor = (verifier: string, level: string) =>
    s.detection.rejections.find(
      (r) => r.verifier === verifier && r.level === level,
    ) ?? null;

  for (const [key, doc] of Object.entries(s.session.verifications ?? {})) {
    rows.push({ key, doc, kind: "accept", rejection: rejectionFor(doc.verifier, doc.level) });
  }
  for (const [key, doc] of Object.entries(s.session.transfers ?? {})) {
    rows.push({ key, doc, kind: "transfer", rejection: rejectionFor(doc.verifier, doc.level) });
  }
  return rows;
}

export interface PositionPoint {
  position: number;
  mismatch: number;
  checks: number;
  rate: number | null;
}

/**
 * Per-message-bit-position error rates for one verification.
 *
 * Which slot carries which bit is not revealed until after transmission, so
 * honest noise has no way to prefer one position over another. A real
 * gradient here is what `position_homogeneity` tests for.
 */
export function positionSeries(doc: VerificationDoc): PositionPoint[] {
  const entries = Object.entries(doc.by_position ?? {});
  return entries
    .map(([k, v]) => ({
      position: Number(k),
      mismatch: v.mismatch,
      checks: v.checks,
      rate: v.checks > 0 ? v.mismatch / v.checks : null,
    }))
    .sort((a, b) => a.position - b.position);
}

/** Summed per-position counts across every verification in the session. */
export function pooledPositionSeries(s: Snapshot): PositionPoint[] {
  const acc = new Map<number, { mismatch: number; checks: number }>();
  for (const row of verifierRows(s)) {
    for (const p of positionSeries(row.doc)) {
      const cur = acc.get(p.position) ?? { mismatch: 0, checks: 0 };
      cur.mismatch += p.mismatch;
      cur.checks += p.checks;
      acc.set(p.position, cur);
    }
  }
  return [...acc.entries()]
    .map(([position, v]) => ({
      position,
      mismatch: v.mismatch,
      checks: v.checks,
      rate: v.checks > 0 ? v.mismatch / v.checks : null,
    }))
    .sort((a, b) => a.position - b.position);
}

// ---------------------------------------------------------------------------
// protocol phases
// ---------------------------------------------------------------------------

export interface PhaseRow {
  key: string;
  label: string;
  seconds: number;
  /** share of total wall time, in [0,1] */
  share: number;
}

const PHASE_LABELS: Record<string, string> = {
  enrol: "Enrol",
  distribute: "Distribute",
  symmetrise: "Symmetrise",
  calibrate: "Calibrate",
  sign: "Sign",
};

/**
 * Timings in protocol order, with verification steps appended.
 *
 * Order is not cosmetic here: a public key is consumable, so the phases must
 * happen once and in sequence. `verify:*` keys are emitted per verifier and
 * level, so they are sorted rather than mapped to fixed labels.
 */
export function phaseRows(s: Snapshot): PhaseRow[] {
  const t = s.session.timings ?? {};
  const total = Object.values(t).reduce((a, b) => a + (b || 0), 0) || 1;
  const fixed = ["enrol", "distribute", "symmetrise", "calibrate", "sign"];
  const rows: PhaseRow[] = [];

  for (const key of fixed) {
    if (typeof t[key] === "number") {
      rows.push({
        key,
        label: PHASE_LABELS[key] ?? key,
        seconds: t[key],
        share: t[key] / total,
      });
    }
  }
  for (const key of Object.keys(t).filter((k) => k.startsWith("verify:")).sort()) {
    const [, verifier, level] = key.split(":");
    rows.push({
      key,
      label: `Verify — ${verifier} (${level})`,
      seconds: t[key],
      share: t[key] / total,
    });
  }
  return rows;
}

// ---------------------------------------------------------------------------
// calibration
// ---------------------------------------------------------------------------

export interface CalibrationRow {
  verifier: string;
  doc: CalibrationDoc;
  /** true when the decoys are inconsistent with the declared floor */
  violation: boolean;
  /** true when the decoy budget certifies anything useful before signing */
  informative: boolean;
}

export function calibrationRows(s: Snapshot): CalibrationRow[] {
  return Object.entries(s.session.calibration ?? {}).map(([verifier, doc]) => ({
    verifier,
    doc,
    violation: Boolean(doc.spec_violation),
    informative: Boolean(doc.informative),
  }));
}

// ---------------------------------------------------------------------------
// scenario comparison
// ---------------------------------------------------------------------------

export interface ComparisonCell {
  scenario: string;
  rule: string;
  flagged: boolean;
}

export interface ComparisonMatrix {
  rules: string[];
  scenarios: ScenarioCard[];
  /** `flagged[ruleIndex][scenarioIndex]` */
  flagged: boolean[][];
  /** rules that fired in at least one scenario */
  activeRules: string[];
  /** rules that never fired anywhere; kept visible so absence is legible */
  silentRules: string[];
}

/**
 * The rule-by-scenario grid.
 *
 * Silent rules are reported separately rather than dropped. A detector that
 * never fires across the whole suite is a real observation about the suite's
 * coverage -- it means no scenario here exercises what that detector looks
 * for -- and hiding the empty rows would present the suite as more
 * comprehensive than it is.
 */
export function comparisonMatrix(index: SnapshotIndex): ComparisonMatrix {
  const rules = index.rules.length > 0 ? index.rules : [...RULE_NAMES];
  const scenarios = index.scenarios;
  const flagged = rules.map((rule) =>
    scenarios.map((s) => s.flagged.includes(rule)),
  );
  const activeRules = rules.filter((_, i) => flagged[i].some(Boolean));
  const silentRules = rules.filter((_, i) => !flagged[i].some(Boolean));
  return { rules, scenarios, flagged, activeRules, silentRules };
}

/** Scenarios grouped by family, families in a fixed reading order. */
export function byFamily(index: SnapshotIndex): Array<{
  family: string;
  label: string;
  scenarios: ScenarioCard[];
}> {
  const order = ["baseline", "attack", "hardware"];
  const labels: Record<string, string> = {
    baseline: "Baseline",
    attack: "Adversarial interventions",
    hardware: "Hardware departures",
  };
  const groups = new Map<string, ScenarioCard[]>();
  for (const s of index.scenarios) {
    const list = groups.get(s.family) ?? [];
    list.push(s);
    groups.set(s.family, list);
  }
  const keys = [...groups.keys()].sort((a, b) => {
    const ia = order.indexOf(a);
    const ib = order.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
  });
  return keys.map((family) => ({
    family,
    label: labels[family] ?? humanRule(family),
    scenarios: groups.get(family) ?? [],
  }));
}

/**
 * Scenarios where an attack went undetected, and clean-channel scenarios
 * where something fired anyway. Both directions matter: the first is the
 * engine's sensitivity limit, the second its false-alarm behaviour, and a
 * dashboard that only showed successful detections would be advertising.
 */
export function coverageGaps(index: SnapshotIndex): {
  missed: ScenarioCard[];
  falseAlarms: ScenarioCard[];
} {
  const missed = index.scenarios.filter(
    (s) => s.family === "attack" && s.verdict === "clean",
  );
  const falseAlarms = index.scenarios.filter(
    (s) => s.family === "baseline" && s.verdict !== "clean",
  );
  return { missed, falseAlarms };
}

// ---------------------------------------------------------------------------
// privileged ground truth
// ---------------------------------------------------------------------------

export interface OracleRow {
  verifier: string;
  meanChsh: number;
  meanBellFidelity: number;
  exactQber: number;
  eveTraceDistance: number;
  eveGuessProbability: number;
}

/**
 * The simulator's ground truth, which no detector is allowed to read.
 *
 * `qds.detect.Evidence` is written so this cannot reach the engine, and the
 * test suite greps the package to prove it. Surfacing it in the dashboard is
 * safe and useful for the opposite reason: it lets a reader check the
 * engine's conclusions against the answer the engine was denied. It must
 * always be labelled as privileged, never mixed into measured quantities.
 */
export function oracleRows(s: Snapshot): OracleRow[] {
  const out: OracleRow[] = [];
  for (const [verifier, dist] of Object.entries(s.session.distribution ?? {})) {
    const o = dist.oracle;
    if (!o) continue;
    out.push({
      verifier,
      meanChsh: o.mean_chsh,
      meanBellFidelity: o.mean_bell_fidelity,
      exactQber: o.exact_qber,
      eveTraceDistance: o.information_disturbance?.eve_trace_distance ?? 0,
      eveGuessProbability: o.information_disturbance?.eve_guess_probability ?? 0.5,
    });
  }
  return out;
}

/** Whether the declared figures were independent of the channel being judged. */
export function declarationIsIndependent(d: DetectionReport): boolean {
  return d.declaration_source === "spec_sheet";
}

/**
 * Problems with the threshold ladder itself.
 *
 * If the ladder is out of order -- transfer below accept, or accept above the
 * forger rate -- then no amount of correct statistics downstream means
 * anything, so this is surfaced before any result.
 */
export function ladderProblems(d: DetectionReport): string[] {
  const p = [...(d.thresholds.problems ?? [])];
  if (d.thresholds.ordered === false && p.length === 0) {
    p.push("The threshold ladder is not in ascending order.");
  }
  return p;
}

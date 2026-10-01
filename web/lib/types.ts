/**
 * The shape of what the engine emits.
 *
 * Every type here was written against real output from
 * `web/scripts/generate_snapshots.py`, not against the Python source. That
 * matters: `RateEstimate` serialises its interval as `ci_low`/`ci_high`, and
 * a contract written from the dataclass field names would have said
 * `low`/`high` and silently produced `undefined` everywhere. Read the JSON.
 *
 * Numbers are `number | null` wherever the engine's `_f()` helper can return
 * null, which it does for anything non-finite. A p-value of `null` is not
 * zero and not missing data -- for `sequential_onset` it means the test has
 * no p-value by construction, because a stopping rule rather than a fixed
 * sample controls its error rates.
 */

// ---------------------------------------------------------------------------
// detection
// ---------------------------------------------------------------------------

/** A rate with an exact Clopper-Pearson interval and its denominator. */
export interface RateEstimate {
  label: string;
  value: number | null;
  /** numerator: mismatches, clicks, dark counts -- depends on `label` */
  k: number;
  /** denominator; `conditioned_on` says which population this is */
  n: number;
  ci_low: number | null;
  ci_high: number | null;
  alpha: number;
  /** plain-language statement of the denominator, straight from the engine */
  conditioned_on: string;
}

/** One detector's result. `detail` is rule-specific and deliberately loose. */
export interface TestOutcome {
  name: RuleName;
  title: string;
  statistic: number | null;
  /** null for tests that have no p-value by construction (sequential_onset) */
  p_value: number | null;
  threshold: number | null;
  flagged: boolean;
  n: number;
  /**
   * False when the test could not run -- too few checks, or a declared
   * figure it needed was absent. An inapplicable test is not a passed test,
   * and the UI must not render it as one.
   */
  applicable: boolean;
  detail: Record<string, unknown>;
  interpretation: string;
}

/** One candidate explanation from the attribution table. */
export interface Attribution {
  label: string;
  /** the detector most responsible; empty string when the match was not rule-driven */
  matched_rule: string;
  rationale: string;
  /** table order; 0 is the most specific match */
  priority: number;
}

/** A recipient who refused the signature, and why. */
export interface Rejection {
  verifier: string;
  level: string;
  rate: number | null;
  threshold: number | null;
  n_checked: number;
  rate_exceeded: boolean;
  identity_ok: boolean;
  mac_ok: boolean;
  /** null when no freshness ledger was consulted */
  fresh: boolean | null;
  reasons: string[];
}

export interface ThresholdLadder {
  spec_rate: number | null;
  accept: number | null;
  transfer: number | null;
  forger_rate: number | null;
  /** false when the ladder is out of order, which invalidates the analysis */
  ordered: boolean | null;
  n_checks: number | null;
  problems: string[];
}

export interface SecurityBits {
  false_alarm: number | null;
  missed_forgery: number | null;
  missed_blind_forgery: number | null;
}

export interface DeclaredFigures {
  yield: number | null;
  dark_fraction: number | null;
  spec_rate_z: number | null;
  spec_rate_x: number | null;
  /** the engine's own caveat about where these figures came from */
  note: string;
}

export type Verdict = "clean" | "suspicious" | "compromised";

export interface DetectionReport {
  verdict: Verdict;
  alarm: boolean;
  /** Fisher's combination; reported but never decisive -- the p-values overlap */
  combined_p_value: number | null;
  alpha: number | null;
  alpha_per_test: number | null;
  n_tests: number;
  n_applicable: number;
  n_flagged: number;
  flagged: RuleName[];
  attribution: Attribution[];
  tests: TestOutcome[];
  estimates: Record<string, RateEstimate>;
  estimates_by_basis: Record<string, RateEstimate>;
  thresholds: ThresholdLadder;
  security_bits: SecurityBits;
  spec_violations: string[];
  protocol_rejects: boolean;
  authentication_failed: boolean;
  rejections: Rejection[];
  /** "spec_sheet" or "simulated_model"; see `DeclaredFigures.note` */
  declaration_source: "spec_sheet" | "simulated_model" | string;
  declared: DeclaredFigures;
  signer: string;
  key_id: string;
  elapsed_seconds: number | null;
}

/** The eleven detectors, in the engine's report order. */
export const RULE_NAMES = [
  "pooled_rate_vs_spec",
  "pooled_rate_vs_transfer",
  "basis_consistency",
  "position_homogeneity",
  "level_homogeneity",
  "yield_vs_declared",
  "dark_excess",
  "correction_uniformity",
  "decoy_vs_signature",
  "recipient_agreement",
  "sequential_onset",
] as const;

export type RuleName = (typeof RULE_NAMES)[number];

/**
 * Detectors whose firing means a premise of the threshold calculus is false,
 * rather than merely that the error rate is high. Mirrors `PREMISE_RULES` in
 * `qds/detect/engine.py`. When one of these fires, "the rate is still under
 * the threshold" stops being reassuring, which is why the engine returns
 * `compromised` even at a low rate.
 */
export const PREMISE_RULES: ReadonlySet<string> = new Set([
  "yield_vs_declared",
  "dark_excess",
  "correction_uniformity",
  "recipient_agreement",
  "decoy_vs_signature",
]);

// ---------------------------------------------------------------------------
// session
// ---------------------------------------------------------------------------

export interface SessionConfigDoc {
  signer: string;
  verifiers: string[];
  message_bits: number;
  L: number;
  signature_slots: number;
  checks_per_signature: number;
  symmetrise: boolean;
  noise_floor_spec: number;
  calibration_alpha: number;
  sampling_epsilon: number;
  threshold_rule: string;
  distribution: {
    check_pairs: number;
    decoy_slots: number;
    storage_intervals: number;
    exact: boolean;
    intervention: { name: string; description: string; [k: string]: unknown };
    predicted_qber: number;
  };
}

/**
 * The simulator's ground truth about one recipient's channel.
 *
 * Nothing in `qds.detect` may read this, and `Evidence.from_session` is
 * written so it cannot. It is surfaced in the dashboard only to let a reader
 * check the engine's conclusions against the truth the engine was denied --
 * which is exactly why the UI labels it as privileged rather than mixing it
 * in with measured quantities.
 */
export interface DistributionOracle {
  check_pairs: number;
  teleported_slots: number;
  mean_chsh: number;
  mean_bell_fidelity: number;
  mean_concurrence: number;
  exact_qber: number;
  exact_qber_by_basis: Record<string, number>;
  exact_qber_decoy: number;
  announcement_error_rate: number;
  information_disturbance: {
    eve_trace_distance: number;
    eve_guess_probability: number;
    disturbance_x_basis: number;
    predicted_trace_distance: number;
  };
}

export interface DistributionSummary {
  verifier: string;
  signer: string;
  key_id: string;
  slots_held: number;
  available: number;
  check_slots: number;
  decoy_slots: number;
  yield_observed: number;
  config: SessionConfigDoc["distribution"];
  oracle?: DistributionOracle;
}

export interface CalibrationDoc {
  verifier: string;
  decoys_measured: number;
  decoy_mismatches: number;
  point_estimate: number;
  lower_confidence: number;
  upper_confidence: number;
  spec_rate: number;
  p_value_vs_spec: number;
  /** true when the channel is provably worse than the sheet it was sold against */
  spec_violation: boolean;
  sampling_margin: number;
  certified_rate: number;
  /** certified_rate < 1/4: whether the decoy budget certifies anything pre-signing */
  informative: boolean;
  alpha: number;
  sampling_epsilon: number;
  predicted_rate: number | null;
}

export interface PolicyDoc {
  honest_error_rate: number;
  /** accept threshold */
  s_a: number;
  /** transfer threshold; must exceed s_a for the ladder to mean anything */
  s_v: number;
  forger_error_rate: number;
  blind_forger_error_rate: number;
  min_checks: number;
  window_seconds: number;
  sound: boolean;
  problems: string[];
}

export interface DeclarationDoc {
  signer: string;
  key_id: string;
  message_hex: string;
  message_bits: number;
  L: number;
  checks: number;
  counter: number;
  nonce: string;
  timestamp: number;
  recipients: string[];
  mac_tags: Record<string, { index: number; tag: string }>;
  declaration_bytes: number;
}

export interface VerificationDoc {
  verifier: string;
  signer: string;
  key_id: string;
  message_hex: string;
  level: string;
  threshold: number;
  accepted: boolean;
  reasons: string[];
  identity_ok: boolean;
  identity_problems: string[];
  mac_ok: boolean;
  mac_problems: string[];
  freshness: { ok: boolean; reasons: string[] };
  measured: boolean;
  slots_addressed: number;
  checks: number;
  mismatches: number;
  missing: number;
  dark: number;
  mismatch_rate: number;
  detection_yield: number;
  /** threshold - rate; negative means inside the threshold */
  margin: number;
  by_basis: Record<string, { mismatch: number; checks: number }>;
  by_position: Record<string, { mismatch: number; checks: number }>;
  p_value_vs_honest: number;
  p_value_vs_forger: number;
  clopper_pearson_95: [number, number];
  failure_bounds: {
    honest_abort: number;
    forgery_one_copy: number;
    forgery_blind: number;
    repudiation: number;
  };
  security_bits: {
    honest_abort: number;
    forgery_one_copy: number;
    forgery_blind: number;
    repudiation: number;
  };
  elapsed_seconds: number;
}

export interface SymmetrisationDoc {
  performed: boolean;
  reason?: string;
  recipients?: string[];
  slots_permuted: number;
  moved: number;
  fraction_moved?: number;
}

export interface SessionResultDoc {
  config: SessionConfigDoc;
  private_key: {
    signer: string;
    key_id: string;
    message_bits: number;
    L: number;
    signature_slots: number;
    private_key_bits: number;
  };
  distribution: Record<string, DistributionSummary>;
  symmetrisation: SymmetrisationDoc;
  calibration: Record<string, CalibrationDoc>;
  policy: PolicyDoc;
  declaration: DeclarationDoc | null;
  verifications: Record<string, VerificationDoc>;
  transfers: Record<string, VerificationDoc>;
  timings: Record<string, number>;
}

// ---------------------------------------------------------------------------
// snapshots
// ---------------------------------------------------------------------------

export type ScenarioFamily = "baseline" | "attack" | "hardware" | string;

export interface ChannelDoc {
  channel_depolarizing: number;
  memory_dephasing: number;
  gate_error: number;
  measurement_error: number;
  misalignment: number;
  transmittance: number;
  detector_efficiency: number;
  dark_count: number;
  yield: number;
  predicted_qber: number;
  /** false when the channel departs from the sheet even with no adversary */
  matches_declared_spec: boolean;
}

export interface Snapshot {
  id: string;
  label: string;
  family: ScenarioFamily;
  blurb: string;
  /** what the scenario was built to demonstrate; compare against `detection.verdict` */
  expectation: string;
  seed: number;
  intervention: {
    name: string;
    kwargs: Record<string, unknown>;
    description: string;
  };
  declared_spec: {
    preset: string;
    predicted_qber: number;
    yield: number;
    transmittance: number;
    detector_efficiency: number;
    dark_count: number;
  };
  channel: ChannelDoc;
  generated_at: number;
  wall_seconds: number;
  session: SessionResultDoc;
  detection: DetectionReport;
}

/** The catalogue entry, small enough to list every scenario without loading them. */
export interface ScenarioCard {
  id: string;
  label: string;
  family: ScenarioFamily;
  blurb: string;
  expectation: string;
  intervention: string;
  verdict: Verdict;
  alarm: boolean;
  pooled_rate: number | null;
  mismatches: number | null;
  checks: number | null;
  ci_low: number | null;
  ci_high: number | null;
  spec_rate: number | null;
  accept_threshold: number | null;
  transfer_threshold: number | null;
  n_flagged: number;
  n_applicable: number;
  n_tests: number;
  flagged: string[];
  primary_attribution: string | null;
  protocol_rejects: boolean;
  authentication_failed: boolean;
  combined_p_value: number | null;
  wall_seconds: number;
}

export interface SnapshotIndex {
  generated_at: number;
  geometry: {
    message_bits: number;
    L: number;
    signature_slots: number;
    checks_per_signature: number;
  };
  declared_spec: { preset: string; predicted_qber: number; yield: number };
  rules: string[];
  scenarios: ScenarioCard[];
}

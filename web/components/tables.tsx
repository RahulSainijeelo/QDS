/**
 * The tabular figures.
 *
 * These are tables because the data is categorical: a detector either fired or
 * did not, a recipient either accepted or did not. Rendering them as tables
 * rather than charts keeps them accessible, selectable, and faithful in print.
 *
 * The editorial rules from `select.ts` are honoured in the markup: an
 * inapplicable test is styled as neither pass nor fail, a premise rule that
 * fired is marked as such, and a detector that never fired across the suite
 * stays visible rather than being dropped so that coverage gaps are legible.
 */

import {
  type CalibrationRow,
  type ComparisonMatrix,
  type EstimateRow,
  type BasisRow,
  type OracleRow,
  type TestRow,
  type VerifierRow,
  humanRule,
  matchesExpectation,
  whyNotApplicable,
} from "../lib/select";
import type { Attribution, ScenarioCard } from "../lib/types";
import { Count, Interval, Num, PValue, Pct, Signed } from "./ui";

// ---------------------------------------------------------------------------
// detector table
// ---------------------------------------------------------------------------

const STATUS_WORD: Record<TestRow["status"], string> = {
  flagged: "flagged",
  passed: "pass",
  inapplicable: "n/a",
};

export function DetectorTable({ rows }: { rows: TestRow[] }) {
  return (
    <table className="tbl tbl--detectors">
      <thead>
        <tr>
          <th scope="col">Detector</th>
          <th scope="col" className="tbl__r">Statistic</th>
          <th scope="col" className="tbl__r">p</th>
          <th scope="col" className="tbl__r">Threshold</th>
          <th scope="col" className="tbl__r">n</th>
          <th scope="col">Status</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => {
          const o = r.outcome;
          const why = whyNotApplicable(o);
          return (
            <tr key={o.name} className={`is-${r.status}${r.premise ? " is-premise" : ""}`}>
              <th scope="row" className="tbl__name">
                {humanRule(o.name)}
                {r.premise ? <abbr className="premise-mark" title="Firing invalidates a premise of the threshold calculus">†</abbr> : null}
              </th>
              <td className="tbl__r"><Num v={o.statistic} /></td>
              <td className="tbl__r">{o.p_value === null ? <span className="num num--absent" title="no p-value by construction">—</span> : <PValue v={o.p_value} />}</td>
              <td className="tbl__r"><Num v={o.threshold} /></td>
              <td className="tbl__r"><Count v={o.n} /></td>
              <td className={`tbl__status status--${r.status}`}>
                {STATUS_WORD[r.status]}
                {r.status === "inapplicable" && why ? <span className="tbl__why">{why}</span> : null}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

/** The firing detectors' one-line readings, which is what an operator acts on. */
export function FiredReadings({ rows }: { rows: TestRow[] }) {
  const fired = rows.filter((r) => r.status === "flagged");
  if (fired.length === 0) return null;
  return (
    <ul className="readings-list">
      {fired.map((r) => (
        <li key={r.outcome.name} className={r.premise ? "is-premise" : undefined}>
          <span className="readings-list__rule">{humanRule(r.outcome.name)}</span>
          <span className="readings-list__text">{r.outcome.interpretation}</span>
        </li>
      ))}
    </ul>
  );
}

// ---------------------------------------------------------------------------
// attribution
// ---------------------------------------------------------------------------

export function AttributionList({ items }: { items: Attribution[] }) {
  if (items.length === 0) return <p className="note">No attribution offered; nothing fired.</p>;
  return (
    <ol className="attrib">
      {items.map((a, i) => (
        <li key={i} className={i === 0 ? "attrib__item attrib__item--primary" : "attrib__item"}>
          <div className="attrib__label">{a.label}</div>
          <p className="attrib__why">{a.rationale}</p>
          {a.matched_rule ? <span className="attrib__rule">via {humanRule(a.matched_rule)}</span> : null}
        </li>
      ))}
    </ol>
  );
}

// ---------------------------------------------------------------------------
// estimates
// ---------------------------------------------------------------------------

export function EstimateTable({ rows }: { rows: EstimateRow[] }) {
  return (
    <table className="tbl tbl--estimates">
      <thead>
        <tr>
          <th scope="col">Quantity</th>
          <th scope="col" className="tbl__r">Estimate</th>
          <th scope="col" className="tbl__r">95% interval</th>
          <th scope="col" className="tbl__r">k / n</th>
          <th scope="col" className="tbl__r">Declared</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.key}>
            <th scope="row" className="tbl__name">
              {r.estimate.label}
              <span className="tbl__cond">{r.estimate.conditioned_on}</span>
            </th>
            <td className="tbl__r"><Num v={r.estimate.value} /></td>
            <td className="tbl__r"><Interval lo={r.estimate.ci_low} hi={r.estimate.ci_high} /></td>
            <td className="tbl__r"><Count v={r.estimate.k} />&thinsp;/&thinsp;<Count v={r.estimate.n} /></td>
            <td className="tbl__r">
              <Num v={r.declared} />
              {r.declaredLabel ? <span className="tbl__cond">{r.declaredLabel}</span> : null}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function BasisTable({ rows }: { rows: BasisRow[] }) {
  return (
    <table className="tbl tbl--basis">
      <thead>
        <tr>
          <th scope="col">Basis</th>
          <th scope="col" className="tbl__r">Rate</th>
          <th scope="col" className="tbl__r">95% interval</th>
          <th scope="col" className="tbl__r">k / n</th>
          <th scope="col" className="tbl__r">Declared (this basis)</th>
          <th scope="col" className="tbl__r">Δ</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => {
          const delta =
            typeof r.estimate.value === "number" && typeof r.declared === "number"
              ? r.estimate.value - r.declared
              : null;
          return (
            <tr key={r.basis}>
              <th scope="row" className="tbl__name">{r.basis}</th>
              <td className="tbl__r"><Num v={r.estimate.value} /></td>
              <td className="tbl__r"><Interval lo={r.estimate.ci_low} hi={r.estimate.ci_high} /></td>
              <td className="tbl__r"><Count v={r.estimate.k} />&thinsp;/&thinsp;<Count v={r.estimate.n} /></td>
              <td className="tbl__r"><Num v={r.declared} /></td>
              <td className="tbl__r"><Signed v={delta} /></td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------------------
// verifications
// ---------------------------------------------------------------------------

export function VerifierTable({ rows }: { rows: VerifierRow[] }) {
  return (
    <table className="tbl tbl--verifiers">
      <thead>
        <tr>
          <th scope="col">Recipient</th>
          <th scope="col">Level</th>
          <th scope="col">Decision</th>
          <th scope="col" className="tbl__r">Error rate</th>
          <th scope="col" className="tbl__r">Threshold</th>
          <th scope="col" className="tbl__r">Margin</th>
          <th scope="col" className="tbl__r">Checks</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.key} className={r.doc.accepted ? "is-accept" : "is-reject"}>
            <th scope="row" className="tbl__name">{r.doc.verifier}</th>
            <td>{r.kind === "transfer" ? "transfer (forwarded)" : "accept (direct)"}</td>
            <td className={r.doc.accepted ? "status--passed" : "status--flagged"}>
              {r.doc.accepted ? "accepted" : "refused"}
            </td>
            <td className="tbl__r"><Num v={r.doc.mismatch_rate} /></td>
            <td className="tbl__r"><Num v={r.doc.threshold} /></td>
            <td className="tbl__r"><Signed v={r.doc.margin} /></td>
            <td className="tbl__r"><Count v={r.doc.checks} /></td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------------------
// calibration
// ---------------------------------------------------------------------------

export function CalibrationTable({ rows }: { rows: CalibrationRow[] }) {
  return (
    <table className="tbl tbl--calibration">
      <thead>
        <tr>
          <th scope="col">Recipient</th>
          <th scope="col" className="tbl__r">Decoy rate</th>
          <th scope="col" className="tbl__r">Certified ≤</th>
          <th scope="col" className="tbl__r">Decoys</th>
          <th scope="col" className="tbl__r">p vs spec</th>
          <th scope="col">Pre-sign certificate</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.verifier} className={r.violation ? "is-flagged" : undefined}>
            <th scope="row" className="tbl__name">{r.verifier}</th>
            <td className="tbl__r"><Num v={r.doc.point_estimate} /></td>
            <td className="tbl__r"><Num v={r.doc.certified_rate} /></td>
            <td className="tbl__r"><Count v={r.doc.decoys_measured} /></td>
            <td className="tbl__r"><PValue v={r.doc.p_value_vs_spec} /></td>
            <td>
              {r.violation ? (
                <span className="status--flagged">worse than the sheet</span>
              ) : r.informative ? (
                <span className="status--passed">certifies &lt; ¼ before signing</span>
              ) : (
                <span className="note-inline">not yet informative</span>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ---------------------------------------------------------------------------
// privileged oracle -- quarantined on purpose
// ---------------------------------------------------------------------------

export function OraclePanel({ rows }: { rows: OracleRow[] }) {
  if (rows.length === 0) return null;
  return (
    <aside className="sealed" aria-label="Privileged ground truth">
      <div className="sealed__banner">
        <span className="sealed__seal" aria-hidden="true">⊘</span>
        Privileged ground truth — the simulator knows this; the detection engine does not.
        Shown only so the engine&rsquo;s conclusions can be checked against the answer it was denied.
      </div>
      <table className="tbl tbl--oracle">
        <thead>
          <tr>
            <th scope="col">Recipient</th>
            <th scope="col" className="tbl__r">CHSH S</th>
            <th scope="col" className="tbl__r">Bell fidelity</th>
            <th scope="col" className="tbl__r">Exact QBER</th>
            <th scope="col" className="tbl__r">Eve trace dist.</th>
            <th scope="col" className="tbl__r">Eve guess p</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.verifier}>
              <th scope="row" className="tbl__name">{r.verifier}</th>
              <td className="tbl__r"><Num v={r.meanChsh} digits={3} /></td>
              <td className="tbl__r"><Num v={r.meanBellFidelity} digits={3} /></td>
              <td className="tbl__r"><Pct v={r.exactQber} /></td>
              <td className="tbl__r"><Num v={r.eveTraceDistance} digits={3} /></td>
              <td className="tbl__r"><Num v={r.eveGuessProbability} digits={3} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </aside>
  );
}

// ---------------------------------------------------------------------------
// rule x scenario matrix (home)
// ---------------------------------------------------------------------------

export function DetectorMatrix({ matrix }: { matrix: ComparisonMatrix }) {
  const premise = new Set(["yield_vs_declared", "dark_excess", "correction_uniformity", "recipient_agreement", "decoy_vs_signature"]);
  return (
    <div className="mx-wrap">
      <table className="tbl mx">
        <thead>
          <tr>
            <th scope="col" className="mx__corner">Detector</th>
            {matrix.scenarios.map((s) => (
              <th key={s.id} scope="col" className={`mx__colhead mx__colhead--${s.verdict === "compromised" ? "alarm" : s.verdict === "suspicious" ? "watch" : "ok"}`}>
                <span className="mx__colhead-text">{s.label}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.rules.map((rule, ri) => {
            const silent = matrix.silentRules.includes(rule);
            return (
              <tr key={rule} className={silent ? "mx__row--silent" : undefined}>
                <th scope="row" className="mx__rowhead">
                  {humanRule(rule)}
                  {premise.has(rule) ? <abbr className="premise-mark" title="Premise rule">†</abbr> : null}
                </th>
                {matrix.scenarios.map((s, ci) => {
                  const on = matrix.flagged[ri][ci];
                  return (
                    <td
                      key={s.id}
                      className={`mx__cell ${on ? (premise.has(rule) ? "mx__cell--premise" : "mx__cell--on") : "mx__cell--off"}`}
                      title={`${humanRule(rule)} × ${s.label}: ${on ? "flagged" : "not flagged"}`}
                    >
                      <span className="sr-only">{on ? "flagged" : "—"}</span>
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="note">
        A filled cell is a detector that fired for that scenario. Rows that are empty across the
        whole suite are kept visible: an empty row means no scenario here exercises what that
        detector looks for, which is a fact about the suite&rsquo;s coverage. Premise rules are
        marked <span className="premise-mark">†</span>.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// coverage gaps + scenario cards (home)
// ---------------------------------------------------------------------------

export function ScenarioRow({ card }: { card: ScenarioCard }) {
  const matched = matchesExpectation(card);
  const tone = card.verdict === "compromised" ? "alarm" : card.verdict === "suspicious" ? "watch" : "ok";
  return (
    <tr>
      <th scope="row" className="tbl__name">
        <a href={`/s/${card.id}`} className="scn-link">{card.label}</a>
        <span className="scn-blurb">{card.blurb}</span>
      </th>
      <td><span className={`verdict verdict--${tone}`}><span className="verdict__label">{card.verdict}</span></span></td>
      <td className="tbl__r"><Pct v={card.pooled_rate} /></td>
      <td className="tbl__r"><Count v={card.n_flagged} />&thinsp;/&thinsp;<Count v={card.n_applicable} /></td>
      <td className="tbl__r">
        {matched === null ? <span className="note-inline">—</span> : matched ? <span className="status--passed">as built</span> : <span className="status--watch">differs</span>}
      </td>
    </tr>
  );
}

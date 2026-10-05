/**
 * Home — the attack comparison.
 *
 * This is the view that answers "does the detector work?", and it is built to
 * be answerable honestly: every scenario is listed with the verdict it was
 * built to produce beside the one the engine returned, and the rule-by-scenario
 * matrix keeps empty detector rows visible so the suite's blind spots are as
 * legible as its catches. The two callouts at the foot name the attacks that
 * slipped through and any clean channel that tripped an alarm, because a
 * comparison that only showed successful detections would be a brochure.
 */

import { loadIndexFromDisk } from "../lib/data";
import { byFamily, comparisonMatrix, coverageGaps } from "../lib/select";
import { Figure } from "../components/ui";
import { DetectorMatrix, ScenarioRow } from "../components/tables";
import { count as fmtCount } from "../lib/format";

export default async function Home() {
  const index = await loadIndexFromDisk();
  const families = byFamily(index);
  const matrix = comparisonMatrix(index);
  const gaps = coverageGaps(index);
  const g = index.geometry;

  return (
    <article className="home">
      <header className="lede">
        <h1 className="lede__title">The detection suite, scenario by scenario</h1>
        <p className="lede__text">
          One signing session is run under each scenario below &mdash; a quiet laboratory link,
          a set of deliberate attacks on the quantum channel, and a set of honest hardware
          departures &mdash; and the same eleven-detector engine judges every one from the
          evidence a verifier would actually hold. Each run signs the same message over{" "}
          {fmtCount(g.signature_slots)} signature slots with {fmtCount(g.checks_per_signature)}{" "}
          checks. The engine is never shown the simulator&rsquo;s ground truth; it reaches its
          verdict from measured mismatch counts alone.
        </p>
      </header>

      {families.map((fam) => (
        <section key={fam.family} className="fam">
          <h2 className="fam__title">{fam.label}</h2>
          <table className="tbl tbl--catalogue">
            <thead>
              <tr>
                <th scope="col">Scenario</th>
                <th scope="col">Verdict</th>
                <th scope="col" className="tbl__r">Pooled rate</th>
                <th scope="col" className="tbl__r">Flagged</th>
                <th scope="col" className="tbl__r">Built to show</th>
              </tr>
            </thead>
            <tbody>
              {fam.scenarios.map((card) => (
                <ScenarioRow key={card.id} card={card} />
              ))}
            </tbody>
          </table>
        </section>
      ))}

      <Figure
        n={1}
        title="Which detector fired in which scenario"
        wide
        caption={
          <>
            Columns are scenarios, tinted by verdict; rows are the eleven detectors. A filled
            cell is a detector that fired. {matrix.silentRules.length > 0 ? (
              <>
                {" "}
                {fmtCount(matrix.silentRules.length)} detector
                {matrix.silentRules.length === 1 ? "" : "s"} never fired anywhere in this suite and
                are kept as empty rows: that is a statement about coverage, not a pass.
              </>
            ) : null}
          </>
        }
      >
        <DetectorMatrix matrix={matrix} />
      </Figure>

      {(gaps.missed.length > 0 || gaps.falseAlarms.length > 0) && (
        <section className="gaps">
          <h2 className="gaps__title">Where the suite is honest about its limits</h2>
          <div className="gaps__grid">
            <div className="gap gap--miss">
              <h3 className="gap__head">Attacks that read clean</h3>
              {gaps.missed.length === 0 ? (
                <p className="gap__none">Every adversarial scenario here raised at least a suspicion.</p>
              ) : (
                <>
                  <p className="gap__text">
                    These interventions evaded the Bonferroni-corrected suite &mdash; the engine
                    returned <em>clean</em>. They mark the sensitivity floor, not a clean bill.
                  </p>
                  <ul className="gap__list">
                    {gaps.missed.map((s) => (
                      <li key={s.id}>
                        <a href={`/s/${s.id}`}>{s.label}</a>
                        <span className="gap__why">{s.blurb}</span>
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </div>
            <div className="gap gap--false">
              <h3 className="gap__head">Clean channels that tripped an alarm</h3>
              {gaps.falseAlarms.length === 0 ? (
                <p className="gap__none">No baseline scenario produced a false alarm.</p>
              ) : (
                <ul className="gap__list">
                  {gaps.falseAlarms.map((s) => (
                    <li key={s.id}>
                      <a href={`/s/${s.id}`}>{s.label}</a>
                      <span className="gap__why">{s.blurb}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        </section>
      )}
    </article>
  );
}

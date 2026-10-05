/**
 * Detection report — why this verdict, detector by detector.
 *
 * The overview says what the engine concluded; this view shows the working.
 * The p-value figure places every detector against the Bonferroni-corrected
 * alpha on one log axis; the table lists all eleven with the one rule the UI
 * refuses to break -- a test that could not run is shown as inapplicable, never
 * as a pass. When a premise detector fired, the page explains why a rate below
 * the threshold stops being reassuring.
 */

import { notFound } from "next/navigation";

import { cardById, loadIndexFromDisk, loadSnapshotFromDisk } from "../../../../lib/data";
import {
  explainVerdict,
  flaggedPremiseRules,
  humanRule,
  rankedAttributions,
  tallyTests,
  testRows,
} from "../../../../lib/select";
import { Scene } from "../../../../components/nav";
import { Count, Figure, Note, Stats } from "../../../../components/ui";
import { PValueFigure } from "../../../../components/figures";
import { AttributionList, DetectorTable, FiredReadings } from "../../../../components/tables";
import { pValue as fmtPValue } from "../../../../lib/format";

export { generateStaticParams } from "../params";

export default async function DetectionPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const index = await loadIndexFromDisk();
  const card = cardById(index, id);
  if (!card) notFound();
  const snap = await loadSnapshotFromDisk(id);

  const d = snap.detection;
  const meaning = explainVerdict(d);
  const rows = testRows(d);
  const tally = tallyTests(d);
  const attributions = rankedAttributions(d);
  const premise = flaggedPremiseRules(d);

  return (
    <>
      <Scene snap={snap} card={card} cards={index.scenarios} tab="detection" />

      <section className={`verdict-line verdict-line--${meaning.tone}`}>
        <p className="verdict-line__headline">{meaning.headline}</p>
        <p className="verdict-line__because">{meaning.because}</p>
      </section>

      <section className="panel">
        <Stats
          items={[
            { k: "Fired", v: <Count v={tally.flagged} />, note: `of ${tally.total} detectors` },
            { k: "Passed", v: <Count v={tally.passed} /> },
            {
              k: "Could not run",
              v: <Count v={tally.inapplicable} />,
              note: "counted as neither pass nor fail",
            },
          ]}
        />
      </section>

      <Figure
        n={1}
        title="Detector p-values against the corrected threshold"
        wide
        caption={
          <>
            A log axis: each detector&rsquo;s p-value, with the Bonferroni-corrected
            α&prime;&nbsp;=&nbsp;{fmtPValue(d.alpha_per_test)} per test drawn as the vertical line.
            Points left of the line fired. A caret pinned at the axis is a p-value that underflowed
            to below the smallest representable value; a row marked &ldquo;no p-value&rdquo; is
            sequential-onset, which has none by construction.
          </>
        }
      >
        <PValueFigure report={d} />
      </Figure>

      <section className="panel">
        <h2 className="panel__head">Every detector</h2>
        <DetectorTable rows={rows} />
        <Note>
          Detectors marked <span className="premise-mark">†</span> are premise rules: when one
          fires it does not merely say the rate is high, it says a premise the thresholds were
          derived under is false. The statistic column is each test&rsquo;s own quantity, so the
          figures are not comparable down the column &mdash; read each against its own threshold.
        </Note>
      </section>

      {rows.some((r) => r.status === "flagged") ? (
        <section className="panel">
          <h2 className="panel__head">What fired, in words</h2>
          <FiredReadings rows={rows} />
        </section>
      ) : null}

      {premise.length > 0 ? (
        <div className="premisebox">
          <h2 className="premisebox__head">Why the verdict holds below the threshold</h2>
          <p className="premisebox__text">
            {premise.map(humanRule).join(", ")} fired. The acceptance threshold is derived assuming
            the loss budget is as declared, the transcript is uniform across slots, positions and
            recipients, and each recipient holds an equivalent key. With one of those premises
            false, a pooled rate under the threshold no longer certifies anything &mdash; the
            calculation it would be checked against does not apply. That is the route by which a
            quiet-looking channel is still called compromised.
          </p>
        </div>
      ) : null}

      <section className="panel">
        <h2 className="panel__head">Attribution</h2>
        <p className="panel__lead">
          Every candidate the engine matched, most specific first. The engine returns all of them,
          not just the top one, because the secondary matches are what let an operator tell a story
          about the primary.
        </p>
        <AttributionList items={attributions} />
      </section>
    </>
  );
}

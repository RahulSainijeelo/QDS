/**
 * Statistics deep-dive — the numbers under the verdict.
 *
 * The four headline estimates with their exact intervals, the same error rate
 * split by measurement basis against each basis's own declared floor, the
 * per-position homogeneity the engine tests for a leak, and the pre-signing
 * decoy calibration. The privileged oracle panel sits last and quarantined:
 * the simulator's ground truth, which the engine was forbidden to read, shown
 * only so a reader can check the verdict against the answer it was denied.
 */

import { notFound } from "next/navigation";

import { cardById, loadIndexFromDisk, loadSnapshotFromDisk } from "../../../../lib/data";
import {
  basisRows,
  calibrationRows,
  estimateRows,
  oracleRows,
  pooledPositionSeries,
} from "../../../../lib/select";
import { Scene } from "../../../../components/nav";
import { Figure, Note } from "../../../../components/ui";
import { PositionFigure } from "../../../../components/figures";
import {
  BasisTable,
  CalibrationTable,
  EstimateTable,
  OraclePanel,
} from "../../../../components/tables";

export { generateStaticParams } from "../params";

export default async function StatisticsPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const index = await loadIndexFromDisk();
  const card = cardById(index, id);
  if (!card) notFound();
  const snap = await loadSnapshotFromDisk(id);

  const estimates = estimateRows(snap);
  const bases = basisRows(snap);
  const positions = pooledPositionSeries(snap);
  const calibration = calibrationRows(snap);
  const oracle = oracleRows(snap);

  const spec = snap.detection.thresholds.spec_rate;
  const accept = snap.detection.thresholds.accept;

  return (
    <>
      <Scene snap={snap} card={card} cards={index.scenarios} tab="statistics" />

      <section className="panel">
        <h2 className="panel__head">Headline estimates</h2>
        <EstimateTable rows={estimates} />
        <Note>
          Every interval is an exact Clopper&ndash;Pearson 95% interval, not a normal
          approximation, so it stays valid at the low counts these denominators reach. Each
          estimate is paired with the figure declared in advance that it is judged against.
        </Note>
      </section>

      <section className="panel">
        <h2 className="panel__head">Error rate by measurement basis</h2>
        <BasisTable rows={bases} />
        <Note>
          The declared floor is one number, but the honest error rate is not equal in both bases:
          memory dephasing spares Z eigenstates while an X-basis check passes through more gates.
          The engine splits the declared floor in the proportion its hardware model predicts, so
          each basis is compared to its own floor rather than to a single pooled number.
        </Note>
      </section>

      <Figure
        n={1}
        title="Error rate by message-bit position"
        wide
        caption={
          <>
            Which slot carries which message bit is not revealed until after transmission, so
            honest noise has no way to prefer one position over another. A real gradient here is
            what <span className="mono">position_homogeneity</span> tests for. The spec and accept
            lines are the declared floor and the acceptance threshold; positions with no checks are
            omitted rather than drawn as zero.
          </>
        }
      >
        <PositionFigure points={positions} specRate={spec} acceptRate={accept} />
      </Figure>

      <section className="panel">
        <h2 className="panel__head">Pre-signing decoy calibration</h2>
        <CalibrationTable rows={calibration} />
        <Note>
          Decoy slots are measured before the message is signed, so they certify the channel
          independently of the signature checks. A certificate is only &ldquo;informative&rdquo;
          when it bounds the rate below one quarter &mdash; the one-copy forger&rsquo;s floor
          &mdash; before anything is committed.
        </Note>
      </section>

      {oracle.length > 0 ? (
        <section className="panel">
          <h2 className="panel__head">Ground truth, for audit only</h2>
          <p className="panel__lead">
            The simulator knows the following exactly. The detection engine does not, and is
            written so it cannot: <span className="mono">Evidence.from_session</span> never copies
            these fields, and the test suite greps the package to prove it. They are shown here
            only so the engine&rsquo;s conclusions can be checked against the truth it was denied.
          </p>
          <OraclePanel rows={oracle} />
        </section>
      ) : null}
    </>
  );
}

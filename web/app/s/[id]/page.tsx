/**
 * Session overview — the landing view for one scenario.
 *
 * It opens with the headline reading (the verdict and why), then the hero
 * figure: the measured error rate and its confidence interval against the
 * four-rung threshold ladder, which is the whole claim the protocol makes on
 * one axis. Below that sit the session's identity facts, the phase timings,
 * and the per-recipient verification outcomes. The deeper statistics and the
 * detector-by-detector reasoning live on the other two tabs.
 */

import { notFound } from "next/navigation";

import { cardById, loadIndexFromDisk, loadSnapshotFromDisk } from "../../../lib/data";
import {
  declarationIsIndependent,
  explainVerdict,
  ladderProblems,
  phaseRows,
  verifierRows,
} from "../../../lib/select";
import { Scene } from "../../../components/nav";
import { Bits, Dur, Figure, Note, SourceTag, Stats, VerdictReading } from "../../../components/ui";
import { PhaseFigure, RateLadder } from "../../../components/figures";
import { VerifierTable } from "../../../components/tables";
import { hexToAscii } from "../../../lib/format";

export { generateStaticParams } from "./params";

export default async function SessionPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const index = await loadIndexFromDisk();
  const card = cardById(index, id);
  if (!card) notFound();
  const snap = await loadSnapshotFromDisk(id);

  const meaning = explainVerdict(snap.detection);
  const problems = ladderProblems(snap.detection);
  const phases = phaseRows(snap);
  const verifiers = verifierRows(snap);
  const cfg = snap.session.config;
  const bits = snap.detection.security_bits;

  const messageHex = snap.session.declaration?.message_hex ?? null;
  const messageText = hexToAscii(messageHex);

  return (
    <>
      <Scene snap={snap} card={card} cards={index.scenarios} tab="overview" />

      <VerdictReading meaning={meaning} />

      {problems.length > 0 ? (
        <div className="alarmbox" role="alert">
          <h2 className="alarmbox__head">The threshold ladder itself is suspect</h2>
          <ul className="alarmbox__list">
            {problems.map((p, i) => (
              <li key={i}>{p}</li>
            ))}
          </ul>
          <p className="alarmbox__text">
            The ladder&rsquo;s rungs must rise spec &lt; accept &lt; transfer &lt; forger for any
            downstream statistic to mean anything. Read every figure below with that in mind.
          </p>
        </div>
      ) : null}

      <Figure
        n={1}
        title="Measured error rate against the threshold ladder"
        wide
        caption={
          <>
            The dot is the pooled error rate across every check; the bar is its exact
            Clopper&ndash;Pearson 95% interval. The rungs are fixed from the declared noise floor
            before the run, never fitted to it. A caret in place of a cap means the interval runs
            past the axis and is drawn clipped, not shrunk.
          </>
        }
      >
        <RateLadder report={snap.detection} />
      </Figure>

      <section className="panel">
        <h2 className="panel__head">The session</h2>
        <Stats
          items={[
            { k: "Signer", v: snap.detection.signer },
            { k: "Key", v: <span className="mono">{snap.detection.key_id}</span> },
            {
              k: "Message",
              v: messageText ? <span className="mono">&ldquo;{messageText}&rdquo;</span> : <span className="num num--absent">—</span>,
              note: messageHex ? <span className="mono">0x{messageHex}</span> : undefined,
            },
            { k: "Recipients", v: cfg.verifiers.join(", ") },
            {
              k: "Geometry",
              v: `${cfg.message_bits}-bit · L=${cfg.L}`,
              note: `${cfg.signature_slots.toLocaleString("en-US")} slots · ${cfg.checks_per_signature.toLocaleString("en-US")} checks`,
            },
            { k: "Wall time", v: <Dur v={snap.wall_seconds} /> },
            {
              k: "Declared figures",
              v: <SourceTag independent={declarationIsIndependent(snap.detection)} />,
              note: "the thresholds are only sound if this is independent of the channel",
            },
          ]}
        />
        <Note>{snap.detection.declared.note}</Note>
      </section>

      <section className="panel">
        <h2 className="panel__head">Who accepted the signature</h2>
        <VerifierTable rows={verifiers} />
        <Note>
          Direct recipients are judged against the acceptance threshold; a forwarded signature is
          judged against the looser transfer threshold, and that gap is the whole content of
          transferability. Margin is threshold minus rate &mdash; negative is inside the threshold.
        </Note>
      </section>

      <Figure
        n={2}
        title="Wall-clock time by protocol phase"
        caption="Phases run once and in order: a quantum public key is consumable, so enrol, distribute, symmetrise, calibrate and sign cannot be repeated or reordered for the same key."
      >
        <PhaseFigure phases={phases} />
      </Figure>

      <section className="panel panel--quiet">
        <h2 className="panel__head">Security margin, as reported</h2>
        <Stats
          items={[
            { k: "Against a one-copy forger", v: <Bits v={bits.missed_forgery} /> },
            { k: "Against a blind forger", v: <Bits v={bits.missed_blind_forgery} /> },
            { k: "False-alarm bound", v: <Bits v={bits.false_alarm} /> },
          ]}
        />
      </section>
    </>
  );
}

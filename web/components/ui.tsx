/**
 * UI primitives shared across the four views.
 *
 * Two rules live here so no page has to remember them:
 *
 *   - every figure (a number) is wrapped in <Num>/<Pct>/<PValue>/... which
 *     routes through `format.ts`, so a null renders as an em dash and nothing
 *     prints more precision than it carries;
 *   - a figure is always a captioned block (<Figure>), numbered like a
 *     whitepaper's, because that is what this dashboard is a view onto.
 *
 * These are server components. Nothing here holds state or touches the
 * browser, which is what lets the whole dashboard export statically.
 */

import type { ReactNode } from "react";

import {
  ABSENT,
  bits as fmtBits,
  count as fmtCount,
  duration as fmtDuration,
  interval as fmtInterval,
  pct as fmtPct,
  pValue as fmtPValue,
  rate as fmtRate,
  signed as fmtSigned,
  small as fmtSmall,
} from "../lib/format";
import type { Verdict } from "../lib/types";
import { type Tone, type VerdictMeaning } from "../lib/select";

// ---------------------------------------------------------------------------
// inline figures -- every number on the page goes through one of these
// ---------------------------------------------------------------------------

type N = number | null | undefined;

function fig(text: string): ReactNode {
  return <span className={`num${text === ABSENT ? " num--absent" : ""}`}>{text}</span>;
}

export const Num = ({ v, digits }: { v: N; digits?: number }) => fig(fmtRate(v, digits ?? 4));
export const Pct = ({ v, digits }: { v: N; digits?: number }) => fig(fmtPct(v, digits));
export const PValue = ({ v }: { v: N }) => fig(fmtPValue(v));
export const Count = ({ v }: { v: N }) => fig(fmtCount(v));
export const Bits = ({ v }: { v: N }) => fig(fmtBits(v));
export const Dur = ({ v }: { v: N }) => fig(fmtDuration(v));
export const Small = ({ v, digits }: { v: N; digits?: number }) => fig(fmtSmall(v, digits));
export const Signed = ({ v, digits }: { v: N; digits?: number }) => fig(fmtSigned(v, digits));
export const Interval = ({ lo, hi, digits }: { lo: N; hi: N; digits?: number }) =>
  fig(fmtInterval(lo, hi, digits));

/** A deliberate em dash, for prose where a value is absent by design. */
export const Dash = () => <span className="num num--absent">{ABSENT}</span>;

// ---------------------------------------------------------------------------
// figure frame
// ---------------------------------------------------------------------------

/**
 * A captioned, numbered figure. The caption is where provenance goes --
 * denominators, confidence levels, how many points were dropped -- because a
 * figure that hides its denominator is decoration.
 */
export function Figure({
  n,
  title,
  caption,
  wide,
  children,
}: {
  n: number;
  title: string;
  caption?: ReactNode;
  wide?: boolean;
  children: ReactNode;
}) {
  return (
    <figure className={`fig${wide ? " fig--wide" : ""}`}>
      <figcaption className="fig__head">
        <span className="fig__num">Fig.&nbsp;{n}</span>
        <span className="fig__title">{title}</span>
      </figcaption>
      <div className="fig__body">{children}</div>
      {caption ? <div className="fig__cap">{caption}</div> : null}
    </figure>
  );
}

// ---------------------------------------------------------------------------
// statistics grid
// ---------------------------------------------------------------------------

export interface StatItem {
  k: ReactNode;
  v: ReactNode;
  /** a muted gloss under the value, e.g. the denominator it was taken over */
  note?: ReactNode;
}

/** A definition list of labelled quantities. */
export function Stats({ items }: { items: StatItem[] }) {
  return (
    <dl className="stats">
      {items.map((it, i) => (
        <div className="stat" key={i}>
          <dt className="stat__k">{it.k}</dt>
          <dd className="stat__v">
            {it.v}
            {it.note ? <span className="stat__note">{it.note}</span> : null}
          </dd>
        </div>
      ))}
    </dl>
  );
}

// ---------------------------------------------------------------------------
// verdict
// ---------------------------------------------------------------------------

const TONE_GLYPH: Record<Tone, string> = {
  ok: "●",
  watch: "◐",
  alarm: "▲",
  muted: "○",
};

/** The verdict as a toned badge. */
export function VerdictBadge({ verdict, tone }: { verdict: Verdict; tone: Tone }) {
  return (
    <span className={`verdict verdict--${tone}`}>
      <span className="verdict__dot" aria-hidden="true">
        {TONE_GLYPH[tone]}
      </span>
      <span className="verdict__label">{verdict}</span>
    </span>
  );
}

/** The verdict with its one-line explanation beneath -- the headline reading. */
export function VerdictReading({ meaning }: { meaning: VerdictMeaning }) {
  return (
    <div className={`reading reading--${meaning.tone}`}>
      <div className="reading__top">
        <VerdictBadge verdict={meaning.verdict} tone={meaning.tone} />
        <span className="reading__headline">{meaning.headline}</span>
      </div>
      <p className="reading__because">{meaning.because}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// notes & provenance
// ---------------------------------------------------------------------------

/** A quiet, small note -- captions, "n of m dropped", provenance. */
export function Note({ children }: { children: ReactNode }) {
  return <p className="note">{children}</p>;
}

/**
 * Where a declared figure came from. The whole threshold calculus depends on
 * this being independent of the channel under test, so it is never buried.
 */
export function SourceTag({ independent }: { independent: boolean }) {
  return (
    <span className={`source ${independent ? "source--spec" : "source--model"}`}>
      {independent ? "declared in advance" : "read from the channel"}
    </span>
  );
}

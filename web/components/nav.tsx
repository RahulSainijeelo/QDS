/**
 * Navigation chrome: the top bar, and the per-scenario header with its tabs.
 *
 * The scenario switcher is a native <details> element, not a scripted
 * dropdown, so it works with no client JavaScript and survives a static
 * export intact. Every tab and switch target is a plain link for the same
 * reason -- this dashboard is a set of documents, and documents have URLs.
 */

import type { ReactNode } from "react";
import Link from "next/link";

import type { ScenarioCard, Snapshot } from "../lib/types";
import { explainVerdict, matchesExpectation } from "../lib/select";
import { VerdictBadge } from "./ui";

export type SceneTab = "overview" | "detection" | "statistics";

const TABS: Array<{ id: SceneTab; label: string; href: (id: string) => string }> = [
  { id: "overview", label: "Session", href: (id) => `/s/${id}` },
  { id: "detection", label: "Detection", href: (id) => `/s/${id}/detection` },
  { id: "statistics", label: "Statistics", href: (id) => `/s/${id}/statistics` },
];

// ---------------------------------------------------------------------------
// top bar
// ---------------------------------------------------------------------------

export function Chrome({ children }: { children: ReactNode }) {
  return (
    <>
      <header className="chrome">
        <Link href="/" className="chrome__brand">
          <span className="chrome__mark" aria-hidden="true">
            QDS
          </span>
          <span className="chrome__name">verifier&rsquo;s instrument</span>
        </Link>
        <span className="chrome__sub">quantum digital signatures — detection analytics</span>
      </header>
      <main className="page">{children}</main>
      <footer className="foot">
        <p>
          Every figure is produced by the detection engine from a real protocol run.
          Numbers are not illustrative; where a measurement is absent it is drawn as an
          em dash, never as zero.
        </p>
      </footer>
    </>
  );
}

// ---------------------------------------------------------------------------
// per-scenario header
// ---------------------------------------------------------------------------

function Switcher({ current, cards }: { current: string; cards: ScenarioCard[] }) {
  const here = cards.find((c) => c.id === current);
  return (
    <details className="switch">
      <summary className="switch__current">
        <span className="switch__label">{here?.label ?? current}</span>
        <span className="switch__caret" aria-hidden="true">
          ▾
        </span>
      </summary>
      <nav className="switch__menu">
        {cards.map((c) => (
          <Link
            key={c.id}
            href={`/s/${c.id}`}
            className={`switch__item${c.id === current ? " is-current" : ""}`}
          >
            <span className="switch__item-label">{c.label}</span>
            <span className={`switch__item-verdict verdict--${toneOf(c)}`}>{c.verdict}</span>
          </Link>
        ))}
      </nav>
    </details>
  );
}

function toneOf(c: ScenarioCard): string {
  return c.verdict === "compromised" ? "alarm" : c.verdict === "suspicious" ? "watch" : "ok";
}

/**
 * The scenario header: title, verdict, blurb, the tab row, and -- when the
 * engine's verdict did not match what the scenario was built to show -- an
 * honest banner saying so, rather than quietly agreeing with itself.
 */
export function Scene({
  snap,
  card,
  cards,
  tab,
}: {
  snap: Snapshot;
  card: ScenarioCard;
  cards: ScenarioCard[];
  tab: SceneTab;
}) {
  const meaning = explainVerdict(snap.detection);
  const matched = matchesExpectation(card);

  return (
    <section className="scene">
      <div className="scene__bar">
        <Switcher current={snap.id} cards={cards} />
        <span className={`scene__family scene__family--${snap.family}`}>{snap.family}</span>
      </div>

      <div className="scene__head">
        <h1 className="scene__title">{snap.label}</h1>
        <VerdictBadge verdict={meaning.verdict} tone={meaning.tone} />
      </div>
      <p className="scene__blurb">{snap.blurb}</p>

      {matched === false ? (
        <p className="scene__mismatch">
          Built to read <em>{snap.expectation}</em>; the engine returned{" "}
          <em>{snap.detection.verdict}</em>. Shown as found.
        </p>
      ) : null}

      <nav className="scenav">
        {TABS.map((t) => (
          <Link
            key={t.id}
            href={t.href(snap.id)}
            className={`scenav__tab${t.id === tab ? " is-active" : ""}`}
            aria-current={t.id === tab ? "page" : undefined}
          >
            {t.label}
          </Link>
        ))}
      </nav>
    </section>
  );
}

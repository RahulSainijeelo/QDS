/**
 * The continuous figures, drawn as hand-rolled SVG over `chart.ts`.
 *
 * Why SVG and not a chart library: every figure here is one of a few shapes,
 * all of them arithmetic, and a library would add a dependency and an opinion
 * about styling in exchange for geometry that fits on a screen. Why these as
 * SVG and the rule matrix as a table: continuous or spatial data is a drawing;
 * a boolean rule-by-scenario grid is a table, and forcing it into SVG would
 * cost accessibility and print fidelity for nothing.
 *
 * Colour lives in CSS classes, never inline, so the same figures re-theme for
 * screen and print and the palette stays in one file.
 *
 * The hero is <RateLadder>: a measured error rate, its confidence interval,
 * and the four-rung threshold ladder it is judged against, on one axis. That
 * single picture is the whole claim the protocol makes, so it is the one place
 * the design spends any boldness.
 */

import {
  frame,
  horizontalBars,
  horizontalErrorBar,
  linearScale,
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
} from "../lib/chart";
import { humanRule, type PhaseRow, type PositionPoint } from "../lib/select";
import type { DetectionReport, ThresholdLadder } from "../lib/types";
import { pct as fmtPct, pValue as fmtPValue } from "../lib/format";

// ---------------------------------------------------------------------------
// the hero: measured rate vs the threshold ladder
// ---------------------------------------------------------------------------

export function RateLadder({ report }: { report: DetectionReport }) {
  const t: ThresholdLadder = report.thresholds;
  const pooled = report.estimates.pooled_rate;
  const W = 720;
  const H = 158;
  const f = frame(W, H, { top: 48, right: 20, bottom: 34, left: 20 });
  const top = f.plot.y;
  const h = f.plot.height;
  const base = f.plot.y + f.plot.height;

  const dom = rateDomain([
    0,
    t.spec_rate,
    t.accept,
    t.transfer,
    t.forger_rate,
    pooled?.value,
    pooled?.ci_low,
    pooled?.ci_high,
  ]);
  const sx = linearScale(dom, [f.plot.x, f.plot.x + f.plot.width]);

  const clamp = (v: number) => Math.max(f.plot.x, Math.min(f.plot.x + f.plot.width, v));
  const zoneDefs: Array<[string, number | null, number | null, string]> = [
    ["within accept", 0, t.accept, "ok"],
    ["transfer only", t.accept, t.transfer, "watch"],
    ["refused", t.transfer, t.forger_rate, "alarm"],
    ["forgeable", t.forger_rate, dom[1], "alarm2"],
  ];
  const zones = zoneDefs
    .filter(([, a, b]) => typeof a === "number" && typeof b === "number" && (b as number) > (a as number))
    .map(([label, a, b, tone]) => {
      const x0 = clamp(sx(a as number));
      const x1 = clamp(sx(b as number));
      return { label, tone, x0, w: Math.max(0, x1 - x0), mid: (x0 + x1) / 2 };
    });

  const markers = thresholdMarkers(t, sx);
  const estY = top + h * 0.56;
  const value = typeof pooled?.value === "number" ? pooled.value : null;
  const eb =
    value !== null
      ? horizontalErrorBar(value, pooled?.ci_low, pooled?.ci_high, sx, estY, 5)
      : null;

  const axisTicks = ticks(dom, 6);

  const ordered = t.ordered !== false;

  return (
    <svg
      className="rl"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      preserveAspectRatio="xMidYMid meet"
      aria-label={
        value !== null
          ? `Measured error rate ${fmtPct(value)} against accept ${fmtPct(t.accept)} and transfer ${fmtPct(t.transfer)} thresholds`
          : "Threshold ladder"
      }
    >
      {/* zones */}
      {zones.map((z) => (
        <g key={z.label}>
          <rect className={`rl-band rl-band--${z.tone}`} x={z.x0} y={top} width={z.w} height={h} />
          {z.w > 66 ? (
            <text className="rl-zone" x={z.mid} y={top + 14} textAnchor="middle">
              {z.label}
            </text>
          ) : null}
        </g>
      ))}

      {/* baseline */}
      <line className="rl-axis" x1={f.plot.x} y1={base} x2={f.plot.x + f.plot.width} y2={base} />

      {/* threshold markers with staggered labels */}
      {markers.map((m) => {
        const x = clamp(m.at);
        const labelY = top - 8 - (m.rank % 2) * 18;
        return (
          <g key={m.key} className={`rl-mark rl-mark--${m.tone}`}>
            <line className="rl-stem" x1={x} y1={labelY + 2} x2={x} y2={base} />
            <text className="rl-mlabel" x={x} y={labelY} textAnchor="middle">
              {m.label}
            </text>
            <text className="rl-mval" x={x} y={base + 22} textAnchor="middle">
              {fmtPct(m.value, m.value < 0.1 ? 1 : 0)}
            </text>
          </g>
        );
      })}

      {/* axis ticks */}
      {axisTicks.map((v, i) => (
        <g key={i} className="rl-tick">
          <line x1={sx(v)} y1={base} x2={sx(v)} y2={base + 4} />
        </g>
      ))}

      {/* the measurement */}
      {eb ? (
        <g className="rl-est">
          <line className="rl-ci" x1={eb.lo} y1={estY} x2={eb.hi} y2={estY} />
          {!eb.clippedLo ? <line className="rl-cap" x1={eb.lo} y1={eb.capLo} x2={eb.lo} y2={eb.capHi} /> : null}
          {eb.clippedHi ? (
            <path className="rl-caret" d={`M${eb.hi} ${estY} l-7 -4 l0 8 z`} />
          ) : (
            <line className="rl-cap" x1={eb.hi} y1={eb.capLo} x2={eb.hi} y2={eb.capHi} />
          )}
          <circle className="rl-dot" cx={clamp(sx(value as number))} cy={estY} r={4.5} />
        </g>
      ) : null}

      {!ordered ? (
        <text className="rl-warn" x={f.plot.x} y={base + 22}>
          ladder out of order — thresholds unreliable
        </text>
      ) : null}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// per-position homogeneity
// ---------------------------------------------------------------------------

export function PositionFigure({
  points,
  specRate,
  acceptRate,
}: {
  points: PositionPoint[];
  specRate: number | null;
  acceptRate: number | null;
}) {
  const W = 720;
  const H = 180;
  const f = frame(W, H, { top: 16, right: 18, bottom: 30, left: 46 });
  const se = series(points.map((p) => ({ x: p.position, y: p.rate })));

  const xs = points.map((p) => p.position);
  const xDom: [number, number] = xs.length ? [Math.min(...xs), Math.max(...xs)] : [0, 1];
  const yMax = Math.max(
    acceptRate ?? 0,
    specRate ?? 0,
    ...se.points.map((p) => p.y),
    0.01,
  );
  const sx = linearScale(xDom, [f.plot.x, f.plot.x + f.plot.width]);
  const sy = linearScale([0, yMax * 1.12], [f.plot.y + f.plot.height, f.plot.y]);

  const yTicks = ticks([0, yMax * 1.12], 4);
  const td = tickDigits(yTicks.map((v) => v * 100));

  return (
    <svg className="pf" viewBox={`0 0 ${W} ${H}`} role="img" preserveAspectRatio="xMidYMid meet"
      aria-label="Per-position error rate across message-bit positions">
      {/* y grid + labels */}
      {yTicks.map((v, i) => (
        <g key={i} className="pf-grid">
          <line x1={f.plot.x} y1={sy(v)} x2={f.plot.x + f.plot.width} y2={sy(v)} />
          <text className="pf-ylabel" x={f.plot.x - 6} y={sy(v)} textAnchor="end" dominantBaseline="middle">
            {fmtPct(v, td)}
          </text>
        </g>
      ))}

      {/* reference lines */}
      {typeof specRate === "number" ? (
        <line className="pf-ref pf-ref--spec" x1={f.plot.x} y1={sy(specRate)} x2={f.plot.x + f.plot.width} y2={sy(specRate)} />
      ) : null}
      {typeof acceptRate === "number" ? (
        <line className="pf-ref pf-ref--accept" x1={f.plot.x} y1={sy(acceptRate)} x2={f.plot.x + f.plot.width} y2={sy(acceptRate)} />
      ) : null}

      {/* the series */}
      <path className="pf-series" d={stepPath(se.points, sx, sy)} fill="none" />
      {se.points.map((p, i) => (
        <circle key={i} className="pf-pt" cx={sx(p.x)} cy={sy(p.y)} r={2.6} />
      ))}

      {/* x axis */}
      <line className="pf-axis" x1={f.plot.x} y1={f.plot.y + f.plot.height} x2={f.plot.x + f.plot.width} y2={f.plot.y + f.plot.height} />
      <text className="pf-xlabel" x={f.plot.x} y={H - 8}>pos {xDom[0]}</text>
      <text className="pf-xlabel" x={f.plot.x + f.plot.width} y={H - 8} textAnchor="end">pos {xDom[1]}</text>
    </svg>
  );
}

// ---------------------------------------------------------------------------
// detector p-values against the corrected alpha
// ---------------------------------------------------------------------------

export function PValueFigure({ report }: { report: DetectionReport }) {
  const tests = report.tests;
  const alpha = report.alpha_per_test;
  const W = 720;
  const rowH = 20;
  const f = frame(W, tests.length * rowH + 40, { top: 22, right: 20, bottom: 18, left: 168 });
  const dom = pValueDomain(tests.map((t) => t.p_value), alpha);
  const sx = logScale(dom, [f.plot.x, f.plot.x + f.plot.width]);
  const laid = pValueLayout(
    tests.map((t) => ({ name: t.name, title: t.title, p_value: t.p_value, flagged: t.flagged, applicable: t.applicable })),
    sx,
  );
  const decades = logTicks(dom, 8);
  const alphaX = typeof alpha === "number" && alpha > 0 ? sx(alpha) : null;

  return (
    <svg className="pv" viewBox={`0 0 ${W} ${f.height}`} role="img" preserveAspectRatio="xMidYMid meet"
      aria-label="Detector p-values on a log axis against the Bonferroni-corrected alpha">
      {/* decade gridlines */}
      {decades.map((d, i) => (
        <g key={i} className="pv-grid">
          <line x1={sx(d)} y1={f.plot.y} x2={sx(d)} y2={f.plot.y + f.plot.height} />
          <text className="pv-xlabel" x={sx(d)} y={f.plot.y - 8} textAnchor="middle">
            {fmtPValue(d)}
          </text>
        </g>
      ))}

      {/* alpha line */}
      {alphaX !== null ? (
        <g className="pv-alpha">
          <line x1={alphaX} y1={f.plot.y - 2} x2={alphaX} y2={f.plot.y + f.plot.height} />
          <text className="pv-alpha-label" x={alphaX} y={f.plot.y + f.plot.height + 12} textAnchor="middle">
            α′ = {fmtPValue(alpha)}
          </text>
        </g>
      ) : null}

      {/* one row per detector */}
      {laid.map((p, i) => {
        const y = f.plot.y + i * rowH + rowH / 2;
        const status = !p.applicable ? "na" : p.flagged ? "flag" : "pass";
        return (
          <g key={p.name} className={`pv-row pv-row--${status}`}>
            <text className="pv-name" x={6} y={y} dominantBaseline="middle">
              {humanRule(p.name)}
            </text>
            {p.at !== null ? (
              <>
                {p.underflow ? (
                  <path className="pv-caret" d={`M${f.plot.x} ${y} l8 -4 l0 8 z`} />
                ) : (
                  <circle className="pv-dot" cx={p.at} cy={y} r={3.4} />
                )}
              </>
            ) : (
              <text className="pv-none" x={f.plot.x + f.plot.width} y={y} textAnchor="end" dominantBaseline="middle">
                no p-value (sequential)
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// phase timings
// ---------------------------------------------------------------------------

export function PhaseFigure({ phases }: { phases: PhaseRow[] }) {
  const W = 720;
  const labelW = 150;
  const barH = 15;
  const gap = 7;
  const plotW = W - labelW - 60;
  const { bars, height, max } = horizontalBars(
    phases.map((p) => ({ key: p.key, label: p.label, value: p.seconds })),
    plotW,
    barH,
    gap,
  );
  const H = Math.max(1, height) + 8;

  return (
    <svg className="ph" viewBox={`0 0 ${W} ${H}`} role="img" preserveAspectRatio="xMidYMid meet"
      aria-label="Wall-clock time spent in each protocol phase">
      {bars.map((b) => {
        const y = b.y + 4;
        return (
          <g key={b.key} className="ph-row">
            <text className="ph-name" x={labelW - 8} y={y + barH / 2} textAnchor="end" dominantBaseline="middle">
              {b.label}
            </text>
            <rect className="ph-track" x={labelW} y={y} width={plotW} height={barH} />
            <rect className="ph-bar" x={labelW} y={y} width={b.width} height={barH} />
            <text className="ph-val" x={labelW + Math.min(b.width + 6, plotW)} y={y + barH / 2} dominantBaseline="middle">
              {b.value < 1 ? `${(b.value * 1000).toFixed(0)}ms` : `${b.value.toFixed(2)}s`}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

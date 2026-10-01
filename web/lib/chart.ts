/**
 * SVG geometry, hand-rolled.
 *
 * There is no charting library here on purpose. Every figure in this
 * dashboard is one of four shapes -- a dot with an error bar, a step series
 * over slot positions, a horizontal threshold marker, a grid cell -- and all
 * four are a handful of arithmetic. A chart library would add a build
 * dependency, a runtime, and an opinion about axis styling, in exchange for
 * geometry that fits in one file.
 *
 * Two rules the callers rely on:
 *
 * 1. A scale never silently accepts a degenerate domain. If min == max the
 *    domain is widened symmetrically, because collapsing it would put every
 *    point at the same pixel and the figure would look like agreement when
 *    it is actually a single measurement.
 * 2. Nothing here throws on absent data. `null` values are dropped from
 *    series and reported via `.dropped`, so a caller can say "3 positions had
 *    no checks" rather than rendering a line through them as if they were
 *    zero.
 *
 * Pure functions, no React, no DOM -- exercised under plain `node` in
 * `web/lib/lib.test.ts`.
 */

// ---------------------------------------------------------------------------
// scales
// ---------------------------------------------------------------------------

export interface Scale {
  /** map a domain value to a pixel coordinate */
  (v: number): number;
  domain: readonly [number, number];
  range: readonly [number, number];
  /** inverse, for reading a pixel back as a value */
  invert(px: number): number;
  kind: "linear" | "log";
}

const EPS = 1e-12;

/**
 * Widen a collapsed domain.
 *
 * A single data point, or several identical ones, gives min == max. Mapping
 * that to a range is a division by zero; mapping it to the midpoint instead
 * draws a flat line that reads as a measured result. Widening by a
 * proportional pad keeps the point where it belongs -- centred, with visible
 * space either side that honestly reflects there is nothing to compare it to.
 */
function widen(lo: number, hi: number): [number, number] {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0, 1];
  if (hi - lo > EPS) return [lo, hi];
  const pad = Math.max(Math.abs(lo), 1e-6) * 0.1;
  return [lo - pad, hi + pad];
}

export function linearScale(
  domain: readonly [number, number],
  range: readonly [number, number],
): Scale {
  const [d0, d1] = widen(domain[0], domain[1]);
  const [r0, r1] = range;
  const m = (r1 - r0) / (d1 - d0);
  const f = ((v: number) => r0 + (v - d0) * m) as Scale;
  f.domain = [d0, d1];
  f.range = [r0, r1];
  f.kind = "linear";
  f.invert = (px: number) => d0 + (px - r0) / (m || EPS);
  return f;
}

/**
 * A log scale, for p-values.
 *
 * p-values in this engine span from ~0.9 down to floating-point underflow, so
 * a linear axis would compress every interesting value into one pixel at the
 * bottom. The floor clamps non-positive inputs: a p-value reported as exactly
 * 0 underflowed, it is not a certainty, and `floor` is where the axis admits
 * it cannot resolve further.
 */
export function logScale(
  domain: readonly [number, number],
  range: readonly [number, number],
  floor = 1e-300,
): Scale {
  const lo = Math.max(Math.min(domain[0], domain[1]), floor);
  const hi = Math.max(Math.max(domain[0], domain[1]), lo * 10);
  const l0 = Math.log10(lo);
  const l1 = Math.log10(hi);
  const [r0, r1] = range;
  const m = (r1 - r0) / (l1 - l0 || EPS);
  const f = ((v: number) => {
    const clamped = Math.max(Number.isFinite(v) ? v : floor, floor);
    return r0 + (Math.log10(clamped) - l0) * m;
  }) as Scale;
  f.domain = [lo, hi];
  f.range = [r0, r1];
  f.kind = "log";
  f.invert = (px: number) => 10 ** (l0 + (px - r0) / (m || EPS));
  return f;
}

// ---------------------------------------------------------------------------
// ticks
// ---------------------------------------------------------------------------

/**
 * Round tick values covering a domain, at 1/2/5 x 10^k steps.
 *
 * The step is chosen from that set rather than from `(hi-lo)/count` directly
 * so labels read as 0.05, 0.10, 0.15 instead of 0.047, 0.094. Ticks are
 * generated across the padded domain and then filtered back into it, which
 * means a requested count is an upper bound, not a promise.
 */
export function ticks(domain: readonly [number, number], count = 5): number[] {
  const [lo, hi] = widen(domain[0], domain[1]);
  const span = hi - lo;
  if (!(span > 0) || count < 1) return [lo];

  const rough = span / count;
  const mag = 10 ** Math.floor(Math.log10(rough));
  const norm = rough / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;

  const out: number[] = [];
  const start = Math.ceil(lo / step) * step;
  // Guard the loop on a count as well as on the bound: a step that underflows
  // relative to the domain would otherwise spin.
  for (let i = 0, v = start; v <= hi + step * 1e-9 && i < 500; i++, v = start + i * step) {
    // Re-round to kill the accumulated binary error that makes 0.30000000000000004.
    out.push(Number(v.toFixed(12)));
  }
  return out;
}

/** Decade ticks for a log axis, thinned so labels never collide. */
export function logTicks(domain: readonly [number, number], maxTicks = 8): number[] {
  const lo = Math.max(Math.min(...domain), 1e-300);
  const hi = Math.max(...domain);
  const e0 = Math.floor(Math.log10(lo));
  const e1 = Math.ceil(Math.log10(hi));
  const decades: number[] = [];
  for (let e = e0; e <= e1; e++) decades.push(e);
  const stride = Math.max(1, Math.ceil(decades.length / maxTicks));
  return decades.filter((_, i) => i % stride === 0).map((e) => 10 ** e);
}

/** How many decimals a tick set needs before two labels read the same. */
export function tickDigits(values: number[]): number {
  for (let d = 0; d <= 8; d++) {
    const seen = new Set(values.map((v) => v.toFixed(d)));
    if (seen.size === values.length) return d;
  }
  return 8;
}

// ---------------------------------------------------------------------------
// plot frames
// ---------------------------------------------------------------------------

export interface Insets {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export interface Frame {
  width: number;
  height: number;
  insets: Insets;
  /** plot area, excluding the axis gutters */
  plot: { x: number; y: number; width: number; height: number };
}

export function frame(
  width: number,
  height: number,
  insets: Partial<Insets> = {},
): Frame {
  const i: Insets = {
    top: insets.top ?? 12,
    right: insets.right ?? 12,
    bottom: insets.bottom ?? 28,
    left: insets.left ?? 52,
  };
  return {
    width,
    height,
    insets: i,
    plot: {
      x: i.left,
      y: i.top,
      width: Math.max(1, width - i.left - i.right),
      height: Math.max(1, height - i.top - i.bottom),
    },
  };
}

// ---------------------------------------------------------------------------
// series
// ---------------------------------------------------------------------------

export interface Pt {
  x: number;
  y: number;
}

export interface Series {
  points: Pt[];
  /** count of input points discarded for having no value */
  dropped: number;
}

/**
 * Turn `(x, y|null)` pairs into plottable points, counting what was dropped.
 *
 * The dropped count is returned rather than logged because it belongs in the
 * figure's caption. A per-position rate series with 4 of 16 positions empty is
 * a different claim from one with all 16 populated, and the reader can only
 * tell if the figure says so.
 */
export function series(
  input: Array<{ x: number; y: number | null | undefined }>,
): Series {
  const points: Pt[] = [];
  let dropped = 0;
  for (const p of input) {
    if (typeof p.y === "number" && Number.isFinite(p.y) && Number.isFinite(p.x)) {
      points.push({ x: p.x, y: p.y });
    } else {
      dropped++;
    }
  }
  points.sort((a, b) => a.x - b.x);
  return { points, dropped };
}

const fx = (n: number) => (Number.isFinite(n) ? Number(n.toFixed(2)) : 0);

/** A polyline path. Empty string for an empty series, which renders nothing. */
export function linePath(points: Pt[], sx: Scale, sy: Scale): string {
  if (points.length === 0) return "";
  return points
    .map((p, i) => `${i === 0 ? "M" : "L"}${fx(sx(p.x))} ${fx(sy(p.y))}`)
    .join(" ");
}

/**
 * A step path, which is the honest rendering for per-slot data.
 *
 * Position 3's error rate is a property of position 3, not of the interval
 * between 3 and 4. A smooth line between them implies an intermediate value
 * that does not exist, so the path holds each level flat across its own band
 * and jumps at the boundary.
 */
export function stepPath(points: Pt[], sx: Scale, sy: Scale): string {
  if (points.length === 0) return "";
  const parts: string[] = [];
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    const x = fx(sx(p.x));
    const y = fx(sy(p.y));
    if (i === 0) {
      parts.push(`M${x} ${y}`);
    } else {
      parts.push(`L${x} ${fx(sy(points[i - 1].y))}`, `L${x} ${y}`);
    }
  }
  return parts.join(" ");
}

/** A filled band between two y series sharing an x axis, for CI envelopes. */
export function bandPath(
  points: Array<{ x: number; lo: number; hi: number }>,
  sx: Scale,
  sy: Scale,
): string {
  const ok = points.filter(
    (p) => Number.isFinite(p.x) && Number.isFinite(p.lo) && Number.isFinite(p.hi),
  );
  if (ok.length === 0) return "";
  const up = ok.map((p, i) => `${i === 0 ? "M" : "L"}${fx(sx(p.x))} ${fx(sy(p.hi))}`);
  const down = [...ok]
    .reverse()
    .map((p) => `L${fx(sx(p.x))} ${fx(sy(p.lo))}`);
  return [...up, ...down, "Z"].join(" ");
}

// ---------------------------------------------------------------------------
// error bars
// ---------------------------------------------------------------------------

export interface ErrorBar {
  /** centre of the estimate */
  cx: number;
  cy: number;
  /** interval extent along the value axis, in pixels */
  lo: number;
  hi: number;
  /** cap ends, in pixels along the category axis */
  capLo: number;
  capHi: number;
  /** true when the interval ran past the axis and was clipped */
  clippedLo: boolean;
  clippedHi: boolean;
}

/**
 * Geometry for one horizontal estimate-with-interval row.
 *
 * Clipping is reported rather than hidden. A Clopper-Pearson interval on a
 * small denominator can easily extend past a plot whose axis was set by the
 * point estimates, and an interval silently cut at the frame edge would look
 * narrower -- i.e. more certain -- than the statistics support. The caller
 * draws an arrowhead where `clipped*` is true.
 */
export function horizontalErrorBar(
  value: number,
  low: number | null | undefined,
  high: number | null | undefined,
  sx: Scale,
  rowCentre: number,
  capHalfHeight = 3,
): ErrorBar {
  const [x0, x1] = sx.range;
  const min = Math.min(x0, x1);
  const max = Math.max(x0, x1);
  const rawLo = typeof low === "number" && Number.isFinite(low) ? sx(low) : sx(value);
  const rawHi = typeof high === "number" && Number.isFinite(high) ? sx(high) : sx(value);
  return {
    cx: sx(value),
    cy: rowCentre,
    lo: Math.max(min, Math.min(rawLo, rawHi)),
    hi: Math.min(max, Math.max(rawLo, rawHi)),
    capLo: rowCentre - capHalfHeight,
    capHi: rowCentre + capHalfHeight,
    clippedLo: Math.min(rawLo, rawHi) < min - 0.5,
    clippedHi: Math.max(rawLo, rawHi) > max + 0.5,
  };
}

// ---------------------------------------------------------------------------
// threshold markers
// ---------------------------------------------------------------------------

export interface Marker {
  key: string;
  label: string;
  value: number;
  /** pixel position along the value axis */
  at: number;
  /** ladder rank, ascending; used to stagger labels so they do not overlap */
  rank: number;
  tone: "spec" | "accept" | "transfer" | "forger";
}

/**
 * The threshold ladder as markers on a rate axis.
 *
 * The four levels are not interchangeable and the order is a security
 * property: spec < accept < transfer < forger. Markers are returned in
 * ascending pixel order with a `rank`, so a caller can stagger labels
 * vertically and a reader can see at a glance whether the ladder is
 * well-formed -- a crossing is visible as an out-of-order label before any
 * text explains it.
 */
export function thresholdMarkers(
  t: {
    spec_rate?: number | null;
    accept?: number | null;
    transfer?: number | null;
    forger_rate?: number | null;
  },
  sx: Scale,
): Marker[] {
  const defs: Array<[string, string, number | null | undefined, Marker["tone"]]> = [
    ["spec_rate", "Declared floor", t.spec_rate, "spec"],
    ["accept", "Accept sₐ", t.accept, "accept"],
    ["transfer", "Transfer sᵥ", t.transfer, "transfer"],
    ["forger_rate", "Forger bound", t.forger_rate, "forger"],
  ];
  const live = defs.filter(
    (d): d is [string, string, number, Marker["tone"]] =>
      typeof d[2] === "number" && Number.isFinite(d[2]),
  );
  return live
    .map(([key, label, value, tone]) => ({
      key,
      label,
      value,
      at: sx(value),
      rank: 0,
      tone,
    }))
    .sort((a, b) => a.value - b.value)
    .map((m, i) => ({ ...m, rank: i }));
}

/** Domain covering every estimate, interval bound and threshold on one axis. */
export function rateDomain(
  values: Array<number | null | undefined>,
  padFraction = 0.08,
): [number, number] {
  const live = values.filter(
    (v): v is number => typeof v === "number" && Number.isFinite(v),
  );
  if (live.length === 0) return [0, 1];
  const lo = Math.min(0, ...live);
  const hi = Math.max(...live);
  const pad = Math.max((hi - lo) * padFraction, 1e-4);
  return [Math.max(0, lo - pad), hi + pad];
}

// ---------------------------------------------------------------------------
// matrix grid
// ---------------------------------------------------------------------------

export interface GridCell<T> {
  row: number;
  col: number;
  x: number;
  y: number;
  width: number;
  height: number;
  datum: T;
}

export interface GridLayout<T> {
  cells: Array<GridCell<T>>;
  width: number;
  height: number;
  cellWidth: number;
  cellHeight: number;
}

/** Cell rectangles for a rule-by-scenario matrix. */
export function gridLayout<T>(
  rows: number,
  cols: number,
  values: (row: number, col: number) => T,
  cellWidth = 34,
  cellHeight = 22,
  gap = 2,
): GridLayout<T> {
  const cells: Array<GridCell<T>> = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      cells.push({
        row: r,
        col: c,
        x: c * (cellWidth + gap),
        y: r * (cellHeight + gap),
        width: cellWidth,
        height: cellHeight,
        datum: values(r, c),
      });
    }
  }
  return {
    cells,
    width: cols > 0 ? cols * (cellWidth + gap) - gap : 0,
    height: rows > 0 ? rows * (cellHeight + gap) - gap : 0,
    cellWidth,
    cellHeight,
  };
}

// ---------------------------------------------------------------------------
// bars
// ---------------------------------------------------------------------------

export interface Bar {
  key: string;
  label: string;
  value: number;
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Horizontal bars sharing one value axis, for phase timings and tallies. */
export function horizontalBars(
  input: Array<{ key: string; label: string; value: number | null | undefined }>,
  plotWidth: number,
  barHeight = 14,
  gap = 6,
  max?: number,
): { bars: Bar[]; height: number; max: number } {
  const live = input.map((d) => ({
    ...d,
    value: typeof d.value === "number" && Number.isFinite(d.value) ? d.value : 0,
  }));
  const peak = max ?? Math.max(1e-12, ...live.map((d) => d.value));
  const bars = live.map((d, i) => ({
    key: d.key,
    label: d.label,
    value: d.value,
    x: 0,
    y: i * (barHeight + gap),
    width: Math.max(0, (d.value / peak) * plotWidth),
    height: barHeight,
  }));
  return {
    bars,
    height: bars.length > 0 ? bars.length * (barHeight + gap) - gap : 0,
    max: peak,
  };
}

// ---------------------------------------------------------------------------
// p-value axis
// ---------------------------------------------------------------------------

export interface PValuePoint {
  name: string;
  title: string;
  p: number | null;
  /** pixel position, or null when the test has no p-value by construction */
  at: number | null;
  flagged: boolean;
  applicable: boolean;
  /** the test underflowed to zero and is pinned at the axis floor */
  underflow: boolean;
}

/**
 * Lay out the detector p-values against the corrected alpha.
 *
 * Three states have to stay distinguishable, and a naive chart would merge
 * them: a test with `p = null` has no p-value by construction (the sequential
 * monitors are stopping rules, so a fixed-sample p-value is not defined for
 * them), a test with `applicable: false` never ran, and a test with `p = 0`
 * underflowed. All three get `at: null` or an explicit `underflow` flag
 * instead of being silently plotted at some default.
 */
export function pValueLayout(
  tests: Array<{
    name: string;
    title: string;
    p_value: number | null;
    flagged: boolean;
    applicable: boolean;
  }>,
  sx: Scale,
): PValuePoint[] {
  const floor = Math.min(...sx.domain);
  return tests.map((t) => {
    const p = t.p_value;
    const missing = p === null || p === undefined || !Number.isFinite(p);
    const underflow = !missing && (p as number) <= floor;
    return {
      name: t.name,
      title: t.title,
      p: missing ? null : (p as number),
      at: missing ? null : sx(underflow ? floor : (p as number)),
      flagged: t.flagged,
      applicable: t.applicable,
      underflow,
    };
  });
}

/** Log-axis domain covering every finite p-value plus the alpha line. */
export function pValueDomain(
  ps: Array<number | null | undefined>,
  alpha: number | null | undefined,
): [number, number] {
  const live = ps.filter(
    (p): p is number => typeof p === "number" && Number.isFinite(p) && p > 0,
  );
  if (typeof alpha === "number" && Number.isFinite(alpha) && alpha > 0) {
    live.push(alpha);
  }
  if (live.length === 0) return [1e-4, 1];
  const lo = Math.min(...live);
  // Floor at 1e-12: below that the exact value is numerically meaningless and
  // an axis decade per order of magnitude wastes the whole figure on noise.
  return [Math.max(1e-12, lo / 3), 1];
}

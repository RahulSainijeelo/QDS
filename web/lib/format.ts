/**
 * Number formatting for an instrument readout.
 *
 * The rule throughout: never print more precision than the measurement
 * carries, and never silently turn a missing number into a plausible one.
 * A null rate renders as an em dash, not as `0.0000`, because a dash is
 * obviously absent and a zero is a claim.
 *
 * Pure functions only -- no React, no DOM -- so `web/lib/lib.test.ts` can
 * exercise all of it under plain `node`.
 */

/** What to print when a number is absent. Not "0", not "N/A". */
export const ABSENT = "—";

const isNum = (x: unknown): x is number =>
  typeof x === "number" && Number.isFinite(x);

/** A rate in [0,1] as a percentage. Absent stays absent. */
export function pct(x: number | null | undefined, digits = 2): string {
  if (!isNum(x)) return ABSENT;
  return `${(x * 100).toFixed(digits)}%`;
}

/** A rate as a bare decimal, for tables where the % sign is noise. */
export function rate(x: number | null | undefined, digits = 4): string {
  if (!isNum(x)) return ABSENT;
  return x.toFixed(digits);
}

/**
 * A p-value.
 *
 * Below 1e-4 the decimal form stops being readable and starts being a row of
 * zeros, so it switches to scientific notation. Values that underflow to
 * exactly 0 print as `<1e-300` rather than `0`, because the engine computes
 * these in floating point and a true zero probability is not what happened.
 */
export function pValue(p: number | null | undefined): string {
  if (p === null || p === undefined) return ABSENT;
  if (!isNum(p)) return ABSENT;
  if (p === 0) return "<1e-300";
  if (p < 1e-4) {
    const [m, e] = p.toExponential(1).split("e");
    return `${m}e${Number(e)}`;
  }
  if (p >= 0.9995) return "1.000";
  return p.toFixed(4);
}

/** `k/n` with thousands separators, or a dash when the denominator is zero. */
export function ratio(k: number | null | undefined, n: number | null | undefined): string {
  if (!isNum(k) || !isNum(n) || n === 0) return ABSENT;
  return `${count(k)}/${count(n)}`;
}

/** An integer with thin thousands separators. */
export function count(x: number | null | undefined): string {
  if (!isNum(x)) return ABSENT;
  return Math.round(x).toLocaleString("en-US");
}

/** A confidence interval as a closed range. */
export function interval(
  low: number | null | undefined,
  high: number | null | undefined,
  digits = 4,
): string {
  if (!isNum(low) || !isNum(high)) return ABSENT;
  return `[${low.toFixed(digits)}, ${high.toFixed(digits)}]`;
}

/**
 * Security expressed in bits, i.e. `-log2(failure probability)`.
 * Larger is safer; the engine reports these alongside the raw bounds.
 */
export function bits(x: number | null | undefined): string {
  if (!isNum(x)) return ABSENT;
  if (x >= 1000) return `${Math.round(x)} bits`;
  return `${x.toFixed(1)} bits`;
}

/** A duration, scaled to whichever unit keeps it legible. */
export function duration(seconds: number | null | undefined): string {
  if (!isNum(seconds)) return ABSENT;
  if (seconds < 1e-3) return `${(seconds * 1e6).toFixed(0)}µs`;
  if (seconds < 1) return `${(seconds * 1e3).toFixed(1)}ms`;
  if (seconds < 60) return `${seconds.toFixed(2)}s`;
  const m = Math.floor(seconds / 60);
  return `${m}m ${(seconds - m * 60).toFixed(0)}s`;
}

/** A small probability such as a dark-count fraction. */
export function small(x: number | null | undefined, digits = 2): string {
  if (!isNum(x)) return ABSENT;
  if (x === 0) return "0";
  if (Math.abs(x) < 1e-4) {
    const [m, e] = x.toExponential(digits).split("e");
    return `${m}e${Number(e)}`;
  }
  return x.toFixed(Math.max(digits, 4));
}

/** Hex message bytes rendered as their ASCII text where printable. */
export function hexToAscii(hex: string | null | undefined): string | null {
  if (!hex || typeof hex !== "string" || hex.length % 2 !== 0) return null;
  let out = "";
  for (let i = 0; i < hex.length; i += 2) {
    const byte = Number.parseInt(hex.slice(i, i + 2), 16);
    if (!Number.isFinite(byte)) return null;
    if (byte < 0x20 || byte > 0x7e) return null;
    out += String.fromCharCode(byte);
  }
  return out;
}

/** `snake_case_name` to `Snake case name`, for rule and attribution labels. */
export function humanise(name: string | null | undefined): string {
  if (!name) return ABSENT;
  const s = name.replace(/_/g, " ").trim();
  return s.charAt(0).toUpperCase() + s.slice(1);
}

/** A unix timestamp as an ISO-ish local string, seconds resolution. */
export function timestamp(unixSeconds: number | null | undefined): string {
  if (!isNum(unixSeconds)) return ABSENT;
  const d = new Date(unixSeconds * 1000);
  if (Number.isNaN(d.getTime())) return ABSENT;
  const p = (n: number) => String(n).padStart(2, "0");
  return (
    `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ` +
    `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
  );
}

/** Signed difference, with an explicit sign so direction is never ambiguous. */
export function signed(x: number | null | undefined, digits = 4): string {
  if (!isNum(x)) return ABSENT;
  const s = x.toFixed(digits);
  return x > 0 ? `+${s}` : s;
}

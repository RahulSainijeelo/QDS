/**
 * Loading snapshots, static-first.
 *
 * The default path reads the JSON that `web/scripts/generate_snapshots.py`
 * wrote into `web/public/data`. That is the whole dataset: committed, diffable,
 * and reproducible by re-running one script. The dashboard needs nothing else
 * to render, which is why a static export is the primary deployment.
 *
 * The optional live path re-runs the engine for a single scenario on demand.
 * It exists for the person who has the Python engine checked out beside the
 * web app and wants to see a fresh run, and it is strictly additive: if the
 * engine is not present, or the app was exported statically, the live path is
 * simply unavailable and the static snapshots stand.
 *
 * The split in this file is deliberate:
 *
 *   - parsing and validation are pure, and throw with the offending field
 *     named, so `web/lib/lib.test.ts` can feed them a real snapshot and a
 *     truncated one under plain `node`;
 *   - every filesystem or subprocess call sits behind a dynamic
 *     `import("node:...")`, so this module carries no static Node dependency
 *     and a client bundle that imports the pure half does not pull in `fs`.
 *
 * Validation is intentionally shallow. It checks the fields whose absence
 * would blank a primary view -- not every nested leaf, which `types.ts`
 * already documents and which deep-validating here would only duplicate and
 * let drift. The goal is to fail at load with "scenarios[2].detection.verdict
 * is missing in honest-lab.json", not to reimplement the schema.
 */

import type { ScenarioCard, Snapshot, SnapshotIndex, Verdict } from "./types.ts";

// ---------------------------------------------------------------------------
// validation helpers (pure)
// ---------------------------------------------------------------------------

/** Thrown when a loaded document is missing a field the UI depends on. */
export class SnapshotError extends Error {
  /** the file the bad document came from, when known */
  readonly source?: string;

  constructor(message: string, source?: string) {
    super(source ? `${message} (in ${source})` : message);
    this.name = "SnapshotError";
    this.source = source;
  }
}

function fail(path: string, want: string, source?: string): never {
  throw new SnapshotError(`${path} ${want}`, source);
}

function isObj(x: unknown): x is Record<string, unknown> {
  return typeof x === "object" && x !== null && !Array.isArray(x);
}

function reqObj(v: unknown, path: string, source?: string): Record<string, unknown> {
  if (!isObj(v)) fail(path, "must be an object", source);
  return v;
}

function reqArr(v: unknown, path: string, source?: string): unknown[] {
  if (!Array.isArray(v)) fail(path, "must be an array", source);
  return v;
}

function reqStr(v: unknown, path: string, source?: string): string {
  if (typeof v !== "string" || v.length === 0) fail(path, "must be a non-empty string", source);
  return v;
}

const VERDICTS: ReadonlySet<string> = new Set<Verdict>(["clean", "suspicious", "compromised"]);

function reqVerdict(v: unknown, path: string, source?: string): Verdict {
  const s = reqStr(v, path, source);
  if (!VERDICTS.has(s)) fail(path, `must be one of clean|suspicious|compromised, got ${JSON.stringify(s)}`, source);
  return s as Verdict;
}

// ---------------------------------------------------------------------------
// parsing (pure)
// ---------------------------------------------------------------------------

/**
 * Validate a parsed `index.json`.
 *
 * Checks the catalogue's spine -- geometry, the rule list, and that every
 * scenario card carries an id and a legal verdict -- because those are what
 * the comparison view and every link out depend on. A card missing its
 * verdict is the kind of fault that otherwise renders as a blank cell three
 * views deep, so it is caught here at the door.
 */
export function parseIndex(raw: unknown, source = "index.json"): SnapshotIndex {
  const o = reqObj(raw, "index", source);
  reqObj(o.geometry, "index.geometry", source);
  const rules = reqArr(o.rules, "index.rules", source).map((r, i) =>
    reqStr(r, `index.rules[${i}]`, source),
  );
  const scenariosRaw = reqArr(o.scenarios, "index.scenarios", source);
  const scenarios = scenariosRaw.map((c, i) => {
    const card = reqObj(c, `index.scenarios[${i}]`, source);
    reqStr(card.id, `index.scenarios[${i}].id`, source);
    reqVerdict(card.verdict, `index.scenarios[${i}].verdict`, source);
    reqArr(card.flagged, `index.scenarios[${i}].flagged`, source);
    return card as unknown as ScenarioCard;
  });
  if (scenarios.length === 0) fail("index.scenarios", "must list at least one scenario", source);
  return { ...(o as object), rules, scenarios } as SnapshotIndex;
}

/**
 * Validate a parsed scenario snapshot.
 *
 * The detection report is the load-bearing half, so its verdict, test array
 * and flagged list are checked explicitly; `session` is required present but
 * not walked, since the views that use it already treat its sub-objects as
 * optional (`verifications ?? {}`) by design.
 */
export function parseSnapshot(raw: unknown, source?: string): Snapshot {
  const o = reqObj(raw, "snapshot", source);
  const id = reqStr(o.id, "snapshot.id", source);
  const src = source ?? `${id}.json`;

  const det = reqObj(o.detection, "snapshot.detection", src);
  reqVerdict(det.verdict, "snapshot.detection.verdict", src);
  reqArr(det.tests, "snapshot.detection.tests", src);
  reqArr(det.flagged, "snapshot.detection.flagged", src);
  reqObj(det.thresholds, "snapshot.detection.thresholds", src);
  reqObj(det.estimates, "snapshot.detection.estimates", src);

  reqObj(o.session, "snapshot.session", src);
  return o as unknown as Snapshot;
}

/** Parse a JSON string into a validated index, naming the source on failure. */
export function parseIndexText(text: string, source = "index.json"): SnapshotIndex {
  return parseIndex(parseJson(text, source), source);
}

/** Parse a JSON string into a validated snapshot, naming the source on failure. */
export function parseSnapshotText(text: string, source?: string): Snapshot {
  return parseSnapshot(parseJson(text, source), source);
}

function parseJson(text: string, source?: string): unknown {
  try {
    return JSON.parse(text);
  } catch (e) {
    throw new SnapshotError(
      `is not valid JSON: ${(e as Error).message}`,
      source,
    );
  }
}

/** The scenario ids an index advertises, in catalogue order. */
export function scenarioIds(index: SnapshotIndex): string[] {
  return index.scenarios.map((s) => s.id);
}

/** Find one card by id. */
export function cardById(index: SnapshotIndex, id: string): ScenarioCard | null {
  return index.scenarios.find((s) => s.id === id) ?? null;
}

// ---------------------------------------------------------------------------
// where the data lives
// ---------------------------------------------------------------------------

/** The public URL a browser fetches a data file from. */
export const dataUrl = {
  index: () => "/data/index.json",
  scenario: (id: string) => `/data/scenarios/${encodeURIComponent(id)}.json`,
};

/**
 * Resolve the on-disk data directory for server-side reads.
 *
 * Next runs the build with `process.cwd()` at the app root (`web/`), so the
 * committed snapshots sit at `public/data`. An explicit argument overrides
 * this, which is the hook the node test uses to point the loaders at the real
 * files without a Next runtime.
 */
async function dataDir(override?: string): Promise<string> {
  if (override) return override;
  const path = await import("node:path");
  return path.join(process.cwd(), "public", "data");
}

// ---------------------------------------------------------------------------
// static loading (server / build time)
// ---------------------------------------------------------------------------

/**
 * Read and validate the catalogue from disk.
 *
 * This is the function a server component calls at build time. It and its
 * siblings reach `node:fs` only through a dynamic import, so importing this
 * module from a client component is harmless until one of these is actually
 * called -- which a client component never does.
 */
export async function loadIndexFromDisk(dir?: string): Promise<SnapshotIndex> {
  const [fs, path] = await Promise.all([
    import("node:fs/promises"),
    import("node:path"),
  ]);
  const base = await dataDir(dir);
  const file = path.join(base, "index.json");
  const text = await readOr(fs, file);
  return parseIndexText(text, file);
}

/** Read and validate one scenario snapshot from disk. */
export async function loadSnapshotFromDisk(id: string, dir?: string): Promise<Snapshot> {
  const [fs, path] = await Promise.all([
    import("node:fs/promises"),
    import("node:path"),
  ]);
  const base = await dataDir(dir);
  const file = path.join(base, "scenarios", `${id}.json`);
  const text = await readOr(fs, file);
  return parseSnapshotText(text, file);
}

/**
 * Every snapshot the index lists, loaded and validated.
 *
 * Reads run concurrently; one bad file rejects the whole batch with its own
 * name attached, which is the right failure for a build step -- a dashboard
 * built from a partially-valid dataset would be worse than one that refused to
 * build.
 */
export async function loadAllFromDisk(dir?: string): Promise<{
  index: SnapshotIndex;
  snapshots: Record<string, Snapshot>;
}> {
  const index = await loadIndexFromDisk(dir);
  const ids = scenarioIds(index);
  const loaded = await Promise.all(ids.map((id) => loadSnapshotFromDisk(id, dir)));
  const snapshots: Record<string, Snapshot> = {};
  ids.forEach((id, i) => {
    snapshots[id] = loaded[i];
  });
  return { index, snapshots };
}

async function readOr(
  fs: typeof import("node:fs/promises"),
  file: string,
): Promise<string> {
  try {
    return await fs.readFile(file, "utf-8");
  } catch (e) {
    const err = e as NodeJS.ErrnoException;
    if (err.code === "ENOENT") {
      throw new SnapshotError(
        "was not found -- run `python3 web/scripts/generate_snapshots.py` to produce the data",
        file,
      );
    }
    throw e;
  }
}

// ---------------------------------------------------------------------------
// static loading (client / runtime fetch)
// ---------------------------------------------------------------------------

/**
 * Fetch and validate the catalogue over HTTP.
 *
 * For a client component in the exported site. `fetch` is global in both the
 * browser and Node 22, so this is also the path the node test exercises when
 * a dev server is up. Non-200 responses throw rather than resolve to an empty
 * index, so a 404 from a missing export surfaces as an error, not a blank
 * page.
 */
export async function fetchIndex(baseUrl = ""): Promise<SnapshotIndex> {
  const url = `${baseUrl}${dataUrl.index()}`;
  const text = await fetchText(url);
  return parseIndexText(text, url);
}

/** Fetch and validate one scenario snapshot over HTTP. */
export async function fetchSnapshot(id: string, baseUrl = ""): Promise<Snapshot> {
  const url = `${baseUrl}${dataUrl.scenario(id)}`;
  const text = await fetchText(url);
  return parseSnapshotText(text, url);
}

async function fetchText(url: string): Promise<string> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new SnapshotError(`fetch failed with HTTP ${res.status}`, url);
  }
  return res.text();
}

// ---------------------------------------------------------------------------
// the optional live path
// ---------------------------------------------------------------------------

export interface LiveResult {
  snapshot: Snapshot;
  /** wall time the engine run took, in seconds, as the generator reported it */
  wallSeconds: number;
  /** stderr from the generator, surfaced so a slow or chatty run is visible */
  log: string;
}

/**
 * Whether a live re-run is even possible from here.
 *
 * True only when the Python engine is checked out beside the web app and we
 * are running in a Node server (not a static export). The route handler calls
 * this first so it can answer a live request with an honest 501 rather than a
 * confusing subprocess error when the engine is simply not there.
 */
export async function liveEngineAvailable(): Promise<boolean> {
  try {
    const [fs, path] = await Promise.all([
      import("node:fs/promises"),
      import("node:path"),
    ]);
    const script = path.join(process.cwd(), "scripts", "generate_snapshots.py");
    await fs.access(script);
    return true;
  } catch {
    return false;
  }
}

/**
 * Re-run the engine for a single scenario and return the fresh snapshot.
 *
 * This shells out to the very same generator that produced the static files,
 * with `--only <id>`, so a live run is byte-for-byte the same computation as
 * the committed one -- not a second, parallel implementation that could drift
 * from it. The generator is told to write into a caller-supplied scratch
 * directory and the result is read straight back; the committed
 * `public/data` is never touched, so a live run cannot corrupt the static
 * dataset it falls back to.
 *
 * Everything Node-specific is dynamically imported, so this file stays
 * importable in a browser bundle even though this particular function can only
 * ever run on a server.
 */
export async function runLiveScenario(id: string, scratchDir?: string): Promise<LiveResult> {
  const [{ spawn }, fs, path, os] = await Promise.all([
    import("node:child_process"),
    import("node:fs/promises"),
    import("node:path"),
    import("node:os"),
  ]);

  // Guard the id: it becomes a path component and a CLI argument, so anything
  // that is not a plain scenario slug is refused before it reaches the shell.
  if (!/^[a-z0-9][a-z0-9-]*$/.test(id)) {
    throw new SnapshotError(`${JSON.stringify(id)} is not a valid scenario id`);
  }

  const out = scratchDir ?? (await fs.mkdtemp(path.join(os.tmpdir(), "qds-live-")));
  const script = path.join(process.cwd(), "scripts", "generate_snapshots.py");

  const log = await new Promise<string>((resolve, reject) => {
    const child = spawn("python3", [script, "--out", out, "--only", id], {
      cwd: process.cwd(),
      env: process.env,
    });
    let stderr = "";
    child.stderr.on("data", (d) => (stderr += String(d)));
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) resolve(stderr);
      else reject(new SnapshotError(`engine exited ${code}: ${stderr.trim()}`));
    });
  });

  const file = path.join(out, "scenarios", `${id}.json`);
  const text = await fs.readFile(file, "utf-8").catch(() => {
    throw new SnapshotError("engine produced no snapshot for this id", file);
  });
  const snapshot = parseSnapshotText(text, file);
  return { snapshot, wallSeconds: snapshot.wall_seconds, log };
}

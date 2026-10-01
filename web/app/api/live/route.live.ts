/**
 * The optional live endpoint.
 *
 * This file is a route handler ONLY when the app is built with QDS_LIVE=1,
 * which is the one mode where `next.config.mjs` adds `live.ts` to
 * `pageExtensions`. In the default static export the filename is not
 * recognised and Next never compiles it, so a dynamic, subprocess-spawning
 * handler can live in the tree without making the static build dynamic.
 *
 *   GET /api/live?id=<scenario>
 *
 * It re-runs the very same Python generator that produced the committed
 * snapshots, for one scenario, into a scratch directory, and returns the fresh
 * snapshot as JSON. The committed `public/data` is never touched. If the
 * engine is not checked out beside the app it answers 501 rather than failing
 * obscurely, so the caller can fall back to the static snapshot cleanly.
 *
 * `force-dynamic` is what makes this a per-request handler; it is also exactly
 * what `output: "export"` refuses, which is the second reason this file must
 * stay invisible to the default build.
 */

import { liveEngineAvailable, runLiveScenario, SnapshotError } from "../../../lib/data";

export const dynamic = "force-dynamic";

function json(body: unknown, status: number): Response {
  return new Response(JSON.stringify(body, null, 2), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

export async function GET(request: Request): Promise<Response> {
  const id = new URL(request.url).searchParams.get("id");
  if (!id) {
    return json({ error: "pass a scenario id, e.g. /api/live?id=honest-lab" }, 400);
  }

  if (!(await liveEngineAvailable())) {
    return json(
      {
        error:
          "the live engine is not available here -- the Python generator " +
          "(scripts/generate_snapshots.py) is not present beside this app, " +
          "so the committed static snapshot stands",
      },
      501,
    );
  }

  try {
    const { snapshot, wallSeconds, log } = await runLiveScenario(id);
    return json({ snapshot, wallSeconds, log }, 200);
  } catch (e) {
    const message = e instanceof SnapshotError ? e.message : (e as Error).message;
    return json({ error: message }, 500);
  }
}

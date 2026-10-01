/**
 * Static params for the per-scenario routes.
 *
 * Declared once and re-exported by every page under `s/[id]`, so the three
 * tabs stay in lockstep: the set of scenarios the site is built for is the set
 * the catalogue lists, and nothing else. With `output: "export"` this is what
 * turns each scenario into a real file on disk at build time.
 */

import { loadIndexFromDisk, scenarioIds } from "../../../lib/data";

export async function generateStaticParams(): Promise<Array<{ id: string }>> {
  const index = await loadIndexFromDisk();
  return scenarioIds(index).map((id) => ({ id }));
}

// @ts-check

/**
 * Two builds from one source tree.
 *
 * The default build is a pure static export: `next build` writes a complete
 * `out/` directory that can be served by any file host, opened from a bucket,
 * or committed. It needs no Node runtime because every page reads its data
 * from the committed JSON in `public/data` at build time. This is the
 * deployment the project is designed around.
 *
 * The live build is opt-in. Setting QDS_LIVE=1 does two things at once:
 *
 *   1. it drops `output: "export"`, so Next runs as a normal server that can
 *      answer a request at runtime; and
 *   2. it widens `pageExtensions` to recognise `*.live.ts`, which is the only
 *      condition under which `app/api/live/route.live.ts` becomes a real route.
 *
 * In the default build that file is not a recognised route filename, so Next
 * ignores it entirely -- which is why a dynamic, subprocess-spawning handler
 * can sit in the tree without breaking the static export. The two modes are
 * mutually exclusive by construction: you cannot accidentally ship the live
 * endpoint in a static build, and you cannot statically export a route that
 * only makes sense against a live engine.
 *
 *   next build              -> static site in out/, no live endpoint
 *   QDS_LIVE=1 next dev      -> dev server with GET /api/live?id=<scenario>
 *   QDS_LIVE=1 next build    -> server build that includes the live endpoint
 */

const LIVE = process.env.QDS_LIVE === "1";

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static export unless a live engine run is explicitly requested.
  output: LIVE ? undefined : "export",

  // Directory-style URLs (s/honest-lab/index.html), which behave on the widest
  // range of static hosts and when opened straight off disk.
  trailingSlash: true,

  // No build-time image optimisation server exists in a static export, and the
  // instrument uses no raster images anyway; declaring this keeps `next build`
  // from objecting if one is ever added.
  images: { unoptimized: true },

  // `*.live.ts` is a route only in live mode; see the header comment.
  pageExtensions: LIVE
    ? ["tsx", "ts", "jsx", "js", "live.tsx", "live.ts"]
    : ["tsx", "ts", "jsx", "js"],

  reactStrictMode: true,
};

export default nextConfig;

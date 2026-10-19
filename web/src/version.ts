/**
 * Build stamp for cache-busting. The CDN in front of the site tells browsers to keep
 * scripts, wasm and zips for hours, so a deploy would otherwise run new HTML against
 * old code. build.ts defines `__BUILD__` as a hash of the build inputs; every asset
 * URL the page or worker fetches carries it as `?v=`.
 */
declare const __BUILD__: string | undefined;

export const BUILD: string = typeof __BUILD__ === "string" ? __BUILD__ : "dev";

/** `url` with the build stamp appended (absolute or relative URLs). */
export function versioned(url: string | URL): string {
  const s = String(url);
  return `${s}${s.includes("?") ? "&" : "?"}v=${BUILD}`;
}

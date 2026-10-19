/**
 * Hand a model to pokefit from another app (e.g. craftmatic: "Print in a Pokéball").
 *
 *   const w = window.open("https://pokefit.party/");
 *   window.addEventListener("message", (e) => {
 *     if (e.source === w && e.data?.type === "pokefit:ready")
 *       w.postMessage({ type: "pokefit:load", name: "build.glb", data: glbArrayBuffer }, "https://pokefit.party");
 *   });
 *
 * pokefit announces `pokefit:ready` to its opener once it can take a file, and loads a `pokefit:load`
 * message's bytes like a dropped file. Only trusted origins are listed: a model is data (nothing in it
 * runs), but a page you did not choose should not be able to fill your pokefit tab.
 * `?url=<model URL>` also works when that server allows cross-origin reads (CORS).
 */

export const TRUSTED_ORIGINS: RegExp[] = [
  /^https:\/\/craftmatic\.click$/,
  /^https:\/\/tribixbite\.github\.io$/,
  /^https:\/\/pokefit\.party$/,
  /^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$/,
];

export const MODEL_SUFFIXES = [".3mf", ".stl", ".obj", ".glb", ".gltf"];

/** Is a message from this origin allowed to load a model? */
export function trustedOrigin(origin: string): boolean {
  return TRUSTED_ORIGINS.some((r) => r.test(origin));
}

/** A `pokefit:load` message as a File, or null if it is not one (or not a model file). */
export function handoffFile(data: unknown): File | null {
  if (!data || typeof data !== "object") return null;
  const msg = data as { type?: unknown; name?: unknown; data?: unknown };
  if (msg.type !== "pokefit:load" || typeof msg.name !== "string") return null;
  const name = msg.name.split(/[\\/]/).pop() ?? "";
  if (!MODEL_SUFFIXES.some((s) => name.toLowerCase().endsWith(s))) return null;
  const bytes = msg.data instanceof ArrayBuffer ? msg.data : ArrayBuffer.isView(msg.data) ? msg.data.buffer : null;
  if (!bytes || bytes.byteLength === 0) return null;
  return new File([bytes as ArrayBuffer], name);
}

/** File name for a `?url=` model: the URL's last path part, if it is a model file. */
export function urlFileName(url: string): string | null {
  try {
    const u = new URL(url);
    if (u.protocol !== "https:" && u.protocol !== "http:") return null;
    const name = decodeURIComponent(u.pathname.split("/").pop() ?? "");
    return MODEL_SUFFIXES.some((s) => name.toLowerCase().endsWith(s)) ? name : null;
  } catch {
    return null;
  }
}

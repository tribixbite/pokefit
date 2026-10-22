/**
 * Per-face colours of the figure preview (engine `preview_faces`): decoding and regrouping a mesh's
 * triangles by colour, so a paint edit recolours the loaded glTF instead of reloading it.
 */
import type { TypedArray } from "three";

/** One draw range of an index buffer (three's BufferGeometry group). */
export interface FaceGroup {
  start: number;
  count: number;
  materialIndex: number;
}

/** Bytes of a base64 string. */
export function base64Bytes(b64: string): Uint8Array {
  const s = atob(b64);
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out;
}

/**
 * Triangles (corner indices `order`, 3 per face) sorted by colour with a counting sort, and the
 * draw groups (index ranges) per colour.
 */
export function groupFaces(order: TypedArray, colour: Uint8Array, n: number):
  { index: TypedArray; groups: FaceGroup[] } {
  const counts = new Uint32Array(n);
  for (const c of colour) if (c < n) counts[c]++;
  const at = new Uint32Array(n);
  const groups: FaceGroup[] = [];
  let pos = 0;
  for (let c = 0; c < n; c++) {
    at[c] = pos;
    if (counts[c]) groups.push({ start: pos * 3, count: counts[c] * 3, materialIndex: c });
    pos += counts[c];
  }
  const index = new (order.constructor as new (n: number) => TypedArray)(pos * 3);
  for (let f = 0; f < colour.length; f++) {
    const c = colour[f];
    if (c >= n) continue;
    const k = at[c]++ * 3;
    index[k] = order[f * 3];
    index[k + 1] = order[f * 3 + 1];
    index[k + 2] = order[f * 3 + 2];
  }
  return { index, groups };
}

/**
 * Dev server for the social-preview renderer: `bun run og`, then open the printed URL.
 * Serves og/index.html and the bundled og.ts, and writes posted images to public/.
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const pub = join(here, "..", "public");
mkdirSync(pub, { recursive: true });

const built = await Bun.build({ entrypoints: [join(here, "og.ts")], target: "browser" });
if (!built.success) {
  for (const l of built.logs) console.error(l);
  process.exit(1);
}
const bundle = await built.outputs[0].text();
const page = `<!doctype html><meta charset="utf-8"><title>og render</title>
<style>body{margin:0;background:#000;color:#ccc;font:14px system-ui}canvas{display:block;max-width:100%}</style>
<p id="status">rendering…</p><script type="module">${bundle.replace(/<\/script/g, "<\\/script")}</script>`;

const ALLOWED = new Set(["og.jpg", "og.gif"]);
const server = Bun.serve({
  port: 8791,
  hostname: "127.0.0.1",
  async fetch(req) {
    const url = new URL(req.url);
    if (req.method === "POST" && url.pathname === "/save") {
      const name = url.searchParams.get("name") ?? "";
      if (!ALLOWED.has(name)) return new Response("bad name", { status: 400 });
      const bytes = new Uint8Array(await req.arrayBuffer());
      writeFileSync(join(pub, name), bytes);
      console.log(`wrote public/${name} (${(bytes.length / 1024).toFixed(0)} KB)`);
      return new Response("ok");
    }
    return new Response(page, { headers: { "content-type": "text/html; charset=utf-8" } });
  },
});
console.log(`open http://${server.hostname}:${server.port}/ to render the social preview`);

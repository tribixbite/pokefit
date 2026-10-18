/** Bundle src/main.ts for the browser and copy the static files into dist/. */
import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const dist = join(here, "dist");
mkdirSync(dist, { recursive: true });

const result = await Bun.build({
  entrypoints: [join(here, "src", "main.ts")],
  outdir: dist,
  target: "browser",
  minify: true,
  sourcemap: "linked",
  naming: "[name].js",
});
if (!result.success) {
  for (const log of result.logs) console.error(log);
  process.exit(1);
}
for (const f of ["index.html", "style.css"]) copyFileSync(join(here, f), join(dist, f));
const size = result.outputs.reduce((n, o) => n + o.size, 0);
console.log(`built ${result.outputs.length} files, ${(size / 1024).toFixed(0)} KB -> ${dist}`);

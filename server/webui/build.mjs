// Bundles src/main.ts and copies the static files into dist/, which
// webui.go embeds. Run through `npm run build`.
import { build } from "esbuild";
import { cpSync, mkdirSync } from "node:fs";

mkdirSync("dist", { recursive: true });
await build({ entryPoints: ["src/main.ts"], bundle: true, minify: true, target: "es2022", outfile: "dist/app.js" });
cpSync("src/index.html", "dist/index.html");
cpSync("src/app.css", "dist/app.css");

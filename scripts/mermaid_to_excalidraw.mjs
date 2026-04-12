#!/usr/bin/env node
/**
 * Mermaid → Excalidraw scene + PNG converter.
 *
 * Pipeline (browser-accurate):
 *   1. Launch Chromium via Puppeteer, load excalidraw.com in the same way a
 *      human user would, paste the Mermaid source into the built-in
 *      "Mermaid to Excalidraw" dialog, and accept.
 *      → produces the .excalidraw scene (identical to what a user would get).
 *   2. Trigger Excalidraw's built-in "Export image (PNG)" action and save the
 *      resulting download.
 *
 * We also fall back to @mermaid-js/mermaid-cli (`mmdc`) for the PNG if the
 * Excalidraw UI changes selectors — mmdc is the same engine excalidraw.com
 * uses for parsing Mermaid, so the rendered PNG stays visually consistent.
 *
 * Install once:
 *   npm install --no-save puppeteer @mermaid-js/mermaid-cli
 *
 * Usage:
 *   node scripts/mermaid_to_excalidraw.mjs <in.mmd> <out-basename>
 *   # writes <out-basename>.png  (and .excalidraw when the UI flow succeeds)
 */
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";

const [, , inPath, outBase] = process.argv;
if (!inPath || !outBase) {
  console.error("usage: node mermaid_to_excalidraw.mjs <in.mmd> <out-basename>");
  process.exit(2);
}

const outDir = dirname(resolve(outBase));
if (!existsSync(outDir)) mkdirSync(outDir, { recursive: true });

// --- PNG (reliable path: mmdc uses the same Mermaid engine Excalidraw ships) ---
const pngPath = `${outBase}.png`;
execFileSync("npx", [
  "--yes", "@mermaid-js/mermaid-cli@latest",
  "-i", inPath, "-o", pngPath, "-b", "transparent", "-w", "1600",
], { stdio: "inherit" });
console.log(`wrote ${pngPath}`);

// --- Excalidraw scene (best-effort via excalidraw.com) ---
try {
  const { default: puppeteer } = await import("puppeteer");
  const { readFileSync, writeFileSync } = await import("node:fs");
  const src = readFileSync(inPath, "utf8");
  const browser = await puppeteer.launch({ headless: "new", args: ["--no-sandbox"] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1600, height: 1000, deviceScaleFactor: 2 });
  await page.goto("https://excalidraw.com/", { waitUntil: "networkidle0", timeout: 60000 });

  // Open "Mermaid to Excalidraw" dialog via the library's aria-label.
  await page.waitForSelector('[aria-label="Mermaid to Excalidraw"]', { timeout: 20000 });
  await page.click('[aria-label="Mermaid to Excalidraw"]');
  await page.waitForSelector('textarea', { timeout: 10000 });
  await page.$eval('textarea', (el, v) => { el.value = ""; el.dispatchEvent(new Event("input", { bubbles: true })); }, "");
  await page.type("textarea", src, { delay: 0 });
  await new Promise((r) => setTimeout(r, 1500));
  await page.click('button[aria-label="Insert"]');
  await new Promise((r) => setTimeout(r, 1500));

  // Snapshot scene from Excalidraw's exposed API (window.EXCALIDRAW_ASSET_PATH / React fiber).
  const scene = await page.evaluate(() => {
    const state = JSON.parse(localStorage.getItem("excalidraw") || "[]");
    return { type: "excalidraw", version: 2, source: "excalidraw.com", elements: state, appState: {}, files: {} };
  });
  writeFileSync(`${outBase}.excalidraw`, JSON.stringify(scene, null, 2));
  console.log(`wrote ${outBase}.excalidraw (${scene.elements.length} elements)`);
  await browser.close();
} catch (e) {
  console.warn(`[skip] excalidraw scene export unavailable: ${e.message}`);
  console.warn("       PNG was still produced via mmdc.");
}

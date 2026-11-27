// Run all four networks on the same cases, each through its own page: its own pen,
// pre-processing and forward pass. Serves the repo itself; needs puppeteer-core and Chrome.
//
// Usage: node bench.js cases.json results.json        (CHROME=/path/to/chrome to override)
const fs = require("fs");
const http = require("http");
const path = require("path");
const puppeteer = require("puppeteer-core");

const ROOT = path.resolve(__dirname, "..", "..");
const CHROME = process.env.CHROME || "/usr/bin/google-chrome";
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".json": "application/json", ".css": "text/css" };

const PLACE = {                                  // where on the pad the digit is drawn (unit square)
  normal: (u) => u,                              // across about 70% of the pad
  half: (u) => 0.06 + 0.5 * (u - 0.15),          // about 35% of the pad, near the top-left corner
  quarter: (u) => 0.06 + 0.25 * (u - 0.15),      // about 18% of the pad
};

// Gemini's CNN page keeps everything inside a closure, so the benchmark serves a copy with a
// test hook that calls the page's own strokes-to-canvas, pre-processing and network.
const HOOK = `  window.__geminiCnnTest = {
    predict(strokes, width) {
      strokesHistory = strokes.map((s) => ({ points: s.map(([x, y]) => ({ x, y })), width: width || brushSize }));
      redrawStrokesToCanvas();
      const pre = extractNormalized28x28(drawCanvas);
      const r = cnn.forward(pre.tensor, cnn.calibratedT);
      return { p: Array.from(r.probs), empty: !!r.isEmpty };
    },
    size: () => drawCanvas.width,
  };

  // Boot`;

function serve() {
  const server = http.createServer((req, res) => {
    const url = new URL(req.url, "http://localhost");
    const file = path.join(ROOT, decodeURIComponent(url.pathname));
    if (!file.startsWith(ROOT)) { res.writeHead(403); res.end(); return; }
    if (url.pathname === "/gemini-cnn/index.html" && url.searchParams.has("bench")) {
      const src = fs.readFileSync(file, "utf8");
      if (src.split("  // Boot").length !== 2) throw new Error("gemini-cnn/index.html changed: no unique '// Boot' marker");
      res.writeHead(200, { "Content-Type": "text/html" });
      res.end(src.replace("  // Boot", HOOK));
      return;
    }
    fs.readFile(file, (err, data) => {
      if (err) { res.writeHead(404); res.end(); return; }
      res.writeHead(200, { "Content-Type": TYPES[path.extname(file)] || "application/octet-stream" });
      res.end(data);
    });
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve(server)));
}

(async () => {
  const [casesPath, outPath] = process.argv.slice(2);
  const { cases } = JSON.parse(fs.readFileSync(casesPath));
  const server = await serve();
  const base = `http://127.0.0.1:${server.address().port}`;
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, args: ["--no-sandbox", "--disable-gpu"] });
  const out = { measured: new Date().toISOString().slice(0, 10), browser: await browser.version(), places: Object.keys(PLACE), apps: {} };

  // Pointer events land on whole pixels, so each placed stroke is snapped to the pad's pixel grid:
  // 310 px for the Claude pads (their desktop size) and 280 px for Gemini's canvases.
  const placedFor = (grid) => {
    const placed = {};
    for (const [name, f] of Object.entries(PLACE)) {
      placed[name] = cases.map((c) => c.strokes.map((s) => s.map(([x, y]) => [Math.round(f(x) * grid) / grid, Math.round(f(y) * grid) / grid])));
    }
    return placed;
  };

  async function runApp(app, url, ready, predict, grid) {
    const page = await browser.newPage();
    const errors = [];
    // Gemini's MLP page reports "tailwind is not defined" when its CDN is blocked; the network is unaffected.
    page.on("pageerror", (e) => { if (!/tailwind is not defined/.test(String(e))) errors.push(String(e)); });
    await page.setRequestInterception(true);
    page.on("request", (req) => (req.url().startsWith(base) ? req.continue() : req.abort()));   // offline: no CDNs
    await page.goto(`${base}/${url}`, { waitUntil: "load" });
    await page.waitForFunction(ready, { timeout: 60000 });
    const probs = {};
    for (const [name, list] of Object.entries(placedFor(grid))) probs[name] = await page.evaluate(predict, list);
    out.apps[app] = { probs, errors };
    if (app === "gemini") {
      out.apps[app].weights = await page.evaluate(() => ({ W1: Array.from(nn.W1), b1: Array.from(nn.b1), W2: Array.from(nn.W2), b2: Array.from(nn.b2), W3: Array.from(nn.W3), b3: Array.from(nn.b3) }));
    }
    await page.close();
    console.log(`${app}: done${errors.length ? ", page errors: " + errors.slice(0, 2).join("; ") : ""}`);
  }

  // Claude's pages: their stroke pipeline fits strokes to a 20x20 box, so pad coordinates go straight in.
  const claude = (list) => list.map((s) => window.__digitReader.predict(s).p);
  await runApp("claude", "claude/index.html", "window.__digitReader", claude, 310);
  await runApp("claude-cnn", "claude/cnn/index.html", "window.__digitReader", claude, 310);

  // Gemini's MLP page: its globals are reachable. Draw on its 280 px canvas as its pointer code does
  // (a dot, then round-capped lines with the default 22 px brush), then its extract and forward pass.
  await runApp("gemini", "gemini/index.html", "typeof nn !== 'undefined' && nn.trainingSteps > 0", (list) => {
    const c = document.getElementById("drawCanvas"), ctx = c.getContext("2d"), W = c.width, out = [];
    for (const strokes of list) {
      ctx.fillStyle = "#000000"; ctx.fillRect(0, 0, c.width, c.height);
      for (const s of strokes) {
        ctx.fillStyle = "#FFFFFF"; ctx.beginPath(); ctx.arc(s[0][0] * W, s[0][1] * W, 11, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = "#FFFFFF"; ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.lineWidth = 22;
        for (let i = 1; i < s.length; i++) { ctx.beginPath(); ctx.moveTo(s[i - 1][0] * W, s[i - 1][1] * W); ctx.lineTo(s[i][0] * W, s[i][1] * W); ctx.stroke(); }
      }
      out.push(Array.from(nn.forward(extractNormalized28x28(c).tensor, 1.0).probs));
    }
    return out;
  }, 280);

  // Gemini's CNN page, through the hooked copy.
  await runApp("gemini-cnn", "gemini-cnn/index.html?bench", "window.__geminiCnnTest", (list) => {
    const W = window.__geminiCnnTest.size();
    return list.map((s) => window.__geminiCnnTest.predict(s.map((st) => st.map(([x, y]) => [x * W, y * W]))).p);
  }, 280);

  fs.writeFileSync(outPath, JSON.stringify(out));
  await browser.close();
  server.close();
})();

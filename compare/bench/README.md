# Benchmark

These scripts score all four networks on identical digits and write the numbers that the Compare tab (`compare/index.html`) and the README table show. Each network is run through its own page in headless Chrome, with that page's own pen, preprocessing and forward pass.

## What it measures

- **MNIST**: the 10,000 test images, fed straight into each network with no drawing involved. Gemini's MLP trains each time its page loads, so `bench.js` reads its weights from the page after training. Because that training runs on canvas renderings, other browsers can give slightly different weights.
- **Pen-drawn digits**: 400 MNIST test digits re-traced as pen strokes, the same picks as `claude/training/build_page.py` (seed 7).
- **Style digits**: 600 hand-designed digits in 24 styles (`claude/training/styles.py`, `make_set(n_per=25, seed=123)`).
- **Drawing size**: the pen-drawn and style digits again at half and a quarter of their normal size, near the pad's top-left corner. Normal fills about 70% of the pad's width.

Strokes are snapped to each pad's pixel grid, as a mouse would be: 310 px for Claude's pads and 280 px for Gemini's.

Two tests favour one side. The pen-drawn digits are re-drawn the way Claude's networks were trained to expect. Gemini's CNN trained on digits rendered from the same 24 style templates, and chose its best epoch partly by its score on them, so its style score is not held out. MNIST is the neutral test.

## Run it

You need Python 3 with NumPy and numba, Node 18 or later, and Chrome or Chromium. You also need `prep.npz`; see [`claude/training/README.md`](../../claude/training/README.md) for how to make it.

```bash
cd compare/bench
npm install                                    # puppeteer-core
python3 make_cases.py ../../claude/training/prep.npz cases.json
node bench.js cases.json results.json          # under a minute; CHROME=/path/to/chrome to override
python3 score.py cases.json results.json ../../claude/training/prep.npz summary.json
python3 build_page.py summary.json             # puts the numbers into ../index.html
```

- `make_cases.py` writes the test digits as strokes in a unit square.
- `bench.js` serves the repo locally, opens each page and draws every digit at each size. It records each network's probabilities and the browser version. Gemini's CNN page keeps its network inside a closure, so `bench.js` adds a small test hook to that page as it serves it; the file on disk is not changed.
- `score.py` turns the probabilities into accuracies and adds MNIST, computed from each network's weights. It writes `summary.json`, the only output kept in the repo.
- `build_page.py` embeds the summary in the Compare page and refills every number in its text that is marked `data-num`. The prose around the numbers is not rewritten, so read it again after a re-run.

The results were measured on 27 November 2026 in Chrome 153.

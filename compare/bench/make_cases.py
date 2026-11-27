"""Benchmark cases, identical for every network.

  mnist400   the 400 MNIST test digits re-drawn as pen strokes (the same picks as
             claude/training/build_page.py, so the README's earlier figures carry over)
  styles600  600 hand-designed digits in 24 styles (styles.make_set(n_per=25, seed=123),
             the set train.py reports)

Strokes are stored in a unit square, placed the way a person draws a digit across most
of the pad, so each app can draw them on its own canvas with its own pen.

Usage: python3 make_cases.py ../../claude/training/prep.npz cases.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "claude", "training"))
import styles  # noqa: E402


def main(prep_path, out_path):
    d = np.load(prep_path)
    te_all, te_off, yte = d["te_all"], d["te_off"], d["yte"].astype(int)
    pick = np.random.default_rng(7).choice(len(yte), 400, replace=False)
    cases = []
    for i in pick:
        segs = te_all[te_off[i]:te_off[i + 1]].astype(np.float64) / 28.0       # 28-pixel frame -> unit square
        strokes = [[[a, b]] if (a == c and b == e) else [[a, b], [c, e]] for a, b, c, e in segs.tolist()]
        cases.append({"set": "mnist400", "label": int(yte[i]), "strokes": strokes})
    for strokes, label, name in styles.make_set(n_per=25, seed=123):
        unit = [[[0.15 + 0.7 * x / 100, 0.15 + 0.7 * y / 100] for x, y in np.asarray(s, float).tolist()] for s in strokes]
        cases.append({"set": "styles600", "style": name, "label": int(label), "strokes": unit})
    json.dump({"cases": cases}, open(out_path, "w"))
    print(f"{len(cases)} cases: {sum(c['set'] == 'mnist400' for c in cases)} mnist400, "
          f"{sum(c['set'] == 'styles600' for c in cases)} styles600")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

"""Embed the trained network in the page and write test fixtures.

Usage: python3 build_page.py runs/m64x32.npz out/
"""
import base64
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import styles  # noqa: E402
from strokes import normalize, render_batch  # noqa: E402

R0, S, TX, TY = 1.2, 4, 14.5, 14.5
EXAMPLE_STYLES = ["0-oval", "1-flag", "2-classic", "3-bumps", "4-open", "5-two-stroke",
                  "6-classic", "7-crossed", "8-figure", "9-stick"]
FIRST_EXAMPLE = 4


def fwd(X, P):
    z1 = X @ P["W1"].T.astype(np.float64) + P["b1"]
    a1 = np.maximum(z1, 0)
    z2 = a1 @ P["W2"].T.astype(np.float64) + P["b2"]
    a2 = np.maximum(z2, 0)
    z3 = a2 @ P["W3"].T.astype(np.float64) + P["b3"]
    return z1, a1, z2, a2, z3


def softmax(z, T):
    z = z / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def main(weights_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    w = np.load(weights_path)
    metrics = json.load(open(weights_path.replace(".npz", ".json")))
    P = {k: w[k].astype(np.float32) for k in ("W1", "b1", "W2", "b2", "W3", "b3")}
    T = float(w["T"])

    d = np.load(os.path.join(HERE, "prep.npz"))
    te_all, te_off, yte = d["te_all"], d["te_off"], d["yte"].astype(int)
    n_te = len(yte)
    Xr = np.zeros((n_te, 28, 28), np.float32)
    render_batch(te_all, te_off, np.arange(n_te), np.zeros((n_te, 8)), np.full(n_te, R0), S, TX, TY, False, Xr,
                 np.zeros((n_te, 3)))
    Xr = Xr.reshape(n_te, 784).astype(np.float64)
    before = fwd(Xr, P)[4]

    # Re-order hidden neurons so the diagram's wiring crosses less. A pure
    # relabelling: the function is unchanged (checked below).
    w3 = np.maximum(P["W3"], 0)                       # (10, 32)
    bary2 = (w3 * np.arange(10)[:, None]).sum(0) / (w3.sum(0) + 1e-9)
    perm2 = np.argsort(bary2, kind="stable")
    P["W2"], P["b2"], P["W3"] = P["W2"][perm2], P["b2"][perm2], P["W3"][:, perm2]
    w2 = np.maximum(P["W2"], 0)                       # (32, 64)
    bary1 = (w2 * np.arange(w2.shape[0])[:, None]).sum(0) / (w2.sum(0) + 1e-9)
    perm1 = np.argsort(bary1, kind="stable")
    P["W1"], P["b1"], P["W2"] = P["W1"][perm1], P["b1"][perm1], P["W2"][:, perm1]
    z1, a1, z2, a2, after = fwd(Xr, P)
    assert np.allclose(before, after, atol=1e-9), "re-ordering changed the network"
    acc_redraw = float((after.argmax(1) == yte).mean())
    print(f"re-ordered neurons; outputs identical (max diff {np.abs(before - after).max():.2e}); "
          f"redraw test acc {acc_redraw:.4f}")

    # Brightness scale for neuron dots: 95th percentile of positive activations
    scale1 = float(np.percentile(a1[a1 > 0], 95))
    scale2 = float(np.percentile(a2[a2 > 0], 95))

    # Examples: one per digit, in varied styles, all read correctly and confidently
    rng = np.random.default_rng(2026)
    examples = []
    for name in EXAMPLE_STYLES:
        label = int(name.split("-")[0])
        base = styles.TEMPLATES[label][name.split("-", 1)[1]]
        for attempt in range(50):
            strokes = styles.jitter(base, rng)
            strokes = [np.round(s, 1) for s in strokes]
            img = normalize(styles.strokes_to_segs(strokes), R0, S, TX, TY).ravel().astype(np.float64)
            pr = softmax(fwd(img[None], P)[4], T)[0]
            if pr.argmax() == label and pr[label] >= 0.95:
                break
        else:
            raise SystemExit(f"no confident example for {name}")
        examples.append({"label": label, "style": name, "p": round(float(pr[label]), 4),
                         "strokes": [s.tolist() for s in strokes]})
    print("examples:", [(e["style"], e["p"], sum(len(s) for s in e["strokes"])) for e in examples])

    blob = b"".join(P[k].astype("<f4").tobytes() for k in ("W1", "b1", "W2", "b2", "W3", "b3"))
    n_params = sum(P[k].size for k in P)
    meta = {"sizes": [784, P["W1"].shape[0], P["W2"].shape[0], P["W3"].shape[0]],
            "T": T, "R": R0, "S": S, "TX": TX, "TY": TY,
            "scale1": round(scale1, 4), "scale2": round(scale2, 4), "firstExample": FIRST_EXAMPLE}

    # Neuron roles measured by neuron_roles.py on these exact (re-ordered) weights
    import hashlib
    roles = None
    roles_path = os.path.join(out_dir, "roles.json")
    if os.path.exists(roles_path):
        roles = json.load(open(roles_path))
        if roles.get("w1hash") != hashlib.sha1(P["W1"].astype("<f4").tobytes()).hexdigest()[:16]:
            print("WARNING: roles.json was measured on different weights; run neuron_roles.py, then build again")
            roles = None
    else:
        print("note: no roles.json yet; run neuron_roles.py on out/final_weights.npz, then build again for tooltips")

    page = open(os.path.join(HERE, "page_template.html")).read()
    fills = {
        "__ROLES__": json.dumps(roles, separators=(",", ":")) if roles else "null",
        "__META__": json.dumps(meta),
        "__EXAMPLES__": json.dumps([{"label": e["label"], "strokes": e["strokes"]} for e in examples],
                                   separators=(",", ":")),
        "__MODEL__": base64.b64encode(blob).decode(),
        "__PARAMS__": f"{n_params:,}",
        "__TEST_ACC__": f"{metrics['test_orig'] * 100:.1f}%",
        "__REDRAW_ACC__": f"{acc_redraw * 100:.1f}%",
    }
    for k, v in fills.items():
        assert page.count(k) == 1, k
        page = page.replace(k, v)
    out_html = os.path.join(out_dir, "neural-digit-reader.html")
    open(out_html, "w").write(page)
    print(f"wrote {out_html} ({len(page) / 1024:.0f} KB), params {n_params:,}, meta {meta}")

    # Standalone document for the repo (the template's head part + body part, plus
    # the small reset the artifact host would otherwise provide).
    cut = page.index("</style>") + len("</style>")
    head = page[:cut].replace('<meta name="claude:viewport-fit-sides" content="cover">\n', "")
    standalone = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        "<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);"
        "padding-bottom:env(safe-area-inset-bottom,0px)}body{margin:0}[hidden]{display:none!important}</style>\n"
        f"{head}\n</head>\n<body>\n{page[cut:].strip()}\n</body>\n</html>\n")
    open(os.path.join(out_dir, "index.html"), "w").write(standalone)
    print(f"wrote {os.path.join(out_dir, 'index.html')} (standalone, {len(standalone) / 1024:.0f} KB)")

    # Fixtures for the browser parity test: the same strokes, run through Python.
    pick = np.random.default_rng(7).choice(n_te, 400, replace=False)
    cases = []
    for i in pick:
        segs = te_all[te_off[i]:te_off[i + 1]].astype(np.float64)
        strokes = [[[a, b]] if (a == c and b == e) else [[a, b], [c, e]] for a, b, c, e in segs.tolist()]
        cases.append({"kind": "mnist-redraw", "label": int(yte[i]), "strokes": strokes})
    for strokes, label, name in styles.make_set(n_per=4, seed=99):
        cases.append({"kind": name, "label": int(label), "strokes": [s.tolist() for s in strokes]})
    for e in examples:
        cases.append({"kind": "example-" + e["style"], "label": e["label"], "strokes": e["strokes"]})
    for c in cases:
        raw = styles.strokes_to_segs([np.array(s, float) for s in c["strokes"]])
        img = normalize(raw, R0, S, TX, TY).ravel()
        c["img"] = img.astype(np.float64).tolist()
        z = fwd(img.astype(np.float64)[None], P)[4]
        c["p"] = softmax(z, T)[0].tolist()
    json.dump({"cases": cases, "examples": examples}, open(os.path.join(out_dir, "fixtures.json"), "w"))
    print(f"wrote {len(cases)} parity cases")
    np.savez(os.path.join(out_dir, "final_weights.npz"), **P, T=np.float32(T))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

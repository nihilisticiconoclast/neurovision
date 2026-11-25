"""Embed the trained convolutional network in its page and write test fixtures.

Usage: python3 cnn_build_page.py runs/cnn8x16.npz out/cnn/
"""
import base64
import hashlib
import json
import os
import sys

import numpy as np
from scipy.signal import correlate2d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import styles  # noqa: E402
from cnn_train import _col2im, forward, pool, unpool_relu  # noqa: E402
from strokes import normalize, render_batch  # noqa: E402

R0, S, TX, TY = 1.2, 4, 14.5, 14.5
EXAMPLE_STYLES = ["0-oval", "1-flag", "2-classic", "3-bumps", "4-open", "5-two-stroke",
                  "6-classic", "7-crossed", "8-figure", "9-stick"]
FIRST_EXAMPLE = 4
KEYS = ("K1", "b1", "K2", "b2", "W3", "b3")


def as64(P):
    return {k: P[k].astype(np.float64) for k in KEYS}


def scores(X, P64, bs=500):
    """Float64 forward pass, as the page computes it (float32 weights, float64 sums)."""
    X = np.asarray(X, np.float64).reshape(-1, 28, 28)
    return np.concatenate([forward(X[i:i + bs], P64)[1] for i in range(0, len(X), bs)])


def softmax(z, T):
    z = z / T
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def sensitivity(img, P64, kind, idx):
    """d(score of digit idx) / d pixels, or d(total of map idx) / d pixels, with the network's
    on/off pattern and pooling winners fixed. Reference for the page's explain()."""
    C1, C2 = len(P64["b1"]), len(P64["b2"])
    cache, _ = forward(np.asarray(img, np.float64).reshape(1, 28, 28), P64)
    if kind == "m1":
        dz1 = np.zeros_like(cache["z1"])
        dz1[..., idx] = cache["z1"][..., idx] > 0
    else:
        if kind == "out":
            dp2 = P64["W3"][idx].reshape(1, C2, 7, 7).transpose(0, 2, 3, 1)
            dz2 = unpool_relu(np.ascontiguousarray(dp2), cache["i2"], cache["z2"])
        else:
            dz2 = np.zeros_like(cache["z2"])
            dz2[..., idx] = cache["z2"][..., idx] > 0
        dc2 = (dz2.reshape(196, C2) @ P64["K2"].reshape(C2, -1)).reshape(1, 14, 14, C1, 5, 5)
        dpp = np.zeros((1, 18, 18, C1))
        _col2im(dc2, dpp)
        dz1 = unpool_relu(np.ascontiguousarray(dpp[:, 2:16, 2:16, :]), cache["i1"], cache["z1"])
    dc1 = (dz1.reshape(784, C1) @ P64["K1"].reshape(C1, -1)).reshape(1, 28, 28, 1, 5, 5)
    dxp = np.zeros((1, 32, 32, 1))
    _col2im(dc1, dxp)
    return dxp[0, 2:30, 2:30, 0].ravel()


def paths(img, P64, kind, idx):
    """Path contributions behind the diagram's wires and the inspector's chips, computed a
    second way (SciPy correlations) as a reference for the page's explain(): Ein[i], E12[j][i],
    E23[k][j] and the target's total, in the page's layouts."""
    C1, C2 = len(P64["b1"]), len(P64["b2"])
    cache, z3 = forward(np.asarray(img, np.float64).reshape(1, 28, 28), P64)
    z1 = cache["z1"][0].transpose(2, 0, 1)
    p1 = pool(np.maximum(cache["z1"], 0))[0][0].transpose(2, 0, 1)
    z2 = cache["z2"][0].transpose(2, 0, 1)
    p2 = cache["f"][0].reshape(C2, 7, 7)
    W3 = P64["W3"].reshape(10, C2, 7, 7)

    def part(j, i):
        return correlate2d(p1[i], P64["K2"][j, i], mode="same")

    def dz2(k):
        dp2 = P64["W3"][k].reshape(1, C2, 7, 7).transpose(0, 2, 3, 1)
        return unpool_relu(np.ascontiguousarray(dp2), cache["i2"], cache["z2"])[0].transpose(2, 0, 1)
    Ein, E12, E23 = np.zeros(C1), np.zeros((C2, C1)), np.zeros((10, C2))
    if kind == "out":
        g = dz2(idx)
        E23[idx] = (W3[idx] * p2).sum((1, 2))
        for j in range(C2):
            for i in range(C1):
                E12[j, i] = (g[j] * part(j, i)).sum()
        Ein = E12.sum(0)
        total = z3[0, idx]
    elif kind == "m2":
        on = z2[idx] > 0
        for i in range(C1):
            E12[idx, i] = (on * part(idx, i)).sum()
        Ein = E12[idx].copy()
        E23[:, idx] = (W3[:, idx] * p2[idx]).sum((1, 2))
        total = z2[idx][on].sum()
    else:
        total = z1[idx][z1[idx] > 0].sum()
        Ein[idx] = total
        gs = [dz2(k) for k in range(10)]
        for j in range(C2):
            pt = part(j, idx)
            E12[j, idx] = ((z2[j] > 0) * pt).sum()
            for k in range(10):
                E23[k, j] = (gs[k][j] * pt).sum()
    return {"Ein": Ein.tolist(), "E12": E12.ravel().tolist(), "E23": E23.ravel().tolist(), "total": float(total)}


def fraction_words(small, big):
    r = big / small
    n = int(round(r))
    words = {2: "half", 3: "a third", 4: "a quarter", 5: "a fifth", 6: "a sixth", 7: "a seventh", 8: "an eighth"}
    return f"about {words[n]}" if n in words and abs(r - n) < 0.35 else f"{small / big:.0%}"


def main(weights_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    w = np.load(weights_path)
    metrics = json.load(open(weights_path.replace(".npz", ".json")))
    P = {k: w[k].astype(np.float32) for k in KEYS}
    T = float(w["T"])
    C1, C2 = len(P["b1"]), len(P["b2"])

    d = np.load(os.path.join(HERE, "prep.npz"))
    te_all, te_off, yte = d["te_all"], d["te_off"], d["yte"].astype(int)
    n_te = len(yte)
    Xr = np.zeros((n_te, 28, 28), np.float32)
    render_batch(te_all, te_off, np.arange(n_te), np.zeros((n_te, 8)), np.full(n_te, R0), S, TX, TY, False, Xr,
                 np.zeros((n_te, 3)))
    before = scores(Xr, as64(P))

    # Re-order filters so the diagram's wiring crosses less. A pure relabelling:
    # the function is unchanged (checked below).
    W3 = P["W3"].reshape(10, C2, 49)
    w3 = np.maximum(W3, 0).sum(2)                                    # (10, C2)
    bary2 = (w3 * np.arange(10)[:, None]).sum(0) / (w3.sum(0) + 1e-9)
    perm2 = np.argsort(bary2, kind="stable")
    P["K2"], P["b2"], P["W3"] = P["K2"][perm2], P["b2"][perm2], np.ascontiguousarray(W3[:, perm2]).reshape(10, -1)
    w2 = np.maximum(P["K2"], 0).sum((2, 3))                           # (C2, C1)
    bary1 = (w2 * np.arange(C2)[:, None]).sum(0) / (w2.sum(0) + 1e-9)
    perm1 = np.argsort(bary1, kind="stable")
    P["K1"], P["b1"], P["K2"] = P["K1"][perm1], P["b1"][perm1], np.ascontiguousarray(P["K2"][:, perm1])
    P64 = as64(P)
    after = scores(Xr, P64)
    assert np.allclose(before, after, atol=1e-9), "re-ordering changed the network"
    acc_redraw = float((after.argmax(1) == yte).mean())
    acc_orig = float((scores(d["Xte"] / 255.0, P64).argmax(1) == yte).mean())
    print(f"re-ordered filters; outputs identical (max diff {np.abs(before - after).max():.2e}); "
          f"redraw test acc {acc_redraw:.4f}, MNIST test acc {acc_orig:.4f}")

    # Brightness scale per map: 95th percentile of its positive activations on test digits
    sub = np.concatenate([Xr[:2000], d["Xte"][:2000].astype(np.float32) / 255.0])
    cache, _ = forward(sub, P)
    a1, a2 = np.maximum(cache["z1"], 0), np.maximum(cache["z2"], 0)
    scale1 = [round(float(np.percentile(a1[..., c][a1[..., c] > 0], 95)), 4) for c in range(C1)]
    scale2 = [round(float(np.percentile(a2[..., c][a2[..., c] > 0], 95)), 4) for c in range(C2)]
    silent1 = [(a1[..., c].reshape(len(sub), -1).max(1) > 0).mean() for c in range(C1)]
    silent2 = [(a2[..., c].reshape(len(sub), -1).max(1) > 0).mean() for c in range(C2)]
    print("share of digits where each map fires somewhere: L1", np.round(silent1, 3), "L2", np.round(silent2, 3))

    # Examples: one per digit, in varied styles, all read correctly and confidently
    rng = np.random.default_rng(2026)
    examples = []
    for name in EXAMPLE_STYLES:
        label = int(name.split("-")[0])
        base = styles.TEMPLATES[label][name.split("-", 1)[1]]
        for attempt in range(50):
            strokes = styles.jitter(base, rng)
            strokes = [np.round(s, 1) for s in strokes]
            img = normalize(styles.strokes_to_segs(strokes), R0, S, TX, TY)
            pr = softmax(scores(img, P64), T)[0]
            if pr.argmax() == label and pr[label] >= 0.95:
                break
        else:
            raise SystemExit(f"no confident example for {name}")
        examples.append({"label": label, "style": name, "p": round(float(pr[label]), 4),
                         "strokes": [s.tolist() for s in strokes]})
    print("examples:", [(e["style"], e["p"], sum(len(s) for s in e["strokes"])) for e in examples])

    blob = b"".join(P[k].astype("<f4").tobytes() for k in KEYS)
    n_params = sum(P[k].size for k in KEYS)
    dense = np.load(os.path.join(HERE, "final_weights.npz"))
    n_dense = sum(dense[k].size for k in ("W1", "b1", "W2", "b2", "W3", "b3"))
    meta = {"c1": C1, "c2": C2, "T": T, "R": R0, "S": S, "TX": TX, "TY": TY,
            "scale1": scale1, "scale2": scale2, "firstExample": FIRST_EXAMPLE}

    # Filter roles measured by cnn_roles.py on these exact (re-ordered) weights
    roles = None
    roles_path = os.path.join(out_dir, "roles.json")
    khash = hashlib.sha1(P["K1"].astype("<f4").tobytes() + P["K2"].astype("<f4").tobytes()).hexdigest()[:16]
    if os.path.exists(roles_path):
        roles = json.load(open(roles_path))
        if roles.get("khash") != khash:
            print("WARNING: roles.json was measured on different weights; run cnn_roles.py, then build again")
            roles = None
    else:
        print("note: no roles.json yet; run cnn_roles.py on out/cnn/final_weights.npz, then build again for tooltips")

    page = open(os.path.join(HERE, "cnn_page_template.html")).read()
    fills = {
        "__ROLES__": json.dumps(roles, separators=(",", ":")) if roles else "null",
        "__META__": json.dumps(meta),
        "__EXAMPLES__": json.dumps([{"label": e["label"], "strokes": e["strokes"]} for e in examples],
                                   separators=(",", ":")),
        "__MODEL__": base64.b64encode(blob).decode(),
        "__PARAMS__": f"{n_params:,}",
        "__PARAMS_NOTE__": f"only {n_params:,} weights, {fraction_words(n_params, n_dense)} of the dense network’s {n_dense:,}",
        "__TEST_ACC__": f"{metrics['test_orig'] * 100:.1f}%",
        "__REDRAW_ACC__": f"{acc_redraw * 100:.1f}%",
    }
    for k, v in fills.items():
        assert page.count(k) == 1, k
        page = page.replace(k, v)
    out_html = os.path.join(out_dir, "convolutional-digit-reader.html")
    open(out_html, "w").write(page)
    print(f"wrote {out_html} ({len(page) / 1024:.0f} KB), params {n_params:,} (dense {n_dense:,}), "
          f"T {T}, scales {scale1} / {scale2}")

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
    pick = np.random.default_rng(7).choice(n_te, 400, replace=False)      # same 400 digits as build_page.py
    cases = []
    for i in pick:
        segs = te_all[te_off[i]:te_off[i + 1]].astype(np.float64)
        strokes = [[[a, b]] if (a == c and b == e) else [[a, b], [c, e]] for a, b, c, e in segs.tolist()]
        cases.append({"kind": "mnist-redraw", "label": int(yte[i]), "strokes": strokes})
    for strokes, label, name in styles.make_set(n_per=4, seed=99):
        cases.append({"kind": name, "label": int(label), "strokes": [s.tolist() for s in strokes]})
    for e in examples:
        cases.append({"kind": "example-" + e["style"], "label": e["label"], "strokes": e["strokes"]})
    erng = np.random.default_rng(11)
    for n, c in enumerate(cases):
        raw = styles.strokes_to_segs([np.array(s, float) for s in c["strokes"]])
        img = normalize(raw, R0, S, TX, TY).ravel()
        c["img"] = img.astype(np.float64).tolist()
        z = scores(img, P64)
        c["p"] = softmax(z, T)[0].tolist()
        if n % 10 == 0:   # evidence fixtures: the guess, one layer-2 map, one layer-1 map
            c["evidence"] = []
            for kind, idx in (("out", int(z.argmax())), ("m2", int(erng.integers(C2))), ("m1", int(erng.integers(C1)))):
                c["evidence"].append({"kind": kind, "idx": idx, "gx": sensitivity(img, P64, kind, idx).tolist(),
                                      **paths(img, P64, kind, idx)})
    acc400 = float(np.mean([np.argmax(c["p"]) == c["label"] for c in cases[:400]]))
    json.dump({"cases": cases, "examples": examples, "acc400": acc400}, open(os.path.join(out_dir, "fixtures.json"), "w"))
    print(f"wrote {len(cases)} parity cases; the 400 re-drawn test digits: {acc400:.4f}")
    np.savez(os.path.join(out_dir, "final_weights.npz"), **P, T=np.float32(T))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

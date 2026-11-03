"""Measure what each hidden neuron responds to, and name it.

For 20,000 test digits (originals plus the same digits re-drawn as pen strokes):
  - score each digit on interpretable shape features (closed loops and where they sit,
    straight strokes and their angle, bars, where strokes end, open sides, ...);
  - record every hidden neuron's activation;
  - rank-correlate each neuron with each feature, and compute which digits it fires for
    and its typical trigger (the activation-weighted average input).
Writes roles_raw.json (all measurements) and contact sheets for review.

Usage: python3 neuron_roles.py out/final_weights.npz out/
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.stats import rankdata
from skimage.morphology import skeletonize
from skimage.transform import radon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from strokes import render_batch  # noqa: E402

TH = np.arange(0, 180, 5)
VERT = np.isin(TH, [170, 175, 0, 5, 10])
SLASH = (TH >= 125) & (TH <= 165)      # "/" (checked on synthetic strokes)
BACK = (TH >= 15) & (TH <= 55)         # "\"
EIGHT = np.ones((3, 3), int)

FEATURES = {
    # name: (label, phrase used in sentences)
    "loop_top": ("Loop at the top", "a closed loop in the upper half"),
    "loop_bottom": ("Loop at the bottom", "a closed loop in the lower half"),
    "loop_big": ("One big loop", "a single large loop round an empty middle"),
    "straight_vert": ("Upright straight stroke", "a long straight upright stroke"),
    "straight_slash": ("Stroke slanting /", "a long straight stroke leaning right, like /"),
    "straight_back": ("Stroke slanting \\", "a long straight stroke leaning left, like \\"),
    "bar_top": ("Bar across the top", "a flat bar along the top"),
    "bar_mid": ("Bar across the middle", "a horizontal bar through the middle"),
    "bar_bottom": ("Stroke along the bottom", "a stroke running along the bottom"),
    "end_tl": ("Stroke ending top left", "a loose stroke end at the top left"),
    "end_tr": ("Stroke ending top right", "a loose stroke end at the top right"),
    "end_bl": ("Stroke ending bottom left", "a loose stroke end at the bottom left"),
    "end_br": ("Stroke ending bottom right", "a loose stroke end at the bottom right"),
    "no_ends": ("Closed shape", "a closed shape with no loose ends"),
    "junctions": ("Strokes crossing or joining", "strokes that cross or join"),
    "empty_centre": ("Empty middle", "an empty middle"),
    "ink_centre": ("Ink through the middle", "ink right through the middle"),
    "narrow": ("Tall, thin shape", "a tall, thin shape"),
    "wide": ("Wide shape", "a wide shape"),
    "open_left": ("Open on the left", "a shape that is open on its left side"),
    "open_right": ("Open on the right", "a shape that is open on its right side"),
    "open_top": ("Open at the top", "a shape that is open at the top"),
    "open_bottom": ("Open at the bottom", "a shape that is open at the bottom"),
    "lots_of_ink": ("Lots of ink", "a lot of ink overall"),
}
NAMES = list(FEATURES)

# Reviewed by eye against the contact sheets (typical trigger, top-firing digits, digit
# preferences). Where the top-correlated feature misdescribed the neuron, the label is
# replaced: either by another measured feature (a string) or by a curated (label, phrase).
OVERRIDES = {
    (1, 4): ("Diagonal through the middle", "a stroke running diagonally through the middle"),
    (1, 7): ("Curl at the bottom", "a curl or loop low down, as in 6s and curly 2s"),
    (1, 14): ("Leaning upright stroke", "a single upright stroke leaning right with nothing across the top, as in 1"),
    (1, 23): "no_ends",
    (1, 27): ("Curves bulging right", "curves that bulge out to the right, as in 3"),
    (1, 31): ("Belly at the bottom right", "a rounded belly at the bottom right, as in 3, 5 and 6"),
    (1, 32): "loop_big",
    (1, 38): "straight_slash",
    (1, 46): ("Loop low down", "a round loop in the lower part, as in 6, 0 and curly 2s"),
    (1, 47): ("Narrow upright shapes", "narrow, upright shapes, though not plain 1s"),
    (1, 49): ("Bowl at the bottom", "a rounded bowl at the bottom, as in 6 and 5"),
    (1, 57): ("Corner then bowl", "a top-left corner leading into a bowl below, as in 5"),
    (1, 60): ("Cap with a stem below", "a cap across the top with a stem dropping below, as in 7 and 9"),
    (2, 8): ("Mixed: 3s and 4s", "a mix of 3 and 4 shapes rather than one clean feature"),
    (2, 20): "open_right",
}


def finalize(raw, W1, out_path):
    """Turn measurements into the page's tooltip data: label, sentence, footnote,
    digit preferences and a 4-bit typical-trigger image per hidden neuron."""
    out = {"h1": [], "h2": []}
    packed = bytearray()
    for ro in raw:
        key = (ro["layer"], ro["idx"] + 1)
        ov = OVERRIDES.get(key, ro["top"][0])
        label, phrase = FEATURES[ov] if isinstance(ov, str) else ov
        pref = np.array(ro["pref"])
        top = [int(k) for k in np.argsort(pref)[::-1] if pref[k] >= 0.6][:3]
        digits = top[0:1] if len(top) == 1 else top
        dtxt = str(digits[0]) if len(digits) == 1 else ", ".join(map(str, digits[:-1])) + f" and {digits[-1]}"
        text = f"Responds to {phrase}. Fires most for {dtxt}."
        neg = ro["neg"]
        foot = f"Active for {ro['rate']:.0%} of digits"
        if ro["corr"][neg] <= -0.3:
            foot += f" · held back by {FEATURES[neg][1]}"
        out["h1" if ro["layer"] == 1 else "h2"].append(
            {"label": label, "text": text, "foot": foot, "pref": [round(v, 2) for v in ro["pref"]], "top": digits})
        sta = np.array(ro["sta"])
        q = np.round((sta - sta.min()) / (sta.max() - sta.min() + 1e-12) * 15).astype(np.uint8)
        packed.extend(((q[0::2] << 4) | q[1::2]).tolist())
    import base64
    out["img"] = base64.b64encode(bytes(packed)).decode()
    import hashlib
    out["w1hash"] = hashlib.sha1(np.asarray(W1, "<f4").tobytes()).hexdigest()[:16]   # ties labels to these exact weights
    json.dump(out, open(out_path, "w"), separators=(",", ":"))
    return out


def longest_run(row):
    best = cur = 0
    for v in row:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def features(x):
    b = x > 0.4
    rows = np.nonzero(b.any(1))[0]
    cols = np.nonzero(b.any(0))[0]
    f = dict.fromkeys(NAMES, 0.0)
    if len(rows) == 0:
        return [0.0] * len(NAMES)
    top, bottom, left, right = rows[0], rows[-1], cols[0], cols[-1]
    h, w = bottom - top + 1, right - left + 1

    lab, n = ndimage.label(~b)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    for k in range(1, n + 1):
        if k in border:
            continue
        ys, xs = np.nonzero(lab == k)
        area = len(ys)
        if area < 3:
            continue
        r = (ys.mean() - top) / h
        if area / (h * w) > 0.18 and 0.3 < r < 0.7:
            f["loop_big"] = max(f["loop_big"], min(1.0, area / 40))
        elif r < 0.5:
            f["loop_top"] = max(f["loop_top"], min(1.0, area / 15))
        else:
            f["loop_bottom"] = max(f["loop_bottom"], min(1.0, area / 15))

    peak = radon(x, theta=TH, circle=False).max(0) / 20
    f["straight_vert"] = peak[VERT].max()
    f["straight_slash"] = peak[SLASH].max()
    f["straight_back"] = peak[BACK].max()

    def band_run(a, c):
        rr = range(int(top + a * h), min(bottom, int(top + c * h)) + 1)
        return min(1.0, max(longest_run(b[r]) for r in rr) / 14) if len(rr) else 0.0
    f["bar_top"] = band_run(0.0, 0.22)
    f["bar_mid"] = band_run(0.38, 0.62)
    f["bar_bottom"] = band_run(0.78, 1.0)

    sk = skeletonize(b)
    nb = ndimage.convolve(sk.astype(int), EIGHT, mode="constant") - sk
    ey, ex = np.nonzero(sk & (nb == 1))
    rx = (ex - left) / max(w - 1, 1)
    ry = (ey - top) / max(h - 1, 1)
    f["end_tl"] = float(np.any((ry < 0.35) & (rx < 0.5)))
    f["end_tr"] = float(np.any((ry < 0.35) & (rx >= 0.5)))
    f["end_bl"] = float(np.any((ry > 0.65) & (rx < 0.5)))
    f["end_br"] = float(np.any((ry > 0.65) & (rx >= 0.5)))
    f["no_ends"] = float(len(ey) == 0)
    f["junctions"] = min(ndimage.label(sk & (nb >= 3), structure=EIGHT)[1], 3) / 3

    c = x[11:17, 11:17].mean()
    f["empty_centre"], f["ink_centre"] = 1 - c, c
    f["narrow"], f["wide"] = 1 - w / 20, w / 20
    mid_r = range(int(top + 0.25 * h), int(top + 0.75 * h) + 1)
    mid_c = range(int(left + 0.25 * w), int(left + 0.75 * w) + 1)
    ol, orr, ot, ob = [], [], [], []
    for r in mid_r:
        xs = np.nonzero(b[r])[0]
        ol.append((xs[0] - left) / w if len(xs) else 1.0)
        orr.append((right - xs[-1]) / w if len(xs) else 1.0)
    for cc in mid_c:
        ys = np.nonzero(b[:, cc])[0]
        ot.append((ys[0] - top) / h if len(ys) else 1.0)
        ob.append((bottom - ys[-1]) / h if len(ys) else 1.0)
    f["open_left"], f["open_right"] = float(np.mean(ol)), float(np.mean(orr))
    f["open_top"], f["open_bottom"] = float(np.mean(ot)), float(np.mean(ob))
    f["lots_of_ink"] = min(1.0, x.sum() / 150)
    return [f[k] for k in NAMES]


def main(weights_path, out_dir):
    W = np.load(weights_path)
    d = np.load(os.path.join(HERE, "prep.npz"))
    yte = d["yte"].astype(int)
    n = len(yte)
    Xr = np.zeros((n, 28, 28), np.float32)
    render_batch(d["te_all"], d["te_off"], np.arange(n), np.zeros((n, 8)), np.full(n, 1.2), 4, 14.5, 14.5, False, Xr,
                 np.zeros((n, 3)))
    X = np.concatenate([Xr.reshape(n, 784), d["Xte"].reshape(n, 784) / 255.0]).astype(np.float64)
    y = np.concatenate([yte, yte])

    a1 = np.maximum(X @ W["W1"].T.astype(np.float64) + W["b1"], 0)
    a2 = np.maximum(a1 @ W["W2"].T.astype(np.float64) + W["b2"], 0)
    A = np.concatenate([a1, a2], 1)                       # (20000, 96): L1 0-63, L2 64-95

    from multiprocessing import Pool
    with Pool(2) as pool:
        F = np.array(pool.map(features, list(X.reshape(-1, 28, 28)), chunksize=500))
    print("features", F.shape, "means", dict(zip(NAMES, F.mean(0).round(2))), flush=True)

    rA = np.apply_along_axis(rankdata, 0, A)
    rF = np.apply_along_axis(rankdata, 0, F)
    zA = (rA - rA.mean(0)) / (rA.std(0) + 1e-9)
    zF = (rF - rF.mean(0)) / (rF.std(0) + 1e-9)
    corr = zA.T @ zF / len(X)                             # (96, n_features) Spearman

    roles = []
    for j in range(A.shape[1]):
        a = A[:, j]
        rate = float((a > 0).mean())
        by_digit = np.array([a[y == k].mean() for k in range(10)])
        pref = by_digit / (by_digit.max() + 1e-9)
        sta = (a[:, None] * X).sum(0) / (a.sum() + 1e-9)
        order = np.argsort(corr[j])[::-1]
        roles.append({
            "layer": 1 if j < 64 else 2, "idx": j if j < 64 else j - 64,
            "rate": round(rate, 3), "pref": pref.round(3).tolist(),
            "corr": {NAMES[k]: round(float(corr[j, k]), 3) for k in range(len(NAMES))},
            "top": [NAMES[k] for k in order[:3]], "neg": NAMES[int(np.argmin(corr[j]))],
            "sta": sta.round(4).tolist(),
            "examples": np.argsort(a)[::-1][:6].tolist(),
        })
    json.dump({"features": FEATURES, "roles": roles}, open(os.path.join(out_dir, "roles_raw.json"), "w"))
    final = finalize(roles, W["W1"], os.path.join(out_dir, "roles.json"))
    for layer in ("h1", "h2"):
        for i, r in enumerate(final[layer]):
            print(f"{layer.upper()} #{i + 1:<2} {r['label']:<28} {r['text'][12:]} | {r['foot']}")

    # Contact sheets for review: typical trigger, top examples, digit preference, proposed label
    def tile(img, s=3):
        return Image.fromarray((255 - np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((28 * s, 28 * s), Image.NEAREST)
    for sheet in range(0, 96, 16):
        canvas = Image.new("RGB", (1180, 16 * 92 + 8), "white")
        dr = ImageDraw.Draw(canvas)
        for r, j in enumerate(range(sheet, min(sheet + 16, 96))):
            ro = roles[j]
            y0 = 4 + r * 92
            sta = np.array(ro["sta"]).reshape(28, 28)
            canvas.paste(tile((sta - sta.min()) / (sta.max() - sta.min() + 1e-9)), (4, y0))
            for e, ix in enumerate(ro["examples"]):
                canvas.paste(tile(X[ix].reshape(28, 28)), (100 + e * 88, y0))
            bx = 640
            for k in range(10):
                hgt = int(60 * ro["pref"][k])
                dr.rectangle([bx + k * 14, y0 + 70 - hgt, bx + k * 14 + 10, y0 + 70], fill=(40, 40, 40))
                dr.text((bx + k * 14 + 2, y0 + 72), str(k), fill=(0, 0, 0))
            c = ro["corr"]
            txt = (f"L{ro['layer']} #{ro['idx'] + 1}  fires {ro['rate']:.0%}\n"
                   f"1: {FEATURES[ro['top'][0]][0]} ({c[ro['top'][0]]:+.2f})\n"
                   f"2: {FEATURES[ro['top'][1]][0]} ({c[ro['top'][1]]:+.2f})\n"
                   f"3: {FEATURES[ro['top'][2]][0]} ({c[ro['top'][2]]:+.2f})\n"
                   f"quiet: {FEATURES[ro['neg']][0]} ({c[ro['neg']]:+.2f})")
            dr.text((800, y0), txt, fill=(0, 0, 0))
        canvas.save(os.path.join(out_dir, f"roles_sheet_{sheet // 16 + 1}.png"))
    print("wrote roles_raw.json and 6 contact sheets")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

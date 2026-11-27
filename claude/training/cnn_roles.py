"""Measure what each filter of the convolutional network responds to, and name it.

Two kinds of measurement, both on the exact (re-ordered) weights used in the page:
  - Probes: small shapes drawn with the page's own pen (strokes.render, radius 1.2):
    straight strokes at 12 angles, stroke ends in 16 directions, corners, curves,
    junctions, crossings, loops and a dot. Each filter's response is read where the shape
    sits, averaged over 16 sub-pixel placements. Layer-1 filters are also swept across
    a stroke to find which side of it they fire on.
  - Digits: 20,000 test digits (originals plus the same digits re-drawn as pen strokes).
    For each filter: which digits make its map fire hardest (mean of each digit's strongest
    spot), how much of its map lights up, where its strongest spots fall (kept in the raw file
    only: most filters fire in several places, so a mean position misleads), and its typical
    trigger (the average patch of input around each firing spot, weighted by activation squared).
The label comes from the strongest probe (with the edge side for layer 1); labels were then
checked by eye against the contact sheets and corrected in OVERRIDES where they misdescribe
the filter. Writes roles_raw.json, roles.json (for the page) and contact sheets.

Usage: python3 cnn_roles.py out/cnn/final_weights.npz out/cnn/
"""
import base64
import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cnn_train import forward  # noqa: E402
from strokes import render, render_batch  # noqa: E402

R0, S = 1.2, 4
P1, P2 = 9, 16          # typical-trigger patch sides for layer 1 (5x5 field + margin) and layer 2 (14x14 field + margin)
DIRS = ["right", "up-right", "up", "up-left", "left", "down-left", "down", "down-right"]
CURVE = {"up": ("Top of a curve", "the top of a curve, bulging upwards"),
         "down": ("Bottom of a curve", "the bottom of a curve, bulging downwards"),
         "left": ("Left side of a curve", "the left side of a curve, bulging out to the left"),
         "right": ("Right side of a curve", "the right side of a curve, bulging out to the right"),
         "up-left": ("Curve bulging up and left", "a curve bulging up and to the left"),
         "up-right": ("Curve bulging up and right", "a curve bulging up and to the right"),
         "down-left": ("Curve bulging down and left", "a curve bulging down and to the left"),
         "down-right": ("Curve bulging down and right", "a curve bulging down and to the right")}
ORIENT = {0: ("flat", "a flat stroke"), 90: ("upright", "an upright stroke"),
          45: ("/", "a stroke slanting like /"), 135: ("\\", "a stroke slanting like \\")}

# Reviewed by eye against the contact sheets (filter weights, typical trigger, the digits that fire it
# hardest with the spot marked, probe responses, digit preferences). Where the strongest probe
# misdescribed the filter, or several filters would share one vague name, the label is replaced by a
# curated (label, phrase). Several layer-2 filters fire on empty space framed by strokes, which no
# single-stroke probe can show.
OVERRIDES = {
    # layer 1: broadly tuned (weakest angle still half its best), so not a flat-stroke detector
    (1, 4): ("Ink in any direction", "ink from a stroke at almost any angle"),
    # layer 1: peaks 1.75 px from the stroke's centre line, outside the ink (pen radius 1.2)
    (1, 8): ("Space above \\ strokes", "the blank space just above a stroke slanting like \\"),
    # layer 2: a / stroke slid across it peaks 1.5 px on its upper-left side (#5 peaks on or below the stroke)
    (2, 1): ("Upper side of / strokes", "a stroke slanting like /, firing along its upper side"),
    # layer 2: fires in empty space; typical trigger is a bar above and a stroke curving away below-left
    (2, 2): ("Pocket under a bar", "an empty pocket with a bar above it and a stroke curving away below, as in the top half of a 5"),
    # layer 2: empty space with a dark arch above and a lighter stroke below (mirror of #9)
    (2, 3): ("Gap under an arch", "the empty space just under a stroke that arches over it, as inside the curl of a 2"),
    # layer 2: strongest probe a corner pointing down-right; typical trigger shows the bend above-left of the spot
    (2, 4): ("Outside a bottom-right bend", "the outside of a stroke that bends round at the bottom right"),
    # layer 2: probes weak and unselective; fires where short bars meet other strokes (4s' crossings, middles of 3 and 5)
    (2, 6): ("Bar meeting a stroke mid-way", "a short flat stroke meeting another stroke halfway up"),
    # layer 2: a corner pointing right and a curve bulging right score equally; the spot sits just outside the bulge
    (2, 7): ("Bulge pointing right", "the outside of a curve or corner that points right, like ) or >"),
    # layer 2: strongest probe the bottom of a curve, but the spot is the gap above it (mirror of #3)
    (2, 9): ("Gap above a base", "the empty space just above the bottom of a curve or a flat base"),
    # layer 2: tuned to 60-75°, steeper than #1 and #5
    (2, 10): ("Steep strokes leaning right", "a long, steep stroke leaning right, like the stem of a 1 or 7"),
    (2, 11): ("Stroke end pointing right", "the loose end of a stroke pointing right, as at the end of a 5’s top bar"),
    # layer 2: strongest probe an upside-down T junction; typical trigger is a slant running into a flat base
    (2, 12): ("Slant meeting a base", "a slanting stroke meeting a flat base, as at the bottom of a 2"),
    # layer 2: probes weak; typical trigger is a stroke coming down and curling left along the bottom
    (2, 13): ("Inside a bottom-right bend", "the inside of a stroke that comes down and curls left along the bottom"),
    (2, 14): ("Top of an arch", "the top of a curve that arches over"),
    (2, 15): ("Upright joined from the left", "an upright stroke with another stroke joining it from the left, as where a 9’s loop meets its stem"),
    (2, 16): ("Point facing left (<)", "two strokes meeting in a point that faces left, like <, as in the middle of a 3"),
}


def u(phi):
    r = np.radians(phi)
    return np.array([np.cos(r), -np.sin(r)])     # screen coordinates: y grows downwards


def dir_name(phi):
    return DIRS[int(round((phi % 360) / 45)) % 8]


def probe_set(key):
    """(family, parameter, polylines) with the probe's key point at `key`."""
    out = []
    for th in range(0, 180, 15):
        out.append(("line", th, [[key - 8 * u(th), key + 8 * u(th)]]))
    for phi in np.arange(0, 360, 22.5):        # stroke leaves the key point towards phi; its free end points the other way
        out.append(("end", (phi + 180) % 360, [[key, key + 10 * u(phi)]]))
    for phi in range(0, 360, 45):              # vertex at the key point, pointing towards phi
        out.append(("corner", phi, [[key + 9 * u(phi + 135), key, key + 9 * u(phi - 135)]]))
    for phi in range(0, 360, 45):              # apex at the key point, bulging towards phi
        for r in (4.0, 6.5):
            c = key - r * u(phi)
            out.append(("curve", phi, [[c + r * u(t) for t in np.arange(phi - 75, phi + 75.1, 3)]]))
    for phi in range(0, 360, 45):              # bar through the key point, stem towards phi
        out.append(("tee", phi, [[key - 8 * u(phi + 90), key + 8 * u(phi + 90)], [key, key + 9 * u(phi)]]))
    for th in (0, 45):
        out.append(("cross", th, [[key - 8 * u(th), key + 8 * u(th)], [key - 8 * u(th + 90), key + 8 * u(th + 90)]]))
    for r in (3.0, 4.5):
        out.append(("loop", r, [[key + r * u(t) for t in np.arange(0, 360.1, 6)]]))
    out.append(("dot", 0, [[key, key]]))
    return out


def polys_to_img(polys):
    segs = []
    for pl in polys:
        pl = np.asarray(pl, float)
        segs.extend(np.concatenate([pl[:-1], pl[1:]], 1) if len(pl) > 1 else [np.concatenate([pl[0], pl[0]])])
    return render(np.asarray(segs, np.float64), R0, S)


def near_max(a, key, layer):
    """Strongest activation among spots within 2 px of the key point (spot centres in pixel units)."""
    if layer == 1:
        c = np.arange(28) + 0.5
    else:
        c = 2 * np.arange(14) + 1.0
    ys = np.abs(c - key[1]) <= 2.0
    xs = np.abs(c - key[0]) <= 2.0
    return a[np.ix_(ys, xs)].reshape(-1, a.shape[-1]).max(0)


def probe_responses(P):
    C1, C2 = len(P["b1"]), len(P["b2"])
    shifts = [(13.5 + 0.5 * i, 13.5 + 0.5 * j) for i in range(4) for j in range(4)]
    names, imgs, keys = None, [], []
    for sx, sy in shifts:
        key = np.array([sx, sy])
        ps = probe_set(key)
        if names is None:
            names = [(f, p) for f, p, _ in ps]
        for _, _, polys in ps:
            imgs.append(polys_to_img(polys))
            keys.append(key)
    X = np.stack(imgs)
    cache, _ = forward(X, P)
    a1, a2 = np.maximum(cache["z1"], 0), np.maximum(cache["z2"], 0)
    n = len(names)
    r1 = np.zeros((n, C1))
    r2 = np.zeros((n, C2))
    for m in range(len(X)):
        r1[m % n] += near_max(a1[m], keys[m], 1) / len(shifts)
        r2[m % n] += near_max(a2[m], keys[m], 2) / len(shifts)
    return names, r1, r2, X[:n]


def edge_profile(P, c, th):
    """Layer-1 filter c's response at a fixed spot as a stroke at angle th slides across it.
    Returns offsets d (along the stroke's normal, which points to the side named by the angle + 90°)
    and responses."""
    ds = np.arange(-3.5, 3.51, 0.25)
    key = np.array([14.5, 14.5])
    n = u(th + 90)
    imgs = [polys_to_img([[key + d * n - 9 * u(th), key + d * n + 9 * u(th)]]) for d in ds]
    cache, _ = forward(np.stack(imgs), P)
    return ds, np.maximum(cache["z1"][:, 14, 14, c], 0)


def side_words(th, d_peak):
    """The stroke sits at +d along normal u(th+90) from the spot, so the spot is on the -normal side."""
    side = dir_name(th + 90 + 180) if d_peak > 0 else dir_name(th + 90)
    return {"up": "top", "down": "bottom", "left": "left", "right": "right",
            "up-left": "upper", "up-right": "upper", "down-left": "lower", "down-right": "lower"}[side]


def label_for(layer, fam, par, prof=None):
    """(label, phrase) from the strongest probe."""
    if fam == "line":
        o = min(ORIENT, key=lambda a: min(abs(a - par), 180 - abs(a - par)))
        word, phrase = ORIENT[o]
        if prof is not None:
            ds, resp = prof
            d_peak = float(ds[np.argmax(resp)])
            if abs(d_peak) >= 1.0:
                side = side_words(o, d_peak)
                short = {"flat": "flat strokes", "upright": "upright strokes", "/": "/ strokes", "\\": "\\ strokes"}[word]
                long = {"flat": "flat strokes", "upright": "upright strokes",
                        "/": "strokes slanting like /", "\\": "strokes slanting like \\"}[word]
                return (f"{side.capitalize()} edges of {short}", f"the {side} edge of {long}")
        return ({"flat": "Flat strokes", "upright": "Upright strokes", "/": "Strokes slanting /", "\\": "Strokes slanting \\"}[word], phrase)
    if fam == "end":
        d = dir_name(par)
        return (f"Stroke end pointing {d}", f"the free end of a stroke, pointing {d}")
    if fam == "corner":
        d = dir_name(par)
        return (f"Corner pointing {d}", f"a corner whose point faces {d}")
    if fam == "curve":
        return CURVE[dir_name(par)]
    if fam == "tee":
        d = dir_name(par)
        return (f"Branch heading {d}", f"a stroke with another branching off it towards the {d}")
    if fam == "cross":
        return ("Crossing strokes", "two strokes crossing")
    if fam == "loop":
        return ("Inside a small loop", "the middle of a small closed loop")
    return ("Dots and blobs", "a dot or short blob of ink")


def main(weights_path, out_dir):
    W = np.load(weights_path)
    P = {k: W[k] for k in ("K1", "b1", "K2", "b2", "W3", "b3")}
    C1, C2 = len(P["b1"]), len(P["b2"])
    d = np.load(os.path.join(HERE, "prep.npz"))
    yte = d["yte"].astype(int)
    n = len(yte)
    Xr = np.zeros((n, 28, 28), np.float32)
    render_batch(d["te_all"], d["te_off"], np.arange(n), np.zeros((n, 8)), np.full(n, R0), S, 14.5, 14.5, False, Xr,
                 np.zeros((n, 3)))
    X = np.concatenate([Xr, d["Xte"].astype(np.float32) / 255.0])
    y = np.concatenate([yte, yte])

    # --- digits ---
    A1 = np.zeros((len(X), 28, 28, C1), np.float32)
    A2 = np.zeros((len(X), 14, 14, C2), np.float32)
    for i in range(0, len(X), 1000):
        cache, _ = forward(X[i:i + 1000], P)
        A1[i:i + 1000] = np.maximum(cache["z1"], 0)
        A2[i:i + 1000] = np.maximum(cache["z2"], 0)
    peak1, peak2 = A1.max((1, 2)), A2.max((1, 2))                       # (N, C): each map's strongest spot per digit
    # typical trigger: average input patch around each firing spot, weighted by activation squared,
    # over the re-drawn digits (one pen width, so strokes stay crisp in the average)
    Xd = X[:n]
    Xp1 = np.pad(Xd, ((0, 0), (P1 // 2, P1 // 2), (P1 // 2, P1 // 2)))
    W1sq = A1[:n] ** 2
    sta1 = np.zeros((C1, P1, P1))
    for dy in range(P1):
        for dx in range(P1):
            sta1[:, dy, dx] = np.einsum("nyxc,nyx->c", W1sq, Xp1[:, dy:dy + 28, dx:dx + 28])
    sta1 /= W1sq.sum((0, 1, 2))[:, None, None]
    Xp2 = np.pad(Xd, ((0, 0), (8, 8), (8, 8)))
    W2sq = A2[:n] ** 2
    sta2 = np.zeros((C2, P2, P2))
    for dy in range(P2):
        for dx in range(P2):
            # layer-2 spot (y2, x2) covers input rows 2*y2-6 .. 2*y2+7; the patch adds one pixel each side
            sta2[:, dy, dx] = np.einsum("nyxc,nyx->c", W2sq, Xp2[:, 1 + dy:1 + dy + 28:2, 1 + dx:1 + dx + 28:2])
    sta2 /= W2sq.sum((0, 1, 2))[:, None, None]
    del W1sq, W2sq
    gy1, gx1 = np.mgrid[0:28, 0:28] + 0.5
    gy2, gx2 = 2 * np.mgrid[0:14, 0:14] + 1.0
    print("digit measurements done", flush=True)

    # --- probes ---
    names, r1, r2, probe_imgs = probe_responses(P)
    fams = sorted(set(f for f, _ in names), key=[f for f, _ in names].index)

    roles = []
    lines = [m for m, (f, _) in enumerate(names) if f == "line"]
    for layer, C, peak, A, sta, (gy, gx), R in ((1, C1, peak1, A1, sta1, (gy1, gx1), r1),
                                                (2, C2, peak2, A2, sta2, (gy2, gx2), r2)):
        for c in range(C):
            by_digit = np.array([peak[y == k, c].mean() for k in range(10)])
            pref = by_digit / (by_digit.max() + 1e-9)
            typ = float(np.percentile(peak[:, c], 99)) + 1e-9
            resp = R[:, c] / typ
            # a 5x5 layer-1 filter can only see a short piece of stroke, so it is named by stroke angle and side
            cand = lines if layer == 1 else range(len(names))
            best = max(cand, key=lambda m: resp[m])
            fam, par = names[best]
            fam_best = {f: float(max(resp[m] for m, (ff, _) in enumerate(names) if ff == f)) for f in fams}
            prof = edge_profile(P, c, par) if layer == 1 else None
            # where it fires: each digit's strongest spot, weighted by how strongly it fires there
            flat = A[..., c].reshape(len(A), -1)
            am, pk = flat.argmax(1), flat.max(1)
            my = float((pk * gy.ravel()[am]).sum() / (pk.sum() + 1e-9))
            mx = float((pk * gx.ravel()[am]).sum() / (pk.sum() + 1e-9))
            top_ix = np.argsort(peak[:n, c])[::-1][:6]            # strongest re-drawn digits (one pen width)
            label, phrase = label_for(layer, fam, par, prof)
            roles.append({
                "layer": layer, "idx": c, "best": [fam, float(par)], "best_resp": round(float(resp[best]), 3),
                "family_best": {k: round(v, 3) for k, v in fam_best.items()},
                "line_tuning": [round(float(resp[m]), 3) for m in lines],
                "top_probes": [[names[m][0], float(names[m][1]), round(float(resp[m]), 3)] for m in np.argsort(resp)[::-1][:5]],
                "edge_profile": [prof[0].tolist(), prof[1].round(4).tolist()] if prof is not None else None,
                "label": label, "phrase": phrase,
                "pref": pref.round(3).tolist(), "fires": round(float((peak[:, c] > 0).mean()), 3),
                "spots": round(float((A[..., c] > 0).mean()), 3), "where": [round(my, 2), round(mx, 2)],
                "sta": sta[c].round(4).tolist(), "examples": top_ix.tolist(),
            })
    json.dump({"probes": [[f, float(p)] for f, p in names], "roles": roles}, open(os.path.join(out_dir, "roles_raw.json"), "w"))
    final = finalize(roles, P, os.path.join(out_dir, "roles.json"))
    for key in ("m1", "m2"):
        for i, r in enumerate(final[key]):
            print(f"{key.upper()} #{i + 1:<2} {r['label']:<34} {r['text'][12:]} | {r['foot']}")
    contact_sheets(roles, P, X, A1, A2, names, probe_imgs, out_dir)


def finalize(raw, P, out_path):
    """Turn measurements into the page's tooltip data: label, sentence, footnote, digit
    preferences and a 4-bit typical-trigger patch per filter."""
    out = {"m1": [], "m2": [], "p1": P1, "p2": P2}
    nibbles = []
    for ro in raw:
        key = (ro["layer"], ro["idx"] + 1)
        label, phrase = OVERRIDES.get(key, (ro["label"], ro["phrase"]))
        pref = np.array(ro["pref"])
        order = [int(k) for k in np.argsort(pref, kind="stable")[::-1]]
        top = [k for k in order if pref[k] >= 0.6][:3]
        dtxt = str(top[0]) if len(top) == 1 else ", ".join(map(str, top[:-1])) + f" and {top[-1]}"
        if pref.min() >= 0.7:
            text = f"Responds to {phrase}. Fires for all ten digits, most for {dtxt}."
        else:
            text = f"Responds to {phrase}. Fires most for {dtxt}."
        foot = f"Lights up {ro['spots']:.0%} of its map on a typical digit · quietest for {order[-1]}"
        out["m1" if ro["layer"] == 1 else "m2"].append(
            {"label": label, "text": text, "foot": foot, "pref": [round(v, 2) for v in ro["pref"]], "top": top})
        sta = np.array(ro["sta"]).ravel()
        q = np.round((sta - sta.min()) / (sta.max() - sta.min() + 1e-12) * 15).astype(np.uint8)
        nibbles.extend(q.tolist())
    if len(nibbles) % 2:
        nibbles.append(0)
    nb = np.array(nibbles, np.uint8)
    out["img"] = base64.b64encode(bytes(((nb[0::2] << 4) | nb[1::2]).tolist())).decode()
    out["khash"] = hashlib.sha1(np.asarray(P["K1"], "<f4").tobytes() + np.asarray(P["K2"], "<f4").tobytes()).hexdigest()[:16]
    json.dump(out, open(out_path, "w"), separators=(",", ":"))
    return out


def contact_sheets(roles, P, X, A1, A2, names, probe_imgs, out_dir):
    """One row per filter: weights, typical trigger, the six strongest firing patches, digit
    preference, probe responses by family, and the proposed label."""
    def gray(img, s):
        img = np.asarray(img, float)
        img = (img - img.min()) / (img.max() - img.min() + 1e-9)
        return Image.fromarray((255 - img * 255).astype(np.uint8)).resize((img.shape[1] * s, img.shape[0] * s), Image.NEAREST)

    def diverging(wt, s):
        m = np.abs(wt).max() + 1e-9
        t = wt / m
        rgb = np.ones(wt.shape + (3,))
        pos, neg = np.clip(t, 0, 1)[..., None], np.clip(-t, 0, 1)[..., None]
        rgb = rgb * (1 - pos) + pos * np.array([0.86, 0.35, 0.08])
        rgb = rgb * (1 - neg) + neg * np.array([0.12, 0.42, 0.81])
        return Image.fromarray((rgb * 255).astype(np.uint8)).resize((wt.shape[1] * s, wt.shape[0] * s), Image.NEAREST)

    fams = sorted(set(f for f, _ in names), key=[f for f, _ in names].index)
    for layer in (1, 2):
        rows = [r for r in roles if r["layer"] == layer]
        A = A1 if layer == 1 else A2
        canvas = Image.new("RGB", (1500, len(rows) * 104 + 10), "white")
        dr = ImageDraw.Draw(canvas)
        for ri, ro in enumerate(rows):
            c, y0 = ro["idx"], 6 + ri * 104
            if layer == 1:
                canvas.paste(diverging(P["K1"][c, 0], 16), (6, y0))
            else:
                for i in range(P["K2"].shape[1]):
                    canvas.paste(diverging(P["K2"][c, i], 8), (6 + (i % 4) * 44, y0 + (i // 4) * 46))
            canvas.paste(gray(np.array(ro["sta"]), 96 // len(ro["sta"])), (188, y0))
            for e, ix in enumerate(ro["examples"]):
                a = A[ix, ..., c]
                yy, xx = np.unravel_index(np.argmax(a), a.shape)
                # whole digit, with the strongest spot marked
                tile = gray(X[ix], 3).convert("RGB")
                td = ImageDraw.Draw(tile)
                cy, cx = ((yy + 0.5) * 3, (xx + 0.5) * 3) if layer == 1 else ((2 * yy + 1) * 3, (2 * xx + 1) * 3)
                td.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], outline=(220, 30, 30), width=2)
                canvas.paste(tile, (296 + e * 96, y0))
            bx = 880
            for k in range(10):
                h = int(60 * ro["pref"][k])
                dr.rectangle([bx + k * 12, y0 + 70 - h, bx + k * 12 + 8, y0 + 70], fill=(40, 40, 40))
                dr.text((bx + k * 12 + 1, y0 + 73), str(k), fill=(0, 0, 0))
            fb = ro["family_best"]
            px0 = 1010
            for fi, f in enumerate(fams):
                h = int(60 * min(1.5, fb[f]) / 1.5)
                dr.rectangle([px0 + fi * 14, y0 + 70 - h, px0 + fi * 14 + 10, y0 + 70], fill=(220, 90, 20))
                dr.text((px0 + fi * 14 + 1, y0 + 73), f[0], fill=(0, 0, 0))
            tp = "; ".join(f"{f} {p:g} {v:.2f}" for f, p, v in ro["top_probes"][:3])
            txt = (f"L{layer} #{c + 1}  fires {ro['fires']:.0%}  spots {ro['spots']:.0%}  where y{ro['where'][0]:.1f} x{ro['where'][1]:.1f}\n"
                   f"{ro['label']}\n{tp}")
            if ro["edge_profile"]:
                ds, rs = ro["edge_profile"]
                txt += f"\nedge peak at d={ds[int(np.argmax(rs))]:+.2f}"
            dr.text((1150, y0), txt, fill=(0, 0, 0))
        canvas.save(os.path.join(out_dir, f"roles_sheet_L{layer}.png"))
    # The probe shapes themselves (first placement), for checking the probe set
    cols = 16
    sheet = Image.new("RGB", (cols * 60, ((len(names) + cols - 1) // cols) * 72), "white")
    dr = ImageDraw.Draw(sheet)
    for m, (f, p) in enumerate(names):
        sheet.paste(gray(probe_imgs[m], 2), ((m % cols) * 60 + 2, (m // cols) * 72 + 2))
        dr.text(((m % cols) * 60 + 2, (m // cols) * 72 + 58), f"{f[:5]} {p:g}", fill=(0, 0, 0))
    sheet.save(os.path.join(out_dir, "probes.png"))
    print("wrote roles_raw.json, roles.json, 2 contact sheets and probes.png")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

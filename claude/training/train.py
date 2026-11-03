"""Train a small fully connected network on MNIST, from scratch in numpy.

Architecture: 784 inputs (28x28 pixels) -> 64 ReLU -> 32 ReLU -> 10 softmax.
Each epoch mixes original MNIST digits (lightly warped) with digits re-drawn
from their skeletons through the page's own stroke pipeline (strokes.py), with
fresh random rotation, shear, aspect, wobble and pen size.

Usage: python3 train.py --out weights.npz [--h1 64 --h2 32 --epochs 30 --mix both]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import styles  # noqa: E402
from strokes import normalize, render_batch, warp_batch  # noqa: E402

R0, S, TX, TY = 1.2, 4, 14.5, 14.5   # pen radius, supersampling, centre of mass target


def relu(z):
    return np.maximum(z, 0)


def forward(X, P):
    z1 = X @ P["W1"].T + P["b1"]
    a1 = relu(z1)
    z2 = a1 @ P["W2"].T + P["b2"]
    a2 = relu(z2)
    z3 = a2 @ P["W3"].T + P["b3"]
    return z1, a1, z2, a2, z3


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def logits(X, P, bs=8192):
    return np.concatenate([forward(X[i:i + bs], P)[4] for i in range(0, len(X), bs)])


def acc(X, y, P):
    return float((logits(X, P).argmax(1) == y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h1", type=int, default=64)
    ap.add_argument("--h2", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--mix", default="both", choices=["both", "redraw", "orig"])
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flag", type=float, default=0.4, help="share of re-drawn 1s given a European flag")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    d = np.load(os.path.join(HERE, "prep.npz"))
    Xtr = d["Xtr"].astype(np.float32) / 255.0
    ytr = d["ytr"].astype(np.int64)
    Xte = d["Xte"].astype(np.float32).reshape(-1, 784) / 255.0
    yte = d["yte"].astype(np.int64)
    tr_all, tr_off, te_all, te_off = d["tr_all"], d["tr_off"], d["te_all"], d["te_off"]
    radii_pool = d["radii"]

    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(Xtr))
    va_idx, tr_idx = perm[:5000], perm[5000:]

    def redraw_plain(seg_all, off, idxs):
        out = np.zeros((len(idxs), 28, 28), np.float32)
        render_batch(seg_all, off, idxs, np.zeros((len(idxs), 8)), np.full(len(idxs), R0), S, TX, TY, False, out,
                     np.zeros((len(idxs), 3)))
        return out.reshape(len(idxs), 784)

    t0 = time.time()
    Xva_o, yva = Xtr[va_idx].reshape(-1, 784), ytr[va_idx]
    Xva_r = redraw_plain(tr_all, tr_off, va_idx)
    Xte_r = redraw_plain(te_all, te_off, np.arange(len(yte)))
    st_items = styles.make_set(n_per=25, seed=123)
    Xst = np.stack([normalize(styles.strokes_to_segs(s), R0, S, TX, TY).ravel() for s, _, _ in st_items])
    yst = np.array([lab for _, lab, _ in st_items])
    st_names = np.array([n for _, _, n in st_items])
    print(f"eval sets ready in {time.time() - t0:.1f}s", flush=True)

    def stroke_aug(m):
        p = np.zeros((m, 8))
        p[:, 0] = rng.uniform(-0.26, 0.26, m)       # rotation, ±15°
        p[:, 1] = rng.uniform(-0.35, 0.35, m)       # shear
        p[:, 2] = rng.uniform(-0.35, 0.35, m)       # log aspect ratio
        p[:, 3] = rng.uniform(0.0, 1.2, m)          # wobble amplitude (px)
        p[:, 4] = rng.uniform(0.12, 0.35, m)
        p[:, 5] = rng.uniform(0, 2 * np.pi, m)
        p[:, 6] = rng.uniform(0.12, 0.35, m)
        p[:, 7] = rng.uniform(0, 2 * np.pi, m)
        radii = np.clip(rng.choice(radii_pool, m) * np.exp(rng.normal(0, 0.08, m)), 0.75, 2.0)
        return p, radii

    def flag_aug(labels):
        m = len(labels)
        ex = np.zeros((m, 3))
        if a.flag > 0:
            ones = (labels == 1) & (rng.random(m) < a.flag)
            k = int(ones.sum())
            ex[ones, 0] = rng.uniform(0.15, 0.4, k)                    # flag length (× height)
            ex[ones, 1] = rng.uniform(np.radians(15), np.radians(55), k)  # flag angle below horizontal
            ex[ones, 2] = np.where(rng.random(k) < 0.3, rng.uniform(0.3, 0.6, k), 0.0)  # optional base
        return ex

    def warp_aug(m):
        th = rng.uniform(-0.17, 0.17, m)
        sh = rng.uniform(-0.15, 0.15, m)
        sc = np.exp(rng.uniform(-0.1, 0.1, m))
        c, s_ = np.cos(th), np.sin(th)
        # inverse map (output -> input) of rotate∘shear∘scale
        f00, f01, f10, f11 = c * sc, (c * sh - s_) * sc, s_ * sc, (s_ * sh + c) * sc
        det = f00 * f11 - f01 * f10
        P = np.stack([f11 / det, -f01 / det, -f10 / det, f00 / det,
                      rng.uniform(-1.5, 1.5, m), rng.uniform(-1.5, 1.5, m)], 1)
        return P

    sizes = [784, a.h1, a.h2, 10]
    P = {}
    for i in range(3):
        P[f"W{i + 1}"] = (rng.normal(0, np.sqrt(2.0 / sizes[i]), (sizes[i + 1], sizes[i]))).astype(np.float32)
        P[f"b{i + 1}"] = np.zeros(sizes[i + 1], np.float32)
    M = {k: np.zeros_like(v) for k, v in P.items()}
    V = {k: np.zeros_like(v) for k, v in P.items()}
    b1, b2, eps, step = 0.9, 0.999, 1e-8, 0

    n_tr = len(tr_idx)
    per_epoch = {"both": 2 * n_tr, "redraw": n_tr, "orig": n_tr}[a.mix]
    total_steps = a.epochs * (per_epoch // a.bs)
    warm = per_epoch // a.bs
    log = []
    for ep in range(a.epochs):
        t = time.time()
        parts_X, parts_y = [], []
        if a.mix in ("both", "orig"):
            out = np.zeros((n_tr, 28, 28), np.float32)
            warp_batch(Xtr, tr_idx, warp_aug(n_tr), out)
            parts_X.append(out.reshape(n_tr, 784))
            parts_y.append(ytr[tr_idx])
        if a.mix in ("both", "redraw"):
            out = np.zeros((n_tr, 28, 28), np.float32)
            p, radii = stroke_aug(n_tr)
            render_batch(tr_all, tr_off, tr_idx, p, radii, S, TX, TY, True, out, flag_aug(ytr[tr_idx]))
            parts_X.append(out.reshape(n_tr, 784))
            parts_y.append(ytr[tr_idx])
        X = np.concatenate(parts_X)
        y = np.concatenate(parts_y)
        t_data = time.time() - t
        order = rng.permutation(len(X))
        tot_loss = 0.0
        for bi in range(len(X) // a.bs):
            idx = order[bi * a.bs:(bi + 1) * a.bs]
            xb, yb = X[idx], y[idx]
            z1, a1, z2, a2, z3 = forward(xb, P)
            p = softmax(z3)
            B = len(yb)
            tot_loss += -np.log(p[np.arange(B), yb] + 1e-12).mean()
            d3 = p
            d3[np.arange(B), yb] -= 1.0
            d3 /= B
            G = {"W3": d3.T @ a2, "b3": d3.sum(0)}
            d2 = (d3 @ P["W3"]) * (z2 > 0)
            G["W2"], G["b2"] = d2.T @ a1, d2.sum(0)
            d1 = (d2 @ P["W2"]) * (z1 > 0)
            G["W1"], G["b1"] = d1.T @ xb, d1.sum(0)
            step += 1
            if step < warm:
                lr = a.lr * (0.1 + 0.9 * step / warm)
            else:
                lr = a.lr * 0.5 * (1 + np.cos(np.pi * (step - warm) / max(1, total_steps - warm)))
            for k in P:
                g = G[k].astype(np.float32)
                M[k] = b1 * M[k] + (1 - b1) * g
                V[k] = b2 * V[k] + (1 - b2) * g * g
                mh = M[k] / (1 - b1 ** step)
                vh = V[k] / (1 - b2 ** step)
                upd = mh / (np.sqrt(vh) + eps)
                if k.startswith("W"):
                    upd = upd + a.wd * P[k]
                P[k] = (P[k] - lr * upd).astype(np.float32)
        rec = {"epoch": ep + 1, "loss": round(float(tot_loss) / (len(X) // a.bs), 4),
               "val_orig": round(acc(Xva_o, yva, P), 4), "val_redraw": round(acc(Xva_r, yva, P), 4),
               "styles": round(acc(Xst, yst, P), 4), "data_s": round(t_data, 1), "s": round(time.time() - t, 1)}
        log.append(rec)
        print(json.dumps(rec), flush=True)

    # Temperature scaling on validation (re-drawn digits = what the page sees)
    zva = logits(Xva_r, P).astype(np.float64)
    best = None
    for T in np.arange(0.5, 3.0001, 0.01):
        nll = -np.log(softmax(zva / T)[np.arange(len(yva)), yva] + 1e-12).mean()
        if best is None or nll < best[1]:
            best = (round(float(T), 2), float(nll))
    T = best[0]

    zst = logits(Xst, P)
    per_style = {}
    for name in np.unique(st_names):
        m = st_names == name
        per_style[str(name)] = round(float((zst[m].argmax(1) == yst[m]).mean()), 2)
    zte_r = logits(Xte_r, P)
    conf = np.zeros((10, 10), int)
    np.add.at(conf, (yte, zte_r.argmax(1)), 1)
    metrics = {"arch": sizes, "mix": a.mix, "epochs": a.epochs, "temperature": T, "val_nll_at_T": round(best[1], 4),
               "test_orig": acc(Xte, yte, P), "test_redraw": float((zte_r.argmax(1) == yte).mean()),
               "styles": float((zst.argmax(1) == yst).mean()), "per_style": per_style,
               "confusion_redraw": conf.tolist(), "log": log}
    np.savez(a.out, **P, T=np.float32(T), R0=np.float32(R0), S=np.int32(S), TX=np.float32(TX), TY=np.float32(TY))
    with open(a.out.replace(".npz", ".json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print("FINAL", json.dumps({k: metrics[k] for k in ("arch", "mix", "temperature", "test_orig", "test_redraw", "styles")}))
    print("per style:", json.dumps(per_style))


if __name__ == "__main__":
    main()

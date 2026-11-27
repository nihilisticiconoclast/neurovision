"""Train a small convolutional network on MNIST, from scratch in numpy.

Architecture:
  28x28 input
  -> 8 filters of 5x5 (zero padding keeps 28x28), ReLU, 2x2 max pool    -> 8 maps of 14x14
  -> 16 filters of 5x5x8 (zero padding keeps 14x14), ReLU, 2x2 max pool -> 16 maps of 7x7 = 784 numbers
  -> 10 scores, each a weighted sum of all 784, softmax
Training uses the same data recipe as train.py: each epoch mixes lightly warped
MNIST digits with digits re-drawn from their skeletons through the page's own
stroke pipeline (strokes.py), with fresh random rotation, shear, aspect, wobble,
pen size and European flags on some 1s.

Maps are channels-last while training; the saved weights are channel-major:
K1 (8,1,5,5), K2 (16,8,5,5), W3 (10, 16*7*7) flattened as [map][row][col].

Usage: python3 cnn_train.py --out runs/cnn8x16.npz [--c1 8 --c2 16 --epochs 30 --mix both]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from numba import njit, prange
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import styles  # noqa: E402
from strokes import normalize, render_batch, warp_batch  # noqa: E402

R0, S, TX, TY = 1.2, 4, 14.5, 14.5   # pen radius, supersampling, centre of mass target
K = 5                                 # filter size
PADK = K // 2                         # zero padding that keeps a map's size


def conv_cols(x):
    """im2col for a 'same' 5x5 convolution. x: (B, H, W, C) -> (B*H*W, C*K*K),
    columns ordered (channel, dy, dx) to match K.reshape(n_out, -1)."""
    B, H, W, C = x.shape
    xp = np.pad(x, ((0, 0), (PADK, PADK), (PADK, PADK), (0, 0)))
    return sliding_window_view(xp, (K, K), axis=(1, 2)).reshape(B * H * W, C * K * K)


@njit(parallel=True, cache=True)
def _pool(a, out, idx):
    B, H, W, C = a.shape
    for b in prange(B):
        for y in range(H // 2):
            for x in range(W // 2):
                for c in range(C):
                    best = a[b, 2 * y, 2 * x, c]
                    bi = 0
                    v = a[b, 2 * y, 2 * x + 1, c]
                    if v > best:
                        best = v
                        bi = 1
                    v = a[b, 2 * y + 1, 2 * x, c]
                    if v > best:
                        best = v
                        bi = 2
                    v = a[b, 2 * y + 1, 2 * x + 1, c]
                    if v > best:
                        best = v
                        bi = 3
                    out[b, y, x, c] = best
                    idx[b, y, x, c] = bi


def pool(a):
    """2x2 max pool, channels last. Ties go to the first maximum in reading order
    (top-left, top-right, bottom-left, bottom-right), as in the page."""
    B, H, W, C = a.shape
    out = np.empty((B, H // 2, W // 2, C), a.dtype)
    idx = np.empty((B, H // 2, W // 2, C), np.int8)
    _pool(a, out, idx)
    return out, idx


@njit(parallel=True, cache=True)
def _unpool_relu(g, idx, z, out):
    B, h, w, C = g.shape
    for b in prange(B):
        for y in range(h):
            for x in range(w):
                for c in range(C):
                    i = idx[b, y, x, c]
                    yy = 2 * y + (i >> 1)
                    xx = 2 * x + (i & 1)
                    if z[b, yy, xx, c] > 0:
                        out[b, yy, xx, c] = g[b, y, x, c]


def unpool_relu(g, idx, z):
    """Route pooled gradients back to the position that won each 2x2 block,
    through the ReLU (zero where the winner wasn't firing)."""
    out = np.zeros(z.shape, g.dtype)
    _unpool_relu(g, idx, z, out)
    return out


@njit(parallel=True, cache=True)
def _col2im(dc, out):
    B, H, W, C, KK, _ = dc.shape
    for b in prange(B):
        for y in range(H):
            for x in range(W):
                for c in range(C):
                    for dy in range(KK):
                        for dx in range(KK):
                            out[b, y + dy, x + dx, c] += dc[b, y, x, c, dy, dx]


def forward(X, P):
    """X: (B, 28, 28). Returns (cache for backprop, scores (B, 10))."""
    B = X.shape[0]
    C1, C2 = len(P["b1"]), len(P["b2"])
    c1 = conv_cols(np.ascontiguousarray(X)[..., None])
    z1 = (c1 @ P["K1"].reshape(C1, -1).T + P["b1"]).reshape(B, 28, 28, C1)
    p1, i1 = pool(np.maximum(z1, 0))
    c2 = conv_cols(p1)
    z2 = (c2 @ P["K2"].reshape(C2, -1).T + P["b2"]).reshape(B, 14, 14, C2)
    p2, i2 = pool(np.maximum(z2, 0))
    f = p2.transpose(0, 3, 1, 2).reshape(B, -1)
    z3 = f @ P["W3"].T + P["b3"]
    return {"c1": c1, "z1": z1, "i1": i1, "c2": c2, "z2": z2, "i2": i2, "f": f}, z3


def backward(cache, d3, P):
    """d3 = dLoss/dscores (B, 10). Returns gradients for every parameter."""
    B = d3.shape[0]
    C1, C2 = len(P["b1"]), len(P["b2"])
    G = {"W3": d3.T @ cache["f"], "b3": d3.sum(0)}
    dp2 = np.ascontiguousarray((d3 @ P["W3"]).reshape(B, C2, 7, 7).transpose(0, 2, 3, 1))
    dz2 = unpool_relu(dp2, cache["i2"], cache["z2"]).reshape(B * 196, C2)
    G["K2"] = (dz2.T @ cache["c2"]).reshape(P["K2"].shape)
    G["b2"] = dz2.sum(0)
    dc2 = (dz2 @ P["K2"].reshape(C2, -1)).reshape(B, 14, 14, C1, K, K)
    dpp = np.zeros((B, 14 + 2 * PADK, 14 + 2 * PADK, C1), d3.dtype)
    _col2im(dc2, dpp)
    dp1 = np.ascontiguousarray(dpp[:, PADK:PADK + 14, PADK:PADK + 14, :])
    dz1 = unpool_relu(dp1, cache["i1"], cache["z1"]).reshape(B * 784, C1)
    G["K1"] = (dz1.T @ cache["c1"]).reshape(P["K1"].shape)
    G["b1"] = dz1.sum(0)
    return G


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def logits(X, P, bs=1000):
    return np.concatenate([forward(X[i:i + bs], P)[1] for i in range(0, len(X), bs)])


def acc(X, y, P):
    return float((logits(X, P).argmax(1) == y).mean())


def init_params(c1, c2, rng):
    shapes = {"K1": (c1, 1, K, K), "K2": (c2, c1, K, K), "W3": (10, c2 * 49)}
    P = {}
    for i, (k, shp) in enumerate(shapes.items()):
        fan_in = int(np.prod(shp[1:]))
        P[k] = rng.normal(0, np.sqrt(2.0 / fan_in), shp).astype(np.float32)
        P[f"b{i + 1}"] = np.zeros(shp[0], np.float32)
    return P


def grad_check():
    """Compare backprop with finite differences on a few random weights (float64)."""
    rng = np.random.default_rng(1)
    P = {k: v.astype(np.float64) for k, v in init_params(3, 4, rng).items()}
    for k in ("b1", "b2"):
        P[k] = P[k] + 0.05
    X = rng.random((2, 28, 28))
    y = np.array([3, 7])

    def loss(P):
        p = softmax(forward(X, P)[1])
        return -np.log(p[np.arange(2), y]).mean()
    cache, z3 = forward(X, P)
    d3 = softmax(z3)
    d3[np.arange(2), y] -= 1
    G = backward(cache, d3 / 2, P)
    worst = 0.0
    for k in P:
        for _ in range(6):
            i = tuple(rng.integers(0, s) for s in P[k].shape)
            old = P[k][i]
            P[k][i] = old + 1e-6
            lp = loss(P)
            P[k][i] = old - 1e-6
            lm = loss(P)
            P[k][i] = old
            num = (lp - lm) / 2e-6
            worst = max(worst, abs(num - G[k][i]) / max(1e-8, abs(num) + abs(G[k][i])))
    print(f"gradient check: worst relative error {worst:.2e}")
    assert worst < 1e-5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1", type=int, default=8)
    ap.add_argument("--c2", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--mix", default="both", choices=["both", "redraw", "orig"])
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--flag", type=float, default=0.4, help="share of re-drawn 1s given a European flag")
    ap.add_argument("--limit", type=int, default=0, help="train on this many digits only (for quick tests)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    grad_check()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    d = np.load(os.path.join(HERE, "prep.npz"))
    Xtr = d["Xtr"].astype(np.float32) / 255.0
    ytr = d["ytr"].astype(np.int64)
    Xte = d["Xte"].astype(np.float32) / 255.0
    yte = d["yte"].astype(np.int64)
    tr_all, tr_off, te_all, te_off = d["tr_all"], d["tr_off"], d["te_all"], d["te_off"]
    radii_pool = d["radii"]

    # Same split as train.py (same seed, same first draw), so validation digits match.
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(Xtr))
    va_idx, tr_idx = perm[:5000], perm[5000:]
    if a.limit:
        tr_idx = tr_idx[:a.limit]

    def redraw_plain(seg_all, off, idxs):
        out = np.zeros((len(idxs), 28, 28), np.float32)
        render_batch(seg_all, off, idxs, np.zeros((len(idxs), 8)), np.full(len(idxs), R0), S, TX, TY, False, out,
                     np.zeros((len(idxs), 3)))
        return out

    t0 = time.time()
    Xva_o, yva = Xtr[va_idx], ytr[va_idx]
    Xva_r = redraw_plain(tr_all, tr_off, va_idx)
    Xte_r = redraw_plain(te_all, te_off, np.arange(len(yte)))
    st_items = styles.make_set(n_per=25, seed=123)
    Xst = np.stack([normalize(styles.strokes_to_segs(s), R0, S, TX, TY) for s, _, _ in st_items])
    yst = np.array([lab for _, lab, _ in st_items])
    st_names = np.array([n for _, _, n in st_items])
    print(f"eval sets ready in {time.time() - t0:.1f}s", flush=True)

    # Augmentations: identical to train.py.
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
        f00, f01, f10, f11 = c * sc, (c * sh - s_) * sc, s_ * sc, (s_ * sh + c) * sc
        det = f00 * f11 - f01 * f10
        return np.stack([f11 / det, -f01 / det, -f10 / det, f00 / det,
                         rng.uniform(-1.5, 1.5, m), rng.uniform(-1.5, 1.5, m)], 1)

    P = init_params(a.c1, a.c2, rng)
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
            parts_X.append(out)
            parts_y.append(ytr[tr_idx])
        if a.mix in ("both", "redraw"):
            out = np.zeros((n_tr, 28, 28), np.float32)
            p, radii = stroke_aug(n_tr)
            render_batch(tr_all, tr_off, tr_idx, p, radii, S, TX, TY, True, out, flag_aug(ytr[tr_idx]))
            parts_X.append(out)
            parts_y.append(ytr[tr_idx])
        X = np.concatenate(parts_X)
        y = np.concatenate(parts_y)
        t_data = time.time() - t
        order = rng.permutation(len(X))
        tot_loss = 0.0
        for bi in range(len(X) // a.bs):
            idx = order[bi * a.bs:(bi + 1) * a.bs]
            xb, yb = X[idx], y[idx]
            cache, z3 = forward(xb, P)
            p = softmax(z3)
            B = len(yb)
            tot_loss += -np.log(p[np.arange(B), yb] + 1e-12).mean()
            p[np.arange(B), yb] -= 1.0
            G = backward(cache, p / B, P)
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
                if k[0] in "KW":
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
    n_params = int(sum(v.size for v in P.values()))
    metrics = {"arch": {"c1": a.c1, "c2": a.c2, "k": K, "params": n_params}, "mix": a.mix, "epochs": a.epochs,
               "temperature": T, "val_nll_at_T": round(best[1], 4),
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

"""Load MNIST, extract stroke skeletons, calibrate pen radius and centring.

Usage: python3 prep.py <mnist_dir> <out_npz>
"""
import gzip
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.ndimage import zoom
from skimage.morphology import skeletonize

U = 4          # upsampling before skeletonising (smoother centre-lines)
THR = 0.3      # ink threshold on the upsampled image
OFFS = [(0, 1), (1, -1), (1, 0), (1, 1)]
ALL8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def load_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    dims = [int.from_bytes(data[4 + 4 * i:8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, np.uint8, offset=4 + 4 * ndim).reshape(dims)


def skel_segs(img):
    """Skeleton of one digit as 8-connected line segments, in the digit's own
    28x28 continuous coordinates (pixel j spans [j, j+1))."""
    up = zoom(img.astype(np.float32) / 255.0, U, order=1)
    sk = skeletonize(up > THR)
    H, W = sk.shape
    P = np.pad(sk, 1)
    segs = []
    for dy, dx in OFFS:
        nb = P[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx]
        ys, xs = np.nonzero(sk & nb)
        if len(ys):
            segs.append(np.stack([xs, ys, xs + dx, ys + dy], 1))
    cnt = np.zeros((H, W), np.int32)
    for dy, dx in ALL8:
        cnt += P[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx]
    ys, xs = np.nonzero(sk & (cnt == 0))
    if len(ys):
        segs.append(np.stack([xs, ys, xs, ys], 1))
    if not segs:
        return np.zeros((0, 4), np.float32)
    s = np.concatenate(segs).astype(np.float64)
    # zoom(order=1) maps output index o to input index o*(28-1)/(28U-1)
    s = s * (27.0 / (28 * U - 1)) + 0.5
    return s.astype(np.float32)


def main(mnist_dir, out_path):
    Xtr = load_idx(f"{mnist_dir}/train-images-idx3-ubyte.gz")
    ytr = load_idx(f"{mnist_dir}/train-labels-idx1-ubyte.gz")
    Xte = load_idx(f"{mnist_dir}/t10k-images-idx3-ubyte.gz")
    yte = load_idx(f"{mnist_dir}/t10k-labels-idx1-ubyte.gz")
    print("loaded", Xtr.shape, Xte.shape, flush=True)

    t = time.time()
    with Pool(2) as pool:
        tr_segs = pool.map(skel_segs, list(Xtr), chunksize=500)
        te_segs = pool.map(skel_segs, list(Xte), chunksize=500)
    print(f"skeletons in {time.time() - t:.1f}s", flush=True)

    def pack(seglist):
        lens = np.array([len(s) for s in seglist])
        offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        return np.concatenate(seglist).astype(np.float32), offsets

    tr_all, tr_off = pack(tr_segs)
    te_all, te_off = pack(te_segs)
    lens = np.diff(tr_off)
    print("segments per digit: median", np.median(lens), "min", lens.min(), "max", lens.max(),
          "empty", int((lens == 0).sum()), flush=True)

    # Where MNIST puts the centre of mass (continuous coords; pixel j centre = j+0.5)
    v = Xtr.astype(np.float64) / 255.0
    tot = v.sum((1, 2))
    cx = (v.sum(1) * (np.arange(28) + 0.5)).sum(1) / tot
    cy = (v.sum(2) * (np.arange(28) + 0.5)).sum(1) / tot
    print(f"MNIST centre of mass: x {cx.mean():.3f}±{cx.std():.3f}  y {cy.mean():.3f}±{cy.std():.3f}")

    # Ink bounding box size, to check the 20px-box convention
    ink = v > 0.1
    rows = ink.any(2)
    cols = ink.any(1)
    h = 28 - rows.argmax(1) - rows[:, ::-1].argmax(1)
    w = 28 - cols.argmax(1) - cols[:, ::-1].argmax(1)
    print("bbox max side (px>0.1): median", np.median(np.maximum(h, w)),
          "pct20", np.mean(np.maximum(h, w) == 20).round(3))

    # Calibrate pen radius: per digit, the R whose rendering of the skeleton
    # (in the digit's own frame) carries the same total ink as the original.
    from strokes import render
    rng = np.random.default_rng(0)
    sample = rng.choice(len(Xtr), 3000, replace=False)
    grid = np.arange(0.4, 2.61, 0.1)
    est = []
    for i in sample:
        segs = tr_all[tr_off[i]:tr_off[i + 1]].astype(np.float64)
        if len(segs) == 0:
            continue
        target = v[i].sum()
        inks = np.array([render(segs, R, 4).sum() for R in grid])
        est.append(np.interp(target, inks, grid))
    est = np.array(est)
    print("equivalent pen radius: median %.3f  IQR %.3f-%.3f  p10 %.3f p90 %.3f" % (
        np.median(est), *np.percentile(est, [25, 75, 10, 90])))

    np.savez_compressed(out_path, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
                        tr_all=tr_all, tr_off=tr_off, te_all=te_all, te_off=te_off,
                        com=np.array([cx.mean(), cy.mean()]), radii=est)
    print("saved", out_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

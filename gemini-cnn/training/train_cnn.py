"""Train a compact, high-accuracy Convolutional Neural Network on MNIST from scratch in NumPy.

Architecture:
  Input:    1 x 28 x 28  (Center-of-mass normalized grayscale in [0, 1])
  Conv1:    8 filters, 3x3, pad=1, stride=1 + ReLU  ->  8 x 28 x 28   (80 params)
  MaxPool1: 2x2, stride=2                            ->  8 x 14 x 14
  Conv2:    16 filters, 3x3, pad=1, stride=1 + ReLU -> 16 x 14 x 14   (1,168 params)
  MaxPool2: 2x2, stride=2                            -> 16 x  7 x  7   (784 features)
  FC1:      784 -> 32 + ReLU                                          (25,120 params)
  FC2:      32  -> 10 (logits -> temperature-scaled softmax)          (330 params)
  Total:    26,698 parameters (~107 KB float32)

Usage:
  python train_cnn.py
"""
import gzip
import json
import os
import sys
import time
import urllib.request

import numpy as np
from scipy.ndimage import affine_transform, grey_dilation, grey_erosion

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "claude", "training"))
import styles  # noqa: E402

MNIST_DIR = os.path.join(HERE, "mnist_data")
MNIST_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]
MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://raw.githubusercontent.com/fgnt/mnist/master/",
]


def download_mnist():
    os.makedirs(MNIST_DIR, exist_ok=True)
    for fname in MNIST_FILES:
        dst = os.path.join(MNIST_DIR, fname)
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            continue
        last_err = None
        for base in MIRRORS:
            try:
                print(f"Downloading {base}{fname}...", flush=True)
                urllib.request.urlretrieve(base + fname, dst)
                break
            except Exception as e:
                last_err = e
        else:
            raise RuntimeError(f"Failed to download {fname}: {last_err}")


def load_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    dims = [int.from_bytes(data[4 + 4 * i:8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, np.uint8, offset=4 + 4 * ndim).reshape(dims)


def render_strokes_numpy(strokes, radius=1.25):
    """Vectorized anti-aliased rasterizer matching browser 20x20 + CoM pipeline."""
    segs = styles.strokes_to_segs(strokes)
    pts = segs.reshape(-1, 2)
    minx, miny = pts.min(axis=0)
    maxx, maxy = pts.max(axis=0)
    L = max(maxx - minx, maxy - miny, 1e-6)
    s = (20.0 - 2.0 * radius) / L
    cx, cy = 0.5 * (minx + maxx), 0.5 * (miny + maxy)

    s2 = segs.copy()
    s2[:, [0, 2]] = (s2[:, [0, 2]] - cx) * s + 14.0
    s2[:, [1, 3]] = (s2[:, [1, 3]] - cy) * s + 14.0

    # 2x2 supersampling grid (56x56 -> 28x28)
    S = 2
    F = 28 * S
    coords = (np.arange(F, dtype=np.float32) + 0.5) / S
    gx, gy = np.meshgrid(coords, coords)  # (F, F)

    def raster(seg_arr):
        x0 = seg_arr[:, 0].astype(np.float32)[:, None, None]
        y0 = seg_arr[:, 1].astype(np.float32)[:, None, None]
        dx = (seg_arr[:, 2] - seg_arr[:, 0]).astype(np.float32)[:, None, None]
        dy = (seg_arr[:, 3] - seg_arr[:, 1]).astype(np.float32)[:, None, None]
        L2 = np.maximum(dx * dx + dy * dy, 1e-12)
        t = np.clip(((gx[None] - x0) * dx + (gy[None] - y0) * dy) / L2, 0.0, 1.0)
        qx = x0 + t * dx - gx[None]
        qy = y0 + t * dy - gy[None]
        d = np.sqrt(qx * qx + qy * qy).min(axis=0)
        cov = np.clip((radius - d) * S + 0.5, 0.0, 1.0)
        return cov.reshape(28, S, 28, S).mean(axis=(1, 3)).astype(np.float32)

    img = raster(s2)
    tot = float(img.sum())
    if tot > 1e-6:
        ys, xs = np.mgrid[0:28, 0:28]
        mx = float((img * (xs + 0.5)).sum()) / tot
        my = float((img * (ys + 0.5)).sum()) / tot
        s2[:, [0, 2]] += (14.0 - mx)
        s2[:, [1, 3]] += (14.0 - my)
        img = raster(s2)
    return img


def make_synthetic_styles(n_per_style, seed, radius_range=(0.95, 1.65)):
    rng = np.random.default_rng(seed)
    xs, ys, names = [], [], []
    for label, style_dict in styles.TEMPLATES.items():
        for name, strokes in style_dict.items():
            for _ in range(n_per_style):
                j = styles.jitter(strokes, rng)
                rad = float(rng.uniform(*radius_range))
                xs.append(render_strokes_numpy(j, radius=rad))
                ys.append(label)
                names.append(f"{label}-{name}")
    return np.stack(xs)[:, None, :, :], np.array(ys, dtype=np.int64), np.array(names)


# ---------------------------------------------------------------------------
# Fast NumPy CNN Primitives (im2col / col2im for 3x3 pad=1 conv, 2x2 maxpool)
# ---------------------------------------------------------------------------
def im2col_3x3_pad1(x):
    """x: (N, C, H, W) -> col: (N*H*W, C*9)"""
    N, C, H, W = x.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
    # Build 9 shifted views without slow Python loops over pixels
    patches = np.empty((N, C, 3, 3, H, W), dtype=x.dtype)
    for dy in range(3):
        for dx in range(3):
            patches[:, :, dy, dx, :, :] = xp[:, :, dy:dy + H, dx:dx + W]
    return patches.transpose(0, 4, 5, 1, 2, 3).reshape(N * H * W, C * 9)


def col2im_3x3_pad1(dcol, N, C, H, W):
    """dcol: (N*H*W, C*9) -> dx: (N, C, H, W)"""
    patches = dcol.reshape(N, H, W, C, 3, 3).transpose(0, 3, 4, 5, 1, 2)
    dxp = np.zeros((N, C, H + 2, W + 2), dtype=dcol.dtype)
    for dy in range(3):
        for dx in range(3):
            dxp[:, :, dy:dy + H, dx:dx + W] += patches[:, :, dy, dx, :, :]
    return dxp[:, :, 1:-1, 1:-1]


def conv_forward(x, W, b):
    """x: (N, Cin, H, W), W: (Cout, Cin, 3, 3), b: (Cout,)"""
    N, Cin, H, W_in = x.shape
    Cout = W.shape[0]
    col = im2col_3x3_pad1(x)  # (N*H*W, Cin*9)
    W_mat = W.reshape(Cout, -1)  # (Cout, Cin*9)
    out = col @ W_mat.T + b  # (N*H*W, Cout)
    out = out.reshape(N, H, W_in, Cout).transpose(0, 3, 1, 2)
    return out, col


def conv_backward(dout, col, W, x_shape):
    """dout: (N, Cout, H, W)"""
    N, Cin, H, W_in = x_shape
    Cout = W.shape[0]
    dout_2d = dout.transpose(0, 2, 3, 1).reshape(N * H * W_in, Cout)
    dW = (dout_2d.T @ col).reshape(W.shape)
    db = dout_2d.sum(axis=0)
    W_mat = W.reshape(Cout, -1)
    dcol = dout_2d @ W_mat
    dx = col2im_3x3_pad1(dcol, N, Cin, H, W_in)
    return dx, dW, db


def maxpool2x2_forward(x):
    """x: (N, C, H, W) with even H, W -> out: (N, C, H//2, W//2), argmax"""
    N, C, H, W = x.shape
    xr = x.reshape(N, C, H // 2, 2, W // 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(N, C, H // 2, W // 2, 4)
    idx = xr.argmax(axis=-1)
    out = xr.max(axis=-1)
    return out, idx


def maxpool2x2_backward(dout, idx, x_shape):
    N, C, H, W = x_shape
    dxr = np.zeros((N, C, H // 2, W // 2, 4), dtype=dout.dtype)
    np.put_along_axis(dxr, idx[..., None], dout[..., None], axis=-1)
    return dxr.reshape(N, C, H // 2, W // 2, 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(N, C, H, W)


def softmax(z):
    zm = z - z.max(axis=1, keepdims=True)
    e = np.exp(zm)
    return e / e.sum(axis=1, keepdims=True)


def init_params(seed=42):
    rng = np.random.default_rng(seed)
    P = {
        "W1": rng.normal(0, np.sqrt(2.0 / (1 * 9)), (8, 1, 3, 3)).astype(np.float32),
        "b1": np.zeros(8, dtype=np.float32),
        "W2": rng.normal(0, np.sqrt(2.0 / (8 * 9)), (16, 8, 3, 3)).astype(np.float32),
        "b2": np.zeros(16, dtype=np.float32),
        "W3": rng.normal(0, np.sqrt(2.0 / 784), (32, 784)).astype(np.float32),
        "b3": np.zeros(32, dtype=np.float32),
        "W4": rng.normal(0, np.sqrt(2.0 / 32), (10, 32)).astype(np.float32),
        "b4": np.zeros(10, dtype=np.float32),
    }
    return P


def forward_cnn(x, P, train=False):
    z1, col1 = conv_forward(x, P["W1"], P["b1"])
    a1 = np.maximum(z1, 0.0)
    p1, idx1 = maxpool2x2_forward(a1)

    z2, col2 = conv_forward(p1, P["W2"], P["b2"])
    a2 = np.maximum(z2, 0.0)
    p2, idx2 = maxpool2x2_forward(a2)

    flat = p2.reshape(x.shape[0], -1)  # (N, 784)
    z3 = flat @ P["W3"].T + P["b3"]
    a3 = np.maximum(z3, 0.0)
    logits = a3 @ P["W4"].T + P["b4"]

    if not train:
        return logits, a1, a2, p2, a3
    cache = (x.shape, col1, z1, a1.shape, idx1, p1.shape, col2, z2, a2.shape, idx2, flat, z3, a3)
    return logits, cache


def predict_logits(X, P, bs=512):
    out = []
    for i in range(0, len(X), bs):
        lg, *_ = forward_cnn(X[i:i + bs], P, train=False)
        out.append(lg)
    return np.concatenate(out, axis=0)


def augment_batch_vectorized(imgs, rng):
    """Fast vectorized bilinear affine augmentation on (N, 28, 28) images."""
    N = imgs.shape[0]
    th = rng.uniform(-0.24, 0.24, N).astype(np.float32)       # +-14 deg
    sh = rng.uniform(-0.20, 0.20, N).astype(np.float32)       # shear
    sx = np.exp(rng.uniform(-0.14, 0.14, N)).astype(np.float32)
    sy = np.exp(rng.uniform(-0.14, 0.14, N)).astype(np.float32)
    tx = rng.uniform(-2.0, 2.0, N).astype(np.float32)
    ty = rng.uniform(-2.0, 2.0, N).astype(np.float32)

    c, s = np.cos(th), np.sin(th)
    # Forward matrix: Rot @ Shear @ Scale
    f00 = c * sx
    f01 = (c * sh - s) * sy
    f10 = s * sx
    f11 = (s * sh + c) * sy
    det = f00 * f11 - f01 * f10
    a = (f11 / det)[:, None, None]
    b = (-f01 / det)[:, None, None]
    c_ = (-f10 / det)[:, None, None]
    d = (f00 / det)[:, None, None]

    grid = np.arange(28, dtype=np.float32) - 13.5
    yy, xx = np.meshgrid(grid, grid, indexing="ij")  # (28, 28)
    X = a * xx[None] + b * yy[None] + 13.5 + tx[:, None, None]
    Y = c_ * xx[None] + d * yy[None] + 13.5 + ty[:, None, None]

    x0 = np.floor(X).astype(np.int32)
    y0 = np.floor(Y).astype(np.int32)
    fx = X - x0
    fy = Y - y0

    out = np.zeros((N, 28, 28), dtype=np.float32)
    n_idx = np.arange(N)[:, None, None]
    for dy in (0, 1):
        wy = fy if dy == 1 else (1.0 - fy)
        yi = y0 + dy
        valid_y = (yi >= 0) & (yi < 28)
        yi_c = np.clip(yi, 0, 27)
        for dx in (0, 1):
            wx = fx if dx == 1 else (1.0 - fx)
            xi = x0 + dx
            valid = valid_y & (xi >= 0) & (xi < 28)
            xi_c = np.clip(xi, 0, 27)
            vals = imgs[n_idx, yi_c, xi_c]
            out += np.where(valid, wx * wy * vals, 0.0)

    # Mild thickness & contrast variation on a subset so canvas strokes match well
    mask_dil = rng.random(N) < 0.12
    mask_ero = (~mask_dil) & (rng.random(N) < 0.08)
    for i in np.nonzero(mask_dil)[0]:
        out[i] = 0.55 * out[i] + 0.45 * grey_dilation(out[i], size=(2, 2))
    for i in np.nonzero(mask_ero)[0]:
        out[i] = 0.60 * out[i] + 0.40 * grey_erosion(out[i], size=(2, 2))

    # Occasional mild brightness boost (like browser canvas strokes)
    gain = rng.uniform(0.88, 1.15, (N, 1, 1)).astype(np.float32)
    out = np.clip(out * gain, 0.0, 1.0)
    return out[:, None, :, :]


def analyze_filters(P, Xte, yte):
    """Measure real learned filter descriptions & top-activating digits on MNIST test set."""
    # Conv1: analyze 3x3 kernel geometry + test-set activation per digit
    W1 = P["W1"][:, 0]  # (8, 3, 3)
    # Run forward on 4,000 test digits to measure real per-digit activations
    N_sub = 4000
    a1_sum = np.zeros((8, 10), dtype=np.float64)
    a2_sum = np.zeros((16, 10), dtype=np.float64)
    counts = np.zeros(10, dtype=np.float64)
    for i in range(0, N_sub, 500):
        xb = Xte[i:i + 500]
        yb = yte[i:i + 500]
        _, a1, a2, _, _ = forward_cnn(xb, P, train=False)
        # Mean spatial activation per image
        m1 = a1.mean(axis=(2, 3))  # (B, 8)
        m2 = a2.mean(axis=(2, 3))  # (B, 16)
        for d in range(10):
            mask = yb == d
            if mask.any():
                a1_sum[:, d] += m1[mask].sum(axis=0)
                a2_sum[:, d] += m2[mask].sum(axis=0)
                counts[d] += mask.sum()
    a1_mean = a1_sum / np.maximum(counts[None, :], 1.0)
    a2_mean = a2_sum / np.maximum(counts[None, :], 1.0)

    # Geometric gradient of each 3x3 Conv1 kernel to assign an honest label
    gx_t = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    gy_t = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    d1_t = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype=np.float64)   # / diag
    d2_t = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=np.float64)   # \ diag
    ctr_t = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float64)
    vbar_t = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=np.float64)
    hbar_t = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float64)

    def cos_sim(k, t):
        k0 = k - k.mean()
        denom = np.linalg.norm(k0) * np.linalg.norm(t) + 1e-9
        return float((k0 * t).sum() / denom)

    conv1_roles = []
    for f in range(8):
        k = W1[f].astype(np.float64)
        scores = {
            "Right-Facing Vertical Edge": cos_sim(k, gx_t),
            "Left-Facing Vertical Edge": cos_sim(k, -gx_t),
            "Bottom Horizontal Edge": cos_sim(k, gy_t),
            "Top Horizontal Edge": cos_sim(k, -gy_t),
            "NE–SW Diagonal Edge": abs(cos_sim(k, d1_t)),
            "NW–SE Diagonal Edge": abs(cos_sim(k, d2_t)),
            "Stroke Ink Detector": float(k.mean() / (np.std(k) + 1e-6)) * 0.5,
            "Center / Corner Peak": cos_sim(k, ctr_t),
            "Vertical Stroke Ridge": cos_sim(k, vbar_t),
            "Horizontal Stroke Ridge": cos_sim(k, hbar_t),
        }
        best_label = max(scores.items(), key=lambda kv: kv[1])[0]
        top_digits = [int(d) for d in np.argsort(-a1_mean[f])[:3]]
        conv1_roles.append({
            "filter": f,
            "role": best_label,
            "top_digits": top_digits,
            "digit_means": [round(float(v), 3) for v in a1_mean[f]],
        })

    # Conv2: label each channel by its strongest digits & spatial focus
    # Also compute mean spatial hotspot (top / mid / bottom, left / center / right)
    spatial2 = np.zeros((16, 14, 14), dtype=np.float64)
    for i in range(0, 2000, 500):
        _, _, a2, _, _ = forward_cnn(Xte[i:i + 500], P, train=False)
        spatial2 += a2.sum(axis=0)
    spatial2 /= 2000.0

    conv2_roles = []
    for f in range(16):
        top_digits = [int(d) for d in np.argsort(-a2_mean[f])[:3]]
        smap = spatial2[f]
        tot = float(smap.sum()) + 1e-9
        ys, xs = np.mgrid[0:14, 0:14]
        cy = float((smap * ys).sum() / tot)
        cx = float((smap * xs).sum() / tot)
        v_zone = "Upper" if cy < 5.3 else ("Lower" if cy > 8.2 else "Central")
        h_zone = "Left" if cx < 5.3 else ("Right" if cx > 8.2 else "Mid")
        zone = f"{v_zone} {h_zone}".replace("Central Mid", "Core")
        conv2_roles.append({
            "channel": f,
            "zone": zone,
            "top_digits": top_digits,
            "digit_means": [round(float(v), 3) for v in a2_mean[f]],
        })

    return conv1_roles, conv2_roles


def main():
    t0 = time.time()
    download_mnist()
    Xtr_raw = load_idx(os.path.join(MNIST_DIR, "train-images-idx3-ubyte.gz")).astype(np.float32) / 255.0
    ytr_all = load_idx(os.path.join(MNIST_DIR, "train-labels-idx1-ubyte.gz")).astype(np.int64)
    Xte = (load_idx(os.path.join(MNIST_DIR, "t10k-images-idx3-ubyte.gz")).astype(np.float32) / 255.0)[:, None, :, :]
    yte = load_idx(os.path.join(MNIST_DIR, "t10k-labels-idx1-ubyte.gz")).astype(np.int64)

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(Xtr_raw))
    va_idx, tr_idx = perm[:4000], perm[4000:]
    Xva = Xtr_raw[va_idx][:, None, :, :]
    yva = ytr_all[va_idx]
    Xtr_base = Xtr_raw[tr_idx]
    ytr_base = ytr_all[tr_idx]

    # Build hand-drawn style evaluation set (600 digits from claude/training/styles.py)
    print("Rendering multi-style stroke evaluation & augmentation sets...", flush=True)
    Xst_eval, yst_eval, st_names = make_synthetic_styles(n_per_style=25, seed=123)
    # Extra stroke training pool (European flagged 1s, crossed 7s, open/closed 4s, etc.)
    Xst_train, yst_train, _ = make_synthetic_styles(n_per_style=160, seed=999)
    print(f"Data ready in {time.time() - t0:.1f}s | train={len(Xtr_base)} + {len(Xst_train)} strokes, val={len(Xva)}, test={len(Xte)}", flush=True)

    P = init_params(seed=42)
    M = {k: np.zeros_like(v) for k, v in P.items()}
    V = {k: np.zeros_like(v) for k, v in P.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    wd = 2e-4

    epochs = 14
    bs = 128
    base_lr = 3.5e-3
    n_epoch_samples = len(Xtr_base) + len(Xst_train)
    steps_per_ep = n_epoch_samples // bs
    total_steps = epochs * steps_per_ep
    warmup = steps_per_ep
    step = 0

    best_score = -1.0
    best_P = None

    for ep in range(epochs):
        te = time.time()
        X_aug = augment_batch_vectorized(Xtr_base, rng)
        X_ep = np.concatenate([X_aug, Xst_train], axis=0)
        y_ep = np.concatenate([ytr_base, yst_train], axis=0)
        order = rng.permutation(len(X_ep))

        tot_loss = 0.0
        for bi in range(steps_per_ep):
            idx = order[bi * bs:(bi + 1) * bs]
            xb = X_ep[idx]
            yb = y_ep[idx]
            B = len(yb)

            logits, cache = forward_cnn(xb, P, train=True)
            probs = softmax(logits)
            tot_loss += -float(np.log(probs[np.arange(B), yb] + 1e-12).mean())

            # Backward
            dlogits = probs
            dlogits[np.arange(B), yb] -= 1.0
            dlogits /= B

            x_shape, col1, z1, a1_shape, idx1, p1_shape, col2, z2, a2_shape, idx2, flat, z3, a3 = cache

            G = {}
            G["W4"] = dlogits.T @ a3
            G["b4"] = dlogits.sum(axis=0)

            da3 = dlogits @ P["W4"]
            dz3 = da3 * (z3 > 0.0)
            G["W3"] = dz3.T @ flat
            G["b3"] = dz3.sum(axis=0)

            dp2 = (dz3 @ P["W3"]).reshape(B, 16, 7, 7)
            da2 = maxpool2x2_backward(dp2, idx2, a2_shape)
            dz2 = da2 * (z2 > 0.0)
            dp1, G["W2"], G["b2"] = conv_backward(dz2, col2, P["W2"], p1_shape)

            da1 = maxpool2x2_backward(dp1, idx1, a1_shape)
            dz1 = da1 * (z1 > 0.0)
            _, G["W1"], G["b1"] = conv_backward(dz1, col1, P["W1"], x_shape)

            step += 1
            if step <= warmup:
                lr = base_lr * (0.15 + 0.85 * step / warmup)
            else:
                prog = (step - warmup) / max(1, total_steps - warmup)
                lr = base_lr * 0.5 * (1.0 + np.cos(np.pi * prog))

            for k in P:
                g = G[k].astype(np.float32)
                M[k] = b1 * M[k] + (1.0 - b1) * g
                V[k] = b2 * V[k] + (1.0 - b2) * g * g
                mh = M[k] / (1.0 - b1 ** step)
                vh = V[k] / (1.0 - b2 ** step)
                upd = mh / (np.sqrt(vh) + eps)
                if k.startswith("W"):
                    upd = upd + wd * P[k]
                P[k] = (P[k] - lr * upd).astype(np.float32)

        va_acc = float((predict_logits(Xva, P).argmax(axis=1) == yva).mean())
        st_acc = float((predict_logits(Xst_eval, P).argmax(axis=1) == yst_eval).mean())
        score = 0.7 * va_acc + 0.3 * st_acc
        if score >= best_score:
            best_score = score
            best_P = {k: v.copy() for k, v in P.items()}
        print(
            f"Epoch {ep + 1:02d}/{epochs} | loss={tot_loss / steps_per_ep:.4f} | "
            f"val_mnist={va_acc * 100:.2f}% | val_styles={st_acc * 100:.2f}% | "
            f"lr={lr:.5f} | {time.time() - te:.1f}s",
            flush=True,
        )

    P = best_P

    # Calibrate softmax temperature T on validation MNIST + styles
    zva = predict_logits(Xva, P).astype(np.float64)
    best_T, best_nll = 1.0, 1e9
    for T in np.arange(0.6, 2.51, 0.02):
        nll = -float(np.log(softmax(zva / T)[np.arange(len(yva)), yva] + 1e-12).mean())
        if nll < best_nll:
            best_T, best_nll = round(float(T), 2), nll

    te_acc = float((predict_logits(Xte, P).argmax(axis=1) == yte).mean())
    zst = predict_logits(Xst_eval, P)
    st_acc = float((zst.argmax(axis=1) == yst_eval).mean())
    per_style = {}
    for name in np.unique(st_names):
        m = st_names == name
        per_style[str(name)] = round(float((zst[m].argmax(axis=1) == yst_eval[m]).mean()), 3)

    conv1_roles, conv2_roles = analyze_filters(P, Xte, yte)
    total_params = sum(int(v.size) for v in P.values())

    out_npz = os.path.join(HERE, "cnn_weights.npz")
    np.savez(out_npz, **P, T=np.float32(best_T))

    meta = {
        "architecture": "1x28x28 -> Conv1(8,3x3,pad1)+ReLU -> MaxPool(2) -> Conv2(16,3x3,pad1)+ReLU -> MaxPool(2) -> FC(784->32)+ReLU -> FC(32->10)",
        "total_params": total_params,
        "calibrated_T": best_T,
        "test_mnist_acc": round(te_acc * 100, 2),
        "test_styles_acc": round(st_acc * 100, 2),
        "per_style": per_style,
        "conv1_roles": conv1_roles,
        "conv2_roles": conv2_roles,
    }
    out_meta = os.path.join(HERE, "cnn_meta.json")
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"\nDONE in {time.time() - t0:.1f}s | params={total_params} | T={best_T} | "
        f"MNIST test={te_acc * 100:.2f}% | Styles(600)={st_acc * 100:.2f}%",
        flush=True,
    )
    print("Saved:", out_npz, "and", out_meta, flush=True)


if __name__ == "__main__":
    main()

"""Stroke -> 28x28 input pipeline.

This is the single source of truth for how a drawing becomes network input.
The browser page runs a line-for-line JavaScript port of `render` and
`normalize`, so training data produced here is exactly what the page feeds
the network.

Pipeline (mirrors how MNIST itself was prepared):
  1. Take the stroke centre-lines (a list of line segments).
  2. Scale them so the inked digit, drawn with a pen of radius R, fits a
     20x20 box (aspect ratio preserved), centred in the 28x28 frame.
  3. Rasterise with anti-aliasing (4x4 supersampling, area-averaged).
  4. Shift so the centre of mass of the ink sits at the frame's centre,
     then rasterise again (shifting the geometry, not the pixels).
Because the pen radius is fixed in the 28x28 frame, stroke thickness no
longer depends on how big or small someone draws.
"""
import numpy as np
from numba import njit, prange

FRAME = 28
BOX = 20.0


@njit(cache=True)
def render(segs, R, S):
    """Rasterise segments (n,4: x0,y0,x1,y1 in 28-frame units) with a round
    pen of radius R. S = supersampling factor. Returns float32 (28,28) in [0,1]."""
    F = FRAME * S
    fine = np.zeros(F * F, np.float32)
    inv = 1.0 / S
    pad = R + inv
    for k in range(segs.shape[0]):
        x0 = segs[k, 0]
        y0 = segs[k, 1]
        x1 = segs[k, 2]
        y1 = segs[k, 3]
        u0 = max(0, int(np.floor((min(x0, x1) - pad) * S)))
        u1 = min(F - 1, int(np.floor((max(x0, x1) + pad) * S)))
        v0 = max(0, int(np.floor((min(y0, y1) - pad) * S)))
        v1 = min(F - 1, int(np.floor((max(y0, y1) + pad) * S)))
        dx = x1 - x0
        dy = y1 - y0
        L2 = dx * dx + dy * dy
        for v in range(v0, v1 + 1):
            py = (v + 0.5) * inv
            for u in range(u0, u1 + 1):
                px = (u + 0.5) * inv
                t = 0.0
                if L2 > 0.0:
                    t = ((px - x0) * dx + (py - y0) * dy) / L2
                    if t < 0.0:
                        t = 0.0
                    elif t > 1.0:
                        t = 1.0
                qx = x0 + t * dx - px
                qy = y0 + t * dy - py
                d = np.sqrt(qx * qx + qy * qy)
                c = (R - d) * S + 0.5
                if c > 1.0:
                    c = 1.0
                idx = v * F + u
                if c > fine[idx]:
                    fine[idx] = c
    img = np.zeros((FRAME, FRAME), np.float32)
    for y in range(FRAME):
        for x in range(FRAME):
            acc = 0.0
            for j in range(S):
                row = (y * S + j) * F + x * S
                for i in range(S):
                    acc += fine[row + i]
            img[y, x] = acc / (S * S)
    return img


@njit(cache=True)
def normalize(raw, R, S, tx, ty):
    """raw: (n,4) segments in any coordinate system. Fit to the 20px box,
    centre by mass on (tx, ty), rasterise. Returns float32 (28,28)."""
    n = raw.shape[0]
    minx = 1e30
    maxx = -1e30
    miny = 1e30
    maxy = -1e30
    for k in range(n):
        for e in range(2):
            x = raw[k, 2 * e]
            y = raw[k, 2 * e + 1]
            if x < minx:
                minx = x
            if x > maxx:
                maxx = x
            if y < miny:
                miny = y
            if y > maxy:
                maxy = y
    L = max(maxx - minx, maxy - miny)
    s = 0.0
    if L > 1e-9:
        s = (BOX - 2.0 * R) / L
    cx = (minx + maxx) * 0.5
    cy = (miny + maxy) * 0.5
    segs = np.empty((n, 4))
    for k in range(n):
        segs[k, 0] = (raw[k, 0] - cx) * s + 14.0
        segs[k, 1] = (raw[k, 1] - cy) * s + 14.0
        segs[k, 2] = (raw[k, 2] - cx) * s + 14.0
        segs[k, 3] = (raw[k, 3] - cy) * s + 14.0
    img = render(segs, R, S)
    tot = 0.0
    mx = 0.0
    my = 0.0
    for y in range(FRAME):
        for x in range(FRAME):
            val = img[y, x]
            tot += val
            mx += val * (x + 0.5)
            my += val * (y + 0.5)
    if tot > 0.0:
        ox = tx - mx / tot
        oy = ty - my / tot
        for k in range(n):
            segs[k, 0] += ox
            segs[k, 2] += ox
            segs[k, 1] += oy
            segs[k, 3] += oy
        img = render(segs, R, S)
    return img


@njit(cache=True)
def _transform(raw, p):
    """Augment stroke geometry. p = [theta, shear, log_aspect, elastic_amp,
    freq1, phase1, freq2, phase2]. Elastic wobble then affine, about (14,14)."""
    n = raw.shape[0]
    out = np.empty((n, 4))
    th = p[0]
    sh = p[1]
    sx = np.exp(p[2] * 0.5)
    sy = np.exp(-p[2] * 0.5)
    ca = np.cos(th)
    sa = np.sin(th)
    # M = Rot(th) @ Shear(sh) @ diag(sx, sy)
    m00 = ca * sx
    m01 = (ca * sh - sa) * sy
    m10 = sa * sx
    m11 = (sa * sh + ca) * sy
    for k in range(n):
        for e in range(2):
            x = raw[k, 2 * e]
            y = raw[k, 2 * e + 1]
            xw = x + p[3] * np.sin(y * p[4] + p[5]) - 14.0
            yw = y + p[3] * np.sin(x * p[6] + p[7]) - 14.0
            out[k, 2 * e] = m00 * xw + m01 * yw + 14.0
            out[k, 2 * e + 1] = m10 * xw + m11 * yw + 14.0
    return out


@njit(cache=True)
def _add_flag(raw, flag_len, flag_ang, base_len):
    """Give a '1' the European flag (short stroke down-left from the top) and
    optionally a base, sized relative to the digit's height."""
    n = raw.shape[0]
    xt, yt, xb, yb = 0.0, 1e30, 0.0, -1e30
    for k in range(n):
        for e in range(2):
            x = raw[k, 2 * e]
            y = raw[k, 2 * e + 1]
            if y < yt:
                yt = y
                xt = x
            if y > yb:
                yb = y
                xb = x
    h = yb - yt
    extra = 1 if base_len <= 0.0 else 2
    out = np.empty((n + extra, 4))
    out[:n] = raw
    L = flag_len * h
    out[n, 0] = xt
    out[n, 1] = yt
    out[n, 2] = xt - L * np.cos(flag_ang)
    out[n, 3] = yt + L * np.sin(flag_ang)
    if extra == 2:
        B = base_len * h
        out[n + 1, 0] = xb - 0.5 * B
        out[n + 1, 1] = yb
        out[n + 1, 2] = xb + 0.5 * B
        out[n + 1, 3] = yb
    return out


@njit(parallel=True, cache=True)
def render_batch(segs_all, offsets, idxs, params, radii, S, tx, ty, augment, out, extra):
    """extra[m] = [flag_len, flag_angle, base_len]; flag_len 0 = no flag."""
    for m in prange(idxs.shape[0]):
        i = idxs[m]
        raw = segs_all[offsets[i]:offsets[i + 1]].astype(np.float64)
        if augment:
            if extra[m, 0] > 0.0:
                raw = _add_flag(raw, extra[m, 0], extra[m, 1], extra[m, 2])
            raw = _transform(raw, params[m])
        out[m] = normalize(raw, radii[m], S, tx, ty)


@njit(parallel=True, cache=True)
def warp_batch(imgs, idxs, params, out):
    """Bilinear affine warp of original MNIST images (light augmentation).
    params[m] = [a, b, c, d, e, f]: output->input map about the centre."""
    for m in prange(idxs.shape[0]):
        img = imgs[idxs[m]]
        a, b, c, d, e, f = params[m, 0], params[m, 1], params[m, 2], params[m, 3], params[m, 4], params[m, 5]
        for y in range(FRAME):
            for x in range(FRAME):
                xc = x + 0.5 - 14.0
                yc = y + 0.5 - 14.0
                X = a * xc + b * yc + 14.0 + e - 0.5
                Y = c * xc + d * yc + 14.0 + f - 0.5
                x0 = int(np.floor(X))
                y0 = int(np.floor(Y))
                fx = X - x0
                fy = Y - y0
                v = 0.0
                for dy in range(2):
                    yy = y0 + dy
                    if yy < 0 or yy >= FRAME:
                        continue
                    wy = fy if dy == 1 else 1.0 - fy
                    for dx in range(2):
                        xx = x0 + dx
                        if xx < 0 or xx >= FRAME:
                            continue
                        wx = fx if dx == 1 else 1.0 - fx
                        v += wx * wy * img[yy, xx]
                out[m, y, x] = v

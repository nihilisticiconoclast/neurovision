"""Hand-designed stroke digits in several writing styles (incl. European
crossed 7s and flagged 1s that MNIST under-represents). Used only for
evaluation: a check on handwriting the training data never showed.
Coordinates: 0..100 box, x right, y down."""
import numpy as np


def line(*pts, n=24):
    out = [np.array(pts[0], float)]
    for a, b in zip(pts[:-1], pts[1:]):
        a, b = np.array(a, float), np.array(b, float)
        t = np.linspace(0, 1, n)[1:, None]
        out.extend(a + (b - a) * t)
    return np.array(out)


def arc(cx, cy, rx, ry, a0, a1, n=48):
    a = np.radians(np.linspace(a0, a1, n))
    return np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a)], 1)


def bez(p0, p1, p2, p3, n=32):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.array(p, float) for p in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def cat(*parts):
    return np.concatenate(parts, 0)


def fig8():
    t = np.linspace(0, 2 * np.pi, 120)
    return np.stack([50 + 25 * np.sin(2 * t), 50 - 44 * np.cos(t)], 1)


TEMPLATES = {
    0: {"oval": [arc(50, 50, 28, 43, -90, 275)],
        "round": [arc(50, 50, 36, 40, -100, 265)],
        "overlap": [arc(50, 52, 30, 42, -60, 330)]},
    1: {"stick": [line((50, 5), (50, 95))],
        "flag": [line((30, 28), (56, 5), (56, 95))],
        "flag+base": [line((30, 28), (56, 5), (56, 95)), line((34, 95), (78, 95))]},
    2: {"classic": [cat(arc(50, 32, 26, 26, 200, 385), line((74, 41), (18, 95), (86, 95)))],
        "swan": [cat(arc(50, 30, 28, 24, 190, 360), bez((78, 30), (78, 55), (30, 70), (18, 95)), line((18, 95), (85, 92)))]},
    3: {"bumps": [cat(arc(48, 28, 22, 22, 200, 450), arc(48, 72, 24, 24, 270, 520))],
        "flat-top": [cat(line((22, 8), (75, 8), (45, 40)), arc(48, 68, 26, 26, -80, 160))]},
    4: {"open": [line((55, 5), (20, 62), (85, 62)), line((65, 30), (65, 97))],
        "closed": [line((68, 97), (68, 5), (18, 65), (88, 65))],
        "box-top": [line((25, 5), (22, 55), (80, 55)), line((70, 8), (70, 97))]},
    5: {"two-stroke": [cat(line((30, 8), (28, 45)), arc(48, 66, 25, 25, 230, 500)), line((30, 8), (78, 8))],
        "one-stroke": [cat(line((78, 8), (30, 8), (28, 45)), arc(48, 66, 25, 25, 230, 500))]},
    6: {"classic": [cat(bez((65, 5), (35, 12), (20, 45), (24, 72)), arc(47, 72, 23, 23, 180, -170))],
        "straight": [cat(line((62, 5), (26, 66)), arc(48, 72, 22, 22, 195, -160))]},
    7: {"classic": [line((18, 10), (80, 10), (42, 95))],
        "crossed": [line((18, 10), (80, 10), (42, 95)), line((44, 52), (80, 52))],
        "curved": [cat(line((18, 12), (80, 12)), bez((80, 12), (60, 40), (50, 65), (48, 95)))]},
    8: {"figure": [fig8()],
        "two-loops": [arc(50, 27, 20, 20, 0, 360), arc(50, 72, 24, 24, 0, 360)]},
    9: {"stick": [cat(arc(45, 30, 22, 22, 0, -360), line((67, 30), (64, 97)))],
        "curved": [cat(arc(45, 30, 22, 22, 0, -360), bez((67, 30), (68, 60), (60, 85), (40, 95)))]},
}


def jitter(strokes, rng):
    th = rng.uniform(-0.17, 0.17)
    sh = rng.uniform(-0.2, 0.2)
    asp = np.exp(rng.uniform(-0.15, 0.15))
    M = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ np.array([[1, sh], [0, 1]]) @ np.diag([asp, 1 / asp])
    out = []
    for s in strokes:
        # smooth wobble along the stroke, like an unsteady hand
        k = len(s)
        noise = np.cumsum(rng.normal(0, 0.6, (k, 2)), 0)
        noise -= np.linspace(0, 1, k)[:, None] * noise[-1]  # pin the ends' drift
        p = (s - 50) @ M.T + 50 + noise
        out.append(p)
    return out


def strokes_to_segs(strokes):
    segs = []
    for s in strokes:
        s = np.asarray(s, float)
        if len(s) == 1:
            segs.append(np.concatenate([s, s], 1))
        else:
            segs.append(np.concatenate([s[:-1], s[1:]], 1))
    return np.concatenate(segs, 0)


def make_set(n_per=25, seed=123):
    rng = np.random.default_rng(seed)
    items = []
    for label, styles in TEMPLATES.items():
        for name, strokes in styles.items():
            for _ in range(n_per):
                items.append((jitter(strokes, rng), label, f"{label}-{name}"))
    return items

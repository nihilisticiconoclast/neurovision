"""Score the benchmark: accuracy per network, test set and drawing size, plus MNIST fed straight in.

Usage: python3 score.py cases.json results.json ../../claude/training/prep.npz summary.json
"""
import json, sys
import numpy as np

HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
REPO = __import__("os").path.join(HERE, "..", "..")
sys.path.insert(0, f"{REPO}/claude/training")
from cnn_train import forward as claude_cnn_forward  # noqa: E402


def mlp(X, W):
    a1 = np.maximum(X @ W["W1"].T + W["b1"], 0)
    a2 = np.maximum(a1 @ W["W2"].T + W["b2"], 0)
    return a2 @ W["W3"].T + W["b3"]


def gemini_cnn(X, W):
    """Gemini's CNN in NumPy: 3x3 'same' convs, 2x2 max pools, 784 -> 32 -> 10 (channel-major, as its page)."""
    from numpy.lib.stride_tricks import sliding_window_view
    def conv(x, K, b):                                      # x (N, C, H, W), K (F, C, 3, 3)
        xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        win = sliding_window_view(xp, (3, 3), axis=(2, 3))  # (N, C, H, W, 3, 3)
        return np.maximum(np.einsum("nchwij,fcij->nfhw", win, K) + b[None, :, None, None], 0)
    def pool(a):
        N, C, H, Wd = a.shape
        return a.reshape(N, C, H // 2, 2, Wd // 2, 2).max((3, 5))
    p2 = pool(conv(pool(conv(X[:, None], W["W1"], W["b1"])), W["W2"], W["b2"])).reshape(len(X), -1)
    return np.maximum(p2 @ W["W3"].T + W["b3"], 0) @ W["W4"].T + W["b4"]


def main(cases_path, results_path, prep_path, out_path):
    cases = json.load(open(cases_path))["cases"]
    res = json.load(open(results_path))
    lab = np.array([c["label"] for c in cases])
    sets = np.array([c["set"] for c in cases])
    style = np.array([c.get("style", "") for c in cases])
    d = np.load(prep_path)
    Xte, yte = d["Xte"].astype(np.float64) / 255.0, d["yte"].astype(int)

    # MNIST test set fed straight in
    cw = np.load(f"{REPO}/claude/training/final_weights.npz")
    ccw = np.load(f"{REPO}/claude/training/cnn_final_weights.npz")
    gcw = np.load(f"{REPO}/gemini-cnn/training/cnn_weights.npz")
    gw = {k: np.array(v) for k, v in res["apps"]["gemini"]["weights"].items()}
    gw["W1"], gw["W2"], gw["W3"] = gw["W1"].reshape(64, 784), gw["W2"].reshape(32, 64), gw["W3"].reshape(10, 32)
    P = {k: ccw[k].astype(np.float64) for k in ("K1", "b1", "K2", "b2", "W3", "b3")}
    z = {
        "gemini": mlp(Xte.reshape(-1, 784), gw),
        "gemini-cnn": np.concatenate([gemini_cnn(Xte[i:i + 1000], {k: gcw[k].astype(np.float64) for k in gcw.files}) for i in range(0, 10000, 1000)]),
        "claude": mlp(Xte.reshape(-1, 784), {k: cw[k].astype(np.float64) for k in cw.files}),
        "claude-cnn": np.concatenate([claude_cnn_forward(Xte[i:i + 1000], P)[1] for i in range(0, 10000, 1000)]),
    }
    params = {"gemini": sum(np.size(v) for v in gw.values()), "gemini-cnn": sum(gcw[k].size for k in gcw.files if k != "T"),
              "claude": sum(cw[k].size for k in cw.files if k != "T"), "claude-cnn": sum(ccw[k].size for k in ccw.files if k != "T")}
    summary = {"measured": res.get("measured") or __import__("datetime").date.today().isoformat(), "browser": res.get("browser", ""),
               "n": {"mnist": int(len(yte)), "mnist400": int((sets == "mnist400").sum()), "styles600": int((sets == "styles600").sum())},
               "apps": {}}
    for app in ("gemini", "gemini-cnn", "claude", "claude-cnn"):
        pr = {pl: np.array(res["apps"][app]["probs"][pl]).argmax(1) for pl in res["places"]}
        s = {"params": int(params[app]),
             "mnist": float((z[app].argmax(1) == yte).mean()),
             "mnist_per_digit": [float((z[app].argmax(1)[yte == k] == k).mean()) for k in range(10)],
             "mnist_confusion": np.bincount(yte * 10 + z[app].argmax(1), minlength=100).reshape(10, 10).tolist()}
        for setname in ("mnist400", "styles600"):
            m = sets == setname
            for pl in res["places"]:
                s[f"{setname}_{pl}"] = float((pr[pl][m] == lab[m]).mean())
            s[f"{setname}_pred_counts"] = np.bincount(pr["normal"][m], minlength=10).tolist()
            s[f"{setname}_per_digit"] = [float((pr["normal"][m & (lab == k)] == k).mean()) for k in range(10)]
        s["styles600_per_style"] = {st: float((pr["normal"][style == st] == lab[style == st]).mean()) for st in sorted(set(style[sets == "styles600"]))}
        s["errors"] = res["apps"][app]["errors"]
        summary["apps"][app] = s
    json.dump(summary, open(out_path, "w"), indent=1)
    print(f"{'':12} {'params':>7} {'MNIST':>7} | {'pen-400 normal':>14} {'half':>6} {'quarter':>7} | {'styles normal':>13} {'half':>6} {'quarter':>7}")
    for app, s in summary["apps"].items():
        print(f"{app:12} {s['params']:7,} {s['mnist']:7.2%} | {s['mnist400_normal']:14.1%} {s['mnist400_half']:6.1%} {s['mnist400_quarter']:7.1%} |"
              f" {s['styles600_normal']:13.1%} {s['styles600_half']:6.1%} {s['styles600_quarter']:7.1%}")
    for app, s in summary["apps"].items():
        print(app, "predictions on the 400 pen-stroke digits (counts of 0-9):", s["mnist400_pred_counts"])


if __name__ == "__main__":
    main(*sys.argv[1:])

"""Embed trained CNN weights and measured filter metadata into gemini-cnn/index.html."""
import base64
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_HTML = os.path.abspath(os.path.join(HERE, "..", "index.html"))
TEMPLATE = os.path.join(HERE, "page_template.html")
WEIGHTS_NPZ = os.path.join(HERE, "cnn_weights.npz")
META_JSON = os.path.join(HERE, "cnn_meta.json")


def b64_f32(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype="<f4").tobytes()).decode("ascii")


def main():
    d = np.load(WEIGHTS_NPZ)
    with open(META_JSON, "r", encoding="utf-8") as f:
        meta = json.load(f)

    payload = {
        "T": float(d["T"]),
        "totalParams": int(meta["total_params"]),
        "testMnistAcc": float(meta["test_mnist_acc"]),
        "testStylesAcc": float(meta["test_styles_acc"]),
        "conv1Roles": meta["conv1_roles"],
        "conv2Roles": meta["conv2_roles"],
        "weights": {
            "W1": b64_f32(d["W1"]),
            "b1": b64_f32(d["b1"]),
            "W2": b64_f32(d["W2"]),
            "b2": b64_f32(d["b2"]),
            "W3": b64_f32(d["W3"]),
            "b3": b64_f32(d["b3"]),
            "W4": b64_f32(d["W4"]),
            "b4": b64_f32(d["b4"]),
        },
    }

    with open(TEMPLATE, "r", encoding="utf-8") as f:
        html = f.read()

    marker = "__CNN_MODEL_JSON__"
    if marker not in html:
        raise ValueError(f"Template missing {marker}")

    out = html.replace(marker, json.dumps(payload, separators=(",", ":")))
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(out)

    print(f"Built {OUT_HTML} ({len(out):,} bytes)")


if __name__ == "__main__":
    main()

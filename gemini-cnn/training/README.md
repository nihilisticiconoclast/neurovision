# NeuroVision Gemini CNN: training code

The convolutional neural network behind `gemini-cnn/index.html`, written and trained from scratch in pure NumPy + SciPy on MNIST plus multi-style synthetic pen strokes.

## Architecture

| Layer | Output Shape | Parameters | Notes |
| :--- | :--- | ---: | :--- |
| Input | `1 × 28 × 28` | — | Center-of-mass normalized grayscale `[0, 1]` |
| Conv1 (`3×3`, pad=1) + ReLU | `8 × 28 × 28` | 80 | 8 learned spatial edge/stroke filters |
| MaxPool1 (`2×2`, stride=2) | `8 × 14 × 14` | — | 2×2 spatial downsampling |
| Conv2 (`3×3`, pad=1) + ReLU | `16 × 14 × 14` | 1,168 | 16 higher-level corner/curve/loop channels |
| MaxPool2 (`2×2`, stride=2) | `16 × 7 × 7` | — | Flattened to 784 features |
| FC1 + ReLU | `32` | 25,120 | Dense hidden representation |
| FC2 + Temperature Softmax | `10` | 330 | Calibrated class probabilities (`0`–`9`) |
| **Total** | | **26,698** | Embedded as float32 base64 in `gemini-cnn/index.html` |

## Reproduce

```bash
pip install numpy scipy
python train_cnn.py   # downloads MNIST to mnist_data/, trains 14 epochs, saves cnn_weights.npz + cnn_meta.json
python build_page.py  # embeds weights & measured filter metadata into ../index.html
```

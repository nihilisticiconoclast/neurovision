# 🧠 NeuroVision // Handwritten Digit Recognition Neural Network

[![License: MIT](https://img.shields.io/badge/License-MIT-cyan.svg)](https://opensource.org/licenses/MIT)
[![Architecture](https://img.shields.io/badge/MLP_Architecture-784_%E2%86%92_64_%E2%86%92_32_%E2%86%92_10-8B5CF6)](https://github.com/nihilisticiconoclast/neurovision)
[![Zero Dependencies](https://img.shields.io/badge/Runtime-Pure_Vanilla_JS-10B981)](https://github.com/nihilisticiconoclast/neurovision)

**NeuroVision** is a real-time, browser-based Handwritten Digit Recognition Laboratory and Neural Network Telemetry Visualizer. Built in a single self-contained HTML5 application with zero external ML framework dependencies, it implements a complete **Multi-Layer Perceptron (`784 → 64 → 32 → 10`)** with live forward-pass signal visualization, authentic **MNIST Center-of-Mass preprocessing**, receptive field heatmaps, and interactive on-device **backpropagation**.

---

## Three versions

This repo holds takes on the same brief from two models: Google Gemini's original NeuroVision in `gemini/`, and two from Anthropic's Claude. Claude's Neural Digit Reader in `claude/` is a dense network like Gemini's. Its Convolutional Digit Reader in `claude/cnn/` reads digits with a convolutional network instead. Open `index.html` and use the switch at the top to flip between them; each keeps its drawing while you look at the others. All three are single HTML files that also run on their own.

| | Gemini (`gemini/index.html`) | Claude (`claude/index.html`) | Claude CNN (`claude/cnn/index.html`) |
| :--- | :--- | :--- | :--- |
| Network | `784 → 64 → 32 → 10`, ReLU, softmax | `784 → 64 → 32 → 10`, ReLU, softmax | Two convolution layers (8, then 16 filters of 5×5, each followed by 2×2 max pooling) `→ 10`, ReLU, softmax |
| Weights | 52,650 | 52,650 | 11,274 |
| Training data | 175 synthetic digits rendered from one stroke template per digit (plus five style variants), shifted and scaled | 55,000 MNIST digits, half of them re-drawn as pen strokes with random slant, wobble and pen size | The same digits and mix as Claude's dense network |
| Where it trains | In your browser, each time the page loads (18 epochs of SGD) | Offline in NumPy ([`claude/training/`](claude/training/)); weights embedded in the page | Offline in NumPy (same folder); weights embedded in the page |
| MNIST test set (10,000 digits) | 43.5% | 98.6% | 99.3% |
| 400 MNIST test digits drawn as pen strokes, through each app's own drawing pipeline | 44.2% | 98.0% | 98.5% |
| Extras | Temperature slider, live re-training on your drawing, brush size, 28×28 pixel inspector, receptive-field maps, presets | Exact per-pixel evidence maps, evidence flow through the network, measured neuron roles on hover, neuron inspector, animated examples | Live feature maps for every filter, exact per-pixel evidence maps, evidence flow through the maps, measured filter roles on hover, each filter's weights in the inspector, animated examples |

All three networks were scored on identical images. Gemini's weights were read from its page after the at-load training in Chrome; because that training runs on canvas renderings, its weights can differ slightly between browsers.

---

## Key Features (Gemini version)

### 1. Interactive Digit Studio & MNIST Preprocessing
- **Smooth Freehand Canvas**: Supports mouse and touch input with adjustable brush thickness (`12px`–`30px`), a one-click **Clear Canvas** action, a **Random** sample generator, and instant handwritten presets (`0`–`9`).
- **Live Bounding-Box & Center-of-Mass (CoM) Reticle**: Automatically detects the bounding box of your drawn stroke, scales it proportionally into a `20×20` inner box, and centers the digit by its pixel center of mass inside a `28×28` sensor matrix.
- **Interactive 28×28 Sensor Matrix Inspector**: Hover over any pixel in the downsampled `28×28` grid to inspect exact `(x, y)` coordinates, normalized grayscale intensity (`0.00`–`1.00`), and toggle raw numeric overlays.

### 2. Real-Time Forward-Pass Topology Visualizer
- **4-Layer Neural Graph (`784 → 64 ReLU → 32 ReLU → 10 Softmax`)**: Watch activations propagate through pooled input regions, hidden ReLU layers, and the 10 output class nodes in real time as you draw.
- **Synaptic Polarity & Density Filtering**: Distinguishes **+Excitatory** (Cyan) and **-Inhibitory** (Violet) weights, with an interactive **Synapse Density** slider to isolate the strongest pathways.

### 3. Structural Feature Maps & Receptive Fields
Displays 6 specialized structural detectors that show *how* the network decomposes a handwritten number:
- **Upper Loop / Curve** (`0, 2, 3, 6, 8, 9`)
- **Lower Loop / Bowl** (`0, 3, 5, 6, 8`)
- **Vertical Stem** (`1, 4, 7, 9`)
- **Horizontal Crossbars** (`2, 4, 5, 7`)
- **Diagonal Stroke** (`2, 4, 7, 8`)
- **Center Intersection** (`3, 4, 8, 9`)

### 4. Classification Telemetry, Softmax Temperature & Live Backpropagation
- **Confidence Distribution**: Displays the top predicted digit, Softmax probability percentage, raw logit ($z$), runner-up comparison, and a live `0`–`9` horizontal probability bar chart.
- **Softmax Temperature ($T$) Control**: Adjust $T \in [0.20, 3.00]$ in real time to observe how temperature scales the probability distribution:
  $$P(y = i \mid \mathbf{x}) = \frac{\exp(z_i / T)}{\sum_{j=0}^{9} \exp(z_j / T)}$$
- **Interactive On-Device Backpropagation**: If the network misclassifies an unusual handwriting style, click the true target label (`0`–`9`) under **Teach / Correct Network** to execute live stochastic gradient descent (Cross-Entropy loss backpropagation) on your drawing.

---

## Quick Start

Because each version is a single HTML file, no build step or server installation is required. The root `index.html` holds the version switcher; `gemini/index.html`, `claude/index.html` and `claude/cnn/index.html` also open on their own.

### Option 1: Open Directly
1. Clone this repository:
   ```bash
   git clone https://github.com/nihilisticiconoclast/neurovision.git
   cd neurovision
   ```
2. Open `index.html` directly in any modern web browser.

### Option 2: Run with a Local Development Server
```bash
# Using Python 3
python3 -m http.server 8080

# Or using Node.js serve
npx serve .
```
Then visit `http://localhost:8080` in your browser.

---

## Architecture Overview

| Layer | Dimensions | Activation | Role |
| :--- | :--- | :--- | :--- |
| **Input Layer** | `28 × 28 = 784` | Normalized `[0, 1]` | Center-of-mass aligned grayscale sensor tensor |
| **Hidden Layer 1** | `64` Neurons | $\text{ReLU}(x) = \max(0, x)$ | Primary stroke, edge, and quadrant loop extraction |
| **Hidden Layer 2** | `32` Neurons | $\text{ReLU}(x) = \max(0, x)$ | High-level topological combinations |
| **Output Layer** | `10` Classes (`0`–`9`) | Temperature-Scaled Softmax | Class probability distribution & logit telemetry |

---

## Claude version: Neural Digit Reader

- Draw with a mouse, finger or stylus. **Clear** (Esc), **Undo** (Ctrl+Z) and **Example**, which animates a sample digit so you can watch the guess change mid-stroke.
- Every neuron is drawn: filled dots are firing, hollow ones are silent, and orange or blue lines show which connections push the selected digit's score up or down.
- Hover over any neuron to see what it responds to, such as "Loop at the top" or "Bar across the top", with the digits that fire it and its typical trigger. The roles are measured, not assigned: 20,000 test digits were scored on shape features and each neuron was labelled by what its activation tracks, then checked by eye. On touch screens, tap the neuron and the inspector shows the same.
- Tap any digit or neuron to inspect it. One map shows how much each inked pixel raised or lowered the score; these add up exactly to the score, because a ReLU network is linear while its on/off pattern is fixed. The other shows where a dab of extra ink would help or hurt.
- Strokes become the 28×28 input the way MNIST was prepared (20×20 box, centred by mass), with pen thickness normalised so drawing size doesn't matter. The browser code is a line-for-line port of the training pipeline, and its inputs are bit-identical to Python's.
- Training code, weights and steps to reproduce: [`claude/training/`](claude/training/).

---

## Claude CNN: Convolutional Digit Reader

- The same instrument as the dense page (pad, examples, answer card, inspector) with a convolutional network inside. Eight 5×5 filters look for edges, sixteen more combine those into shapes, and each digit's score weighs the sixteen 7×7 maps that result.
- Every feature map is drawn as a tile that lights up wherever its filter fires on your drawing. Orange or blue wires show which maps push the selected digit's score up or down.
- Hover over a map to see what its filter responds to, such as "Upright strokes", "Top of an arch" or "Pocket under a bar". The roles are measured: each filter was shown 65 probe shapes drawn with the page's own pen (strokes at 12 angles, stroke ends, corners, curves, junctions, loops) and scored on 20,000 test digits, then checked by eye. Several layer-2 filters turn out to fire on empty space framed by strokes, such as the pocket under a 5's top bar.
- Tap any digit or map to inspect it: the filter's own weights, where it fires in your drawing, and which pixels produced its output. As on the dense page, the pixel maps add up exactly to the score, because a ReLU and max-pool network is linear while its on/off pattern and pooling winners are fixed.
- It has a fifth of the dense network's weights (11,274 against 52,650) and scores higher: 99.3% against 98.6% on the MNIST test set, and 98.7% against 98.1% on the test digits re-drawn as pen strokes. On the 600 hand-designed digits in other styles it gets 599, missing one flagged 1; the dense network gets all 600.
- Same stroke pipeline as the dense page, so its inputs are bit-identical to Python's too. Training code, weights and steps to reproduce: the `cnn_` files in [`claude/training/`](claude/training/).

---

## License

Distributed under the MIT License. Feel free to fork, experiment with hidden layer dimensions, or embed the visualizer in educational machine learning projects.

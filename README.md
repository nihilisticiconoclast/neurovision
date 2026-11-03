# 🧠 NeuroVision // Handwritten Digit Recognition Neural Network

[![License: MIT](https://img.shields.io/badge/License-MIT-cyan.svg)](https://opensource.org/licenses/MIT)
[![Architecture](https://img.shields.io/badge/MLP_Architecture-784_%E2%86%92_64_%E2%86%92_32_%E2%86%92_10-8B5CF6)](https://github.com/nihilisticiconoclast/neurovision)
[![Zero Dependencies](https://img.shields.io/badge/Runtime-Pure_Vanilla_JS-10B981)](https://github.com/nihilisticiconoclast/neurovision)

**NeuroVision** is a real-time, browser-based Handwritten Digit Recognition Laboratory and Neural Network Telemetry Visualizer. Built in a single self-contained HTML5 application with zero external ML framework dependencies, it implements a complete **Multi-Layer Perceptron (`784 → 64 → 32 → 10`)** with live forward-pass signal visualization, authentic **MNIST Center-of-Mass preprocessing**, receptive field heatmaps, and interactive on-device **backpropagation**.

---

## Two versions

This repo holds two takes on the same brief: Google Gemini's original NeuroVision in `gemini/`, and Anthropic Claude's Neural Digit Reader in `claude/`. Open `index.html` and use the switch at the top to flip between them; each keeps its drawing while you look at the other. Both are single HTML files that also run on their own.

| | Gemini (`gemini/index.html`) | Claude (`claude/index.html`) |
| :--- | :--- | :--- |
| Network | `784 → 64 → 32 → 10`, ReLU, softmax | `784 → 64 → 32 → 10`, ReLU, softmax |
| Training data | 175 synthetic digits rendered from one stroke template per digit (plus five style variants), shifted and scaled | 55,000 MNIST digits, half of them re-drawn as pen strokes with random slant, wobble and pen size |
| Where it trains | In your browser, each time the page loads (18 epochs of SGD) | Offline in NumPy ([`claude/training/`](claude/training/)); weights embedded in the page |
| MNIST test set (10,000 digits) | 43.5% | 98.6% |
| 400 MNIST test digits drawn as pen strokes, through each app's own drawing pipeline | 44.2% | 98.0% |
| Extras | Temperature slider, live re-training on your drawing, brush size, 28×28 pixel inspector, receptive-field maps, presets | Exact per-pixel evidence maps, evidence flow through the network, measured neuron roles on hover, neuron inspector, animated examples |

Both networks were scored on identical images. Gemini's weights were read from its page after the at-load training in Chrome; because that training runs on canvas renderings, its weights can differ slightly between browsers.

---

## ✨ Key Features (Gemini version)

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

## 🚀 Quick Start

Because each version is a single HTML file, no build step or server installation is required. The root `index.html` holds the version switcher; `gemini/index.html` and `claude/index.html` also open on their own.

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

## 🛠️ Architecture Overview

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

## 📄 License

Distributed under the MIT License. Feel free to fork, experiment with hidden layer dimensions, or embed the visualizer in educational machine learning projects.

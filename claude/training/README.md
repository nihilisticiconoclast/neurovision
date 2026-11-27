# Neural Digit Reader: training code

The network behind `claude/index.html`: a 784 → 64 → 32 → 10 fully connected network (ReLU, softmax), written and trained from scratch in NumPy on MNIST, plus the build step that embeds it in the page. The convolutional network behind `claude/cnn/index.html` lives here too and shares the data and stroke pipeline; see [Convolutional network](#convolutional-network) below.

## Results

| Test set | Accuracy |
| :--- | :--- |
| MNIST test set (10,000 digits) | 98.6% |
| The same test digits re-drawn as pen strokes through the page's pipeline | 98.1% |
| Hand-designed digits in other styles (crossed 7s, flagged 1s, closed 4s…), 600 digits | 100% |

## Files

- `prep.py` loads MNIST, skeletonises every digit into stroke segments, and calibrates the pen radius (1.2 px) and the centre-of-mass target (14.5, 14.5) from the data.
- `strokes.py` turns strokes into the 28×28 input: fit to a 20×20 box, anti-alias with 4×4 supersampling, centre by mass, the way MNIST was prepared. The page runs a line-for-line JavaScript port, and its inputs are bit-identical to this file's.
- `styles.py` holds hand-designed digits in other writing styles, used only for evaluation.
- `train.py` is the network itself: forward pass, backprop, AdamW with a cosine schedule, and temperature scaling for honest probabilities. Each epoch mixes lightly warped MNIST digits with digits re-drawn from their skeletons with random slant, shear, wobble and pen size, and gives 40% of the re-drawn 1s a European flag.
- `build_page.py` re-orders hidden neurons for a tidier diagram (outputs unchanged), picks the examples, and embeds the float32 weights into `page_template.html`.
- `neuron_roles.py` works out what each hidden neuron responds to, for the hover tooltips. It scores 20,000 test digits on shape features (closed loops and where they sit, straight strokes and their angle, bars, where strokes end, open sides, an empty middle), rank-correlates every neuron's activation with each feature, and records which digits fire it and its typical trigger (the activation-weighted average input). Labels come from the best-matching feature; 14 were corrected after checking the contact sheets by eye, listed in `OVERRIDES` with the reason. Neurons with no single clean feature are labelled as mixed.
- `final_weights.npz` holds the trained weights as used in the page: `W1` (64×784), `b1`, `W2` (32×64), `b2`, `W3` (10×32), `b3`, and temperature `T`. Weight matrices are `[out, in]`, so a layer is `relu(x @ W.T + b)`.

## Reproduce

```bash
pip install numpy scipy scikit-image numba pillow
# put the four MNIST .gz files (e.g. from github.com/fgnt/mnist) in ./mnist
python3 prep.py ./mnist prep.npz          # ~40 s
python3 train.py --out runs/m64x32.npz    # ~5 min on 2 CPU cores
python3 build_page.py runs/m64x32.npz out/          # writes out/final_weights.npz
python3 neuron_roles.py out/final_weights.npz out/  # measures neuron roles (~25 s), writes contact sheets
python3 build_page.py runs/m64x32.npz out/          # builds again, now with the tooltips
cp out/index.html ../index.html
```

`train.py` options: `--h1`, `--h2` (layer sizes), `--epochs`, `--mix both|redraw|orig`, `--flag` (share of 1s given a flag).

## Convolutional network

The network behind `claude/cnn/index.html`: two convolution layers (8 filters of 5×5, then 16 filters of 5×5×8, each with ReLU and 2×2 max pooling) and a 784 → 10 readout, 11,274 weights in all. It is written and trained from scratch in NumPy on exactly the same data mix as the dense network, and uses the same `prep.py`, `strokes.py` and `styles.py`.

| Test set | Accuracy |
| :--- | :--- |
| MNIST test set (10,000 digits) | 99.3% |
| The same test digits re-drawn as pen strokes through the page's pipeline | 98.7% |
| Hand-designed digits in other styles (crossed 7s, flagged 1s, closed 4s…), 600 digits | 99.8% (one flagged 1 missed) |

- `cnn_train.py` is the network itself: im2col convolutions, max pooling whose ties go to the first maximum in reading order (as in the page), backprop (checked against finite differences at start-up), AdamW with a cosine schedule, and temperature scaling. Its split, augmentation and evaluation are copied from `train.py`, so both networks see the same digits.
- `cnn_build_page.py` re-orders filters for a tidier diagram (outputs unchanged), picks the examples, embeds the float32 weights into `cnn_page_template.html`, and writes parity fixtures. The fixtures include the evidence the page shows (pixel sensitivities and the contributions behind every wire), computed a second way with SciPy.
- `cnn_roles.py` works out what each filter responds to, for the hover tooltips. It measures responses to 65 probe shapes drawn with the page's pen (straight strokes at 12 angles, stroke ends, corners, curves, junctions, crossings, loops, a dot), averaged over 16 sub-pixel placements. On 20,000 test digits it also records which digits fire each map hardest, how much of the map lights up, and the typical trigger (the average input patch around each firing spot, weighted by activation squared). Layer-1 filters are named by stroke angle and by which side of the stroke they fire on, found by sliding a stroke across them. Labels come from the strongest probe; 16 were corrected or reworded after checking the contact sheets by eye, listed in `OVERRIDES` with the reason. Several layer-2 filters fire on empty space framed by strokes, which no single-stroke probe can show.
- `cnn_final_weights.npz` holds the trained weights as used in the page: `K1` (8×1×5×5), `b1`, `K2` (16×8×5×5), `b2`, `W3` (10×784, inputs ordered map, row, column), `b3`, and temperature `T`. Convolutions zero-pad by 2, so each map keeps its input's size.

### Reproduce

After `prep.py` (above):

```bash
python3 cnn_train.py --out runs/cnn8x16.npz                  # ~16 min on 2 CPU cores
python3 cnn_build_page.py runs/cnn8x16.npz out/cnn/          # writes out/cnn/final_weights.npz
python3 cnn_roles.py out/cnn/final_weights.npz out/cnn/      # measures filter roles (~40 s), writes contact sheets
python3 cnn_build_page.py runs/cnn8x16.npz out/cnn/          # builds again, now with the tooltips
cp out/cnn/index.html ../cnn/index.html
```

`cnn_train.py` options: `--c1`, `--c2` (filters per layer), `--epochs`, `--mix both|redraw|orig`, `--flag`.

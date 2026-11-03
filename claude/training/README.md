# Neural Digit Reader: training code

The network behind `claude/index.html`: a 784 → 64 → 32 → 10 fully connected network (ReLU, softmax), written and trained from scratch in NumPy on MNIST, plus the build step that embeds it in the page.

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
- `final_weights.npz` holds the trained weights as used in the page: `W1` (64×784), `b1`, `W2` (32×64), `b2`, `W3` (10×32), `b3`, and temperature `T`. Weight matrices are `[out, in]`, so a layer is `relu(x @ W.T + b)`.

## Reproduce

```bash
pip install numpy scipy scikit-image numba
# put the four MNIST .gz files (e.g. from github.com/fgnt/mnist) in ./mnist
python3 prep.py ./mnist prep.npz          # ~40 s
python3 train.py --out runs/m64x32.npz    # ~5 min on 2 CPU cores
python3 build_page.py runs/m64x32.npz out/
cp out/index.html ../index.html
```

`train.py` options: `--h1`, `--h2` (layer sizes), `--epochs`, `--mix both|redraw|orig`, `--flag` (share of 1s given a flag).

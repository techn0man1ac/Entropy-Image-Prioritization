# RGB Image Block Entropy Prioritization

A small, lightweight Python prototype that uses **Shannon entropy of real RGB colors** to estimate how visually varied each image block is, then turns that value into a simple **priority ranking**.

The project is intentionally simple. It uses only **OpenCV, NumPy, and Matplotlib**. There is no machine learning model, no neural network, and no heavy image-processing pipeline.

![Screenshot with Milka cat](https://raw.githubusercontent.com/techn0man1ac/Entropy-Image-Prioritization/refs/heads/main/Python/Figure_1.png)

---

## What does the project do?

The program follows five simple steps:

1. Load a color image.
2. Split it into square blocks, for example `100 x 100` pixels.
3. Calculate Shannon entropy from the **full RGB color of every pixel** in each block.
4. Convert the entropy values into a relative `0-100` priority score and rank the blocks.
5. Show the result as a color heatmap and highlight the highest-priority blocks.

In simple words:

> **The more different RGB colors a block contains, the more information it gets in the entropy score, and the higher its priority can become.**

This is a lightweight way to decide which image regions could be inspected first by a more expensive algorithm later.

---

## Important: this is real RGB entropy

The original prototype converted the image to grayscale before calculating entropy. This version does **not** do that for entropy.

For every pixel, the three color channels are kept together:

```text
RGB = (R, G, B)
```

The code packs each RGB triplet into one integer and builds a histogram from the actual colors that appear in the block.

So these colors are treated as different values:

```text
(255, 0, 0)     red
(0, 255, 0)     green
(0, 0, 255)     blue
```

This keeps the calculation simple while using the full color information instead of reducing everything to grayscale.

The Shannon entropy formula is:

```text
H = -sum(p * log2(p))
```

where `p` is the probability of each RGB color inside the block.

---

## What is the priority score?

The prototype uses a deliberately simple rule:

```text
priority score = normalized RGB entropy
```

The lowest entropy found in the image becomes `0`, and the highest becomes `100`.

This is a **relative score inside the current image**. A score of `100` means that the block has the highest entropy in that image, not that it has a universal or absolute importance of `100`.

The program then selects the top percentage of blocks, for example:

```text
Top 10% blocks → prioritized
Remaining 90% → not prioritized
```

The highest-priority blocks are marked with red borders and rank numbers in `Figure_1.png`. The same figure is displayed automatically when the function runs, so the map can be inspected immediately without opening the PNG manually.

This is intentionally not a machine-learning prediction. It is a small and explainable ranking rule.

---

## Why use entropy for prioritization?

A block with very similar pixels is usually visually simple:

```text
100 100 101 100
100 101 100 100
```

A block containing many different colors is more varied:

```text
 20 180  70 230
 90  40 210 120
```

Entropy gives a compact number for this difference.

This can be useful as a **cheap first filter** before running a more expensive task such as object detection, OCR, segmentation, or detailed analysis.

The important limitation is that entropy is not the same as semantic importance:

```text
high entropy != important object
low entropy  != unimportant object
```

For example, image noise or a very detailed texture can also have high entropy.

---

## Why not use a machine-learning model?

This repository is intentionally a **small educational/prototype image-processing project**.

The goal is to keep the idea:

- easy to read;
- easy to run;
- easy to explain;
- fast enough for a simple computer;
- free from large model downloads.

The entropy ranking can later become one input to a larger computer-vision system, but this project does not need a neural network to demonstrate the idea.

---

## How the code stays lightweight

The implementation avoids expensive techniques such as:

- neural networks;
- dense per-pixel optimization;
- large pre-trained models;
- large intermediate images for every block.

For each block, it only needs to count the RGB colors that actually occur in that block and then apply the Shannon entropy formula.

The entropy map is also written directly into the matching image coordinates, so the block boundaries stay aligned with the original image. No `cv2.resize()` step is used to stretch a smaller map over the image.

---

## Installation

Python 3.8+ is recommended.

Install the three required libraries:

```bash
pip install opencv-python numpy matplotlib
```

---

## Run the demo

Keep these five files in the same project folder:

```text
├── LICENSE
├── README.md
└── Python/
    ├── Figure_1.png
    ├── MilkaCat.jpg
    └── main.py
```

Then run:

```bash
python main.py
```

The script automatically uses `MilkaCat.jpg` next to `main.py`, saves the updated result to `Figure_1.png`, and **opens the generated image-map on screen**.

The program also prints the highest-priority blocks to the terminal, including their coordinates, RGB entropy, and relative priority score.

The map display is enabled by default through `show_result=True`. For a headless machine or a server without a graphical display, call the function with `show_result=False`; the PNG will still be generated normally.

---

## Main parameters

The example at the bottom of `main.py` uses:

```python
block_size=100
alpha=0.30
top_percent=10.0
```

| Parameter | Type | Default | Meaning |
| --- | --- | --- | --- |
| `image_path` | `str` or `Path` | required | Input image path |
| `output_path` | `str` or `Path` | `Figure_1.png` | Output figure path |
| `block_size` | `int` | `100` | Width and height of each square block in pixels |
| `alpha` | `float` | `0.30` | Heatmap opacity from `0.0` to `1.0` |
| `top_percent` | `float` | `10.0` | Percentage of blocks selected for priority |
| `show_result` | `bool` | `True` | Display the generated map window after saving |

### Choosing the block size

The block size controls the balance between detail and simplicity.

- Smaller blocks give more local detail but create more blocks.
- Larger blocks are faster and smoother but less local.

`100 x 100` is a simple starting point for the included example.

---

## Output

`Figure_1.png` contains two views:

### Original Image

The untouched input image.

### RGB Entropy + Priority

The right side combines three simple pieces of information:

1. **Heatmap color** — relative RGB entropy of each block.
2. **White grid** — exact block boundaries.
3. **Red borders and rank numbers** — blocks selected by the priority rule.

This makes the prototype useful both as a visual explanation and as a starting point for a later image-processing pipeline.

---

## Example priority flow

```text
Color image
    |
    v
Split into blocks
    |
    v
Exact RGB color counts
    |
    v
Shannon entropy
    |
    v
Normalize to 0-100
    |
    v
Sort blocks
    |
    v
Select top N%
    |
    v
Use these blocks first in a later task
```

For example, a future pipeline could be:

```text
RGB entropy priority
        |
        v
Top 10% blocks
        |
        v
Object detector / OCR / segmentation
```

The current repository stops at the priority ranking step. It does not try to decide what object is inside a block.

---

## Limitations

This is still a **small educational/prototype project**, not a complete production prioritization system.

RGB entropy can favor:

- detailed textures;
- strong edges;
- colorful areas;
- image noise.

It does not understand the meaning of the image. A high-priority block is simply a block with relatively high RGB entropy in the current image.

Also, the priority score is normalized separately for each image, so scores should not be treated as absolute values across unrelated images.

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/techn0man1ac/Entropy-Image-Prioritization?tab=MIT-1-ov-file) file for details.

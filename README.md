# RGB Image Block Entropy Prioritization

A small, lightweight **image-processing prototype** that uses **multivariate Shannon entropy of RGB colors** to estimate how varied each image block is and then prioritize the most varied blocks.

The project is intentionally simple. It uses only:

- OpenCV
- NumPy
- Matplotlib

There is **no machine learning model, no neural network, and no heavy processing pipeline**.

![Screenshot with Milka cat](https://raw.githubusercontent.com/techn0man1ac/Entropy-Image-Prioritization/refs/heads/main/Python/Figure_1.png)

## What does the project do?

The program follows this simple pipeline:

```text
Color image
    ↓
100×100 pixel blocks
    ↓
H(R), H(G), H(B)
    ↓
Joint multivariate H(R,G,B)
    ↓
Priority score 0–100
    ↓
Top 10% blocks
    ↓
Heatmap + red priority borders
```

In simple words:

> **The program looks at the complete RGB colors inside every block. Blocks with a more varied distribution of full RGB colors get higher joint entropy and therefore higher priority.**

The final image is shown in the same simple two-panel style as the reference project:

- left: the original image;
- right: RGB entropy heatmap + block grid + prioritized blocks.

The result is also saved as `Figure_1.png`.

## What is multivariate RGB entropy?

The important part is that the three color channels are kept together.

For each pixel we use the complete observation:

```text
(R, G, B)
```

The program calculates four values for every block:

```text
H(R)       -> red-channel entropy
H(G)       -> green-channel entropy
H(B)       -> blue-channel entropy
H(R,G,B)   -> joint multivariate RGB entropy
```

The **joint value `H(R,G,B)` is the priority signal**.

The Shannon entropy formula is:

```text
H = -sum(p * log2(p))
```

For the joint calculation, `p` is the probability of each complete RGB color that occurs in the block.

This is different from grayscale entropy:

```text
RGB -> grayscale -> H(gray)
```

and different from simply averaging three independent measurements:

```text
(H(R) + H(G) + H(B)) / 3
```

Here the full RGB color is treated as one three-dimensional observation.

## Why use RGB instead of grayscale?

Converting to grayscale removes direct color information.

This prototype keeps the original RGB values so the entropy can react to changes in color as well as changes in brightness.

For example, these are different RGB observations:

```text
(220, 40, 40)
(40, 220, 40)
(40, 40, 220)
```

A joint RGB histogram keeps those color combinations separate.

## Why is the calculation still lightweight?

A full 8-bit RGB image has:

```text
256 × 256 × 256 = 16,777,216
```

possible exact RGB colors.

The code does **not** allocate a large 256×256×256 array. Instead, it packs each RGB pixel into one 24-bit integer and uses NumPy's `unique(..., return_counts=True)` to count the colors that actually occur in the current block.

For the small prototype image this is simple, readable, and fast enough for an ordinary computer.

## What is image prioritization here?

After calculating `H(R,G,B)` for every block, the blocks are sorted from highest entropy to lowest entropy.

The entropy values are also converted to a relative score from `0` to `100`:

```text
lowest entropy  -> 0
highest entropy -> 100
```

By default, the top **10%** of blocks are selected.

Those blocks receive rank numbers:

```text
#1  highest joint RGB entropy
#2  second highest
#3  third highest
...
```

They are marked with red borders in the final image.

This is a deliberately simple and explainable prioritization rule.

## What does a high priority mean?

It means:

> **This block contains a more varied distribution of complete RGB colors than most other blocks in the same image.**

It does **not** automatically mean:

```text
high priority = important object
```

Entropy does not understand the semantic meaning of an image.

For example, a detailed texture, foliage, hair, JPEG artifacts, or image noise can all produce high entropy.

Therefore this project should be viewed as a **cheap first-pass filter**, not as an object detector.

## Why calculate H(R), H(G), and H(B) too?

The three marginal values are useful for understanding the result.

A block might have:

```text
H(R) = 4.8
H(G) = 5.2
H(B) = 3.9
H(R,G,B) = 7.7
```

The individual values help explain variation in each channel, while the joint value is used for the final ranking.

## Why use blocks?

The image is divided into fixed square regions, for example:

```text
100 × 100 pixels
```

Each block receives one entropy value.

This creates a simple spatial map:

```text
[ 4.1 ][ 5.2 ][ 2.9 ]
[ 6.4 ][ 7.0 ][ 3.8 ]
[ 4.0 ][ 6.1 ][ 5.0 ]
```

The heatmap shows those values visually.

The block size is deliberately configurable. Smaller blocks provide more local detail; larger blocks are coarser but reduce the number of calculations.

## Output

When `main.py` finishes, it:

1. calculates the entropy for all blocks;
2. ranks the blocks;
3. creates the heatmap and priority overlay;
4. saves the result to `Figure_1.png`;
5. **opens the final figure using Matplotlib** so the result is immediately visible when running the script normally.

The figure has the same basic presentation style as the reference implementation:

```text
┌──────────────────────┬────────────────────────────┐
│                      │                            │
│    Original Image    │  RGB Entropy + Priority    │
│                      │                            │
│        photo         │ heatmap + grid + #1 #2...  │
│                      │                            │
└──────────────────────┴────────────────────────────┘
```

## How to run

Put all five files in the same directory:

```text
README.md
main.py
MilkaCat.jpg
Figure_1.png
LICENSE
```

Install the three required packages:

```bash
pip install numpy opencv-python matplotlib
```

Then run:

```bash
python main.py
```

The program automatically uses `MilkaCat.jpg` next to `main.py`.

## Main parameters

The defaults are defined near the top of `main.py`:

```python
DEFAULT_BLOCK_SIZE = 100
DEFAULT_ALPHA = 0.30
DEFAULT_TOP_PERCENT = 10.0
```

### `DEFAULT_BLOCK_SIZE`

Controls the block size in pixels.

```text
50  -> smaller, more local blocks
100 -> balanced default
200 -> larger, coarser blocks
```

### `DEFAULT_ALPHA`

Controls how strongly the entropy heatmap is blended with the original image.

```text
0.0 -> original image only
1.0 -> heatmap only
0.3 -> balanced overlay
```

### `DEFAULT_TOP_PERCENT`

Controls how many blocks become prioritized.

```text
5  -> only the highest 5%
10 -> highest 10%
20 -> highest 20%
```

## Project structure

The project intentionally remains a small educational/prototype image-processing project:

```text
├── LICENSE
├── README.md
└── Python/
    ├── Figure_1.png
    ├── MilkaCat.jpg
    └── main.py
```

There are no extra modules or services.

## Limitations

This prototype is intentionally simple.

It does not:

- detect objects;
- recognize faces;
- understand the meaning of a scene;
- perform segmentation;
- use machine learning;
- guarantee that the highest-entropy blocks are the most important blocks.

Another limitation is that exact RGB entropy can be influenced by tiny color variations introduced by JPEG compression, sensors, or lighting. This prototype intentionally keeps the exact RGB calculation because it is the clearest demonstration of **joint multivariate RGB entropy** and remains practical for the small example image.

## Possible future extension

A future version could combine joint RGB entropy with another very cheap signal such as edge density or local sharpness:

```text
RGB entropy
     +
edge density
     +
sharpness
     ↓
priority score
```

That would still be lightweight, while making the priority estimate less dependent on color diversity alone.

## Project category

**Small educational/prototype image-processing project.**

The goal is to demonstrate a simple, explainable idea:

> **Use local multivariate RGB entropy as a cheap first-pass signal for deciding which image regions deserve attention first.**

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/techn0man1ac/Entropy-Image-Prioritization?tab=MIT-1-ov-file) file for details.

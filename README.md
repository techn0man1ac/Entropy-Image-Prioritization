# Smooth Image Entropy Heatmap Generator

An optimized Python tool that computes spatial Shannon Entropy over local image regions and generates a seamless, high-resolution heat map overlay on top of the original image using OpenCV, NumPy, and Matplotlib.

## Overview

Spatial Shannon Entropy measures local visual information density. Regions with rich textures, sharp edges, or high detail exhibit higher entropy, while smooth or uniform backgrounds have lower values.

Unlike traditional grid-based methods that suffer from rigid boundary artifacts, this implementation features dynamic scaling, vectorized sliding windows, bicubic interpolation, and Gaussian smoothing to yield clean, organic heatmaps.

## Key Features

- **Seamless Visuals:** Replaces harsh block transitions with smooth color gradients via bicubic interpolation and Gaussian filtering.
- **Dynamic Sizing:** Automatically calculates the optimal processing block size based on relative image dimensions.
- **50% Window Overlap:** Evaluates overlapping visual fields to prevent edge loss at block boundaries.
- **Vectorized Performance:** Uses `numpy.lib.stride_tricks` to process image patches rapidly without heavy pure-Python loops.
- **Safe Execution:** Includes explicit resource cleanup (`plt.close('all')`) to prevent window locks.

## Mathematical Formulation

For a grayscale block $B$ of size $W \times H$, the discrete intensity probability $p(x)$ for luminance levels $x \in [0, 255]$ is:

$$p(x) = \frac{\text{hist}[x]}{\sum_{i=0}^{255} \text{hist}[i]}$$

The Shannon Entropy $H(B)$ in bits is defined as:

$$H(B) = -\sum_{x=0}^{255} p(x) \log_2(p(x)) \quad \text{where } p(x) > 0$$

The resulting entropy grid $H$ is normalized using Min-Max scaling to an 8-bit scale $[0, 255]$:

$$H_{\text{norm}} = \frac{H - H_{\min}}{H_{\max} - H_{\min}} \times 255$$

## Installation

Ensure you have Python 3.8+ installed along with the required dependencies:

```bash
pip install numpy opencv-python matplotlib

```

## Quick Start

```python
from smooth_entropy import generate_smooth_entropy_overlay

# Generate overlay with dynamic scaling and smooth gradients
generate_smooth_entropy_overlay(
    image_path="path/to/your/image.jpg",
    block_scale=0.03,  # 3% of the image's smallest dimension
    alpha=0.35,        # 35% heatmap opacity
    blur_radius=25     # Gaussian smoothing radius
)

```

## Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `image_path` | `str` | Required | Path to the target image file. |
| `block_scale` | `float` | `0.03` | Block size ratio relative to `min(height, width)`. |
| `alpha` | `float` | `0.3` | Heatmap transparency overlay weight ($0.0$ to $1.0$). |
| `blur_radius` | `int` | `15` | Kernel radius for Gaussian smoothing (automatically converted to an odd integer). |

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/techn0man1ac/Entropy-Image-Prioritization?tab=MIT-1-ov-file) file for details.

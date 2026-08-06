# Image Block Entropy Heatmap Generator

A clear, lightweight Python tool that calculates spatial **Shannon Entropy** over discrete image blocks and visualizes the result as a side-by-side comparison using OpenCV, NumPy, and Matplotlib.

Unlike smooth gradient visualizers, this implementation preserves exact pixel-aligned grid boundaries ($N \times N$), making it ideal for discrete spatial analysis, image processing pipelines, and region-of-interest (ROI) prioritizing.

![Screenshot with Milka cat](https://raw.githubusercontent.com/techn0man1ac/Entropy-Image-Prioritization/refs/heads/main/Figure_1.png)

---

##  Demo Features

* **Side-by-Side View:** Displays the original color image next to the block-entropy heatmap in a clean $800 \times 600$ window.
* **Pixel-Accurate Grid:** Uses direct slice-based map assignment (eliminating `cv2.resize` artifacts), ensuring block color boundaries align perfectly with grid lines.
* **Adjustable Parameters:** Easily customize block dimensions and heatmap overlay opacity ($\alpha$).

---

##  How It Works

1. High entropy values (red/warm areas) highlight complex, detailed visual regions (e.g., textures, edges, faces).
2. Low entropy values (blue/cold areas) correspond to uniform or flat regions (e.g., plain backgrounds, sky).

---

##  Installation

Make sure you have Python 3.8+ installed along with the required libraries:

```bash
pip install opencv-python numpy matplotlib

```

---

##  Usage

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np


def calculate_entropy(block):
    if block.size == 0:
        return 0.0
    hist, _ = np.histogram(block, bins=256, range=[0, 256])
    prob = hist[hist > 0] / block.size
    return -np.sum(prob * np.log2(prob))


def generate_entropy_comparison(image_path, block_size=100, alpha=0.3):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("Error: Could not load image. Check the file path!")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape

    # 1. Compute entropy map at full image resolution
    entropy_full = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = img_gray[y : y + block_size, x : x + block_size]
            val = calculate_entropy(block)
            entropy_full[y : y + block_size, x : x + block_size] = val

    # 2. Normalize and apply JET colormap
    norm_map = cv2.normalize(
        entropy_full, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    heatmap_color = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)

    # 3. Blend overlay and draw crisp grid borders
    overlay = cv2.addWeighted(heatmap_color, alpha, img_bgr, 1 - alpha, 0)

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            cv2.rectangle(
                overlay,
                (x, y),
                (min(x + block_size, w), min(y + block_size, h)),
                (255, 255, 255),
                1,
            )

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # 4. Display in 800x600 window (8x6 inches at 100 DPI)
    fig, axes = plt.subplots(1, 2, figsize=(8, 6), dpi=100)

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(overlay_rgb)
    axes[1].set_title(f"Entropy Grid ({block_size}px)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close("all")


# Run script
generate_entropy_comparison("path/to/your/image.jpg", block_size=100, alpha=0.3)

```

---

##  Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `image_path` | `str` | *Required* | Path to the target image file. |
| `block_size` | `int` | `100` | Size of each square block in pixels. |
| `alpha` | `float` | `0.3` | Transparency of heatmap layer ($0.0$ = transparent, $1.0$ = opaque heatmap). |

---

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/techn0man1ac/Entropy-Image-Prioritization?tab=MIT-1-ov-file) file for details.

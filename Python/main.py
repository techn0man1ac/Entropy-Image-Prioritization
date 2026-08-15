"""Lightweight RGB entropy analysis and image-block prioritization prototype."""

from pathlib import Path
from typing import Dict, List, Union, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_BLOCK_SIZE = 100
DEFAULT_ALPHA = 0.30
DEFAULT_TOP_PERCENT = 10.0


def calculate_rgb_entropy(block_rgb: np.ndarray) -> float:
    """Return Shannon entropy of exact RGB colors in one image block.

    Each RGB pixel is packed into one integer, so the histogram is built from
    full colors rather than grayscale values or separate channels.
    """
    if block_rgb.size == 0:
        return 0.0

    # Pack R, G and B into one 24-bit integer per pixel.
    packed = (
        (block_rgb[:, :, 0].astype(np.uint32) << 16)
        | (block_rgb[:, :, 1].astype(np.uint32) << 8)
        | block_rgb[:, :, 2].astype(np.uint32)
    ).reshape(-1)

    # Count each distinct RGB color present in the block.
    _, counts = np.unique(packed, return_counts=True)
    probabilities = counts / packed.size

    return float(-np.sum(probabilities * np.log2(probabilities)))


def normalize_scores(values: np.ndarray) -> np.ndarray:
    """Scale an array to the 0-100 range without expensive processing."""
    minimum = float(values.min())
    maximum = float(values.max())

    if maximum == minimum:
        return np.full(values.shape, 100.0, dtype=np.float32)

    return ((values - minimum) / (maximum - minimum) * 100.0).astype(np.float32)


def calculate_block_entropies(
    img_rgb: np.ndarray, block_size: int
) -> Tuple[np.ndarray, List[Dict[str, Union[float, int]]]]:
    """Calculate one exact RGB entropy value for every image block."""
    height, width = img_rgb.shape[:2]
    entropy_map = np.zeros((height, width), dtype=np.float32)
    blocks: List[Dict[str, Union[float, int]]] = []

    block_id = 1

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = img_rgb[y : y + block_size, x : x + block_size]
            entropy = calculate_rgb_entropy(block)

            y2 = min(y + block_size, height)
            x2 = min(x + block_size, width)
            entropy_map[y:y2, x:x2] = entropy

            blocks.append(
                {
                    "id": block_id,
                    "x": x,
                    "y": y,
                    "width": x2 - x,
                    "height": y2 - y,
                    "entropy": entropy,
                }
            )
            block_id += 1

    return entropy_map, blocks


def prioritize_blocks(
    blocks: List[Dict[str, Union[float, int]]], top_percent: float
) -> List[Dict[str, Union[float, int]]]:
    """Rank blocks by RGB entropy and mark the highest-priority blocks."""
    entropy_values = np.array(
        [float(block["entropy"]) for block in blocks], dtype=np.float32
    )
    scores = normalize_scores(entropy_values)

    for block, score in zip(blocks, scores):
        block["priority_score"] = float(score)
        block["priority"] = 0

    requested = max(1, int(np.ceil(len(blocks) * top_percent / 100.0)))
    ranked = sorted(blocks, key=lambda item: float(item["priority_score"]), reverse=True)

    for rank, block in enumerate(ranked[:requested], start=1):
        block["priority"] = rank

    return ranked


def draw_priority_overlay(
    image_bgr: np.ndarray,
    blocks: List[Dict[str, Union[float, int]]],
    top_percent: float,
) -> np.ndarray:
    """Draw the block grid and highlight the selected high-priority blocks."""
    overlay = image_bgr.copy()

    selected = {
        int(block["priority"]): block
        for block in blocks
        if int(block["priority"]) > 0
    }

    # Draw all grid lines first so the selected blocks stay easy to read.
    height, width = image_bgr.shape[:2]
    for block in blocks:
        x = int(block["x"])
        y = int(block["y"])
        x2 = min(x + int(block["width"]), width - 1)
        y2 = min(y + int(block["height"]), height - 1)
        cv2.rectangle(overlay, (x, y), (x2, y2), (255, 255, 255), 1)

    # Highlight only the top-ranked blocks. This turns the heatmap into a
    # real, simple prioritization result instead of visualization only.
    for rank, block in selected.items():
        x = int(block["x"])
        y = int(block["y"])
        x2 = min(x + int(block["width"]), width - 1)
        y2 = min(y + int(block["height"]), height - 1)
        cv2.rectangle(overlay, (x, y), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            overlay,
            f"#{rank}",
            (x + 5, min(y + 20, height - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        overlay,
        f"Top {top_percent:g}% blocks",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return overlay


def generate_entropy_comparison(
    image_path: str | Path,
    output_path: str | Path = "Figure_1.png",
    block_size: int = DEFAULT_BLOCK_SIZE,
    alpha: float = DEFAULT_ALPHA,
    top_percent: float = DEFAULT_TOP_PERCENT,
    show_result: bool = True,
) -> List[Dict[str, Union[float, int]]]:
    """Analyze, prioritize, visualize, save and optionally display the result."""
    image_path = Path(image_path)
    output_path = Path(output_path)

    if block_size <= 0:
        raise ValueError("block_size must be greater than 0")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0")
    if not 0.0 < top_percent <= 100.0:
        raise ValueError("top_percent must be greater than 0 and at most 100")

    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    height, width = img_rgb.shape[:2]

    # 1. Calculate exact RGB entropy for every block.
    entropy_map, blocks = calculate_block_entropies(img_rgb, block_size)

    # 2. Convert entropy values into a relative 0-100 priority score.
    ranked_blocks = prioritize_blocks(blocks, top_percent)

    # 3. Turn the score map into a readable heatmap.
    norm_map = cv2.normalize(
        entropy_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    heatmap_bgr = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
    heatmap_overlay_bgr = cv2.addWeighted(
        heatmap_bgr, alpha, img_bgr, 1.0 - alpha, 0
    )

    # 4. Add the crisp block grid and highlight prioritized blocks.
    priority_overlay_bgr = draw_priority_overlay(
        heatmap_overlay_bgr, ranked_blocks, top_percent
    )
    priority_overlay_rgb = cv2.cvtColor(priority_overlay_bgr, cv2.COLOR_BGR2RGB)

    # 5. Save the figure instead of requiring a GUI window.
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=100)
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(priority_overlay_rgb)
    axes[1].set_title(
        f"RGB Entropy + Priority (block {block_size}px)"
    )
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")

    # Show the generated map when the function is called interactively.
    # Setting show_result=False keeps the same function usable on headless machines.
    if show_result:
        plt.show()

    plt.close(fig)

    # Print a compact ranking so the result is useful outside the image view.
    print(f"Image: {image_path}")
    print(f"Size: {width}x{height} | Blocks: {len(ranked_blocks)}")
    print(
        f"RGB entropy: {min(b['entropy'] for b in ranked_blocks):.2f} "
        f"to {max(b['entropy'] for b in ranked_blocks):.2f} bits"
    )
    print(f"Priority selection: top {top_percent:g}%")
    print("Top priority blocks:")
    for block in ranked_blocks[: min(10, len(ranked_blocks))]:
        print(
            f"  #{int(block['priority']) or '-':>2} "
            f"block {int(block['id']):>3} | "
            f"x={int(block['x']):>4}, y={int(block['y']):>4} | "
            f"entropy={float(block['entropy']):.2f} | "
            f"score={float(block['priority_score']):>6.1f}"
        )

    print(f"Saved figure: {output_path}")
    return ranked_blocks


if __name__ == "__main__":
    # Keep the demo self-contained: the example image sits next to this file.
    project_dir = Path(__file__).resolve().parent
    generate_entropy_comparison(
        image_path=project_dir / "MilkaCat.jpg",
        output_path=project_dir / "Figure_1.png",
        block_size=DEFAULT_BLOCK_SIZE,
        alpha=DEFAULT_ALPHA,
        top_percent=DEFAULT_TOP_PERCENT,
        show_result=True,
    )

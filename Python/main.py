"""Lightweight multivariate RGB entropy image prioritization prototype."""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_BLOCK_SIZE = 100
DEFAULT_ALPHA = 0.30
DEFAULT_TOP_PERCENT = 10.0

Number = Union[int, float]
Block = Dict[str, Number]


def shannon_entropy_from_counts(counts: np.ndarray) -> float:
    """Calculate Shannon entropy from non-zero histogram counts."""
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0

    probabilities = counts.astype(np.float64) / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def calculate_rgb_entropies(
    block_rgb: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Return H(R), H(G), H(B), and joint multivariate H(R,G,B).

    The image stays in full RGB. The joint entropy treats each complete
    (R, G, B) color as one three-dimensional observation.
    """
    if block_rgb.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    r = block_rgb[:, :, 0].reshape(-1)
    g = block_rgb[:, :, 1].reshape(-1)
    b = block_rgb[:, :, 2].reshape(-1)

    # Calculate marginal entropy for each RGB channel.
    entropy_r = shannon_entropy_from_counts(np.bincount(r, minlength=256))
    entropy_g = shannon_entropy_from_counts(np.bincount(g, minlength=256))
    entropy_b = shannon_entropy_from_counts(np.bincount(b, minlength=256))

    # Pack the full 8-bit RGB color into one 24-bit integer.
    # This creates a compact 1-D representation of the 3-D RGB histogram.
    packed_rgb = (
        (r.astype(np.uint32) << 16)
        | (g.astype(np.uint32) << 8)
        | b.astype(np.uint32)
    )

    # np.unique returns the count of every complete RGB color in the block.
    # No grayscale conversion and no channel averaging are used here.
    _, color_counts = np.unique(packed_rgb, return_counts=True)
    entropy_rgb = shannon_entropy_from_counts(color_counts)

    return entropy_r, entropy_g, entropy_b, entropy_rgb


def normalize_scores(values: np.ndarray) -> np.ndarray:
    """Scale values to a simple 0-100 relative priority score."""
    minimum = float(values.min())
    maximum = float(values.max())

    if maximum == minimum:
        return np.full(values.shape, 100.0, dtype=np.float32)

    return ((values - minimum) / (maximum - minimum) * 100.0).astype(np.float32)


def calculate_block_entropies(
    img_rgb: np.ndarray,
    block_size: int,
) -> Tuple[np.ndarray, List[Block]]:
    """Calculate RGB entropy values for every image block."""
    height, width = img_rgb.shape[:2]
    entropy_map = np.zeros((height, width), dtype=np.float32)
    blocks: List[Block] = []

    block_id = 1

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            block = img_rgb[y : y + block_size, x : x + block_size]
            entropy_r, entropy_g, entropy_b, entropy_rgb = calculate_rgb_entropies(block)

            y2 = min(y + block_size, height)
            x2 = min(x + block_size, width)

            # Fill the complete block with its one joint RGB entropy value.
            entropy_map[y:y2, x:x2] = entropy_rgb

            blocks.append(
                {
                    "id": block_id,
                    "x": x,
                    "y": y,
                    "width": x2 - x,
                    "height": y2 - y,
                    "entropy_r": entropy_r,
                    "entropy_g": entropy_g,
                    "entropy_b": entropy_b,
                    "entropy_rgb": entropy_rgb,
                }
            )
            block_id += 1

    return entropy_map, blocks


def prioritize_blocks(
    blocks: List[Block],
    top_percent: float,
) -> List[Block]:
    """Rank blocks by joint RGB entropy and mark the highest-priority ones."""
    entropy_values = np.array(
        [float(block["entropy_rgb"]) for block in blocks],
        dtype=np.float32,
    )
    priority_scores = normalize_scores(entropy_values)

    for block, score in zip(blocks, priority_scores):
        block["priority_score"] = float(score)
        block["priority_rank"] = 0

    # Highest joint RGB entropy gets highest priority.
    # Coordinates make ties deterministic between repeated runs.
    ranked = sorted(
        blocks,
        key=lambda block: (
            -float(block["entropy_rgb"]),
            int(block["y"]),
            int(block["x"]),
        ),
    )

    selected_count = max(1, int(np.ceil(len(ranked) * top_percent / 100.0)))

    for rank, block in enumerate(ranked[:selected_count], start=1):
        block["priority_rank"] = rank

    return ranked


def draw_priority_overlay(
    image_bgr: np.ndarray,
    blocks: List[Block],
    top_percent: float,
) -> np.ndarray:
    """Draw the block grid and highlight the prioritized blocks."""
    overlay = image_bgr.copy()
    height, width = overlay.shape[:2]

    # Draw the full block grid first.
    for block in blocks:
        x = int(block["x"])
        y = int(block["y"])
        x2 = min(x + int(block["width"]) - 1, width - 1)
        y2 = min(y + int(block["height"]) - 1, height - 1)
        cv2.rectangle(overlay, (x, y), (x2, y2), (255, 255, 255), 1)

    # Draw red borders and rank labels for selected blocks only.
    for block in blocks:
        rank = int(block["priority_rank"])
        if rank <= 0:
            continue

        x = int(block["x"])
        y = int(block["y"])
        x2 = min(x + int(block["width"]) - 1, width - 1)
        y2 = min(y + int(block["height"]) - 1, height - 1)

        cv2.rectangle(overlay, (x, y), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            overlay,
            f"#{rank}",
            (x + 4, min(y + 18, height - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # Keep the label simple and visible on the heatmap.
    cv2.putText(
        overlay,
        f"Top {top_percent:g}% blocks",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
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
) -> List[Block]:
    """Analyze, prioritize, save, and display the final comparison figure."""
    image_path = Path(image_path)
    output_path = Path(output_path)

    if block_size <= 0:
        raise ValueError("block_size must be greater than 0")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0")
    if not 0.0 < top_percent <= 100.0:
        raise ValueError("top_percent must be greater than 0 and at most 100")

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    # 1. Calculate marginal RGB entropies and the joint H(R,G,B) per block.
    entropy_map, blocks = calculate_block_entropies(image_rgb, block_size)

    # 2. Rank blocks using joint multivariate RGB entropy.
    ranked_blocks = prioritize_blocks(blocks, top_percent)

    # 3. Convert the entropy map into a heatmap and blend it with the image.
    normalized_map = cv2.normalize(
        entropy_map,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )
    heatmap_bgr = cv2.applyColorMap(normalized_map, cv2.COLORMAP_JET)
    heatmap_overlay_bgr = cv2.addWeighted(
        heatmap_bgr,
        alpha,
        image_bgr,
        1.0 - alpha,
        0,
    )

    # 4. Draw the same clear grid + priority overlay style as the reference.
    result_bgr = draw_priority_overlay(
        heatmap_overlay_bgr,
        ranked_blocks,
        top_percent,
    )
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    # 5. Build the final side-by-side figure.
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=100)
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(result_rgb)
    axes[1].set_title(f"RGB Entropy + Priority (block {block_size}px)")
    axes[1].axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")

    # Show the result when the script finishes, matching the interactive
    # behavior of the reference project.
    plt.show()
    plt.close(fig)

    min_entropy = min(float(block["entropy_rgb"]) for block in ranked_blocks)
    max_entropy = max(float(block["entropy_rgb"]) for block in ranked_blocks)
    selected_count = sum(int(block["priority_rank"]) > 0 for block in ranked_blocks)

    print(f"Image: {image_path.name}")
    print(f"Size: {width}x{height} | Blocks: {len(ranked_blocks)}")
    print(f"Joint RGB entropy H(R,G,B): {min_entropy:.2f} to {max_entropy:.2f} bits")
    print(f"Priority selection: top {top_percent:g}% ({selected_count} blocks)")
    print("Top priority blocks:")

    for block in ranked_blocks[: min(10, len(ranked_blocks))]:
        rank = int(block["priority_rank"])
        print(
            f"  #{rank or '-':>2} block {int(block['id']):>3} | "
            f"x={int(block['x']):>4}, y={int(block['y']):>4} | "
            f"H(R)={float(block['entropy_r']):.2f} | "
            f"H(G)={float(block['entropy_g']):.2f} | "
            f"H(B)={float(block['entropy_b']):.2f} | "
            f"H(R,G,B)={float(block['entropy_rgb']):.2f} | "
            f"score={float(block['priority_score']):>5.1f}"
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
    )

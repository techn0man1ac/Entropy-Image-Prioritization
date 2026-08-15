"""Small RGB entropy prioritization prototype."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


BLOCK_SIZE = 100
ALPHA = 0.30
TOP_PERCENT = 10.0


def entropy(counts: np.ndarray) -> float:
    """Return Shannon entropy for non-zero histogram counts."""
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0

    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def rgb_entropy(block: np.ndarray) -> tuple[float, float, float, float]:
    """Return H(R), H(G), H(B), and joint H(R,G,B)."""
    if block.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    r, g, b = (block[:, :, i].ravel() for i in range(3))

    # Marginal entropy of each channel.
    h_r = entropy(np.bincount(r, minlength=256))
    h_g = entropy(np.bincount(g, minlength=256))
    h_b = entropy(np.bincount(b, minlength=256))

    # Pack (R, G, B) so each complete color becomes one histogram value.
    color = (r.astype(np.uint32) << 16) | (g.astype(np.uint32) << 8) | b
    _, counts = np.unique(color, return_counts=True)
    h_rgb = entropy(counts)

    return h_r, h_g, h_b, h_rgb


def analyze_blocks(image: np.ndarray, block_size: int) -> tuple[np.ndarray, list[dict]]:
    """Calculate joint RGB entropy and metadata for every image block."""
    height, width = image.shape[:2]
    entropy_map = np.zeros((height, width), dtype=np.float32)
    blocks = []

    for block_id, (y, x) in enumerate(
        ((y, x) for y in range(0, height, block_size) for x in range(0, width, block_size)),
        start=1,
    ):
        y2, x2 = min(y + block_size, height), min(x + block_size, width)
        h_r, h_g, h_b, h_rgb = rgb_entropy(image[y:y2, x:x2])

        entropy_map[y:y2, x:x2] = h_rgb
        blocks.append(
            {
                "id": block_id,
                "x": x,
                "y": y,
                "width": x2 - x,
                "height": y2 - y,
                "entropy_r": h_r,
                "entropy_g": h_g,
                "entropy_b": h_b,
                "entropy_rgb": h_rgb,
            }
        )

    return entropy_map, blocks


def prioritize(blocks: list[dict], top_percent: float) -> list[dict]:
    """Rank blocks by joint RGB entropy and mark the top percentage."""
    values = np.array([b["entropy_rgb"] for b in blocks], dtype=np.float32)
    low, high = values.min(), values.max()
    scores = np.full_like(values, 100.0) if low == high else (values - low) / (high - low) * 100

    for block, score in zip(blocks, scores):
        block["priority_score"] = float(score)
        block["priority_rank"] = 0

    ranked = sorted(blocks, key=lambda b: (-b["entropy_rgb"], b["y"], b["x"]))
    selected = max(1, int(np.ceil(len(ranked) * top_percent / 100)))

    for rank, block in enumerate(ranked[:selected], start=1):
        block["priority_rank"] = rank

    return ranked


def overlay_priority(image_bgr: np.ndarray, blocks: list[dict], top_percent: float) -> np.ndarray:
    """Draw the block grid and red borders for prioritized blocks."""
    result = image_bgr.copy()
    height, width = result.shape[:2]

    for block in blocks:
        x, y = block["x"], block["y"]
        x2 = min(x + block["width"] - 1, width - 1)
        y2 = min(y + block["height"] - 1, height - 1)

        # White grid for all blocks.
        cv2.rectangle(result, (x, y), (x2, y2), (255, 255, 255), 1)

        rank = block["priority_rank"]
        if rank:
            # Red border + rank for prioritized blocks.
            cv2.rectangle(result, (x, y), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                result,
                f"#{rank}",
                (x + 4, min(y + 18, height - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    cv2.putText(
        result,
        f"Top {top_percent:g}% blocks",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return result


def run(image_path: str | Path, output_path: str | Path = "Figure_1.png") -> list[dict]:
    """Analyze the image, save the final figure, and display it."""
    if BLOCK_SIZE <= 0:
        raise ValueError("BLOCK_SIZE must be greater than 0")
    if not 0 <= ALPHA <= 1:
        raise ValueError("ALPHA must be between 0 and 1")
    if not 0 < TOP_PERCENT <= 100:
        raise ValueError("TOP_PERCENT must be between 0 and 100")

    image_path = Path(image_path)
    output_path = Path(output_path)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    entropy_map, blocks = analyze_blocks(image_rgb, BLOCK_SIZE)
    ranked = prioritize(blocks, TOP_PERCENT)

    # Convert entropy values into a heatmap and blend it with the photo.
    normalized = cv2.normalize(entropy_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(heatmap, ALPHA, image_bgr, 1 - ALPHA, 0)
    result = overlay_priority(blended, ranked, TOP_PERCENT)

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=100)
    axes[0].imshow(image_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"RGB Entropy + Priority (block {BLOCK_SIZE}px)")
    axes[1].axis("off")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    selected = sum(block["priority_rank"] > 0 for block in ranked)
    print(f"Image: {image_path.name}")
    print(f"Size: {image_rgb.shape[1]}x{image_rgb.shape[0]} | Blocks: {len(ranked)}")
    print(
        "Joint RGB entropy H(R,G,B): "
        f"{min(b['entropy_rgb'] for b in ranked):.2f} to "
        f"{max(b['entropy_rgb'] for b in ranked):.2f} bits"
    )
    print(f"Priority selection: top {TOP_PERCENT:g}% ({selected} blocks)")
    print(f"Saved figure: {output_path}")

    return ranked


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    run(project_dir / "MilkaCat.jpg", project_dir / "Figure_1.png")

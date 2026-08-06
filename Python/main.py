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

    # 1. Створюємо матрицю ентропії у повному розмірі зображення (без cv2.resize!)
    entropy_full = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = img_gray[y : y + block_size, x : x + block_size]
            val = calculate_entropy(block)
            # Заповнюємо точно той самий прямокутник
            entropy_full[y : y + block_size, x : x + block_size] = val

    # 2. Нормалізуємо та накладаємо кольорову мапу
    norm_map = cv2.normalize(
        entropy_full, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    heatmap_color = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)

    # 3. Змішуємо шар оверлею та наносимо лінії строго по межах
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

    # 4. Відображення вікна 800x600 (8x6 дюймів при dpi=100)
    fig, axes = plt.subplots(1, 2, figsize=(8, 6), dpi=100)

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(overlay_rgb)
    axes[1].set_title(f"Entropy Grid ({block_size}px)")
    axes[1].axis("off")

    plt.tight_layout()
    print("Waiting for user to close the plot window...")
    plt.show()
    plt.close("all")


# Запуск
generate_entropy_comparison(
    "C:/Projects/vscode-basics/Python/MilkaCat.jpg", # Here you picture 
    block_size=100,
    alpha=0.3,
)
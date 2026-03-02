import os
import random

import matplotlib.pyplot as plt


def main():
    # score on X axis, accuracy on Y axis
    points = [
        ("FP32 Resnet20", 0.1926, 92.13),
        ("FP16 Resnet18 (Baseline)", 1.0000, 94.00),
        ("FP16 Resnet20", 0.0963, 92.69),
        ("FP8 Resnet20", 0.0481, 92.69),
        ("BinaryConnect Resnet20", 0.1459, 91.42),
        ("FP8 Resnet20 (Pu 0.7 Ps 0.0)", 0.0392, 90.27),
        ("FP16 Resnet20 (Pu 0.8 Ps 0.1)", 0.0694, 90.36),
        ("FP1w4a Resnet20 (Pu 0.1 Ps 0.05)", 0.0185, 90.59),
    ]

    plt.figure(figsize=(12, 7))
    palette = list(plt.get_cmap("tab10").colors)
    rng = random.Random()

    # one color per model + legend entry
    for idx, (name, score, acc) in enumerate(points):
        color = palette[idx % len(palette)]
        plt.scatter(
            score,
            acc,
            s=120,
            color=color,
            edgecolors="black",
            linewidth=1.0,
            zorder=3,
            label=name,
        )
        dx, dy = 0, 0
        while abs(dx) < 8 and abs(dy) < 6:
            dx = rng.randint(-26, 26)
            dy = rng.randint(-16, 16)
        plt.annotate(
            f"{score:.4f}",
            (score, acc),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7, linewidth=0.2),
            arrowprops=dict(arrowstyle="-", color=color, linewidth=0.6, alpha=0.8),
        )

    plt.xlabel("Score")
    plt.ylabel("Accuracy (%)")
    plt.title("Model Score vs Accuracy")
    plt.grid(True, linestyle="--", alpha=0.35, zorder=0)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=8)
    plt.tight_layout()

    output_path = "/home/antonio/imt_efficient_deep_learning/antonio/experiments/quantization/score_vs_accuracy_custom.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()

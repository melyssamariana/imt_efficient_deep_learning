import argparse
import copy
import json
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from models.resnet_s import resnet20


@dataclass
class PruningResult:
    pruning_ratio: float
    weight_sparsity_percent: float
    accuracy_percent: float
    avg_loss: float


def parse_args():
    parser = argparse.ArgumentParser(
        description="Global L1 pruning on FP16-quantized ResNet20 (CIFAR-10, no retraining)."
    )
    parser.add_argument(
        "--checkpoint",
        default="/home/antonio/imt_efficient_deep_learning/antonio/best_model.pth",
        help="Path to FP32 checkpoint (.pth).",
    )
    parser.add_argument(
        "--cifar-root",
        default="/opt/img/effdl-cifar10/",
        help="Root folder for CIFAR-10 dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size used during evaluation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
        help="Pruning ratios to evaluate (global amount in [0, 1)).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--output-json",
        default="/home/antonio/imt_efficient_deep_learning/antonio/experiments/quantization/global_pruning_fp16_results.json",
        help="Where to save pruning results.",
    )
    parser.add_argument(
        "--output-plot",
        default="/home/antonio/imt_efficient_deep_learning/antonio/experiments/quantization/global_pruning_fp16_plot.png",
        help="Where to save pruning plot.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def clean_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict) and "state_dict" in checkpoint_obj and isinstance(checkpoint_obj["state_dict"], dict):
        state_dict = checkpoint_obj["state_dict"]
    elif isinstance(checkpoint_obj, dict) and "model_state_dict" in checkpoint_obj and isinstance(checkpoint_obj["model_state_dict"], dict):
        state_dict = checkpoint_obj["model_state_dict"]
    elif isinstance(checkpoint_obj, dict):
        state_dict = checkpoint_obj
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint_obj)}")

    cleaned = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        cleaned_key = key[6:] if key.startswith("model.") else key
        cleaned[cleaned_key] = value
    return cleaned


def load_resnet20_from_checkpoint(checkpoint_path: str) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = resnet20()
    model.load_state_dict(clean_state_dict(checkpoint))
    model.eval()
    return model


def build_test_loader(cifar_root: str, batch_size: int, num_workers: int):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    dataset = CIFAR10(cifar_root, train=False, download=True, transform=transform_test)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def prepare_fp16_model(base_model: nn.Module, device: torch.device):
    if device.type == "cuda":
        return copy.deepcopy(base_model).half(), torch.float16, "native FP16 (model + inputs)"

    emulated = copy.deepcopy(base_model).float()
    with torch.no_grad():
        for module in emulated.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.weight.copy_(module.weight.to(torch.float16).to(torch.float32))
                if module.bias is not None:
                    module.bias.copy_(module.bias.to(torch.float16).to(torch.float32))
    return emulated, torch.float32, "FP16 storage emulated on CPU (FP32 compute)"


def get_prunable_parameters(model: nn.Module):
    params = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and module.weight is not None:
            params.append((module, "weight"))
    return params


def apply_global_pruning_and_remove(model: nn.Module, amount: float):
    params_to_prune = get_prunable_parameters(model)
    prune.global_unstructured(
        params_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    for module, name in params_to_prune:
        prune.remove(module, name)


def compute_weight_sparsity_percent(model: nn.Module) -> float:
    total_weights = 0
    total_zeros = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and module.weight is not None:
                w = module.weight
                total_weights += w.numel()
                total_zeros += (w == 0).sum().item()

    if total_weights == 0:
        return 0.0
    return 100.0 * total_zeros / total_weights


def evaluate(model: nn.Module, loader, device: torch.device, input_dtype: torch.dtype):
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if input_dtype == torch.float16:
                inputs = inputs.half()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / total
    return accuracy, avg_loss


def validate_ratios(ratios):
    for r in ratios:
        if r < 0.0 or r >= 1.0:
            raise ValueError(f"Invalid pruning ratio {r}. Ratios must be in [0, 1).")


def print_results_table(results):
    print("\n=== Global Pruning Results (FP16 Quantized ResNet20, no retraining) ===")
    header = (
        f"{'Pruning ratio':>14}"
        f"{'Weight sparsity (%)':>22}"
        f"{'Accuracy (%)':>16}"
        f"{'Loss':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.pruning_ratio:>14.2f}"
            f"{r.weight_sparsity_percent:>22.2f}"
            f"{r.accuracy_percent:>16.2f}"
            f"{r.avg_loss:>12.4f}"
        )
    print()


def save_pruning_plot(results, output_path: str):
    ratios = [r.pruning_ratio for r in results]
    accuracies = [r.accuracy_percent for r in results]
    losses = [r.avg_loss for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(ratios, accuracies, marker="o", linewidth=2, color="#1f77b4")
    axes[0].set_title("Accuracy vs Pruning Ratio")
    axes[0].set_xlabel("Pruning ratio")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(alpha=0.3)

    axes[1].plot(ratios, losses, marker="o", linewidth=2, color="#d62728")
    axes[1].set_title("Loss vs Pruning Ratio")
    axes[1].set_xlabel("Pruning ratio")
    axes[1].set_ylabel("Cross-entropy loss")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    validate_ratios(args.ratios)
    ratios = sorted(set(args.ratios))

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    test_loader = build_test_loader(args.cifar_root, args.batch_size, args.num_workers)
    fp32_model = load_resnet20_from_checkpoint(args.checkpoint)
    fp16_model_template, input_dtype, precision_note = prepare_fp16_model(fp32_model, device)
    print(f"Precision path: {precision_note}")

    results = []
    for ratio in ratios:
        pruned_model = copy.deepcopy(fp16_model_template)
        if ratio > 0.0:
            apply_global_pruning_and_remove(pruned_model, amount=ratio)

        accuracy, avg_loss = evaluate(pruned_model, test_loader, device, input_dtype)
        sparsity = compute_weight_sparsity_percent(pruned_model)
        results.append(
            PruningResult(
                pruning_ratio=ratio,
                weight_sparsity_percent=sparsity,
                accuracy_percent=accuracy,
                avg_loss=avg_loss,
            )
        )
        print(
            f"[ratio={ratio:.2f}] sparsity={sparsity:.2f}% | "
            f"acc={accuracy:.2f}% | loss={avg_loss:.4f}"
        )

    print_results_table(results)

    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    payload = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "precision_note": precision_note,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "ratios": ratios,
        "results": [r.__dict__ for r in results],
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved JSON results: {args.output_json}")

    plot_dir = os.path.dirname(args.output_plot)
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
    save_pruning_plot(results, args.output_plot)
    print(f"Saved plot: {args.output_plot}")


if __name__ == "__main__":
    main()

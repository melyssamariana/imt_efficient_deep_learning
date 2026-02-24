import argparse
import copy
import json
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from models.resnet_s import resnet20


@dataclass
class FilterPruningResult:
    pruning_ratio: float
    filter_sparsity_percent: float
    weight_sparsity_percent: float
    oneshot_accuracy_percent: float
    oneshot_loss: float
    gradual_accuracy_percent: float
    gradual_loss: float


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Gradual structured filter pruning (Li et al., 2016 style) with retraining across "
            "Conv layers on ResNet20, starting from FP16-quantized storage."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default="/home/antonio/imt_efficient_deep_learning/antonio/experiments/quantization/best_model.pth",
        help="Path to checkpoint (.pth).",
    )
    parser.add_argument(
        "--cifar-root",
        default="/opt/img/effdl-cifar10/",
        help="Root folder for CIFAR-10 dataset.",
    )
    parser.add_argument("--train-batch-size", type=int, default=128, help="Batch size for retraining.")
    parser.add_argument("--test-batch-size", type=int, default=256, help="Batch size for evaluation.")
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader workers.")
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0.2, 0.4, 0.6, 0.8],
        help="Per-layer filter pruning ratio (applied to each Conv2d output channels).",
    )
    parser.add_argument(
        "--layer-finetune-epochs",
        type=int,
        default=1,
        help="Retraining epochs after pruning each Conv layer.",
    )
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate for retraining.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for retraining.")
    parser.add_argument(
        "--train-subset-size",
        type=int,
        default=None,
        help="Optional subset size of CIFAR-10 train for faster experiments.",
    )
    parser.add_argument("--seed", type=int, default=2147483647, help="Seed for subset sampling.")
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional max number of train batches per epoch.",
    )
    parser.add_argument(
        "--max-test-batches",
        type=int,
        default=None,
        help="Optional max number of test batches for quick checks.",
    )
    parser.add_argument(
        "--eval-precision",
        choices=["fp16", "fp32"],
        default="fp16",
        help="Precision used for evaluation metrics.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--output-json",
        default="/home/antonio/imt_efficient_deep_learning/antonio/experiments/quantization/filter_pruning_gradual_retrain_fp16_results.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--output-plot",
        default="/home/antonio/imt_efficient_deep_learning/antonio/experiments/quantization/filter_pruning_gradual_retrain_fp16_plot.png",
        help="Output plot path.",
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


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


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


def load_resnet20_from_checkpoint(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = resnet20()
    model.load_state_dict(clean_state_dict(checkpoint))
    model.eval()
    return model


def build_loaders(cifar_root: str, train_batch_size: int, test_batch_size: int, num_workers: int, subset_size, seed: int):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"),
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = CIFAR10(cifar_root, train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10(cifar_root, train=False, download=True, transform=test_transform)

    if subset_size is not None and 0 < subset_size < len(train_dataset):
        indices = np.arange(len(train_dataset))
        rng = np.random.RandomState(seed=seed)
        rng.shuffle(indices)
        train_dataset = torch.utils.data.Subset(train_dataset, indices[:subset_size].tolist())

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def clone_and_quantize_storage(model: nn.Module, dtype: torch.dtype):
    quantized = copy.deepcopy(model).cpu()
    with torch.no_grad():
        for module in quantized.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.weight.copy_(module.weight.to(dtype).to(torch.float32))
                if module.bias is not None:
                    module.bias.copy_(module.bias.to(dtype).to(torch.float32))
    quantized.eval()
    return quantized


def get_conv_layers(model: nn.Module):
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            layers.append((name, module))
    return layers


def prune_conv_filters_ln_structured(module: nn.Conv2d, amount: float):
    if amount <= 0.0:
        return
    prune.ln_structured(module, name="weight", amount=amount, n=1, dim=0)


def remove_pruning_if_present(module: nn.Module, name: str = "weight"):
    if hasattr(module, f"{name}_orig"):
        prune.remove(module, name)


def get_eval_note(device: torch.device, precision: str):
    if precision == "fp32":
        return "FP32 eval"
    if device.type == "cuda":
        return "FP16 eval via CUDA autocast"
    return "FP16 requested on CPU; evaluated in FP32 compute"


def evaluate(model: nn.Module, loader, device: torch.device, precision: str, max_batches=None):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    use_amp = precision == "fp16" and device.type == "cuda"

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(inputs)
            else:
                outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss_sum += loss.item() * labels.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    avg_loss = loss_sum / total
    return accuracy, avg_loss


def finetune_model(
    model: nn.Module,
    train_loader,
    device: torch.device,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    max_train_batches=None,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    for epoch in range(epochs):
        model.train()
        run_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        avg_loss = run_loss / total
        train_acc = 100.0 * correct / total
        print(
            f"      Finetune epoch {epoch + 1:02d}/{epochs:02d} | "
            f"loss={avg_loss:.4f} | acc={train_acc:.2f}% | lr={scheduler.get_last_lr()[0]:.6f}"
        )


def apply_oneshot_filter_pruning(model: nn.Module, amount: float):
    for _, module in get_conv_layers(model):
        prune_conv_filters_ln_structured(module, amount=amount)
    for _, module in get_conv_layers(model):
        remove_pruning_if_present(module, "weight")


def apply_gradual_filter_pruning_with_retrain(
    model: nn.Module,
    amount: float,
    train_loader,
    device: torch.device,
    layer_finetune_epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    max_train_batches=None,
):
    conv_layers = get_conv_layers(model)
    for layer_idx, (name, module) in enumerate(conv_layers, start=1):
        prune_conv_filters_ln_structured(module, amount=amount)
        print(f"    Layer {layer_idx:02d}/{len(conv_layers):02d} pruned: {name}")
        if layer_finetune_epochs > 0:
            finetune_model(
                model=model,
                train_loader=train_loader,
                device=device,
                epochs=layer_finetune_epochs,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                max_train_batches=max_train_batches,
            )
        remove_pruning_if_present(module, "weight")


def compute_conv_filter_sparsity_percent(model: nn.Module):
    total_filters = 0
    zero_filters = 0
    with torch.no_grad():
        for _, module in get_conv_layers(model):
            w = module.weight
            out_channels = w.size(0)
            filt_norm = w.view(out_channels, -1).abs().sum(dim=1)
            total_filters += out_channels
            zero_filters += (filt_norm == 0).sum().item()
    return 0.0 if total_filters == 0 else 100.0 * zero_filters / total_filters


def compute_weight_sparsity_percent(model: nn.Module):
    total = 0
    zeros = 0
    with torch.no_grad():
        for _, module in get_conv_layers(model):
            w = module.weight
            total += w.numel()
            zeros += (w == 0).sum().item()
    return 0.0 if total == 0 else 100.0 * zeros / total


def validate_ratios(ratios):
    for ratio in ratios:
        if ratio < 0.0 or ratio >= 1.0:
            raise ValueError(f"Invalid ratio {ratio}. Expected 0 <= ratio < 1.")


def print_results_table(results):
    print("\n=== Filter Pruning Results (One-shot vs Gradual+Retrain) ===")
    header = (
        f"{'Ratio':>8}"
        f"{'Filter sparsity':>18}"
        f"{'Weight sparsity':>18}"
        f"{'Acc oneshot':>14}"
        f"{'Acc gradual':>14}"
        f"{'Loss oneshot':>14}"
        f"{'Loss gradual':>14}"
        f"{'Delta acc':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        delta = r.gradual_accuracy_percent - r.oneshot_accuracy_percent
        print(
            f"{r.pruning_ratio:>8.2f}"
            f"{r.filter_sparsity_percent:>18.2f}"
            f"{r.weight_sparsity_percent:>18.2f}"
            f"{r.oneshot_accuracy_percent:>14.2f}"
            f"{r.gradual_accuracy_percent:>14.2f}"
            f"{r.oneshot_loss:>14.4f}"
            f"{r.gradual_loss:>14.4f}"
            f"{delta:>12.2f}"
        )
    print()


def save_plot(results, output_path: str):
    ratios = [r.pruning_ratio for r in results]
    oneshot_acc = [r.oneshot_accuracy_percent for r in results]
    gradual_acc = [r.gradual_accuracy_percent for r in results]
    oneshot_loss = [r.oneshot_loss for r in results]
    gradual_loss = [r.gradual_loss for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(ratios, oneshot_acc, marker="o", linewidth=2, label="One-shot filter prune", color="#1f77b4")
    axes[0].plot(ratios, gradual_acc, marker="o", linewidth=2, label="Gradual prune + retrain", color="#2ca02c")
    axes[0].set_title("Accuracy vs Pruning Ratio")
    axes[0].set_xlabel("Per-layer filter pruning ratio")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(ratios, oneshot_loss, marker="o", linewidth=2, label="One-shot filter prune", color="#ff7f0e")
    axes[1].plot(ratios, gradual_loss, marker="o", linewidth=2, label="Gradual prune + retrain", color="#d62728")
    axes[1].set_title("Loss vs Pruning Ratio")
    axes[1].set_xlabel("Per-layer filter pruning ratio")
    axes[1].set_ylabel("Cross-entropy loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    validate_ratios(args.ratios)
    ratios = sorted(set(args.ratios))

    set_seed(args.seed)
    device = resolve_device(args.device)
    eval_note = get_eval_note(device, args.eval_precision)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Eval mode: {eval_note}")

    train_loader, test_loader = build_loaders(
        cifar_root=args.cifar_root,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        subset_size=args.train_subset_size,
        seed=args.seed,
    )

    base_model = load_resnet20_from_checkpoint(args.checkpoint)
    fp16_start_model = clone_and_quantize_storage(base_model, torch.float16)

    results = []
    for ratio in ratios:
        print(f"\n--- Ratio {ratio:.2f} ---")

        oneshot_model = copy.deepcopy(fp16_start_model).to(device).float()
        apply_oneshot_filter_pruning(oneshot_model, amount=ratio)
        oneshot_acc, oneshot_loss = evaluate(
            model=oneshot_model,
            loader=test_loader,
            device=device,
            precision=args.eval_precision,
            max_batches=args.max_test_batches,
        )
        print(f"  One-shot prune:        acc={oneshot_acc:.2f}% | loss={oneshot_loss:.4f}")

        gradual_model = copy.deepcopy(fp16_start_model).to(device).float()
        apply_gradual_filter_pruning_with_retrain(
            model=gradual_model,
            amount=ratio,
            train_loader=train_loader,
            device=device,
            layer_finetune_epochs=args.layer_finetune_epochs,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            max_train_batches=args.max_train_batches,
        )
        gradual_acc, gradual_loss = evaluate(
            model=gradual_model,
            loader=test_loader,
            device=device,
            precision=args.eval_precision,
            max_batches=args.max_test_batches,
        )

        filter_sparsity = compute_conv_filter_sparsity_percent(gradual_model)
        weight_sparsity = compute_weight_sparsity_percent(gradual_model)
        print(
            f"  Gradual + retrain:     acc={gradual_acc:.2f}% | loss={gradual_loss:.4f} | "
            f"filter_sparsity={filter_sparsity:.2f}%"
        )

        results.append(
            FilterPruningResult(
                pruning_ratio=ratio,
                filter_sparsity_percent=filter_sparsity,
                weight_sparsity_percent=weight_sparsity,
                oneshot_accuracy_percent=oneshot_acc,
                oneshot_loss=oneshot_loss,
                gradual_accuracy_percent=gradual_acc,
                gradual_loss=gradual_loss,
            )
        )

    print_results_table(results)

    output_json_dir = os.path.dirname(args.output_json)
    if output_json_dir:
        os.makedirs(output_json_dir, exist_ok=True)
    output_plot_dir = os.path.dirname(args.output_plot)
    if output_plot_dir:
        os.makedirs(output_plot_dir, exist_ok=True)

    payload = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "eval_precision": args.eval_precision,
        "eval_note": eval_note,
        "layer_finetune_epochs": args.layer_finetune_epochs,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "ratios": ratios,
        "train_subset_size": args.train_subset_size,
        "max_train_batches": args.max_train_batches,
        "max_test_batches": args.max_test_batches,
        "results": [r.__dict__ for r in results],
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved JSON results: {args.output_json}")

    save_plot(results, args.output_plot)
    print(f"Saved plot: {args.output_plot}")


if __name__ == "__main__":
    main()

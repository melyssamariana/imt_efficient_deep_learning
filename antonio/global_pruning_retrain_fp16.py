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
class RetrainResult:
    pruning_ratio: float
    weight_sparsity_percent: float
    pre_retrain_accuracy_percent: float
    pre_retrain_loss: float
    post_retrain_accuracy_percent: float
    post_retrain_loss: float


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Global magnitude pruning + retraining (Han et al. style) on ResNet20, "
            "with FP16 quantization-aware start and FP16 evaluation."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default="/home/antonio/imt_efficient_deep_learning/antonio/experiments/quantization/best_model.pth",
        help="Path to trained ResNet20 checkpoint.",
    )
    parser.add_argument(
        "--cifar-root",
        default="/opt/img/effdl-cifar10/",
        help="Root folder for CIFAR-10 dataset.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=128,
        help="Batch size for retraining.",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=256,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Dataloader workers.",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=[0.3, 0.5, 0.7, 0.9],
        help="Global pruning ratios to evaluate.",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=10,
        help="Retraining epochs after pruning.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.005,
        help="Learning rate for retraining.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD retraining.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for retraining.",
    )
    parser.add_argument(
        "--train-subset-size",
        type=int,
        default=None,
        help="Optional number of training samples for faster experiments.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2147483647,
        help="Seed for subset selection.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Optional cap on batches per training epoch (debug/speed).",
    )
    parser.add_argument(
        "--max-test-batches",
        type=int,
        default=None,
        help="Optional cap on evaluation batches (debug/speed).",
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
        default="/home/antonio/imt_efficient_deep_learning/antonio/experiments/quantization/global_pruning_retrain_fp16_results.json",
        help="Where to save JSON results.",
    )
    parser.add_argument(
        "--output-plot",
        default="/home/antonio/imt_efficient_deep_learning/antonio/experiments/quantization/global_pruning_retrain_fp16_plot.png",
        help="Where to save comparison plot.",
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


def get_prunable_parameters(model: nn.Module):
    params = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and module.weight is not None:
            params.append((module, "weight"))
    return params


def apply_global_pruning(model: nn.Module, amount: float):
    if amount <= 0.0:
        return
    params_to_prune = get_prunable_parameters(model)
    prune.global_unstructured(params_to_prune, pruning_method=prune.L1Unstructured, amount=amount)


def remove_pruning_reparam(model: nn.Module):
    for module, name in get_prunable_parameters(model):
        if hasattr(module, f"{name}_orig"):
            prune.remove(module, name)


def compute_weight_sparsity_percent(model: nn.Module) -> float:
    total = 0
    zeros = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and module.weight is not None:
                w = module.weight
                total += w.numel()
                zeros += (w == 0).sum().item()
    return 0.0 if total == 0 else 100.0 * zeros / total


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

    acc = 100.0 * correct / total
    avg_loss = loss_sum / total
    return acc, avg_loss


def finetune_pruned_model(
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
        running_loss = 0.0
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

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        avg_loss = running_loss / total
        train_acc = 100.0 * correct / total
        print(
            f"    Finetune epoch {epoch + 1:02d}/{epochs:02d} | "
            f"loss={avg_loss:.4f} | acc={train_acc:.2f}% | lr={scheduler.get_last_lr()[0]:.6f}"
        )


def validate_ratios(ratios):
    for ratio in ratios:
        if ratio < 0.0 or ratio >= 1.0:
            raise ValueError(f"Invalid ratio {ratio}. Expected 0 <= ratio < 1.")


def print_results_table(results):
    print("\n=== Global Pruning + Retrain Results ===")
    header = (
        f"{'Ratio':>8}"
        f"{'Sparsity (%)':>14}"
        f"{'Acc pre':>12}"
        f"{'Acc post':>12}"
        f"{'Loss pre':>12}"
        f"{'Loss post':>12}"
        f"{'Delta acc':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        delta = r.post_retrain_accuracy_percent - r.pre_retrain_accuracy_percent
        print(
            f"{r.pruning_ratio:>8.2f}"
            f"{r.weight_sparsity_percent:>14.2f}"
            f"{r.pre_retrain_accuracy_percent:>12.2f}"
            f"{r.post_retrain_accuracy_percent:>12.2f}"
            f"{r.pre_retrain_loss:>12.4f}"
            f"{r.post_retrain_loss:>12.4f}"
            f"{delta:>12.2f}"
        )
    print()


def save_plot(results, output_path: str):
    ratios = [r.pruning_ratio for r in results]
    pre_acc = [r.pre_retrain_accuracy_percent for r in results]
    post_acc = [r.post_retrain_accuracy_percent for r in results]
    pre_loss = [r.pre_retrain_loss for r in results]
    post_loss = [r.post_retrain_loss for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(ratios, pre_acc, marker="o", linewidth=2, label="After prune (no retrain)", color="#1f77b4")
    axes[0].plot(ratios, post_acc, marker="o", linewidth=2, label="After retrain", color="#2ca02c")
    axes[0].set_title("Accuracy vs Pruning Ratio")
    axes[0].set_xlabel("Pruning ratio")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(ratios, pre_loss, marker="o", linewidth=2, label="After prune (no retrain)", color="#ff7f0e")
    axes[1].plot(ratios, post_loss, marker="o", linewidth=2, label="After retrain", color="#d62728")
    axes[1].set_title("Loss vs Pruning Ratio")
    axes[1].set_xlabel("Pruning ratio")
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
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    train_loader, test_loader = build_loaders(
        cifar_root=args.cifar_root,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        subset_size=args.train_subset_size,
        seed=args.seed,
    )

    base_model = load_resnet20_from_checkpoint(args.checkpoint)
    # Start from FP16-quantized storage as requested, but keep float32 train compute for stability.
    fp16_start_model = clone_and_quantize_storage(base_model, torch.float16)

    results = []
    eval_note = get_eval_note(device, args.eval_precision)
    for ratio in ratios:
        print(f"\n--- Ratio {ratio:.2f} ---")
        model = copy.deepcopy(fp16_start_model).to(device).float()
        apply_global_pruning(model, amount=ratio)

        pre_acc, pre_loss = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            precision=args.eval_precision,
            max_batches=args.max_test_batches,
        )
        print(f"  After prune (no retrain): acc={pre_acc:.2f}% | loss={pre_loss:.4f}")

        if args.finetune_epochs > 0:
            finetune_pruned_model(
                model=model,
                train_loader=train_loader,
                device=device,
                epochs=args.finetune_epochs,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                max_train_batches=args.max_train_batches,
            )

        remove_pruning_reparam(model)
        sparsity = compute_weight_sparsity_percent(model)

        post_acc, post_loss = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            precision=args.eval_precision,
            max_batches=args.max_test_batches,
        )
        print(f"  After retrain:          acc={post_acc:.2f}% | loss={post_loss:.4f} | sparsity={sparsity:.2f}%")

        results.append(
            RetrainResult(
                pruning_ratio=ratio,
                weight_sparsity_percent=sparsity,
                pre_retrain_accuracy_percent=pre_acc,
                pre_retrain_loss=pre_loss,
                post_retrain_accuracy_percent=post_acc,
                post_retrain_loss=post_loss,
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
        "finetune_epochs": args.finetune_epochs,
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

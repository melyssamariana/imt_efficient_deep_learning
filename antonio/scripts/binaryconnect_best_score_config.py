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
class RunResult:
    pruning_ratio: float
    p_u: float
    pre_retrain_accuracy_percent: float
    pre_retrain_loss: float
    post_retrain_accuracy_percent: float
    post_retrain_loss: float
    raw_param_term: float
    raw_ops_term: float
    raw_score: float
    reference_score: float
    score_vs_reference_percent: float


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply the best-scoring strategy found so far (global pruning + retrain) "
            "to a BinaryConnect-trained ResNet20 checkpoint."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default="/home/antonio/imt_efficient_deep_learning/antonio/best_model_binaryconnect.pth",
        help="Path to BinaryConnect checkpoint.",
    )
    parser.add_argument(
        "--cifar-root",
        default="/opt/img/effdl-cifar10/",
        help="Root folder for CIFAR-10 dataset.",
    )
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[0.7],
        help="Global unstructured pruning ratios. Default is 0.7 (best previous config).",
    )
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-subset-size", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2147483647)
    parser.add_argument("--q-w", type=int, default=1, help="Weight quantization bits (BinaryConnect -> 1).")
    parser.add_argument("--q-a", type=int, default=32, help="Activation quantization bits.")
    parser.add_argument("--w-ref", type=float, default=5.6e6, help="Reference param score (ResNet18 FP16).")
    parser.add_argument("--f-ref", type=float, default=2.8e8, help="Reference ops score (ResNet18 FP16).")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output-json",
        default="/home/antonio/imt_efficient_deep_learning/antonio/experiments/quantization/binaryconnect_best_score_config_results.json",
    )
    parser.add_argument(
        "--output-plot",
        default="/home/antonio/imt_efficient_deep_learning/antonio/experiments/quantization/binaryconnect_best_score_config_plot.png",
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


def load_model_from_checkpoint(checkpoint_path: str) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = resnet20()
    model.load_state_dict(clean_state_dict(checkpoint))
    model.eval()
    return model


def build_loaders(cifar_root, train_batch_size, test_batch_size, num_workers, subset_size, seed):
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


def get_prunable_modules(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear)) and m.weight is not None]


def apply_global_pruning_collect_masks(model: nn.Module, ratio: float):
    modules = get_prunable_modules(model)
    params = [(m, "weight") for m in modules]
    if ratio > 0.0:
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=ratio)

    masks = {}
    for m in modules:
        if hasattr(m, "weight_mask"):
            masks[m] = m.weight_mask.detach().clone()
            prune.remove(m, "weight")
        else:
            masks[m] = torch.ones_like(m.weight.detach())
    return masks


def enforce_masks(masks):
    with torch.no_grad():
        for module, mask in masks.items():
            module.weight.mul_(mask.to(module.weight.device, dtype=module.weight.dtype))


def bc_binarize_with_masks(masks):
    saved = []
    with torch.no_grad():
        for module, mask in masks.items():
            w = module.weight
            saved.append((module, w.detach().clone()))
            binary = torch.where(w >= 0, torch.ones_like(w), -torch.ones_like(w))
            binary = binary * mask.to(w.device, dtype=w.dtype)
            w.copy_(binary)
    return saved


def bc_restore(saved):
    with torch.no_grad():
        for module, full_precision in saved:
            module.weight.copy_(full_precision)


def bc_clip_and_mask(masks):
    with torch.no_grad():
        for module, mask in masks.items():
            module.weight.clamp_(-1.0, 1.0)
            module.weight.mul_(mask.to(module.weight.device, dtype=module.weight.dtype))


def evaluate_binary_masked(model: nn.Module, loader, device: torch.device, masks, max_batches=None):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            saved = bc_binarize_with_masks(masks)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            bc_restore(saved)

            loss_sum += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total, loss_sum / total


def train_binaryconnect_masked(
    model: nn.Module,
    train_loader,
    device: torch.device,
    masks,
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
            saved = bc_binarize_with_masks(masks)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            bc_restore(saved)

            optimizer.step()
            bc_clip_and_mask(masks)

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(
            f"    Finetune epoch {epoch + 1:02d}/{epochs:02d} | "
            f"loss={epoch_loss:.4f} | acc={epoch_acc:.2f}% | lr={scheduler.get_last_lr()[0]:.6f}"
        )


def compute_pu_from_masks(masks):
    total = 0
    zeros = 0
    for mask in masks.values():
        total += mask.numel()
        zeros += (mask == 0).sum().item()
    return 0.0 if total == 0 else zeros / total


def count_model_weights(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_model_macs(model: nn.Module, input_shape=(3, 32, 32)):
    macs = 0
    handles = []

    def conv_hook(module: nn.Conv2d, _inputs, output):
        nonlocal macs
        out_h, out_w = output.shape[2], output.shape[3]
        kernel_ops = (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        macs += output.shape[0] * module.out_channels * out_h * out_w * kernel_ops

    def linear_hook(module: nn.Linear, _inputs, output):
        nonlocal macs
        macs += output.shape[0] * module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        model(torch.randn(1, *input_shape))
    for h in handles:
        h.remove()
    return int(macs)


def compute_raw_score(p_s, p_u, q_w, q_a, w, f):
    raw_param = (1.0 - (p_s + p_u)) * (q_w / 32.0) * w
    raw_ops = (1.0 - p_s) * (max(q_w, q_a) / 32.0) * f
    return raw_param, raw_ops, raw_param + raw_ops


def save_plot(results, output_path: str):
    ratios = [r.pruning_ratio for r in results]
    pre_acc = [r.pre_retrain_accuracy_percent for r in results]
    post_acc = [r.post_retrain_accuracy_percent for r in results]
    rel_score = [r.score_vs_reference_percent for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(ratios, pre_acc, marker="o", linewidth=2, label="After prune", color="#1f77b4")
    axes[0].plot(ratios, post_acc, marker="o", linewidth=2, label="After retrain", color="#2ca02c")
    axes[0].set_title("Accuracy vs Pruning Ratio")
    axes[0].set_xlabel("Pruning ratio")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(ratios, rel_score, marker="o", linewidth=2, color="#d62728")
    axes[1].set_title("Score (% of ResNet18 FP16 ref)")
    axes[1].set_xlabel("Pruning ratio")
    axes[1].set_ylabel("Score (%), lower is better")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    for ratio in args.ratios:
        if ratio < 0.0 or ratio >= 1.0:
            raise ValueError(f"Invalid pruning ratio: {ratio}")

    set_seed(args.seed)
    ratios = sorted(set(args.ratios))
    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print("Strategy: global unstructured pruning + retrain (best previous config baseline)")

    train_loader, test_loader = build_loaders(
        cifar_root=args.cifar_root,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        subset_size=args.train_subset_size,
        seed=args.seed,
    )

    base_model = load_model_from_checkpoint(args.checkpoint)
    w = count_model_weights(base_model)
    f = count_model_macs(base_model, input_shape=(3, 32, 32))
    reference_score = args.w_ref + args.f_ref

    print(f"w={w} | f={f}")
    print(f"q_w={args.q_w} bits | q_a={args.q_a} bits")
    print(f"Reference raw score (ResNet18 FP16): {reference_score:.2f}")

    results = []
    for ratio in ratios:
        print(f"\n--- Ratio {ratio:.2f} ---")
        model = copy.deepcopy(base_model).to(device)

        masks = apply_global_pruning_collect_masks(model, ratio=ratio)
        enforce_masks(masks)
        p_u = compute_pu_from_masks(masks)

        pre_acc, pre_loss = evaluate_binary_masked(
            model=model,
            loader=test_loader,
            device=device,
            masks=masks,
            max_batches=args.max_test_batches,
        )
        print(f"  After prune (binary eval): acc={pre_acc:.2f}% | loss={pre_loss:.4f} | p_u={p_u:.4f}")

        if args.finetune_epochs > 0:
            train_binaryconnect_masked(
                model=model,
                train_loader=train_loader,
                device=device,
                masks=masks,
                epochs=args.finetune_epochs,
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                max_train_batches=args.max_train_batches,
            )

        post_acc, post_loss = evaluate_binary_masked(
            model=model,
            loader=test_loader,
            device=device,
            masks=masks,
            max_batches=args.max_test_batches,
        )

        p_s = 0.0
        raw_param, raw_ops, raw_score = compute_raw_score(
            p_s=p_s,
            p_u=p_u,
            q_w=args.q_w,
            q_a=args.q_a,
            w=w,
            f=f,
        )
        score_vs_ref_pct = 100.0 * raw_score / reference_score

        print(
            f"  After retrain (binary eval): acc={post_acc:.2f}% | loss={post_loss:.4f} | "
            f"raw_score={raw_score:.2f} ({score_vs_ref_pct:.2f}% of ref)"
        )

        results.append(
            RunResult(
                pruning_ratio=ratio,
                p_u=p_u,
                pre_retrain_accuracy_percent=pre_acc,
                pre_retrain_loss=pre_loss,
                post_retrain_accuracy_percent=post_acc,
                post_retrain_loss=post_loss,
                raw_param_term=raw_param,
                raw_ops_term=raw_ops,
                raw_score=raw_score,
                reference_score=reference_score,
                score_vs_reference_percent=score_vs_ref_pct,
            )
        )

    out_json_dir = os.path.dirname(args.output_json)
    if out_json_dir:
        os.makedirs(out_json_dir, exist_ok=True)
    out_plot_dir = os.path.dirname(args.output_plot)
    if out_plot_dir:
        os.makedirs(out_plot_dir, exist_ok=True)

    payload = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "strategy": "global_pruning_retrain_binaryconnect",
        "ratios": ratios,
        "finetune_epochs": args.finetune_epochs,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "q_w": args.q_w,
        "q_a": args.q_a,
        "w": w,
        "f": f,
        "w_ref": args.w_ref,
        "f_ref": args.f_ref,
        "results": [r.__dict__ for r in results],
    }
    with open(args.output_json, "w", encoding="utf-8") as fobj:
        json.dump(payload, fobj, indent=2)
    print(f"\nSaved JSON results: {args.output_json}")

    save_plot(results, args.output_plot)
    print(f"Saved plot: {args.output_plot}")


if __name__ == "__main__":
    main()

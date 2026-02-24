import argparse
import copy
import json
import os
from dataclasses import dataclass
from typing import List

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
class SearchResult:
    p_s_target: float
    p_u_target_relative: float
    p_u_score_equivalent: float
    measured_filter_sparsity_percent: float
    measured_weight_sparsity_percent: float
    pre_retrain_accuracy_percent: float
    pre_retrain_loss: float
    post_retrain_accuracy_percent: float
    post_retrain_loss: float
    raw_param_term: float
    raw_ops_term: float
    raw_score: float
    normalized_param_term: float
    normalized_ops_term: float
    normalized_score: float


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Search best score with combined FP8 (weights+activations emulated) + structured pruning "
            "+ unstructured pruning + retraining on ResNet20/CIFAR-10."
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
    parser.add_argument("--train-batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--structured-ratios",
        nargs="+",
        type=float,
        default=[0.0, 0.2, 0.4],
        help="Structured pruning ratios (p_s target, per Conv layer filters).",
    )
    parser.add_argument(
        "--unstructured-ratios",
        nargs="+",
        type=float,
        default=[0.5, 0.7, 0.9],
        help="Global unstructured pruning ratio over remaining connections.",
    )
    parser.add_argument("--finetune-epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-subset-size", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2147483647)
    parser.add_argument("--acc-threshold", type=float, default=90.0)
    parser.add_argument(
        "--fp8-format",
        choices=["e4m3", "e5m2"],
        default="e4m3",
        help="FP8 format for emulation.",
    )
    parser.add_argument(
        "--q-w",
        type=int,
        default=8,
        help="Bits used in score for weights (FP8 -> 8).",
    )
    parser.add_argument(
        "--q-a",
        type=int,
        default=8,
        help="Bits used in score for activations (FP8 -> 8).",
    )
    parser.add_argument(
        "--w-ref",
        type=float,
        default=1.0,
        help="Reference denominator for param term (use 1.0 for raw score).",
    )
    parser.add_argument(
        "--f-ref",
        type=float,
        default=1.0,
        help="Reference denominator for ops term (use 1.0 for raw score).",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--output-json",
        default="/home/antonio/imt_efficient_deep_learning/antonio/experiments/quantization/fp8_structured_unstructured_search_results.json",
    )
    parser.add_argument(
        "--output-plot",
        default="/home/antonio/imt_efficient_deep_learning/antonio/experiments/quantization/fp8_structured_unstructured_search_plot.png",
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


def load_resnet20(checkpoint_path: str):
    model = resnet20()
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(state))
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


def choose_fp8_dtype(fp8_format: str):
    if fp8_format == "e4m3":
        if hasattr(torch, "float8_e4m3fn"):
            return torch.float8_e4m3fn
    elif fp8_format == "e5m2":
        if hasattr(torch, "float8_e5m2"):
            return torch.float8_e5m2
    raise RuntimeError("Requested FP8 dtype is not available in this PyTorch build.")


def quantize_weights_fp8_inplace(model: nn.Module, fp8_dtype):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if module.weight is not None:
                    module.weight.copy_(module.weight.to(fp8_dtype).to(torch.float32))
                if module.bias is not None:
                    module.bias.copy_(module.bias.to(fp8_dtype).to(torch.float32))


class FP8ActivationEmulator:
    def __init__(self, model: nn.Module, fp8_dtype):
        self.model = model
        self.fp8_dtype = fp8_dtype
        self.handles = []

    def _pre_hook(self, _module, inputs):
        if not inputs:
            return inputs
        x = inputs[0]
        if not torch.is_tensor(x) or not x.is_floating_point():
            return inputs
        qx = x.to(self.fp8_dtype).to(torch.float32)
        if len(inputs) == 1:
            return (qx,)
        return (qx, *inputs[1:])

    def enable(self):
        if self.handles:
            return
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.handles.append(module.register_forward_pre_hook(self._pre_hook))

    def disable(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def get_conv_layers(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, nn.Conv2d)]


def get_prunable_params(model: nn.Module):
    params = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and module.weight is not None:
            params.append((module, "weight"))
    return params


def apply_combined_pruning(model: nn.Module, p_s: float, p_u_relative: float):
    if p_s > 0.0:
        for module in get_conv_layers(model):
            prune.ln_structured(module, name="weight", amount=p_s, n=1, dim=0)
    if p_u_relative > 0.0:
        params = get_prunable_params(model)
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=p_u_relative)


def remove_pruning_reparam(model: nn.Module):
    for module, name in get_prunable_params(model):
        if hasattr(module, f"{name}_orig"):
            prune.remove(module, name)


def compute_filter_sparsity_percent(model: nn.Module):
    total_filters = 0
    zero_filters = 0
    with torch.no_grad():
        for module in get_conv_layers(model):
            w = module.weight
            norms = w.view(w.size(0), -1).abs().sum(dim=1)
            total_filters += w.size(0)
            zero_filters += (norms == 0).sum().item()
    return 0.0 if total_filters == 0 else 100.0 * zero_filters / total_filters


def compute_weight_sparsity_percent(model: nn.Module):
    total = 0
    zeros = 0
    with torch.no_grad():
        for module, _ in get_prunable_params(model):
            w = module.weight
            total += w.numel()
            zeros += (w == 0).sum().item()
    return 0.0 if total == 0 else 100.0 * zeros / total


def evaluate(model, loader, device, act_emulator: FP8ActivationEmulator, max_batches=None):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    act_emulator.enable()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss_sum += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    act_emulator.disable()

    return 100.0 * correct / total, loss_sum / total


def finetune(
    model,
    train_loader,
    device,
    act_emulator: FP8ActivationEmulator,
    fp8_dtype,
    epochs,
    lr,
    momentum,
    weight_decay,
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
        total = 0
        correct = 0
        running_loss = 0.0

        act_emulator.enable()
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

            # Keep weight storage quantized to FP8 emulation after each update.
            quantize_weights_fp8_inplace(model, fp8_dtype)

            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        act_emulator.disable()

        scheduler.step()
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(
            f"    Finetune epoch {epoch + 1:02d}/{epochs:02d} | "
            f"loss={epoch_loss:.4f} | acc={epoch_acc:.2f}% | lr={scheduler.get_last_lr()[0]:.6f}"
        )


def count_weights(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_macs(model: nn.Module, input_shape=(3, 32, 32)):
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


def compute_score_terms(p_s, p_u_score, q_w, q_a, w, f, w_ref, f_ref):
    raw_param = (1.0 - (p_s + p_u_score)) * (q_w / 32.0) * w
    raw_ops = (1.0 - p_s) * (max(q_w, q_a) / 32.0) * f
    raw_score = (raw_param / w_ref) + (raw_ops / f_ref)
    norm_param = raw_param / w_ref
    norm_ops = raw_ops / f_ref
    norm_score = norm_param + norm_ops
    return raw_param, raw_ops, raw_score, norm_param, norm_ops, norm_score


def save_scatter_plot(results: List[SearchResult], output_path: str, acc_threshold: float):
    scores = [r.raw_score for r in results]
    accs = [r.post_retrain_accuracy_percent for r in results]
    colors = [r.p_s_target for r in results]
    unique_ps = sorted(set(colors))

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(scores, accs, c=colors, cmap="viridis", s=80)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("p_s target")
    cbar.set_ticks(unique_ps)
    cbar.set_ticklabels([f"{v:.2f}" for v in unique_ps])

    for r in results:
        label = f"(ps={r.p_s_target:.2f}, pu={r.p_u_target_relative:.2f})"
        ax.annotate(label, (r.raw_score, r.post_retrain_accuracy_percent), fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.axhline(acc_threshold, color="red", linestyle="--", linewidth=1, label=f"Acc threshold {acc_threshold:.1f}%")
    ax.set_xlabel("Raw score (lower is better)")
    ax.set_ylabel("Post-retrain accuracy (%)")
    ax.set_title("FP8 + Structured + Unstructured Search")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_results_table(results: List[SearchResult]):
    print("\n=== Search Results (sorted by raw score) ===")
    header = (
        f"{'p_s':>6}"
        f"{'p_u_rel':>9}"
        f"{'p_u_eq':>9}"
        f"{'Acc pre':>10}"
        f"{'Acc post':>10}"
        f"{'Sparsity W%':>13}"
        f"{'Sparsity F%':>13}"
        f"{'Raw Score':>14}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.p_s_target:>6.2f}"
            f"{r.p_u_target_relative:>9.2f}"
            f"{r.p_u_score_equivalent:>9.3f}"
            f"{r.pre_retrain_accuracy_percent:>10.2f}"
            f"{r.post_retrain_accuracy_percent:>10.2f}"
            f"{r.measured_weight_sparsity_percent:>13.2f}"
            f"{r.measured_filter_sparsity_percent:>13.2f}"
            f"{r.raw_score:>14.2f}"
        )
    print()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    fp8_dtype = choose_fp8_dtype(args.fp8_format)



    for p_s in args.structured_ratios:
        if p_s < 0.0 or p_s >= 1.0:
            raise ValueError(f"Invalid p_s value: {p_s}")
    for p_u in args.unstructured_ratios:
        if p_u < 0.0 or p_u >= 1.0:
            raise ValueError(f"Invalid p_u value: {p_u}")

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"FP8 format: {args.fp8_format} ({fp8_dtype})")
    print("Method: FP8 emulation (weights + activations) + structured pruning + unstructured pruning + retrain")

    train_loader, test_loader = build_loaders(
        args.cifar_root,
        args.train_batch_size,
        args.test_batch_size,
        args.num_workers,
        args.train_subset_size,
        args.seed,
    )

    base_model = load_resnet20(args.checkpoint)
    w = count_weights(base_model)
    f = count_macs(base_model, input_shape=(3, 32, 32))
    args.w_ref = 5.6 * 10**6
    args.f_ref = 2.8 * 10**8
    print(f"w={w} | f={f} | q_w={args.q_w} | q_a={args.q_a} | w_ref={args.w_ref} | f_ref={args.f_ref}")

    all_results = []
    for p_s in sorted(set(args.structured_ratios)):
        for p_u_rel in sorted(set(args.unstructured_ratios)):
            p_u_score = (1.0 - p_s) * p_u_rel
            if p_s + p_u_score >= 1.0:
                print(f"Skipping invalid combo p_s={p_s:.2f}, p_u_rel={p_u_rel:.2f} (p_s+p_u_eq >= 1).")
                continue

            print(f"\n--- Combo p_s={p_s:.2f}, p_u_rel={p_u_rel:.2f} ---")
            model = copy.deepcopy(base_model).to(device).float()
            quantize_weights_fp8_inplace(model, fp8_dtype)
            act_emulator = FP8ActivationEmulator(model, fp8_dtype)

            apply_combined_pruning(model, p_s=p_s, p_u_relative=p_u_rel)
            pre_acc, pre_loss = evaluate(
                model=model,
                loader=test_loader,
                device=device,
                act_emulator=act_emulator,
                max_batches=args.max_test_batches,
            )
            print(f"  After prune: acc={pre_acc:.2f}% | loss={pre_loss:.4f}")

            if args.finetune_epochs > 0:
                finetune(
                    model=model,
                    train_loader=train_loader,
                    device=device,
                    act_emulator=act_emulator,
                    fp8_dtype=fp8_dtype,
                    epochs=args.finetune_epochs,
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    max_train_batches=args.max_train_batches,
                )

            post_acc, post_loss = evaluate(
                model=model,
                loader=test_loader,
                device=device,
                act_emulator=act_emulator,
                max_batches=args.max_test_batches,
            )

            # Measure sparsity after consolidation.
            remove_pruning_reparam(model)
            weight_sparsity = compute_weight_sparsity_percent(model)
            filter_sparsity = compute_filter_sparsity_percent(model)

            raw_param, raw_ops, raw_score, norm_param, norm_ops, norm_score = compute_score_terms(
                p_s=p_s,
                p_u_score=p_u_score,
                q_w=args.q_w,
                q_a=args.q_a,
                w=w,
                f=f,
                w_ref=args.w_ref,
                f_ref=args.f_ref,
            )
            print(
                f"  After retrain: acc={post_acc:.2f}% | loss={post_loss:.4f} | "
                f"raw_score={raw_score:.2f} | norm_score={norm_score:.6f}"
            )

            all_results.append(
                SearchResult(
                    p_s_target=p_s,
                    p_u_target_relative=p_u_rel,
                    p_u_score_equivalent=p_u_score,
                    measured_filter_sparsity_percent=filter_sparsity,
                    measured_weight_sparsity_percent=weight_sparsity,
                    pre_retrain_accuracy_percent=pre_acc,
                    pre_retrain_loss=pre_loss,
                    post_retrain_accuracy_percent=post_acc,
                    post_retrain_loss=post_loss,
                    raw_param_term=raw_param,
                    raw_ops_term=raw_ops,
                    raw_score=raw_score,
                    normalized_param_term=norm_param,
                    normalized_ops_term=norm_ops,
                    normalized_score=norm_score,
                )
            )

    if not all_results:
        raise RuntimeError("No valid experiment combination was run.")

    ranked = sorted(all_results, key=lambda x: x.raw_score)
    eligible = [r for r in ranked if r.post_retrain_accuracy_percent >= args.acc_threshold]

    print_results_table(ranked)
    if eligible:
        best = eligible[0]
        print(
            f"Best eligible (acc >= {args.acc_threshold:.2f}%): "
            f"p_s={best.p_s_target:.2f}, p_u_rel={best.p_u_target_relative:.2f}, "
            f"acc={best.post_retrain_accuracy_percent:.2f}%, raw_score={best.raw_score:.2f}"
        )
    else:
        best = ranked[0]
        print(
            f"No eligible combo reached {args.acc_threshold:.2f}%. "
            f"Best raw score overall: p_s={best.p_s_target:.2f}, p_u_rel={best.p_u_target_relative:.2f}, "
            f"acc={best.post_retrain_accuracy_percent:.2f}%, raw_score={best.raw_score:.2f}"
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
        "fp8_format": args.fp8_format,
        "q_w": args.q_w,
        "q_a": args.q_a,
        "w": w,
        "f": f,
        "w_ref": args.w_ref,
        "f_ref": args.f_ref,
        "acc_threshold": args.acc_threshold,
        "structured_ratios": sorted(set(args.structured_ratios)),
        "unstructured_ratios": sorted(set(args.unstructured_ratios)),
        "finetune_epochs": args.finetune_epochs,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "results_sorted_by_raw_score": [r.__dict__ for r in ranked],
        "best_eligible": eligible[0].__dict__ if eligible else None,
        "best_overall": ranked[0].__dict__,
    }
    with open(args.output_json, "w", encoding="utf-8") as fobj:
        json.dump(payload, fobj, indent=2)
    print(f"Saved JSON results: {args.output_json}")

    save_scatter_plot(ranked, args.output_plot, args.acc_threshold)
    print(f"Saved plot: {args.output_plot}")


if __name__ == "__main__":
    main()

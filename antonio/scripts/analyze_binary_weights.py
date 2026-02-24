import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from models.resnet_s import resnet20

BITS_FLOAT = 32
BITS_BINARY = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze BinaryConnect weights and compare baseline vs binarized ResNet20."
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="best_model.pth",
        help="Checkpoint analyzed for detailed weight distribution.",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        default="best_model.pth",
        help="Baseline (full precision) checkpoint for size comparison.",
    )
    parser.add_argument(
        "--binary-checkpoint",
        default="best_model_binarized.pth",
        help="Binarized checkpoint for size comparison.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for operation counting (default: 1).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip histogram generation.",
    )
    return parser.parse_args()


def load_clean_state_dict(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")

    cleaned = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        cleaned_key = key[6:] if key.startswith("model.") else key
        cleaned[cleaned_key] = value
    return cleaned


def load_resnet20_checkpoint(checkpoint_path):
    model = resnet20()
    state_dict = load_clean_state_dict(checkpoint_path)
    model.load_state_dict(state_dict)
    return model


def format_mib(byte_count):
    return byte_count / (1024 ** 2)


def conv_linear_weight_names(model):
    names = set()
    for module_name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            weight_name = f"{module_name}.weight" if module_name else "weight"
            names.add(weight_name)
    return names


def count_params_for_size(model):
    binary_weight_names = conv_linear_weight_names(model)
    total_params = 0
    binary_candidate_params = 0

    for name, param in model.named_parameters():
        count = param.numel()
        total_params += count
        if name in binary_weight_names:
            binary_candidate_params += count

    non_binary_params = total_params - binary_candidate_params
    baseline_bits = total_params * BITS_FLOAT
    binaryconnect_bits = binary_candidate_params * BITS_BINARY + non_binary_params * BITS_FLOAT

    return {
        "total_params": total_params,
        "binary_candidate_params": binary_candidate_params,
        "non_binary_params": non_binary_params,
        "baseline_bits": baseline_bits,
        "binaryconnect_bits": binaryconnect_bits,
        "baseline_bytes": baseline_bits / 8.0,
        "binaryconnect_bytes": binaryconnect_bits / 8.0,
    }


def exact_binary_ratio(model):
    weights = []
    for name, param in model.named_parameters():
        if "weight" in name:
            weights.append(param.detach().cpu().numpy().ravel())
    if not weights:
        return 0.0
    flat = np.concatenate(weights)
    exact = np.sum(np.isclose(flat, -1.0, atol=1e-6) | np.isclose(flat, 1.0, atol=1e-6))
    return 100.0 * exact / len(flat)


def analyze_weight_distribution(model):
    all_weights = []
    layer_stats = []

    for name, param in model.named_parameters():
        if "weight" not in name:
            continue
        weights = param.detach().cpu().numpy().flatten()
        all_weights.extend(weights)

        min_val = weights.min()
        max_val = weights.max()
        mean_val = weights.mean()
        std_val = weights.std()

        outside_range = np.sum((weights < -1.0) | (weights > 1.0))
        total = len(weights)
        pct_outside = 100.0 * outside_range / total

        exactly_minus_one = np.sum(np.isclose(weights, -1.0, atol=1e-6))
        exactly_plus_one = np.sum(np.isclose(weights, 1.0, atol=1e-6))
        in_between = total - exactly_minus_one - exactly_plus_one

        distances_to_binary = np.minimum(np.abs(weights - 1.0), np.abs(weights + 1.0))
        avg_distance = distances_to_binary.mean()

        layer_stats.append(
            {
                "name": name,
                "min": min_val,
                "max": max_val,
                "mean": mean_val,
                "std": std_val,
                "outside_range": outside_range,
                "total": total,
                "pct_outside": pct_outside,
                "exactly_minus_one": exactly_minus_one,
                "exactly_plus_one": exactly_plus_one,
                "in_between": in_between,
                "avg_distance": avg_distance,
            }
        )

    return np.array(all_weights), layer_stats


def estimate_inference_ops(model, batch_size=1, image_size=32):
    model = model.eval().cpu()
    totals = {
        "conv_linear_mul": 0,
        "conv_linear_add": 0,
        "residual_add": 0,
        "output_elements": 0,
    }
    layer_ops = []
    hooks = []

    def make_conv_hook(name):
        def hook(module, inputs, outputs):
            x = inputs[0]
            y = outputs
            batch = x.shape[0]
            out_c, out_h, out_w = y.shape[1], y.shape[2], y.shape[3]
            k_h, k_w = module.kernel_size
            kernel_size = (module.in_channels // module.groups) * k_h * k_w
            output_elems = batch * out_c * out_h * out_w
            muls = output_elems * kernel_size
            adds = output_elems * (kernel_size - 1 + (1 if module.bias is not None else 0))

            totals["conv_linear_mul"] += muls
            totals["conv_linear_add"] += adds
            totals["output_elements"] += output_elems
            layer_ops.append({"name": name, "type": "Conv2d", "muls": muls, "adds": adds})

        return hook

    def make_linear_hook(name):
        def hook(module, inputs, outputs):
            x = inputs[0]
            batch = x.shape[0]
            output_elems = batch * module.out_features
            muls = output_elems * module.in_features
            adds = output_elems * (module.in_features - 1 + (1 if module.bias is not None else 0))

            totals["conv_linear_mul"] += muls
            totals["conv_linear_add"] += adds
            totals["output_elements"] += output_elems
            layer_ops.append({"name": name, "type": "Linear", "muls": muls, "adds": adds})

        return hook

    def make_basicblock_hook(name):
        def hook(module, inputs, outputs):
            residual_adds = outputs.numel()
            totals["residual_add"] += residual_adds
            layer_ops.append({"name": name, "type": "ResidualAdd", "muls": 0, "adds": residual_adds})

        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(make_conv_hook(name)))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_linear_hook(name)))
        elif module.__class__.__name__ == "BasicBlock":
            hooks.append(module.register_forward_hook(make_basicblock_hook(name)))

    dummy = torch.randn(batch_size, 3, image_size, image_size)
    with torch.no_grad():
        _ = model(dummy)

    for hook in hooks:
        hook.remove()

    totals["total_add"] = totals["conv_linear_add"] + totals["residual_add"]
    return totals, layer_ops


def maybe_checkpoint_size(path):
    if os.path.exists(path):
        return os.path.getsize(path)
    return None


def maybe_binary_ratio_from_checkpoint(path):
    if not os.path.exists(path):
        return None
    try:
        model = load_resnet20_checkpoint(path)
        return exact_binary_ratio(model)
    except Exception:
        return None


def print_size_and_ops_comparison(model, baseline_checkpoint, binary_checkpoint, op_totals):
    size_stats = count_params_for_size(model)
    baseline_ckpt_size = maybe_checkpoint_size(baseline_checkpoint)
    binary_ckpt_size = maybe_checkpoint_size(binary_checkpoint)
    baseline_ckpt_binary_ratio = maybe_binary_ratio_from_checkpoint(baseline_checkpoint)
    binary_ckpt_binary_ratio = maybe_binary_ratio_from_checkpoint(binary_checkpoint)

    baseline_bits = size_stats["baseline_bits"]
    bc_bits = size_stats["binaryconnect_bits"]
    compression = baseline_bits / bc_bits if bc_bits > 0 else float("inf")

    baseline_mul = op_totals["conv_linear_mul"]
    baseline_add = op_totals["total_add"]
    binary_xnor = baseline_mul
    binary_popcount = op_totals["output_elements"]

    print(f"\n{'='*80}")
    print("BASELINE VS BINARYCONNECT (RESNET20)")
    print(f"{'='*80}\n")

    print("Model size (theoretical in-memory):")
    print(f"  Total parameters: {size_stats['total_params']:,}")
    print(f"  Binarizable params (Conv/Linear weights): {size_stats['binary_candidate_params']:,}")
    print(f"  Non-binarized params (BN/bias/others): {size_stats['non_binary_params']:,}")
    print(
        f"  Baseline float32: {format_mib(size_stats['baseline_bytes']):.4f} MiB "
        f"({baseline_bits:,} bits)"
    )
    print(
        f"  BinaryConnect (1-bit Conv/Linear + 32-bit others): "
        f"{format_mib(size_stats['binaryconnect_bytes']):.4f} MiB ({bc_bits:,} bits)"
    )
    print(f"  Theoretical compression: {compression:.2f}x")

    print("\nCheckpoint size on disk:")
    if baseline_ckpt_size is None:
        print(f"  Baseline checkpoint ({baseline_checkpoint}): file not found")
    else:
        print(
            f"  Baseline checkpoint ({baseline_checkpoint}): "
            f"{format_mib(baseline_ckpt_size):.4f} MiB"
        )
    if binary_ckpt_size is None:
        print(f"  Binarized checkpoint ({binary_checkpoint}): file not found")
    else:
        print(
            f"  Binarized checkpoint ({binary_checkpoint}): "
            f"{format_mib(binary_ckpt_size):.4f} MiB"
        )
    if baseline_ckpt_size is not None and binary_ckpt_size is not None:
        if binary_ckpt_size > 0:
            print(f"  On-disk compression observed: {baseline_ckpt_size / binary_ckpt_size:.2f}x")
        print("  Note: .pth stores tensors as float32 unless explicitly bit-packed.")
    print("Checkpoint binarization level (exact {-1, +1} in all weight tensors):")
    if baseline_ckpt_binary_ratio is None:
        print(f"  Baseline checkpoint ({baseline_checkpoint}): unavailable")
    else:
        print(f"  Baseline checkpoint ({baseline_checkpoint}): {baseline_ckpt_binary_ratio:.2f}%")
    if binary_ckpt_binary_ratio is None:
        print(f"  Binarized checkpoint ({binary_checkpoint}): unavailable")
    else:
        print(f"  Binarized checkpoint ({binary_checkpoint}): {binary_ckpt_binary_ratio:.2f}%")

    print("\nInference operations per forward pass (Conv/Linear + residual adds):")
    print(f"  Baseline ResNet20:")
    print(f"    Multiplications: {baseline_mul:,}")
    print(f"    Additions:       {baseline_add:,}")
    print(f"    Total (Mul+Add): {baseline_mul + baseline_add:,}")
    print(f"  BinaryConnect ResNet20:")
    print(f"    Float multiplications: 0 (replaced by binary ops)")
    print(f"    XNOR-equivalent ops:   {binary_xnor:,}")
    print(f"    POPCOUNT-equivalent ops: {binary_popcount:,}")
    print(f"    Additions (incl. residual): {baseline_add:,}")
    print("  Note: BN/ReLU/Pooling ops are not included in these counts.")


def main():
    args = parse_args()

    print(f"\n{'='*80}")
    print("BinaryConnect Weight Analysis")
    print(f"{'='*80}\n")
    print(f"Analyzing checkpoint: {args.checkpoint}\n")

    model = load_resnet20_checkpoint(args.checkpoint)
    print(f"✓ Loaded model from {args.checkpoint}\n")

    print(f"{'='*80}")
    print("WEIGHT ANALYSIS")
    print(f"{'='*80}\n")

    all_weights, layer_stats = analyze_weight_distribution(model)

    print("Layer-by-Layer Statistics:")
    print(f"{'-'*80}")
    for stat in layer_stats:
        print(f"\n{stat['name']}:")
        print(f"  Range: [{stat['min']:.6f}, {stat['max']:.6f}]")
        print(f"  Mean: {stat['mean']:.6f}, Std: {stat['std']:.6f}")
        print(f"  Total weights: {stat['total']}")
        print(f"  Exactly -1: {stat['exactly_minus_one']} ({100.0*stat['exactly_minus_one']/stat['total']:.2f}%)")
        print(f"  Exactly +1: {stat['exactly_plus_one']} ({100.0*stat['exactly_plus_one']/stat['total']:.2f}%)")
        print(f"  In between: {stat['in_between']} ({100.0*stat['in_between']/stat['total']:.2f}%)")
        print(f"  Outside [-1, 1]: {stat['outside_range']} ({stat['pct_outside']:.2f}%)")
        print(f"  Avg distance to nearest binary: {stat['avg_distance']:.6f}")

    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    print(f"{'='*80}\n")
    print(f"Total weight parameters: {len(all_weights)}")
    print(f"Range: [{all_weights.min():.6f}, {all_weights.max():.6f}]")
    print(f"Mean: {all_weights.mean():.6f}")
    print(f"Std: {all_weights.std():.6f}")
    print(f"Median: {np.median(all_weights):.6f}")

    exactly_minus_one_total = np.sum(np.isclose(all_weights, -1.0, atol=1e-6))
    exactly_plus_one_total = np.sum(np.isclose(all_weights, 1.0, atol=1e-6))
    in_between_total = len(all_weights) - exactly_minus_one_total - exactly_plus_one_total
    outside_range_total = np.sum((all_weights < -1.0) | (all_weights > 1.0))

    print(f"\nValue Distribution:")
    print(f"  Exactly -1: {exactly_minus_one_total} ({100.0*exactly_minus_one_total/len(all_weights):.2f}%)")
    print(f"  Exactly +1: {exactly_plus_one_total} ({100.0*exactly_plus_one_total/len(all_weights):.2f}%)")
    print(f"  In between [-1, +1]: {in_between_total} ({100.0*in_between_total/len(all_weights):.2f}%)")
    print(f"  Outside [-1, +1]: {outside_range_total} ({100.0*outside_range_total/len(all_weights):.2f}%)")

    distances_to_binary_total = np.minimum(np.abs(all_weights - 1.0), np.abs(all_weights + 1.0))
    print(f"\nAverage distance to nearest binary value: {distances_to_binary_total.mean():.6f}")
    print(f"Exact +/-1 ratio in analyzed checkpoint: {exact_binary_ratio(model):.2f}%")

    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}\n")

    if outside_range_total == 0 and in_between_total < len(all_weights) * 0.01:
        print("✓ Weights appear to be BINARIZED (all values are at or very close to -1 or +1)")
    elif outside_range_total == 0:
        print("⚠ Weights are CLIPPED to [-1, 1] but NOT fully binarized")
        print(f"  {100.0*in_between_total/len(all_weights):.2f}% of weights are between -1 and +1")
    else:
        print("✗ Weights are NOT binarized")
        print(f"  {100.0*outside_range_total/len(all_weights):.2f}% of weights are outside [-1, 1]")
        print("  This suggests the model was trained with standard methods, not BinaryConnect")

    op_totals, _ = estimate_inference_ops(model, batch_size=args.batch_size, image_size=32)
    print_size_and_ops_comparison(
        model,
        baseline_checkpoint=args.baseline_checkpoint,
        binary_checkpoint=args.binary_checkpoint,
        op_totals=op_totals,
    )

    print(f"\n{'='*80}\n")

    if args.no_plot:
        print("Skipping histogram (--no-plot).")
        return

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(all_weights, bins=100, edgecolor="black", alpha=0.7)
    plt.xlabel("Weight Value")
    plt.ylabel("Count")
    plt.title("Distribution of All Weights")
    plt.axvline(-1, color="r", linestyle="--", label="Binary values (-1, +1)")
    plt.axvline(1, color="r", linestyle="--")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(all_weights, bins=200, range=(-1.5, 1.5), edgecolor="black", alpha=0.7)
    plt.xlabel("Weight Value")
    plt.ylabel("Count")
    plt.title("Distribution of Weights (Zoomed to [-1.5, 1.5])")
    plt.axvline(-1, color="r", linestyle="--", label="Binary values")
    plt.axvline(1, color="r", linestyle="--")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    checkpoint_basename = os.path.splitext(os.path.basename(args.checkpoint))[0]
    output_filename = f"weight_distribution_{checkpoint_basename}.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"✓ Histogram saved to '{output_filename}'")
    print()


if __name__ == "__main__":
    main()

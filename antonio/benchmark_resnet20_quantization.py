import argparse
import copy
import json
import os
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from models.resnet_s import resnet20


@dataclass
class EvalResult:
    name: str
    accuracy: float
    avg_loss: float
    timed_images: int
    avg_batch_latency_ms: float
    avg_image_latency_ms: float
    throughput_img_s: float
    timing_note: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark ResNet20 baseline vs FP16/FP8-quantized variants on CIFAR-10."
    )
    parser.add_argument(
        "--checkpoint",
        default="/home/antonio/imt_efficient_deep_learning/antonio/experiments/quantization/best_model.pth",
        help="Path to the trained ResNet20 checkpoint (.pth).",
    )
    parser.add_argument(
        "--cifar-root",
        default="/opt/img/effdl-cifar10/",
        help="Root folder for CIFAR-10 dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/antonio/imt_efficient_deep_learning/antonio/experiments/quantization",
        help="Directory to save plot and JSON results.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for inference benchmark.")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of dataloader workers.")
    parser.add_argument(
        "--warmup-batches",
        type=int,
        default=10,
        help="Number of initial batches excluded from timing.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional limit of evaluation batches (for quick tests).",
    )
    parser.add_argument(
        "--fp8-format",
        choices=["e4m3", "e5m2"],
        default="e4m3",
        help="FP8 format for weight quantization emulation.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device.",
    )
    return parser.parse_args()


def resolve_device(device_arg):
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


def load_resnet20_from_checkpoint(checkpoint_path):
    model = resnet20()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint))
    model.eval()
    return model


def build_test_loader(cifar_root, batch_size, num_workers):
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


def clone_and_quantize_storage(model, dtype):
    quantized = copy.deepcopy(model).cpu()
    with torch.no_grad():
        for module in quantized.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if module.weight is not None and module.weight.is_floating_point():
                    module.weight.copy_(module.weight.to(dtype).to(torch.float32))
                if module.bias is not None and module.bias.is_floating_point():
                    module.bias.copy_(module.bias.to(dtype).to(torch.float32))
    quantized.eval()
    return quantized


def choose_fp8_dtype(fp8_format):
    if fp8_format == "e4m3":
        if hasattr(torch, "float8_e4m3fn"):
            return torch.float8_e4m3fn
    else:
        if hasattr(torch, "float8_e5m2"):
            return torch.float8_e5m2
    raise RuntimeError(
        "This PyTorch build does not expose the requested FP8 dtype. "
        "Use a newer PyTorch version with float8 dtype support."
    )


def evaluate_model(model, model_name, loader, device, input_dtype=torch.float32, warmup_batches=10, max_batches=None):
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    loss_sum = 0.0

    timed_batch_latencies_ms = []
    timed_images = 0

    use_cuda_events = device.type == "cuda"
    if use_cuda_events:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if input_dtype == torch.float16:
                inputs = inputs.half()

            if use_cuda_events:
                torch.cuda.synchronize()
                starter.record()
                outputs = model(inputs)
                ender.record()
                torch.cuda.synchronize()
                elapsed_ms = starter.elapsed_time(ender)
            else:
                start = time.perf_counter()
                outputs = model(inputs)
                elapsed_ms = (time.perf_counter() - start) * 1000.0

            if batch_idx >= warmup_batches:
                timed_batch_latencies_ms.append(elapsed_ms)
                timed_images += labels.size(0)

            loss = criterion(outputs, labels)
            loss_sum += loss.item() * labels.size(0)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total
    avg_loss = loss_sum / total

    total_timed_ms = sum(timed_batch_latencies_ms)
    if timed_images == 0:
        avg_batch_latency_ms = 0.0
        avg_image_latency_ms = 0.0
        throughput_img_s = 0.0
        timing_note = "No timed batches (increase --max-batches or decrease --warmup-batches)."
    else:
        avg_batch_latency_ms = total_timed_ms / len(timed_batch_latencies_ms)
        avg_image_latency_ms = total_timed_ms / timed_images
        throughput_img_s = timed_images / (total_timed_ms / 1000.0)
        timing_note = "Forward-pass latency excludes data loading and host->device transfer."

    return EvalResult(
        name=model_name,
        accuracy=accuracy,
        avg_loss=avg_loss,
        timed_images=timed_images,
        avg_batch_latency_ms=avg_batch_latency_ms,
        avg_image_latency_ms=avg_image_latency_ms,
        throughput_img_s=throughput_img_s,
        timing_note=timing_note,
    )


def save_comparison_plot(results, output_path):
    names = [r.name for r in results]
    accs = [r.accuracy for r in results]
    latencies = [r.avg_image_latency_ms for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bars1 = axes[0].bar(names, accs, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[0].set_title("CIFAR-10 Accuracy")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(min(accs) - 2.0, max(accs) + 2.0)
    for bar, val in zip(bars1, accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{val:.2f}", ha="center", va="bottom")

    bars2 = axes[1].bar(names, latencies, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    axes[1].set_title("Inference Latency")
    axes[1].set_ylabel("Latency (ms/image)")
    for bar, val in zip(bars2, latencies):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{val:.4f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_results_table(results):
    print("\n=== Quantization Benchmark Results (CIFAR-10) ===")
    header = (
        f"{'Model':<24}"
        f"{'Accuracy (%)':>14}"
        f"{'Loss':>12}"
        f"{'Latency (ms/img)':>18}"
        f"{'Throughput (img/s)':>20}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name:<24}"
            f"{r.accuracy:>14.2f}"
            f"{r.avg_loss:>12.4f}"
            f"{r.avg_image_latency_ms:>18.4f}"
            f"{r.throughput_img_s:>20.2f}"
        )
    print()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    test_loader = build_test_loader(args.cifar_root, args.batch_size, args.num_workers)
    base_model = load_resnet20_from_checkpoint(args.checkpoint)

    baseline_result = evaluate_model(
        model=copy.deepcopy(base_model),
        model_name="Baseline FP32",
        loader=test_loader,
        device=device,
        input_dtype=torch.float32,
        warmup_batches=args.warmup_batches,
        max_batches=args.max_batches,
    )

    if device.type == "cuda":
        fp16_model = copy.deepcopy(base_model).half()
        fp16_input_dtype = torch.float16
        fp16_name = "FP16 (native)"
    else:
        fp16_model = clone_and_quantize_storage(base_model, torch.float16)
        fp16_input_dtype = torch.float32
        fp16_name = "FP16 (emulated)"

    fp16_result = evaluate_model(
        model=fp16_model,
        model_name=fp16_name,
        loader=test_loader,
        device=device,
        input_dtype=fp16_input_dtype,
        warmup_batches=args.warmup_batches,
        max_batches=args.max_batches,
    )

    fp8_dtype = choose_fp8_dtype(args.fp8_format)
    fp8_model = clone_and_quantize_storage(base_model, fp8_dtype)
    fp8_result = evaluate_model(
        model=fp8_model,
        model_name=f"FP8-{args.fp8_format} (emulated)",
        loader=test_loader,
        device=device,
        input_dtype=torch.float32,
        warmup_batches=args.warmup_batches,
        max_batches=args.max_batches,
    )

    results = [baseline_result, fp16_result, fp8_result]
    print_results_table(results)

    plot_path = os.path.join(args.output_dir, "quantization_comparison.png")
    save_comparison_plot(results, plot_path)
    print(f"Saved comparison plot: {plot_path}")

    json_path = os.path.join(args.output_dir, "quantization_benchmark_results.json")
    payload = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "batch_size": args.batch_size,
        "warmup_batches": args.warmup_batches,
        "max_batches": args.max_batches,
        "fp8_format": args.fp8_format,
        "results": [r.__dict__ for r in results],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved JSON results: {json_path}")

    print("\nNotes:")
    print("- FP16 uses native half precision only on CUDA; on CPU it is storage-quantized emulation.")
    print("- FP8 result is emulated by FP8 casting of Conv/Linear parameters and float32 inference.")
    print("- This script benchmarks inference only (no retraining / QAT).")


if __name__ == "__main__":
    main()

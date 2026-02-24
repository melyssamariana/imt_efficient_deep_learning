import argparse
import glob
import json
import os
from dataclasses import asdict, dataclass
from typing import List, Tuple

import torch
import torch.nn as nn

from models.resnet_s import resnet20


@dataclass
class ScoreEntry:
    source_file: str
    experiment: str
    variant: str
    accuracy_percent: float
    p_s: float
    p_u: float
    q_w: int
    q_a: int
    w: int
    f: int
    param_term: float
    ops_term: float
    score: float


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute CIFAR-10 compression score from experiment JSON results "
            "(quantization/pruning scripts)."
        )
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "/home/antonio/imt_efficient_deep_learning/antonio/experiments/quantization/*results.json",
            "/home/antonio/imt_efficient_deep_learning/antonio/experiments/quantization/quantization_benchmark_results.json",
        ],
        help="Input JSON files or glob patterns.",
    )
    parser.add_argument(
        "--acc-threshold",
        type=float,
        default=80.0,
        help="Only keep models with accuracy >= threshold in the final ranking.",
    )
    parser.add_argument(
        "--default-qw",
        type=int,
        default=16,
        help="Default weight bit-width used for pruning scripts.",
    )
    parser.add_argument(
        "--default-qa",
        type=int,
        default=16,
        help="Default activation bit-width used for pruning scripts.",
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=3,
        default=[3, 32, 32],
        metavar=("C", "H", "W"),
        help="Model input shape for MAC count.",
    )
    parser.add_argument(
        "--w-override",
        type=int,
        default=None,
        help="Manual override for number of weights (w).",
    )
    parser.add_argument(
        "--f-override",
        type=int,
        default=None,
        help="Manual override for number of MACs (f).",
    )
    parser.add_argument(
        "--output-json",
        default="/home/antonio/imt_efficient_deep_learning/antonio/experiments/quantization/score_ranking.json",
        help="Where to save score table JSON.",
    )
    parser.add_argument(
        "--w-ref",
        type=float,
        default=5.6e6,
        help="Reference weighted parameter budget (ResNet18 half precision).",
    )
    parser.add_argument(
        "--f-ref",
        type=float,
        default=2.8e8,
        help="Reference weighted MAC budget (ResNet18 half precision).",
    )
    return parser.parse_args()


def resolve_input_files(patterns: List[str]) -> List[str]:
    files = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            files.extend(matches)
        elif os.path.isfile(pattern):
            files.append(pattern)
    unique = sorted(set(files))
    if not unique:
        raise FileNotFoundError("No input JSON files found. Check --inputs patterns.")
    return unique


def count_model_weights(model: nn.Module) -> int:
    # Uses all learnable parameters, aligned with common parameter counting.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_model_macs(model: nn.Module, input_shape: Tuple[int, int, int]) -> int:
    macs = 0
    handles = []

    def conv_hook(module: nn.Conv2d, _inputs, output):
        nonlocal macs
        out_h, out_w = output.shape[2], output.shape[3]
        kernel_ops = (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        macs += output.shape[0] * module.out_channels * out_h * out_w * kernel_ops

    def linear_hook(module: nn.Linear, _inputs, output):
        nonlocal macs
        batch = output.shape[0]
        macs += batch * module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, *input_shape)
        model(dummy)

    for h in handles:
        h.remove()
    return int(macs)


def compute_score(p_s: float, p_u: float, q_w: int, q_a: int, w: int, f: int, w_ref: float, f_ref: float):
    if p_s < 0 or p_u < 0 or p_s + p_u > 1:
        raise ValueError(f"Invalid pruning fractions: p_s={p_s}, p_u={p_u}")
    if w_ref <= 0 or f_ref <= 0:
        raise ValueError(f"Invalid references: w_ref={w_ref}, f_ref={f_ref}. Both must be > 0.")

    # Normalize by ResNet18 FP16 references from the assignment formula.
    param_term = ((1.0 - (p_s + p_u)) * (q_w / 32.0) * w) / w_ref
    ops_term = ((1.0 - p_s) * (max(q_w, q_a) / 32.0) * f) / f_ref
    return param_term, ops_term, param_term + ops_term


def bits_from_precision(eval_precision: str, default_qw: int, default_qa: int):
    if eval_precision == "fp32":
        return 32, 32
    if eval_precision == "fp16":
        return 16, 16
    return default_qw, default_qa


def bits_from_quantization_name(name: str):
    lower = name.lower()
    if "baseline fp32" in lower:
        return 32, 32
    if "fp16" in lower:
        return 16, 16
    if "fp8" in lower:
        # In your benchmark script FP8 is weight-only emulation (activations kept FP32).
        return 8, 32
    return 32, 32


def parse_quantization_benchmark(data: dict, source_file: str, w: int, f: int, w_ref: float, f_ref: float):
    entries = []
    for item in data.get("results", []):
        acc = float(item["accuracy"])
        q_w, q_a = bits_from_quantization_name(item["name"])
        p_s = 0.0
        p_u = 0.0
        param_term, ops_term, score = compute_score(p_s, p_u, q_w, q_a, w, f, w_ref, f_ref)
        entries.append(
            ScoreEntry(
                source_file=source_file,
                experiment="quantization_benchmark",
                variant=item["name"],
                accuracy_percent=acc,
                p_s=p_s,
                p_u=p_u,
                q_w=q_w,
                q_a=q_a,
                w=w,
                f=f,
                param_term=param_term,
                ops_term=ops_term,
                score=score,
            )
        )
    return entries


def parse_global_pruning(
    data: dict, source_file: str, w: int, f: int, default_qw: int, default_qa: int, w_ref: float, f_ref: float
):
    entries = []
    for item in data.get("results", []):
        acc = float(item["accuracy_percent"])
        # p_u uses measured sparsity for robustness.
        p_u = float(item["weight_sparsity_percent"]) / 100.0
        p_s = 0.0
        q_w = default_qw
        q_a = default_qa
        param_term, ops_term, score = compute_score(p_s, p_u, q_w, q_a, w, f, w_ref, f_ref)
        entries.append(
            ScoreEntry(
                source_file=source_file,
                experiment="global_pruning_no_retrain",
                variant=f"ratio={item['pruning_ratio']:.2f}",
                accuracy_percent=acc,
                p_s=p_s,
                p_u=p_u,
                q_w=q_w,
                q_a=q_a,
                w=w,
                f=f,
                param_term=param_term,
                ops_term=ops_term,
                score=score,
            )
        )
    return entries


def parse_global_pruning_retrain(
    data: dict, source_file: str, w: int, f: int, default_qw: int, default_qa: int, w_ref: float, f_ref: float
):
    entries = []
    eval_precision = data.get("eval_precision", None)
    q_w, q_a = bits_from_precision(eval_precision, default_qw, default_qa)

    for item in data.get("results", []):
        p_s = 0.0
        p_u = float(item["weight_sparsity_percent"]) / 100.0

        for phase, acc_key in [
            ("pre_retrain", "pre_retrain_accuracy_percent"),
            ("post_retrain", "post_retrain_accuracy_percent"),
        ]:
            acc = float(item[acc_key])
            param_term, ops_term, score = compute_score(p_s, p_u, q_w, q_a, w, f, w_ref, f_ref)
            entries.append(
                ScoreEntry(
                    source_file=source_file,
                    experiment="global_pruning_retrain",
                    variant=f"ratio={item['pruning_ratio']:.2f}|{phase}",
                    accuracy_percent=acc,
                    p_s=p_s,
                    p_u=p_u,
                    q_w=q_w,
                    q_a=q_a,
                    w=w,
                    f=f,
                    param_term=param_term,
                    ops_term=ops_term,
                    score=score,
                )
            )
    return entries


def parse_filter_pruning(
    data: dict, source_file: str, w: int, f: int, default_qw: int, default_qa: int, w_ref: float, f_ref: float
):
    entries = []
    eval_precision = data.get("eval_precision", None)
    q_w, q_a = bits_from_precision(eval_precision, default_qw, default_qa)

    for item in data.get("results", []):
        p_s = float(item["filter_sparsity_percent"]) / 100.0
        p_u = 0.0

        for phase, acc_key in [
            ("oneshot", "oneshot_accuracy_percent"),
            ("gradual_retrain", "gradual_accuracy_percent"),
        ]:
            acc = float(item[acc_key])
            param_term, ops_term, score = compute_score(p_s, p_u, q_w, q_a, w, f, w_ref, f_ref)
            entries.append(
                ScoreEntry(
                    source_file=source_file,
                    experiment="filter_pruning_gradual_retrain",
                    variant=f"ratio={item['pruning_ratio']:.2f}|{phase}",
                    accuracy_percent=acc,
                    p_s=p_s,
                    p_u=p_u,
                    q_w=q_w,
                    q_a=q_a,
                    w=w,
                    f=f,
                    param_term=param_term,
                    ops_term=ops_term,
                    score=score,
                )
            )
    return entries


def parse_any_result_file(path: str, w: int, f: int, default_qw: int, default_qa: int, w_ref: float, f_ref: float):
    with open(path, "r", encoding="utf-8") as fobj:
        data = json.load(fobj)

    if "results" not in data:
        return []

    results = data["results"]
    if not results:
        return []

    first = results[0]
    if isinstance(first, dict) and "accuracy" in first and "name" in first:
        return parse_quantization_benchmark(data, path, w, f, w_ref, f_ref)
    if isinstance(first, dict) and "accuracy_percent" in first and "pruning_ratio" in first:
        return parse_global_pruning(data, path, w, f, default_qw, default_qa, w_ref, f_ref)
    if isinstance(first, dict) and "pre_retrain_accuracy_percent" in first and "post_retrain_accuracy_percent" in first:
        return parse_global_pruning_retrain(data, path, w, f, default_qw, default_qa, w_ref, f_ref)
    if isinstance(first, dict) and "oneshot_accuracy_percent" in first and "gradual_accuracy_percent" in first:
        return parse_filter_pruning(data, path, w, f, default_qw, default_qa, w_ref, f_ref)
    return []


def print_table(entries: List[ScoreEntry], title: str):
    print(f"\n=== {title} ===")
    header = (
        f"{'Rank':>6}"
        f"{'Experiment':>34}"
        f"{'Variant':>26}"
        f"{'Acc (%)':>10}"
        f"{'p_s':>8}"
        f"{'p_u':>8}"
        f"{'q_w':>6}"
        f"{'q_a':>6}"
        f"{'Score':>12}"
    )
    print(header)
    print("-" * len(header))
    for idx, e in enumerate(entries, start=1):
        print(
            f"{idx:>6}"
            f"{e.experiment:>34}"
            f"{e.variant:>26}"
            f"{e.accuracy_percent:>10.2f}"
            f"{e.p_s:>8.3f}"
            f"{e.p_u:>8.3f}"
            f"{e.q_w:>6d}"
            f"{e.q_a:>6d}"
            f"{e.score:>12.5f}"
        )
    print()


def print_best_case_details(entry: ScoreEntry, w_ref: float, f_ref: float):
    raw_param = ((1.0 - (entry.p_s + entry.p_u)) * (entry.q_w / 32.0) * entry.w)
    raw_ops = ((1.0 - entry.p_s) * (max(entry.q_w, entry.q_a) / 32.0) * entry.f)

    print("\n--- Best Case Detailed Variables ---")
    print(f"source_file         = {entry.source_file}")
    print(f"experiment          = {entry.experiment}")
    print(f"variant             = {entry.variant}")
    print(f"accuracy_percent    = {entry.accuracy_percent:.4f}")
    print(f"p_s                 = {entry.p_s:.8f}")
    print(f"p_u                 = {entry.p_u:.8f}")
    print(f"q_w                 = {entry.q_w}")
    print(f"q_a                 = {entry.q_a}")
    print(f"w                   = {entry.w}")
    print(f"f                   = {entry.f}")
    print(f"w_ref               = {w_ref}")
    print(f"f_ref               = {f_ref}")
    print(f"raw_param_term      = {raw_param:.8f}")
    print(f"raw_ops_term        = {raw_ops:.8f}")
    print(f"param_term          = {entry.param_term:.8f}")
    print(f"ops_term            = {entry.ops_term:.8f}")
    print(f"score               = {entry.score:.8f}")


def main():
    args = parse_args()
    files = resolve_input_files(args.inputs)

    model = resnet20()
    w = int(args.w_override) if args.w_override is not None else count_model_weights(model)
    f = int(args.f_override) if args.f_override is not None else count_model_macs(model, tuple(args.input_shape))

    print(f"Input files ({len(files)}):")
    for path in files:
        print(f"  - {path}")
    print(f"\nUsing constants:")
    print(f"  w (weights) = {w}")
    print(f"  f (MACs)    = {f}")
    print(f"  w_ref       = {args.w_ref}")
    print(f"  f_ref       = {args.f_ref}")

    entries = []
    skipped = []
    for path in files:
        parsed = parse_any_result_file(path, w, f, args.default_qw, args.default_qa, args.w_ref, args.f_ref)
        if parsed:
            entries.extend(parsed)
        else:
            skipped.append(path)

    if skipped:
        print("\nSkipped files (unsupported format):")
        for path in skipped:
            print(f"  - {path}")

    if not entries:
        raise RuntimeError("No score entries were parsed from input files.")

    all_ranked = sorted(entries, key=lambda x: x.score)
    eligible = sorted([e for e in entries if e.accuracy_percent >= args.acc_threshold], key=lambda x: x.score)

    print_table(all_ranked, "All Candidates (sorted by score)")
    if eligible:
        print_table(eligible, f"Eligible Candidates (accuracy >= {args.acc_threshold:.2f}%)")
        best = eligible[0]
        print("Best eligible candidate:")
        print(
            f"  {best.experiment} | {best.variant} | "
            f"acc={best.accuracy_percent:.2f}% | score={best.score:.5f}"
        )
        print_best_case_details(best, args.w_ref, args.f_ref)
    else:
        print(f"\nNo candidates reached accuracy threshold {args.acc_threshold:.2f}%.")
        best = all_ranked[0]
        print("Best overall candidate (no eligible candidate):")
        print(
            f"  {best.experiment} | {best.variant} | "
            f"acc={best.accuracy_percent:.2f}% | score={best.score:.5f}"
        )
        print_best_case_details(best, args.w_ref, args.f_ref)

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    payload = {
        "acc_threshold": args.acc_threshold,
        "w": w,
        "f": f,
        "w_ref": args.w_ref,
        "f_ref": args.f_ref,
        "inputs": files,
        "skipped": skipped,
        "all_candidates": [asdict(e) for e in all_ranked],
        "eligible_candidates": [asdict(e) for e in eligible],
    }
    with open(args.output_json, "w", encoding="utf-8") as fobj:
        json.dump(payload, fobj, indent=2)
    print(f"\nSaved ranking JSON: {args.output_json}")


if __name__ == "__main__":
    main()

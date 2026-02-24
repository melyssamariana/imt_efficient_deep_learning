import copy
import json
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.utils.prune as prune
from torch import nn


ParamRef = Tuple[nn.Module, str]


class _PrunningBase:
    def __init__(self, cfgModel, ratios, params_to_prune):
        self.cfgModel = cfgModel
        self.ratios = list(ratios)
        self.params_to_prune = params_to_prune

        self.result_ratio: List[float] = []
        self.result_sparsity: List[float] = []
        self.result_acc: List[float] = []
        self.result_loss: List[float] = []

    def _reset_results(self):
        self.result_ratio = []
        self.result_sparsity = []
        self.result_acc = []
        self.result_loss = []

    def _ensure_output_dir(self):
        os.makedirs(self.cfgModel.path_backup, exist_ok=True)

    def _conv_params(self, model: nn.Module) -> List[ParamRef]:
        return [
            (m, "weight")
            for m in model.modules()
            if isinstance(m, nn.Conv2d) and m.weight is not None
        ]

    def _all_prunable_params(self, model: nn.Module) -> List[ParamRef]:
        return [
            (m, "weight")
            for m in model.modules()
            if isinstance(m, (nn.Conv2d, nn.Linear)) and m.weight is not None
        ]

    def _remove_reparam(self, params: List[ParamRef]):
        for module, name in params:
            if hasattr(module, f"{name}_orig"):
                prune.remove(module, name)

    def _evaluate_copy(self, model: nn.Module) -> Tuple[float, float]:
        original_model = self.cfgModel.model
        try:
            self.cfgModel.model = model.to(self.cfgModel.device)
            loss, acc = self.cfgModel.evaluate()
        finally:
            self.cfgModel.model = original_model
        return loss, acc

    def compute_sparsity(self, model: nn.Module) -> float:
        total = 0
        zeros = 0
        with torch.no_grad():
            for module, name in self._all_prunable_params(model):
                w = getattr(module, name)
                total += w.numel()
                zeros += (w == 0).sum().item()
        return 0.0 if total == 0 else (100.0 * zeros / total)

    def _count_total_and_zeros(self, model: nn.Module) -> Tuple[int, int]:
        total = 0
        zeros = 0
        with torch.no_grad():
            for module, name in self._all_prunable_params(model):
                w = getattr(module, name)
                total += w.numel()
                zeros += (w == 0).sum().item()
        return total, zeros

    def _add_result(self, ratio: float, sparsity: float, acc: float, loss: float):
        self.result_ratio.append(ratio)
        self.result_sparsity.append(sparsity)
        self.result_acc.append(acc)
        self.result_loss.append(loss)

    def print_results_table(self, title: str):
        print(f"\n=== {title} ===")
        header = (
            f"{'Ratio':>10}"
            f"{'Sparsity (%)':>16}"
            f"{'Accuracy (%)':>16}"
            f"{'Loss':>12}"
        )
        print(header)
        print("-" * len(header))
        for i in range(len(self.result_ratio)):
            print(
                f"{self.result_ratio[i]:>10.2f}"
                f"{self.result_sparsity[i]:>16.2f}"
                f"{self.result_acc[i]:>16.2f}"
                f"{self.result_loss[i]:>12.4f}"
            )
        print()

    def save_results(self, mode: str):
        self._ensure_output_dir()
        output_path = os.path.join(
            self.cfgModel.path_backup,
            f"pruning_results_{mode}_{self.cfgModel.label}.json",
        )
        payload = {
            "mode": mode,
            "device": str(self.cfgModel.device),
            "batch_size": self.cfgModel.batch_size,
            "ratios": self.result_ratio,
            "results": [
                {
                    "ratio": self.result_ratio[i],
                    "sparsity": self.result_sparsity[i],
                    "accuracy": self.result_acc[i],
                    "loss": self.result_loss[i],
                }
                for i in range(len(self.result_ratio))
            ],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved JSON results: {output_path}")

    def plot(self, mode: str):
        if not self.result_ratio:
            return
        self._ensure_output_dir()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        axes[0].plot(self.result_ratio, self.result_acc, marker="o", linewidth=2, color="#1f77b4")
        axes[0].set_title("Accuracy vs Ratio")
        axes[0].set_xlabel("Ratio")
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].grid(alpha=0.3)

        axes[1].plot(self.result_ratio, self.result_loss, marker="o", linewidth=2, color="#d62728")
        axes[1].set_title("Loss vs Ratio")
        axes[1].set_xlabel("Ratio")
        axes[1].set_ylabel("Cross-entropy loss")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        output_plot = os.path.join(
            self.cfgModel.path_backup,
            f"prune_plot_{mode}_{self.cfgModel.label}.png",
        )
        plt.savefig(output_plot, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot: {output_plot}")

    def _finalize(self, mode: str, title: str, last_model: Optional[nn.Module]):
        self.print_results_table(title=title)
        self.save_results(mode=mode)
        self.plot(mode=mode)

        if last_model is not None:
            self._ensure_output_dir()
            output_model = os.path.join(
                self.cfgModel.path_backup,
                f"pruned_model_{mode}_{self.cfgModel.label}.pth",
            )
            torch.save(last_model.state_dict(), output_model)
            print(f"Saved pruned model state_dict: {output_model}")


class PrunningUnstructured(_PrunningBase):
    def unstructured(self, ratios: Optional[List[float]] = None):
        run_ratios = self.ratios if ratios is None else list(ratios)
        self._reset_results()
        last_model = None

        print(f"Running unstructured pruning for ratios: {run_ratios}")
        for ratio in run_ratios:
            pruned_model = copy.deepcopy(self.cfgModel.model)

            if ratio > 0.0:
                params = self._all_prunable_params(pruned_model)
                prune.global_unstructured(
                    params,
                    pruning_method=prune.L1Unstructured,
                    amount=ratio,
                )
                self._remove_reparam(params)

            loss, acc = self._evaluate_copy(pruned_model)
            sparsity = self.compute_sparsity(pruned_model)
            self._add_result(ratio=ratio, sparsity=sparsity, acc=acc, loss=loss)
            last_model = pruned_model

            print(
                f"[unstructured ratio={ratio:.2f}] "
                f"sparsity={sparsity:.2f}% | acc={acc:.2f}% | loss={loss:.4f}"
            )

        self._finalize(
            mode="unstructured",
            title="Unstructured Pruning Results",
            last_model=last_model,
        )


class PrunningStructured(PrunningUnstructured):
    def count_weights(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def count_macs(self, model: nn.Module, input_shape=(3, 32, 32)):
        macs = 0
        handles = []

        def conv_hook(module: nn.Conv2d, _inputs, output):
            nonlocal macs
            out_h, out_w = output.shape[2], output.shape[3]
            kernel_ops = (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
            macs += output.shape[0] * module.out_channels * out_h * out_w * kernel_ops

        def linear_hook(module: nn.Linear, _inputs, _output):
            nonlocal macs
            macs += module.in_features * module.out_features

        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                handles.append(m.register_forward_hook(conv_hook))
            elif isinstance(m, nn.Linear):
                handles.append(m.register_forward_hook(linear_hook))

        model_device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            model(torch.randn(1, *input_shape, device=model_device))
        for h in handles:
            h.remove()
        return int(macs)

    def structured(self, ratios: Optional[List[float]] = None):
        run_ratios = self.ratios if ratios is None else list(ratios)
        self._reset_results()
        last_model = None

        print(f"Running structured pruning for ratios: {run_ratios}")
        for ratio in run_ratios:
            pruned_model = copy.deepcopy(self.cfgModel.model)

            if ratio > 0.0:
                conv_params = self._conv_params(pruned_model)
                for module, name in conv_params:
                    prune.ln_structured(module, name=name, amount=ratio, n=1, dim=0)
                self._remove_reparam(conv_params)

            loss, acc = self._evaluate_copy(pruned_model)
            sparsity = self.compute_sparsity(pruned_model)
            self._add_result(ratio=ratio, sparsity=sparsity, acc=acc, loss=loss)
            last_model = pruned_model

            print(
                f"[structured ratio={ratio:.2f}] "
                f"sparsity={sparsity:.2f}% | acc={acc:.2f}% | loss={loss:.4f}"
            )

        self._finalize(
            mode="structured",
            title="Structured Pruning Results",
            last_model=last_model,
        )

    def combined(
        self,
        structured_ratios: Optional[List[float]] = None,
        unstructured_ratios: Optional[List[float]] = None,
        avoid_overlap: bool = True,
    ):
        s_ratios = self.ratios if structured_ratios is None else list(structured_ratios)
        u_ratios = s_ratios if unstructured_ratios is None else list(unstructured_ratios)
        if len(s_ratios) != len(u_ratios):
            raise ValueError(
                "structured_ratios and unstructured_ratios must have same length "
                f"(got {len(s_ratios)} and {len(u_ratios)})."
            )

        self._reset_results()
        last_model = None

        print(
            "Running combined pruning (structured + unstructured). "
            f"avoid_overlap={avoid_overlap}"
        )
        for p_s, p_u in zip(s_ratios, u_ratios):
            pruned_model = copy.deepcopy(self.cfgModel.model)

            if p_s > 0.0:
                conv_params = self._conv_params(pruned_model)
                for module, name in conv_params:
                    prune.ln_structured(module, name=name, amount=p_s, n=1, dim=0)
                self._remove_reparam(conv_params)

            total_before_u, zeros_before_u = self._count_total_and_zeros(pruned_model)
            expected_new_zeros = 0

            if p_u > 0.0:
                params = self._all_prunable_params(pruned_model)
                if avoid_overlap:
                    importance_scores = {}
                    nonzero = 0
                    with torch.no_grad():
                        for module, name in params:
                            w = getattr(module, name)
                            nz = w != 0
                            nonzero += nz.sum().item()
                            scores = w.detach().abs().clone()
                            scores[~nz] = float("inf")
                            importance_scores[(module, name)] = scores

                    expected_new_zeros = min(int(nonzero * p_u), nonzero)
                    if expected_new_zeros > 0:
                        prune.global_unstructured(
                            params,
                            pruning_method=prune.L1Unstructured,
                            amount=expected_new_zeros,
                            importance_scores=importance_scores,
                        )
                        self._remove_reparam(params)
                else:
                    prune.global_unstructured(
                        params,
                        pruning_method=prune.L1Unstructured,
                        amount=p_u,
                    )
                    self._remove_reparam(params)

            total_after_u, zeros_after_u = self._count_total_and_zeros(pruned_model)
            new_zeros = zeros_after_u - zeros_before_u
            if total_before_u != total_after_u:
                raise RuntimeError("Unexpected parameter count change during pruning.")

            print(
                f"[verify p_s={p_s:.2f}, p_u={p_u:.2f}] "
                f"total={total_after_u} | zeros_before_u={zeros_before_u} | "
                f"zeros_after_u={zeros_after_u} | new_zeros={new_zeros} | "
                f"expected_new_zeros={expected_new_zeros}"
            )

            loss, acc = self._evaluate_copy(pruned_model)
            sparsity = self.compute_sparsity(pruned_model)
            effective_ratio = p_s + (1.0 - p_s) * p_u
            self._add_result(ratio=effective_ratio, sparsity=sparsity, acc=acc, loss=loss)
            last_model = pruned_model

            print(
                f"[combined p_s={p_s:.2f}, p_u={p_u:.2f}, eff={effective_ratio:.2f}] "
                f"sparsity={sparsity:.2f}% | acc={acc:.2f}% | loss={loss:.4f}"
            )

        self._finalize(
            mode="combined",
            title="Combined Structured + Unstructured Results",
            last_model=last_model,
        )

    # Backward-compatible alias used by current main.py versions.
    def structured_unstructured_no_overlap(self, unstructured_ratios=None):
        self.combined(
            structured_ratios=self.ratios,
            unstructured_ratios=unstructured_ratios,
            avoid_overlap=True,
        )

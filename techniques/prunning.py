import copy
from typing import List, Optional, Tuple

import torch
import torch.nn.utils.prune as prune
from torch import nn


ParamRef = Tuple[nn.Module, str]
ParamSpec = Tuple[str, str, bool]  # (module_name, param_name, is_conv)


class _PrunningBase:
    def __init__(self, cfgModel, ratios, params_to_prune):
        self.cfgModel = cfgModel
        self.ratios = list(ratios)
        self.params_to_prune = params_to_prune
        self._param_specs = self._build_param_specs(cfgModel.model, params_to_prune)

        self.result_ratio: List[float] = []
        self.result_sparsity: List[float] = []
        self.result_acc: List[float] = []
        self.result_loss: List[float] = []

    def _reset_results(self):
        self.result_ratio = []
        self.result_sparsity = []
        self.result_acc = []
        self.result_loss = []

    def _build_param_specs(self, model: nn.Module, params_to_prune) -> List[ParamSpec]:
        specs = []
        if not params_to_prune:
            return specs

        id_to_name = {id(module): name for name, module in model.named_modules()}
        for module, param_name in params_to_prune:
            module_name = id_to_name.get(id(module))
            if module_name is None:
                continue
            if not hasattr(module, param_name):
                continue
            if getattr(module, param_name) is None:
                continue

            specs.append((module_name, param_name, isinstance(module, nn.Conv2d)))
        return specs

    def _params_from_specs(self, model: nn.Module, conv_only: bool = False) -> List[ParamRef]:
        if not self._param_specs:
            return []

        name_to_module = dict(model.named_modules())
        params: List[ParamRef] = []
        for module_name, param_name, is_conv in self._param_specs:
            if conv_only and not is_conv:
                continue
            module = name_to_module.get(module_name)
            if module is None:
                continue
            if conv_only and not isinstance(module, nn.Conv2d):
                continue
            if not hasattr(module, param_name):
                continue
            if getattr(module, param_name) is None:
                continue
            params.append((module, param_name))
        return params

    def _conv_params(self, model: nn.Module) -> List[ParamRef]:
        params = self._params_from_specs(model, conv_only=True)
        if params:
            return params
        return [
            (m, "weight")
            for m in model.modules()
            if isinstance(m, nn.Conv2d) and m.weight is not None
        ]

    def _all_prunable_params(self, model: nn.Module) -> List[ParamRef]:
        params = self._params_from_specs(model, conv_only=False)
        if params:
            return params
        return [
            (m, "weight")
            for m in model.modules()
            if isinstance(m, (nn.Conv2d, nn.Linear)) and m.weight is not None
        ]

    def _remove_reparam(self, params: List[ParamRef]):
        for module, name in params:
            if hasattr(module, f"{name}_orig"):
                prune.remove(module, name)

    def _resolve_target_dtype(self):
        cfg_dtype = getattr(self.cfgModel, "input_dtype", None)
        if isinstance(cfg_dtype, torch.dtype):
            return cfg_dtype
        if isinstance(cfg_dtype, str):
            key = cfg_dtype.strip().lower()
            if key in {"fp16", "half", "float16", "bf16", "bfloat16"}:
                return torch.float16
            if key in {"fp32", "float", "float32", "binary", "bc"}:
                return torch.float32

        try:
            return next(self.cfgModel.model.parameters()).dtype
        except StopIteration:
            return None

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        model = model.to(self.cfgModel.device)
        target_dtype = self._resolve_target_dtype()
        if target_dtype in (torch.float16, torch.float32):
            model = model.to(dtype=target_dtype)
        return model

    def _evaluate_copy(self, model: nn.Module) -> Tuple[float, float]:
        original_model = self.cfgModel.model
        try:
            self.cfgModel.model = self._prepare_model(model)
            loss, acc = self.cfgModel.evaluate()
        finally:
            self.cfgModel.model = original_model
        return loss, acc

    def _weight_stats(self, model: nn.Module) -> Tuple[int, int]:
        total = 0
        zeros = 0
        with torch.no_grad():
            for module, name in self._all_prunable_params(model):
                w = getattr(module, name)
                total += w.numel()
                zeros += (w == 0).sum().item()
        return total, zeros

    def compute_sparsity(self, model: nn.Module) -> float:
        total, zeros = self._weight_stats(model)
        return 0.0 if total == 0 else (100.0 * zeros / total)

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

    def _finalize(self, title: str, last_model: Optional[nn.Module], set_as_current_model: bool):
        self.print_results_table(title=title)

        if last_model is not None:
            prepared = self._prepare_model(last_model)
            if set_as_current_model:
                self.cfgModel.model = prepared
            return prepared
        return None


class PrunningUnstructured(_PrunningBase):
    def unstructured(self, ratios: Optional[List[float]] = None, set_as_current_model: bool = True):
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

        return self._finalize(
            title="Unstructured Pruning Results",
            last_model=last_model,
            set_as_current_model=set_as_current_model,
        )


class PrunningStructured(PrunningUnstructured):
    def structured(self, ratios: Optional[List[float]] = None, set_as_current_model: bool = True):
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

        return self._finalize(
            title="Structured Pruning Results",
            last_model=last_model,
            set_as_current_model=set_as_current_model,
        )

    def combined(
        self,
        structured_ratios: Optional[List[float]] = None,
        unstructured_ratios: Optional[List[float]] = None,
        avoid_overlap: bool = True,
        set_as_current_model: bool = True,
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

            total_before_u, zeros_before_u = self._weight_stats(pruned_model)
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

            total_after_u, zeros_after_u = self._weight_stats(pruned_model)
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

        return self._finalize(
            title="Combined Structured + Unstructured Results",
            last_model=last_model,
            set_as_current_model=set_as_current_model,
        )

    # Backward-compatible alias used by current main.py versions.
    def structured_unstructured_no_overlap(self, unstructured_ratios=None, set_as_current_model: bool = True):
        return self.combined(
            structured_ratios=self.ratios,
            unstructured_ratios=unstructured_ratios,
            avoid_overlap=True,
            set_as_current_model=set_as_current_model,
        )

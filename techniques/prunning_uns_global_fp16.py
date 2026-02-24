import copy
import torch.nn.utils.prune as prune
import torch
from torch import nn
import utils
from utils import ConfigModel
import os
import json
import matplotlib.pyplot as plt

class Prunning:
    def __init__(self, cfgModel, pruning_percentage, ratios, params_to_prune):
        # configuration object (model, loaders, device, etc.)
        self.cfgModel = cfgModel

        self.pruning_percentage = pruning_percentage
        self.ratios = ratios
        # initial list of (module, "weight") for the base model
        self.params_to_prune = params_to_prune
        self.result_ratio = []
        self.result_sparsity = []
        self.result_acc = []
        self.result_loss = []
    
    def compute_sparsity(self, model) -> float:
        total_weights = 0
        total_zeros = 0
        with torch.no_grad():
            # Check
            for module in model.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)) and module.weight is not None:
                    w = module.weight
                    total_weights += w.numel()
                    total_zeros += (w == 0).sum().item()

        if total_weights == 0:
            return 0.0
        return 100.0 * total_zeros / total_weights
            
    def print_results_table(self):
        print("\n=== Global Pruning Results (FP16 Quantized ResNet20, no retraining) ===")
        header = (
            f"{'Pruning ratio':>14}"
            f"{'Weight sparsity (%)':>22}"
            f"{'Accuracy (%)':>16}"
            f"{'Loss':>12}"
        )
        print(header)
        print("-" * len(header))
        for r in self.result_ratio:
            idx = self.result_ratio.index(r)
            print(
                f"{r:>14.2f}"
                f"{self.result_sparsity[idx]:>22.2f}"
                f"{self.result_acc[idx]:>16.2f}"
                f"{self.result_loss[idx]:>12.4f}"
            )
        print()

    def save_results(self):
        output_dir = os.path.dirname(self.cfgModel.path_backup)
        output_path = os.path.join(
            self.cfgModel.path_backup, 
            "pruning_results_"+self.cfgModel.label+".json")
        output_dir = os.path.dirname(self.cfgModel.path_backup)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        payload = {
            "device": str(self.cfgModel.device),
            "batch_size": self.cfgModel.batch_size,
            "ratios": self.ratios,
            "results": [
                {
                    "pruning_ratio": self.result_ratio[i],
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

    def plot(self):
        ratios = self.result_ratio
        accuracies = self.result_acc
        losses = self.result_loss

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
        plt.savefig(self.cfgModel.path_backup + "/prune_plot_"+self.cfgModel.label+".png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    
    def unstructured(self):
        print(f"Pruning {self.pruning_percentage}% of the model parameters.")
        for ratio in self.ratios:
            # Copy to avoid modifying the original model, and prune from the same start point
            pruned_model = copy.deepcopy(self.cfgModel.model)
            if ratio > 0.0:
                # build pruning params for THIS copy of the model
                params_to_prune = [
                    (module, "weight")
                    for module in pruned_model.modules()
                    if isinstance(module, (nn.Conv2d, nn.Linear)) and module.weight is not None
                ]
                prune.global_unstructured(
                    params_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=ratio,
                )
                for module, name in params_to_prune:
                    prune.remove(module, name)

            # Temporarily swap the model inside cfgModel to reuse its evaluate()
            original_model = self.cfgModel.model
            try:
                self.cfgModel.model = pruned_model.to(self.cfgModel.device)
                loss, acc = self.cfgModel.evaluate()
            finally:
                self.cfgModel.model = original_model

            sparsity = self.compute_sparsity(pruned_model)
            self.result_ratio.append(ratio)
            self.result_sparsity.append(sparsity)
            self.result_acc.append(acc)
            self.result_loss.append(loss)
            
            print(
                f"[ratio={ratio:.2f}] sparsity={sparsity:.2f}% | "
                f"acc={acc:.2f}% | loss={loss:.4f}"
            )

        self.print_results_table()
        self.save_results()
        self.plot()

        # save model
        output_path = os.path.join(
            self.cfgModel.path_backup, 
            "pruned_model_"+self.cfgModel.label+".pth")
        torch.save(self.cfgModel.model.state_dict(), output_path)
        print(f"Saved pruned model state_dict: {output_path}")

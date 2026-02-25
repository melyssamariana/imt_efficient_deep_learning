### See http://papers.nips.cc/paper/5647-binaryconnect-training-deep-neural-networks-with-b
### for a complete description of the algorithm.

import torch
import torch.nn as nn


class BinaryActivationSTE(nn.Module):
    def forward(self, x):
        # Forward binario em {-1, +1}; backward aproximado por identidade (STE).
        x_bin = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
        return x + (x_bin - x).detach()


class BC(nn.Module):
    def __init__(
        self,
        model,
        binarize_activations=True,
        preserve_pruned_zeros=True,
        auto_mask_from_zeros=True,
    ):
        super().__init__()
        self.model = model
        self.binarize_activations = binarize_activations
        self.preserve_pruned_zeros = preserve_pruned_zeros
        self.auto_mask_from_zeros = auto_mask_from_zeros

        if self.binarize_activations:
            self._replace_relu_with_binary(self.model)

        self.target_modules = []
        self.num_of_params = 0
        self.saved_params = []
        self.pruning_masks = []
        self._mask_initialized = False
        self.refresh_binary_state(reset_mask_from_current_zeros=False)

    def _replace_relu_with_binary(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, name, BinaryActivationSTE())
            else:
                self._replace_relu_with_binary(child)

    def _rebuild_target_modules(self):
        self.target_modules = []
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and module.weight is not None:
                self.target_modules.append(module.weight)
        self.num_of_params = len(self.target_modules)

    def refresh_binary_state(self, reset_mask_from_current_zeros=True):
        """
        Rebuild internal BC buffers after structural weight changes (e.g., pruning remove()).
        """
        self._rebuild_target_modules()
        self.saved_params = [param.detach().clone() for param in self.target_modules]
        self.pruning_masks = [torch.ones_like(param) for param in self.target_modules]
        self._mask_initialized = False
        if reset_mask_from_current_zeros and self.preserve_pruned_zeros and self.auto_mask_from_zeros:
            self.set_pruning_masks_from_current_weights()

    def set_pruning_masks_from_current_weights(self):
        """
        Freeze current zero pattern as pruning mask:
        1 -> active connection, 0 -> pruned connection.
        """
        with torch.no_grad():
            for idx, param in enumerate(self.target_modules):
                self.pruning_masks[idx] = (param != 0).to(dtype=param.dtype, device=param.device)
        self._mask_initialized = True

    def _ensure_pruning_masks(self):
        if not self.preserve_pruned_zeros or self._mask_initialized:
            return
        if self.auto_mask_from_zeros:
            self.set_pruning_masks_from_current_weights()

    def _apply_pruning_masks(self):
        if not self.preserve_pruned_zeros or not self._mask_initialized:
            return
        with torch.no_grad():
            for idx, param in enumerate(self.target_modules):
                mask = self.pruning_masks[idx]
                if mask.shape != param.shape or mask.device != param.device or mask.dtype != param.dtype:
                    mask = mask.to(dtype=param.dtype, device=param.device)
                    self.pruning_masks[idx] = mask
                param.mul_(mask)

    def save_params(self):
        # Keep a full-precision shadow copy aligned with current device/dtype.
        with torch.no_grad():
            for index, param in enumerate(self.target_modules):
                saved = self.saved_params[index]
                if (
                    saved.shape != param.shape
                    or saved.device != param.device
                    or saved.dtype != param.dtype
                ):
                    self.saved_params[index] = param.detach().clone()
                else:
                    self.saved_params[index].copy_(param.detach())

    def binarization(self):
        # (1) Save full-precision parameters.
        self.save_params()
        self._ensure_pruning_masks()

        # (2) Replace active weights with binary values in {-1, +1}.
        with torch.no_grad():
            for idx, param in enumerate(self.target_modules):
                binary = torch.where(
                    param >= 0,
                    torch.ones_like(param),
                    -torch.ones_like(param),
                )
                if self.preserve_pruned_zeros and self._mask_initialized:
                    binary.mul_(self.pruning_masks[idx].to(dtype=param.dtype, device=param.device))
                param.copy_(binary)

    def restore(self):
        # Restore full-precision weights.
        with torch.no_grad():
            for index, param in enumerate(self.target_modules):
                param.copy_(self.saved_params[index])
        self._apply_pruning_masks()

    def clip(self):
        # Clip all parameters to the range [-1, 1] and re-apply pruning mask.
        with torch.no_grad():
            for param in self.target_modules:
                param.copy_(nn.Hardtanh()(param))
        self._apply_pruning_masks()

    def forward(self, x):
        return self.model(x)

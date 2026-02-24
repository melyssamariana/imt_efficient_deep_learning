### See http://papers.nips.cc/paper/5647-binaryconnect-training-deep-neural-networks-with-b
### for a complete description of the algorithm.

import torch
import torch.nn as nn


class BC(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        # Track all Conv2d/Linear weights to binarize.
        self.target_modules = []
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                self.target_modules.append(module.weight)

        self.num_of_params = len(self.target_modules)
        self.saved_params = [param.detach().clone() for param in self.target_modules]

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

        # (2) Replace weights with strictly binary values in {-1, +1}.
        with torch.no_grad():
            for param in self.target_modules:
                binary = torch.where(
                    param >= 0,
                    torch.ones_like(param),
                    -torch.ones_like(param),
                )
                param.copy_(binary)

    def restore(self):
        # Restore full-precision weights.
        with torch.no_grad():
            for index, param in enumerate(self.target_modules):
                param.copy_(self.saved_params[index])

    def clip(self):
        ## Clip all parameters to the range [-1,1] using Hard Tanh 
        ## you can use the nn.Hardtanh function
        with torch.no_grad():
            for param in self.target_modules:
                param.copy_(nn.Hardtanh()(param))

    def forward(self, x):
        return self.model(x)

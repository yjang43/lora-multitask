""" LoRA core module implementation """

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
class Linear(nn.Linear):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 4, 
        lora_alpha: int = 1,
        **kwargs
    ):
        super().__init__(in_features, out_features, **kwargs)

        if not r > 0:
            raise ValueError("Rank r should be greater than zero.")
        
        self.rank = r
        self.lora_a = nn.Parameter(torch.empty((r, in_features)))
        self.lora_b = nn.Parameter(torch.empty((out_features, r)))
        self.scaling = lora_alpha / r

        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.reset_lora_parameters()

    @classmethod
    def from_module(
        cls,
        module: nn.Linear,
        r: int = 4,
        lora_alpha: int = 1
    ):
        if not isinstance(module, nn.Linear):
            raise ValueError("Module should be an instance of nn.Linear.")
        in_features = module.in_features
        out_features = module.out_features
        linear = cls(in_features, out_features, r, lora_alpha)
        linear.weight = module.weight
        linear.bias = module.bias
        return linear


    def reset_lora_parameters(self):
        # Initialize A the same way as the default for nn.Linear and B to zero.
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor):

        result = F.linear(x, self.weight, bias=self.bias)
        result += (x @ self.lora_a.transpose(0, 1) @ self.lora_b.transpose(0, 1)) * self.scaling
        return result

"""LoRA finetuning util"""

from typing import (
    Dict,
    List,
    Tuple,
)

import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import Linear

REQUIRED_KEYS = (
    "model_name_or_path",
    "prompt_format",
    "lora_parameters",
    "lora_scaling",
)

def initialize_finetuning(
        model: transformers.AutoModelForCausalLM,
        module_names: List[str],
        r: int = 4
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """TODO
    """

    lora_parameter, lora_scaling = {}, {}
    for module_name in module_names:
        module = model.get_submodule(module_name)

        # Initialize LoRA.
        if isinstance(module, nn.Linear):
            lora_module = Linear.from_module(module, r)
        else:
            # NOTE: Add as more types are supported.
            raise NotImplementedError("Only supported modules are nn.Linear.")

        module_name_split = module_name.split(".")
        parent_module_name, attr = ".".join(module_name_split[:-1]), module_name_split[-1]
        parent_module = model.get_submodule(parent_module_name)
        setattr(parent_module, attr, lora_module)

        lora_scaling[module_name] = lora_module.scaling
        lora_parameter[module_name + ".lora_a"] = lora_module.lora_a
        lora_parameter[module_name + ".lora_b"] = lora_module.lora_b
    
    # Set requires_grad to False to reduce memory usage in computation graph.
    for parameter_name, parameter in model.named_parameters():
        if ".lora_" not in parameter_name:
            parameter.requires_grad = False

    return lora_parameter, lora_scaling

def save_finetuning_checkpoint(checkpoint: Dict, chekcpoint_path: str, device="cpu"):
    """TODO
    """

    _validate_checkpoint(checkpoint)
    for key in checkpoint:
        if isinstance(checkpoint[key], torch.Tensor):
            checkpoint[key].to(device)
    torch.save(checkpoint, chekcpoint_path)

def load_finetuning_checkpoint(checkpoint_path: str, device="cpu") -> Dict:
    """TODO
    """

    checkpoint = torch.load(checkpoint_path, map_location=device)
    _validate_checkpoint(checkpoint)
    return checkpoint

def _validate_checkpoint(checkpoint):
    if set(checkpoint.keys()).issuperset(set(REQUIRED_KEYS)):
        raise ValueError("checkpoint does not have required keys within.")

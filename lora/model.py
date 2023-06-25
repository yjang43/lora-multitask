"""Model with dynamic LoRA loading"""

from typing import Dict

import json
import functools
import contextlib

import transformers
import torch
import torch.nn as nn


class MultiTaskModel(nn.Module):
    """Multi-task model that injects LoRA dynamically. The model is initialized 
    with pretrained model. When the function, generate_response, is called, 
    LoRA checkpoint for the specified downstream task will be applied and generate 
    the response accordingly. The checkpoint mapper that maps a downstream task 
    to a checkpoint path is required. In the format of the following:

    ```
    # downstream_mapper.json
    {
        "question_answering": "path_to_the_checkpoint",
        "summary": "path_to_the_checkpoint",
        ...
    }
    ```

    Then, for example, a user can generate a summary of an article by calling
    a function `article_summary = generate_response("summary", article)`.
    To reduce the IO bottleneck, LoRA checkpoints are cached once called,
    so generating a response for the same downstream task will be faster.
    """

    def __init__(
            self,
            model_name_or_path: str,
            checkpoint_mapper_path: str,
            device: str = "cpu"
        ):
        """
        Args:
            model_name_or_path: Name or path of huggingface pretrained model.
            checkpoint_mapper_path: Path to a file that maps downstream to checkpoint.
            device: A device to load tensor (cuda, mps, cpu)
        """

        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.prompt_format = ""
        self.device = device

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        self.pretrained_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path)
        self.pretrained_model.to(device)
        self.generation_config = self._load_generation_config()

        self._activate_downstream = functools.partial(self._apply_lora, mode="add")
        self._deactivate_downstream = functools.partial(self._apply_lora, mode="subtract")
        self._checkpoint_mapper = self._load_checkpoint_mapper(checkpoint_mapper_path)

    def generate_response(
            self, 
            downstream: str,
            instruction: str = "",
            input: str = "",
            **generation_kwargs
        ) -> str:
        """Generate response suitable for the downstrea task specified.

        Args:
            downstream: Downstream task ID.
            instruction: Instruction given to the model.
            input: Input given to the model.
            **generation_kwargs: Extra options for generation config.
        Returns:
            generated_response: Response generated.
        """

        if downstream not in self._checkpoint_mapper:
            raise ValueError("The downstream task is not registered in config file.")
        checkpoint_path = self._checkpoint_mapper[downstream]
        downstream_checkpoint = self._load_downstream_checkpoint(checkpoint_path)

        with self.apply_downstream(downstream_checkpoint) as downstream_model:
            prompt = self.prompt_format.format_map({"instruction": instruction, "input": input})
            input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
            outputs = downstream_model.generate(
                input_ids=input_ids.to(self.device),
                generation_config=self.generation_config,
                **generation_kwargs
            )
            input_length = input_ids.shape[1]
            generated_tokens = outputs.sequences[0, input_length:]
            generated_response = self.tokenizer.decode(
                generated_tokens,
                # skip_special_tokens=True
            )
            
        return generated_response

    @contextlib.contextmanager
    def apply_downstream(self, downstream_checkpoint: Dict) -> transformers.AutoModelForCausalLM:
        """Apply checkpoint LoRA on pretrained model weight.

        Args:
            downstream_checkoint: 
                Downstream checkpoint containing lora_parameter, lora_scaling, and prompt_format.
        Yields:
            downstream_model: LoRA applied model.
        """

        lora_parameters = downstream_checkpoint["lora_parameters"]
        lora_scalings = downstream_checkpoint["lora_scalings"]
        self.prompt_format = downstream_checkpoint["prompt_format"]
        self._activate_downstream(lora_parameters, lora_scalings)
        
        try:
            downstream_model = self.pretrained_model
            yield downstream_model
        finally:
            self._deactivate_downstream(lora_parameters, lora_scalings)
            self.prompt_format = ""
  
    def _load_generation_config(self):
        generation_config = transformers.GenerationConfig(
            temperature=0.5,
            top_p=0.75,
            num_bemas=4,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            max_new_tokens=128,
        )
        return generation_config

    def _load_checkpoint_mapper(self, checkpoint_mapper_path):
        with open(checkpoint_mapper_path) as f:
            checkpoint_mapper = json.load(f)
        return checkpoint_mapper
    
    def _apply_lora(self, lora_parameters, lora_scalings, mode):
        if mode == "add":
            sign = 1
        elif mode == "subtract":
            sign = -1
        else:
            raise ValueError("mode should be either activate or deactivate")
        
        for module_name in lora_scalings:
            module = self.pretrained_model.get_submodule(module_name)
            lora_a = lora_parameters[f"{module_name}.lora_a"]
            lora_b = lora_parameters[f"{module_name}.lora_b"]
            lora_scaling = lora_scalings[module_name]

            # Decode weight offset and apply to pretrained weight.
            with torch.no_grad():
                if isinstance(module, nn.Linear):
                    module.weight += sign * lora_b @ lora_a * lora_scaling
                else:
                    # NOTE: Add as more types are supported.
                    raise NotImplementedError("Only supported modules are nn.Linear.")
    
    def _validate_checkpoint(self, checkpoint):
        if self.pretrained_model.name_or_path != checkpoint["model_name_or_path"]:
            raise ValueError("model_name_or_path of the checkpoint does not match pretrained model.")
        
    def _checkpoint_cache_warning(self, parameter_dict):
        size_byte = 0
        for parameter in parameter_dict.values():
            size_byte += parameter.nelement() * parameter.element_size()
        size_mb = size_byte / 1024**2
        if size_mb > 100:
            print(
                "Warning: LoRA weight is bigger than 100MB."
                "You may need to reduce checkpoint cache size")

    @functools.lru_cache(maxsize=16)
    def _load_downstream_checkpoint(self, checkpoint_path):
        downstream_checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self._validate_checkpoint(downstream_checkpoint)
        self._checkpoint_cache_warning(downstream_checkpoint["lora_parameters"])

        return downstream_checkpoint

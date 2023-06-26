import os
import argparse
import random

import tqdm
import torch
import transformers
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dutil

import dataset as ft_dataset
import lora
import lora.train_util

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)




# TODO: DEBUG LOAD FROM CHECKPOINT FEATURE!


def load_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--target_modules_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--prompt_format_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--validation_data_path",
        type=str,
    )
    parser.add_argument(
        "--load_from_checkpoint",
        default="",
        type=str
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoint",
        type=str
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int
    )
    parser.add_argument(
        "--lora_rank",
        default=4,
        type=int
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float
    )
    parser.add_argument(
        "--num_epochs",
        default=1,
        type=int
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str
    )
    parser.add_argument(
        # This argument is important as applying LoRA on pretrained model
        # for inference should be as fast as possible.
        "--checkpoint_device",
        default="cuda",
        type=str
    )
    parser.add_argument(
        "--evaluate",
        default=False,
        type=bool,
        action=argparse.BooleanOptionalAction
    )
    args = parser.parse_args()
    return args
    
def _validate_target_modules(target_module_names, model: nn.Module):
    for module_name in target_module_names:
        module = model.get_submodule(module_name)
        if not any([
            isinstance(module, available_module) 
            for available_module in lora.AVAILABLE_MODULES
        ]):
            raise ValueError(f"{type(module)} is not available for LoRA.")

def _load_dataloader(data_path, tokenizer, prompt_format, batch_size, collate_fn, shuffle):
    dataset = ft_dataset.SupervisedDataset(data_path, tokenizer, prompt_format)
    dataloader = dutil.DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
    return dataloader

def _compute_gradient_norm(parameter_dict):
    total_gradient_norm = 0
    for _, parameter in parameter_dict.items():
        gradient_norm = parameter.grad.data.norm(2)
        total_gradient_norm += gradient_norm ** 2
    total_gradient_norm = total_gradient_norm ** .5
    return total_gradient_norm
        
def _validate_prompt_format(prompt):
    # TODO: Validate if {instruction} and (optional) {input} is in the format.
    pass

def _load_prompt_format(prompt_path):
    with open(prompt_path, "r") as f:
        prompt_format = f.read()
    _validate_prompt_format(prompt_format)
    return prompt_format

def _load_target_modules(target_modules_path, model):
    with open(target_modules_path, "r") as f:
        target_modules: list = f.read().split("\n")
        target_modules = [module for module in target_modules if module.strip("\n\t ")]
    _validate_target_modules(target_modules, model)
    return target_modules



# TODO: evaluate implementation

def _warning_different_args(old_args, new_args):
    # TODO: Throw warning if arg from checkpoint is different.
    pass

def _count_num_parameters(parameters):
    count = 0
    for parameter in parameters:
        count += parameter.nelement()
    return count


def finetune(args, evaluate=False):
    if args.load_from_checkpoint:
        last_checkpoint = lora.train_util.load_finetuning_checkpoint(
            args.load_from_checkpoint, device=args.device)
        _warning_different_args(last_checkpoint["args"], args)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, local_files_only=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path, local_files_only=True)
    ft_dataset.check_for_special_tokens(tokenizer, model)

    prompt_format = _load_prompt_format(args.prompt_format_path)
    collator = ft_dataset.DataCollatorForSupervisedDataset(tokenizer)
    train_dataloader = _load_dataloader(
        args.train_data_path, tokenizer, prompt_format,
        batch_size=args.batch_size, collate_fn=collator, shuffle=True)
    if args.evaluate:
        validation_dataloader = _load_dataloader(
            args.validation_data_path, tokenizer, prompt_format,
            batch_size=args.batch_size, collate_fn=collator, shuffle=False)

    target_modules = _load_target_modules(args.target_modules_path, model)
    
    if args.load_from_checkpoint:
        lora_parameters, lora_scalings = lora.train_util.initialize_finetuning(
            model, target_modules, r=args.lora_rank
        )
        model.load_state_dict(last_checkpoint["lora_parameters"], strict=False)
        # assert all(torch.equal(
        #     lora_parameters[k],last_checkpoint["lora_parameters"][k]) for k in lora_parameters)
    else:
        lora_parameters, lora_scalings = lora.train_util.initialize_finetuning(
            model, target_modules, r=args.lora_rank
        )

    # Print model parameter.
    pretrained_model_parameter_count = _count_num_parameters(model.parameters())
    lora_parameter_count = _count_num_parameters(lora_parameters.values())
    print(
        f"The number of trainable parameter, {lora_parameter_count // 1e3}K "
        f"is {lora_parameter_count / pretrained_model_parameter_count:.3f}th of pretrained model.")


    optimizer = optim.AdamW(
        params=[lora_parameters[name] for name in lora_parameters],
        lr=args.learning_rate
    )
    if args.load_from_checkpoint:
        optimizer.load_state_dict(last_checkpoint["optimizer_state_dict"])
        # Ad-hoc patch. There should be a better way...
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(args.device)

    # Train starts
    model.to(args.device)
    model.train()
    loss_history = last_checkpoint["loss_history"] if args.load_from_checkpoint else []

    start_epoch = last_checkpoint["epoch"] if args.load_from_checkpoint else 0
    prog_bar = tqdm.tqdm(range((args.num_epochs-start_epoch) * len(train_dataloader)))
    for epoch in range(start_epoch, args.num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            # Forward and backprop
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()

            gradient_norm = _compute_gradient_norm(lora_parameters)

            loss_history.append(loss.item())
            prog_bar.set_description(
                f"epoch: {epoch} | loss: {loss.item(): .3f} | gradient: {gradient_norm: .3f}")
            prog_bar.update()
        
        if args.evaluate:
            # TODO: Evaluation code here
            pass

        if args.checkpoint_dir:
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            checkpoint = {
                # Required keys
                "model_name_or_path": args.model_name_or_path,
                "prompt_format": prompt_format,
                "lora_parameters": lora_parameters,
                "lora_scalings": lora_scalings,

                # For continue training
                "args": args,
                "epoch": epoch + 1,
                "loss_history": loss_history,
                "optimizer_state_dict": optimizer.state_dict(),

            }
            lora.train_util.save_finetuning_checkpoint(
                checkpoint,
                os.path.join(args.checkpoint_dir, f"{epoch}.pt"),
                device=args.checkpoint_device
            )


if __name__ == "__main__":
    args = load_arguments()
    set_seed(args.seed)

    finetune(args, evaluate=True)

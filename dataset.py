""" Dataset module for finetuning.
Much of the code is modified from stanford alpaca.
"""

import os
import copy
import json
from typing import Dict, List

import torch
import transformers
from torch.utils.data import Dataset

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def check_for_special_tokens(
        tokenizer: transformers.AutoTokenizer,
        model: transformers.AutoModelForCausalLM
):
    # NOTE: Realized special tokens trained during finetune
    # NOTE: will not be maintained when applying LoRA as these special tokens
    # NOTE: are by default missing in the pretrained model

    # special_tokens_dict = {}
    # if tokenizer.pad_token is None:
    #     special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    # if tokenizer.eos_token is None:
    #     special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    # if tokenizer.bos_token is None:
    #     special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    # if tokenizer.unk_token is None:
    #     special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # if special_tokens_dict:
    #     # Print warning.
    #     print(
    #         f"Tokenizer missing {', '.join(special_tokens_dict.keys())}."
    #         f"Adding these to tokenizer and resizing model embedding.")
    #     tokenizer.add_special_tokens(special_tokens_dict)
    #     model.resize_token_embeddings(len(tokenizer))

    if tokenizer.eos_token is None:
        raise ValueError("eos_token should be set for the pretrained model.")
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if special_tokens_dict:
        # Print warning.
        print(
            f"Tokenizer missing {', '.join(special_tokens_dict.keys())}. "
            f"Adding these to tokenizer and resizing model embedding.")
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
    

def _tokenize_fn(strings, tokenizer: transformers.PreTrainedTokenizer):
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            # If model_max_length was never set, value is too big.
            max_length=min(tokenizer.model_max_length, 4096),
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        # Ignore pad tokens when counting input length.
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: List[str],
    targets: List[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict[str, torch.Tensor]:
    """Preprocess the data by tokenizing.
    TODO
    """

    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    # Mask input part of label
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, prompt: str):
        """TODO
        """

        super().__init__()
        
        if ".json" != os.path.splitext(data_path)[1]:
            raise ValueError("data_path should be JSON file.")
        with open(data_path, "r") as file:
            list_data_dict = json.load(file)

        sources = [
            prompt.format_map(example) for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer):
        """TODO
        """

        self.tokenizer = tokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )



# lora-pytorch
Simple implementation of LoRA and mutitask-like model with fast checkpoint swapping.

## Overview
LoRA is a parameter efficient finetuning technique.
Some types of moudles of a pretrained moel can be specified to inject LoRA.
Then, finetuning with LoRA has some benfits.

* Reduce memory footprint.
* No change in inference time.
* Small checkpoint for each downstream task.
* Less catastrophic forgetting.
* Somewhat better generalization (personal thought).
* Somewhat faster finetuning process (but seems to converge slower).
* Provides in-depth perspective of understanding model weight.

The fact that the number of parameters for each checkopint is so much smaller to the pretrained model 
allows saving multiple checkpoints in a single resource.
Additionally, swapping from a checkpoint of one downstream task to that of another downstream task can be done fast as it is modifiy very very small portion of pretrained weight.
Thus, it is possible to mock multitask model with multiple checkpoints.

For example, assume there are checkpoints for question answering, summarization, and so on.
The base pretrained model can be loaded and then dynamically load either question answering or summarization downstream checkpoints 
to generate response accordingly.


## Install
```bash
git clone https://github.com/yjang43/lora-pytorch.git
pip install -r requirements.txt
```

## Finetune
Prior to start finetuning, train datset, prompt format, and target modules should be specified.

Train dataset is in the same format introduced in Stanford Alpaca.
List of dictionaries with three keys: `instruction`, `input`, `output`.

Prompt format should contain `{instruction}` and `{input}` each of which will be replaced with train data. For example, prompt could look like the following.
```
# promopt_format.txt
Article:
{instruction}

Summarize of the article: 

```

Target modules are names of modules each of which will be injected with LoRA.
Get the name of modules to inject LoRA with `model.named_moduels()`.

Once requirements for finetuning is set, start finetuning for each downstream task.
```bash
python finetune.py \
    --model_name_or_path "../huggingface_model/gpt-neo-125m/" \
    --target_modules_path "downstream_example/target_modules.txt" \
    --prompt_format_path "downstream_example/prompt_format.txt" \
    --train_data_path "downstream_example/data.json" \      # downstream task 1
    --checkpoint_dir "downstream_example/checkpoint" \
    --num_epochs "1"
```

## Multitask model with LoRA
Once finetuning for multiple downstream tasks is finished, it is ready to create mutitask-like model instance.
Create downstream task to checkpoint path mapper before instantiating the model.
Then, multitask model can be used like the following.

```python
import lora

model_path = "../huggingface_model/polyglot-ko-1.3b/"
checkpoint_mapper_path = "downstream_checkpoint_mapper.json"
multitask_model = lora.MultiTaskModel(
    model_path, checkpoint_mapper_path, device="cuda")


# Question answering downstream task
response = multitask_model.generate_response(
    downstream="question_answering", instruction="...")

# Summarization downstream task
response = multitask_model.generate_response(
    downstream="summarization", instruction="...", input="...")
```

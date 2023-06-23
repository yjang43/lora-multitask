python finetune.py \
    --model_name_or_path "../huggingface_model/gpt-neo-125m/" \
    --target_modules_path "downstream_example/target_modules.txt" \
    --prompt_format_path "downstream_example/prompt_format.txt" \
    --train_data_path "downstream_example/data.json" \
    --checkpoint_dir "downstream_example/checkpoint" \
    --num_epochs "1"
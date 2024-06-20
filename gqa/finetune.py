import copy
import json
import os
import sys
from typing import List

import fire
import torch
import transformers
import wandb
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from datasets import load_dataset

wandb.init(mode="disabled")


def train(
        # model/data params
        base_model: str = "",  # the only required argument
        data_path: str = "./alpaca_data_cleaned.json",
        output_dir: str = "./lora-alpaca",
        # training hyperparams
        batch_size: int = 128,
        micro_batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 3e-4,
        cutoff_len: int = 512,
        val_set_size: int = 2000,
        # lora hyperparams
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
        ],
        # llm hyperparams
        train_on_inputs: bool = True,  # if False, masks out inputs in loss
        group_by_length: bool = True,  # faster, but produces an odd training loss curve
        # other
        mask: bool = False,
):
    print(
        f"Training Alpaca-LoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"data_path: {data_path}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"val_set_size: {val_set_size}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"lora_target_modules: {lora_target_modules}\n"
        f"train_on_inputs: {train_on_inputs}\n"
        f"group_by_length: {group_by_length}\n"
    )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=True,
        device_map=device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference

    def encode_with_prompt_completion_format(example):
        '''
        Here we assume each example has 'prompt' and 'completion' fields.
        We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated
        and it doesn't make sense to follow directly with the completion.
        '''
        # Don't add fucking space
        example_text = example['prompt'] + example['completion'] + tokenizer.eos_token
        tokenized_example = tokenizer(example_text, max_length=cutoff_len, truncation=True)
        input_ids = tokenized_example.input_ids
        labels = copy.deepcopy(input_ids)
        attention_mask = tokenized_example.attention_mask

        # mask the prompt part for avoiding loss
        tokenized_prompt = tokenizer(example['prompt'], max_length=cutoff_len, truncation=True)
        input_length = len(tokenized_prompt.input_ids)
        assert labels[:input_length] == tokenized_prompt.input_ids, \
            "Sanity check failed at " + json.dumps(example)
        labels[:input_length] = [-100, ] * input_length

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    data = load_dataset("json", data_files=data_path)

    if val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(encode_with_prompt_completion_format)
        val_data = train_val["test"].shuffle().map(encode_with_prompt_completion_format)
    else:
        train_data = data["train"].shuffle().map(encode_with_prompt_completion_format)
        val_data = None

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if val_set_size > 0 else None,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()

    model.save_pretrained(output_dir)

    print("\n If there's a warning about missing keys above, please disregard :)")


if __name__ == "__main__":
    fire.Fire(train)

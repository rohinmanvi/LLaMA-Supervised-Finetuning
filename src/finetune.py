# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
from dataclasses import dataclass, field
import datasets
from typing import Optional, Dict, Sequence

from peft import (
    LoraConfig,
    prepare_model_for_int8_training,
    get_peft_model,
)

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer

import io
import json
import typing
import pathlib


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


@dataclass
class LoraArguments:
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: typing.List[str] = ["q_proj", "v_proj"],
    lora_weight_path: str = ""


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    (model_args, data_args, training_args,
     lora_args) = parser.parse_args_into_dataclasses()

    model = transformers.LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        load_in_8bit=True, 
        torch_dtype=torch.float16, 
        device_map={'':0}
    )
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenizer = transformers.LlamaTokenizer.from_pretrained(model_args.model_name_or_path, add_eos_token=True)
    tokenizer.pad_token_id = 0
    
    text = []

    with open(data_args.data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            text.append(data)

    data = datasets.Dataset.from_list(text)
    data = data.shuffle().map(lambda x: tokenizer(x['text'], padding='max_length', max_length=training_args.model_max_length, truncation=True))

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data,
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()

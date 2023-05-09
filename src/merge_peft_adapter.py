from dataclasses import dataclass, field
from typing import Optional

import peft
import torch
from peft import PeftConfig, PeftModel
import transformer

base_model_name = "decapoda-research/llama-7b-hf"
model_name = "models/highway-driver-final-2"

model = transformers.LlamaForCausalLM.from_pretrained(base_model_name, load_in_8bit=True, torch_dtype=torch.float16, device_map={'':0})
model = peft.PeftModel.from_pretrained(model, model_name, torch_dtype=torch.float16)
tokenizer = transformers.LlamaTokenizer.from_pretrained(base_model_name)

key_list = [key for key, _ in model.base_model.model.named_modules() if "lora" not in key]
for key in key_list:
    parent, target, target_name = model.base_model._get_submodules(key)
    if isinstance(target, peft.tuners.lora.Linear):
        bias = target.bias is not None
        new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
        model.base_model._replace_module(parent, target_name, new_module, target)

model = model.base_model.model

model.save_pretrained("models/merged-highway-driver-final-2")
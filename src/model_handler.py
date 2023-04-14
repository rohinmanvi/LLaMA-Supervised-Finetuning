import os
import gc
import random
import torch
import transformers
import peft
import datasets

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import respond_to_batch


class ModelHandler:
    def __init__(self, base_model_name):
        self.base_model_name = base_model_name
        self.model = None
        self.current_peft_model = None
        
        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(self.base_model_name)

        self.load_base_model()

    def load_base_model(self):
        print('Loading base model...')
        self.model = transformers.LlamaForCausalLM.from_pretrained(self.base_model_name, load_in_8bit=True, torch_dtype=torch.float16, device_map={'':0})

    def load_peft_model(self, model_name):
        print('Loading peft model ' + model_name + '...')
        self.model = peft.PeftModel.from_pretrained(self.model, model_name, torch_dtype=torch.float16)

    def reset_model(self):
        del self.model
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
        self.model = None
        self.current_peft_model = None

    def generate_text(self, peft_model, text, generation_config):
        if peft_model == 'None':
            peft_model = None

        if self.current_peft_model != peft_model:
            if self.current_peft_model is not None or self.model is not None:
                self.reset_model()

            self.load_base_model()
            self.current_peft_model = peft_model
            if peft_model is not None:
                self.load_peft_model(peft_model)

        if self.model is None:
            self.load_base_model()

        inputs = self.tokenizer(text, return_tensors="pt")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )

        response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        return response
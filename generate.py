from model_handler import ModelHandler

model_handler = ModelHandler("decapoda-research/llama-7b-hf")

def generate_prompt(instruction: str, input_ctxt: str = None) -> str:
    if input_ctxt:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_ctxt}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""

instruction = "What is the meaning of life?"
input_ctxt = None

prompt = generate_prompt(instruction, input_ctxt)

response = model_handler.generate_text(
    peft_model='None',
    text=prompt,
    temperature=0.1,
    top_p=0.75,
    top_k=50,
    max_new_tokens=128
)

print(response)
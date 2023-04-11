from model_handler import ModelHandler

model_handler = ModelHandler("decapoda-research/llama-7b-hf")

prompt = "\"Our vehicle is going 5.8 m/s with a steering angle of 7.9\u00b0 to the left. The other vehicle is 14.9 m away and is 1.0\u00b0 to the left. It is going 7.0 m/s with a direction of 34.4\u00b0 to the left.\" ->"

response = model_handler.generate_text(
    peft_model='llama-driver',
    text=prompt,
    temperature=0.1,
    top_p=0.75,
    top_k=50,
    max_new_tokens=32
)

print(response)
from transformers import AutoModelForCausalLM, AutoTokenizer

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("distillbert/distilgpt2", torch_dtype="auto", device_map="auto")

    print("[DEBUG] Model loaded")
    print(model)

    print("-" * 64)

    tokenizer = AutoTokenizer.from_pretrained("distillbert/distilgpt2")

    print("[DEBUG] Tokenizer loaded")
    print(tokenizer)

    print("-" * 64)

    prompt = "The secret to baking a good cake is "
    model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    generated_outputs = model.generate(**model_inputs, max_new_tokens=100)
    decoded_outputs = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)

    print("[DEBUG] Generated outputs")
    print(f"prompt : {prompt}")
    print(f"model_inputs : {model_inputs}")
    print(f"generated_outputs : {generated_outputs}")
    print(f"generated_outputs[0] : {generated_outputs[0]}")
    print(f"decoded_outputs : {decoded_outputs}")
    print(f"decoded_outputs[0] : {decoded_outputs[0]}")

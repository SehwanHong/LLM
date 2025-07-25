from transformers import AutoModelForMaskedLM, AutoTokenizer
from datasets import load_dataset
from transformers import pipeline
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import Trainer

import torch
import math

if __name__ == "__main__": 
    model = AutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")
    eli5 = load_dataset("eli5_category", split="train[:5000]")

    eli5 = eli5.train_test_split(test_size=0.2)
    
    print(f"[DEBUG] ELI5-Category Dataset load")

    print(f'{eli5["train"][0]}')

    print("-" * 64)

    def preprocess_function(examples): 
        return tokenizer([" ".join(x) for x in examples["answers.text"]])

    eli5 = eli5.flatten()

    print(f"[DEBUG] ELI-5 Flatten")

    print(f"eli5 flatten = {eli5}")

    print("-" * 64)
    
    tokenized_eli5 = eli5.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=eli5["train"].column_names,
    )  

    print(f"[DEBUG] Tokenized ELI5-Category Dataset load")

    print(f'{tokenized_eli5["train"][0]}')

    print("-" * 64)

    block_size = 128

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    lm_dataset = tokenized_eli5.map(
        group_texts,
        batched=True,
        num_proc=4,
    )

    print(f"[DEBUG] Grouped ELI5-Category Dataset load")

    print(f'{lm_dataset["train"][0]}')

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    print(f"[DEBUG] DataCollatorForLanguageModeling load")

    print(f'{data_collator([lm_dataset["train"][i] for i in range(2)])}')

    print("-" * 64)

    training_args = TrainingArguments(
        output_dir="distilbert-eli5-category",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        push_to_hub=False,
        logging_dir="logs",
        logging_steps=10,
        save_steps=10,
        save_total_limit=2,
        local_rank=-1,
    )

    print("[DEBUG] Training arguments")
    print(training_args)

    print("-" * 64)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    print("[DEBUG] Training completed")
    eval_results = trainer.evaluate()

    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    print("-" * 64)

    print("[DEBUG] Fill-Mask by pipeline")
    text = "The Milky Way is a <mask> galaxy."
    generator = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    generated_result = generator(text, top_k=5)

    for result in generated_result:
        print(result)

    print("-" * 64)

    print("[DEBUG] Text generation step by step")
    inputs = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    
    print(inputs)

    mask_token_index = torch.where(inputs == tokenizer.mask_token_id)[1]
    print(f"[DEBUG] Mask token index: {mask_token_index}")

    logits = model(inputs).logits
    mask_token_logits = logits[0, mask_token_index, :]
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

    for token in top_5_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))
    
    print("-" * 64)

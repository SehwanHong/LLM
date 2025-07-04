from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from transformers import pipeline
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import Trainer

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import math

if __name__ == "__main__": 
    model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
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
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print(f"[DEBUG] DataCollatorForLanguageModeling load")

    print(f'{data_collator([lm_dataset["train"][i] for i in range(2)])}')

    print("-" * 64)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    train_dataloader = DataLoader(lm_dataset["train"], shuffle=True, collate_fn=data_collator, batch_size=8)
    eval_dataloader = DataLoader(lm_dataset["test"], collate_fn=data_collator, batch_size=8)

    for epoch in range(10):
        print(f"[DEBUG] Training {epoch} epoch")

        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch:02d} Train ")
        for batch in progress_bar:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        model.eval()
        
        with torch.no_grad():
            total_eval_loss = 0

            progress_bar = tqdm(eval_dataloader, desc=f"Epoch {epoch:02d} Eval ")
            for batch in eval_dataloader:
                batch = {k: v.to(model.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_eval_loss += loss.item()
            
            avg_eval_loss = total_eval_loss / len(eval_dataloader)
            print(f"[EPOCH {epoch:03i}] Average Evaluation Loss: {avg_eval_loss}")

    print("-" * 64)

    print("[DEBUG] Text generation by pipeline")
    prompt = "Somatic hypermutation allows the immune system to"
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    print(generator(prompt, max_new_tokens=50))

    print("-" * 64)

    print("[DEBUG] Text generation step by step")
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    print(f"[DEBUG] Inputs: {inputs}")

    outputs = model.generate(inputs, max_new_tokens=50)
    print(f"[DEBUG] Outputs: {outputs}")

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    print("-" * 64)

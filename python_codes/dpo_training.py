from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig

import torch

from datasets import load_dataset

from trl import DPOTrainer, DPOConfig, DPODataCollatorWithPadding

def return_prompt_and_responses(samples):
    return {
        "prompt": [
            question
            for question in samples["question"]
        ],
        "chosen": samples["chosen"],   # chosen response
        "rejected": samples["rejected"], # rejected response
    }

if __name__ == "__main__":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "distilbert/distilgpt2",
        quantization_config=bnb_config,
    )

    print("[DEBUG] Model loaded")
    print(model)

    print("-" * 64)

    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[DEBUG] Tokenizer loaded")
    print(tokenizer)

    print("-" * 64)

    dataset = load_dataset("Intel/orca_dpo_pairs")

    print("[DEBUG] Dataset loaded")
    print(dataset)

    print("[DEBUG] Check dataset")
    print(dataset["train"][0])

    print("-" * 64)

    peft_config = LoraConfig(
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = dataset.map(return_prompt_and_responses, batched=True)

    print("[DEBUG] Check dataset after map")
    print(dataset["train"][0])

    print("-" * 64)

    model_ref = AutoModelForCausalLM.from_pretrained(
        "distilbert/distilgpt2",
        quantization_config=bnb_config,
    )

    dpo_config = DPOConfig(
        output_dir="dpo_training_intel_orca",
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        max_prompt_length=1024,
        max_length=1024,
        batch_size=16,
        beta=0.1,
        learning_rate=1e-4,
        num_train_epochs=1,
        data_collator=DPODataCollatorWithPadding(
            tokenizer=tokenizer,
        ),
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=10,
        report_to="wandb",
        wandb_project="dpo_training_intel_orca",
        wandb_name="dpo_training_intel_orca",
    )

    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=dpo_config,
        peft_config=peft_config,
    )

    dpo_trainer.train()

    print("[DEBUG] Check model after training")
    print(model)

    print("-" * 64)
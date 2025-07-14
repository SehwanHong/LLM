from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig

import torch

from datasets import load_dataset

from trl import DPOTrainer
from qwen_vl_utils import process_vision_info

def return_prompt_and_responses(samples) -> Dict[str, str, str]:
    return {
        "prompt": [
            "Question: " + question + "\n\nAnswer: "
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
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
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
        ref_model=model_ref,
        beta=0.1,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )

    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=dpo_config,
        beta=0.1,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    dpo_trainer.train()

    print("[DEBUG] Check model after training")
    print(model)

    print("-" * 64)
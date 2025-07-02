from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from transformers import DataCollatorWithPadding
from transformers import TrainingArguments
from transformers import Trainer

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2", torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    eli5 = load_dataset("eli5_category", split="train[:5000]", trust_remote_code=True)

    eli5 = eli5.train_test_split(test_size=0.2)
    
    print(f"[DEBUG] ELI5-Category Dataset load")

    print(f'{eli5["train"][0]}')

    print("-" * 64)


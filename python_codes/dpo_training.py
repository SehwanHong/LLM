from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig

from datasets import load_dataset

from trl import SFTConfig, SFTTrainer
from transformers import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2", torch_dtype="auto", device_map="auto")

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

    model_inputs = tokenizer(dataset["train"]["prompt"], return_tensors="pt").to("cuda")

    print("[DEBUG] Model inputs")
    print(model_inputs)

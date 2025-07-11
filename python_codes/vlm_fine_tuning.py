from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from datasets import load_dataset
from transformers import pipeline
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import Trainer
from peft import LoraConfig

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset

from trl import SFTConfig, SFTTrainer
from transformers import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info


PROMPT= """Create a Short Product description based on the provided ##PRODUCT NAME## and ##CATEGORY## and image. 
Only return description. The description should be SEO optimized and for a better mobile search experience.
 
##PRODUCT NAME##: {product_name}
##CATEGORY##: {category}"""


SYSTEM_MESSAGE = "You are an expert product description writer for Amazon."

# Create a data collator to encode text and image pairs
def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    image_inputs = [process_vision_info(example["messages"])[0] for example in examples]
 
    # Tokenize the texts and process the images
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
 
    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100  #
    # Ignore the image token index in the loss computation (model specific)
    if isinstance(processor, Qwen2VLProcessor):
        image_tokens = [151652,151653,151655]
    else: 
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100
    batch["labels"] = labels
 
    return batch

# Convert dataset to OAI messages       
def format_data(sample):
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT.format(product_name=sample["Product Name"], category=sample["Category"]),
                    },{
                        "type": "image",
                        "image": sample["image"],
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["description"]}],
            },
        ],
    }

if __name__ == "__main__": 
    # Load dataset from the hub
    dataset_id = "philschmid/amazon-product-descriptions-vlm"
    dataset = load_dataset("philschmid/amazon-product-descriptions-vlm", split="train")
    
    # Convert dataset to OAI messages
    # need to use list comprehension to keep Pil.Image type, .mape convert image to bytes
    dataset = [format_data(sample) for sample in dataset]
    
    print(f"[DEBUG] Dataset loaded")
    print(f"Sample data : {dataset[345]['messages']}")

    print("-" * 64)


    # Hugging Face model id
    model_id = "Qwen/Qwen2-VL-7B-Instruct" 
    
    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load model and tokenizer
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        # attn_implementation="flash_attention_2", # not supported for training
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Preparation for inference
    text = processor.apply_chat_template(
        dataset[2]["messages"], tokenize=False, add_generation_prompt=False
    )

    print("[DEBUG] Check apply chat template")
    print(f"text : {text}")

    print("-" * 64)

 
    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=8,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM", 
    )
    
    print("[DEBUG] PEFT Config")
    print(f"PEFT Config : {peft_config}")

    print("-" * 64)

    args = SFTConfig(
        output_dir="qwen2-7b-instruct-amazon-description", # directory to save and repository id
        num_train_epochs=3,                     # number of training epochs
        per_device_train_batch_size=4,          # batch size per device during training
        gradient_accumulation_steps=8,          # number of steps before performing a backward/update pass
        gradient_checkpointing=True,            # use gradient checkpointing to save memory
        optim="adamw_torch_fused",              # use fused adamw optimizer
        logging_steps=5,                       # log every 10 steps
        save_strategy="epoch",                  # save checkpoint every epoch
        learning_rate=2e-4,                     # learning rate, based on QLoRA paper
        bf16=True,                              # use bfloat16 precision
        tf32=True,                              # use tf32 precision
        max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",           # use constant learning rate scheduler
        push_to_hub=False,                       # push model to hub
        gradient_checkpointing_kwargs = {"use_reentrant": False}, # use reentrant checkpointing
        dataset_text_field="", # need a dummy field for collator
        dataset_kwargs = {"skip_prepare_dataset": True} # important for collator
    )
    args.remove_unused_columns=False

    print("[DEBUG] SFTConfig")
    print(f"SFTConfig : {args}")

    print("-" * 64)
    
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
    )

    print("[DEBUG] SFTTrainer")
    print(f"SFTTrainer : {trainer}")

    print("-" * 64)

    trainer.train()

    trainer.save_model(args.output_dir)

    print(f"[DEBUG] Model saved to {args.output_dir}")

    print("-" * 64)

    print("[DEBUG] Check model")
    print(f"Model : {model}")

    # free the memory again
    del model
    del trainer
    torch.cuda.empty_cache()

    sample = {
        "product_name": "Hasbro Marvel Avengers-Serie Marvel Assemble Titan-Held, Iron Man, 30,5 cm Actionfigur",
        "catergory": "Toys & Games | Toy Figures & Playsets | Action Figures",
        "image": "https://m.media-amazon.com/images/I/81+7Up7IWyL._AC_SY300_SX300_.jpg"
    }
    
    # prepare message
    messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image"],
                },
                {"type": "text", "text": PROMPT.format(product_name=sample["product_name"], category=sample["catergory"])},
            ],
        }
    ]

    print("[DEBUG] Check messages")
    print(f"Messages : {messages}")

    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        # attn_implementation="flash_attention_2", # not supported for training
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )

    print("-" * 64)
    def generate_description(sample, model, processor):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
            {"role": "user", "content": [
                {"type": "image","image": sample["image"]},
                {"type": "text", "text": PROMPT.format(product_name=sample["product_name"], category=sample["catergory"])},
            ]},
        ]
        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=256, top_p=1.0, do_sample=True, temperature=0.8)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]
    
    print("[DEBUG] Check generate description")
    base_description = generate_description(sample, model, processor)
    
    print(f"Base Generated description : {base_description}")

    print("-" * 64)

    fine_tuned_model = AutoModelForVision2Seq.from_pretrained(
        args.output_dir,
        # attn_implementation="flash_attention_2", # not supported for training
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )

    print("[DEBUG] Check fine-tuned model") 

    fine_tuned_description = generate_description(sample, fine_tuned_model, processor)

    print(f"Fine-tuned Generated description : {fine_tuned_description}")

    print("-" * 64)

    

---
title: "VLM Fine Tuning with Qwen2-VL using Amazon Product Description Dataset"
date: 2025-07-11
layout: default   # 또는 page, default 등 테마에 따라
tags: [vlm, fine-tuning, llm, transformers, qwen2-vl,  huggingface, transformers-tutorial, ]
---

## Introduction

이번 포스트에서는 Qwen2-VL을 활용하여 데이터셋을 활용하여 훈련을 하는 방법을 알아보자. 참고자료[1]를 기반으로 훈련을 해본것이다.

## 환경 준비

```bash
pip install --upgrade "torch==2.4.0" torchvision tensorboard pillow

pip install  --upgrade "transformers==4.45.1" "datasets==3.0.1" "accelerate==0.34.2" "evaluate==0.4.3" "bitsandbytes==0.44.0" "trl==0.11.1" "peft==0.13.0" "qwen-vl-utils"
```

일단 코드를 실행하기 전에 환경을 맞춰 주어야 한다. 일단은 참고자료에 있는 코드를 기반으로 설정을 했으니 그 상황에 맞춰서 설정을 해주자.


## 모델 준비

```python
# Hugging Face model id
model_id = "Qwen/Qwen2-VL-7B-Instruct" 

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    device_map="auto",
    # attn_implementation="flash_attention_2", # not supported for training
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
processor = AutoProcessor.from_pretrained(model_id)
```

일단 모델을 준비해 보자. 기본적으로 Vision2Seq 모델을 사용하고 있는 것이고, 이때 어떤 방식으로 양자화를 할까에 대한 설정을 BitsAndBytes 라이브러리를 사용해서 설정을 해준다.

이렇게 사용하면 모델을 불러 올수 있다.

## 데이터 준비

```python
# Load dataset from the hub
dataset_id = "philschmid/amazon-product-descriptions-vlm"
dataset = load_dataset("philschmid/amazon-product-descriptions-vlm", split="train")

# Convert dataset to OAI messages
# need to use list comprehension to keep Pil.Image type, .mape convert image to bytes
dataset = [format_data(sample) for sample in dataset]

print(f"[DEBUG] Dataset loaded")
print(f"Sample data : {dataset[345]['messages']}")

print("-" * 64)
```

데이터셋을 준비해보자. 기존의 HuggingFace 튜토리얼에서 사용했던 것과 동일하게 테이터셋을 불러온다. 다만 우리가 원하는 상황에 맞게 데이터셋을 변경해 준다.

```python
PROMPT= """Create a Short Product description based on the provided ##PRODUCT NAME## and ##CATEGORY## and image. 
Only return description. The description should be SEO optimized and for a better mobile search experience.
 
##PRODUCT NAME##: {product_name}
##CATEGORY##: {category}"""

SYSTEM_MESSAGE = "You are an expert product description writer for Amazon."

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
```

우리가 여기서 원하는 것은 이미지를 사용해서 아마존의 상품 설명을 생성하려고 하는 것이다. 그렇기 때문에 입력 형식을 ChatBot 형식으로 변경시킨다. 사용자가 입력으로 넣을 때 이미지와 택스트를 넣으면 그에 대한 설명을 생성하는 것으로 변경을 해준다.

```text
[DEBUG] Check apply chat template
text : <|im_start|>system
You are an expert product description writer for Amazon.<|im_end|>
<|im_start|>user
Create a Short Product description based on the provided ##PRODUCT NAME## and ##CATEGORY## and image.
Only return description. The description should be SEO optimized and for a better mobile search experience.

##PRODUCT NAME##: Barbie Fashionistas Doll Wear Your Heart
##CATEGORY##: Toys & Games | Dolls & Accessories | Dolls<|vision_start|><|image_pad|><|vision_end|><|im_end|>
|im_start|>assistant
Express your style with Barbie Fashionistas Doll Wear Your Heart! This fashionable doll boasts a unique outfit and accessories, perfect for imaginative play.  A great gift for kids aged 3+.  Collect them all! #Barbie #Fashionistas #Doll #Toys #GirlsToys #FashionDoll #Play<|im_end|>
```

이렇게 하면 데이터는 준비가 되었다.

## 훈련 준비

```python
# LoRA config based on QLoRA paper & Sebastian Raschka experiment
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM", 
)
```

모델 전체를 새롭게 훈련시키기에는 너무나도 많은 연산량이 필요하기 때문에 LoRA를 사용하여 훈련을 해보자. 

```txt
[DEBUG] PEFT Config
PEFT Config : LoraConfig(peft_type=<PeftType.LORA: 'LORA'>, auto_mapping=None, base_model_name_or_path=None, revision=None, task_type='CAUSAL_LM', inference_mode=False, r=8, target_modules={'v_proj', 'q_proj'}, lora_alpha=16, lora_dropout=0.05, fan_in_fan_out=False, bias='none', use_rslora=False, modules_to_save=None, init_lora_weights=True, layers_to_transform=None, layers_pattern=None, rank_pattern={}, alpha_pattern={}, megatron_config=None, megatron_core='megatron.core', loftq_config={}, use_dora=False, layer_replication=None, runtime_config=LoraRuntimeConfig(ephemeral_gpu_offload=False))
```

## 훈련 시작

```python
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
    report_to="tensorboard",                # report metrics to tensorboard
    gradient_checkpointing_kwargs = {"use_reentrant": False}, # use reentrant checkpointing
    dataset_text_field="", # need a dummy field for collator
    dataset_kwargs = {"skip_prepare_dataset": True} # important for collator
)
args.remove_unused_columns=False

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=collate_fn,
    peft_config=peft_config,
)
```

이전에 사용했던 Trainer를 사용하는 것이 아니라 SFTTrainer를 사용해서 훈련하면 된다. SFTTrainer는 기존의 Trainer를 기반으로 만들어진 클레스로 Fine-Tuning을 위해서 HuggingFace에서 만들어준 Class이다.

## 훈련 결과

```python
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
```

이제 훈련된 모델을 사용해서 생성을 해보자. 

```txt
[DEBUG] Check generate description
Base Generated description : Experience the thrilling world of Marvel with the Hasbro Marvel Avengers Series Marvel Assemble Titan-Held, Iron Man Action Figure. This 30.5 cm tall figure is a must-have for any fan of the iconic Iron Man. Crafted with detailed design and premium materials, this action figure is perfect for play and display. Bring the power of Iron Man to life with this impressive figure, ready to take on any adventure. Perfect for collectors and kids alike, this action figure is a great addition to any toy collection.
----------------------------------------------------------------
[DEBUG] Check fine-tuned model
Fine-tuned Generated description : Unleash the power of Iron Man with this Hasbro Marvel Avengers Titan Hero Series Iron Man action figure!  At 30.5 cm tall, this highly detailed Iron Man figure is perfect for epic battles and imaginative play.  Collect all the Avengers for ultimate showdowns!
```

기존의 모델과 훈련된 모델을 사용하여 생성을 해서 비교를 해보면, 훈련된 모델이 깔끔한 형식으로 설명을 해주고 있는 것을 보여준다. 기본모델도 괜찮은 설명을 해주고 있지만, 훈련이 된 모델이 상품설명에 조금은 더 적합한 설명을 해주고 있는 것을 볼 수 있다.

## 참고자료

[1] [https://www.philschmid.de/fine-tune-multimodal-llms-with-trl](https://www.philschmid.de/fine-tune-multimodal-llms-with-trl)

[실행 코드 전체](https://github.com/sehwanhong/LLM/blob/main/python_codes/vlm_fine_tuning.py)
---
title: "Hugging Face Transformers Tutorial - 04"
date: 2025-07-03
layout: default   # 또는 page, default 등 테마에 따라
tags: [transformers, huggingface, transformers-tutorial, llm]
---

## 1. Introduction

이전에는 Causal Language Model (CLM) 방식을 훈련하는 방식을 알아보았다. 이번에는 Masked Language Model (MLM) 방식을 훈련하는 방식을 알아보자.

이전에도 설명을 했지만 MLM 방식에 대하여 알아보자. MLM 방식은 토큰을 마스킹하고, 마스킹된 토큰을 예측하는 방식이다.

```txt
"Fly me to the moon, and let me [MASK] among the stars"
```

위 같은 문장이 있을 때, 저 부분에 비어 있는 `[MASK]`{:.txt} 부분에 대하여 어떤 단어가 들어가는지 예측하는 모델을 의미한다.

## 2. MLM을 훈련해 보자

CLM과 MLM이 달라지는 부분은 DataCollator를 통해서 데이터를 처리하는 방식만 달라진다. CLM은 다음 단어를 추축을 하는 반면 MLM은 마스킹 된 토큰을 예측하는 것임으로 DataCollator에서 토큰에 마스킹을 추가해 줘야 한다.

```python
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
```

위에서 `mlm_probability`{:.python}는 마스킹된 토큰의 비율을 의미한다. 즉 15%의 토큰을 마스킹하고 예측하는 방식이다. 

이제 훈련을 해보자. 이번에는 모델을 불러올 때 `AutoModelForMaskedLM`{:.python}을 사용하면 된다.

```python
model = AutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base", torch_dtype="auto")

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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
```

이렇게 되면 훈련이 자동으로 완료가 된다. 완료가 된 정보들을 두가지 방식으로 생성을 할 수 있다.

하나는 아래와 같은 방식으로 pipeline을 통해서 생성을 할 수 있다.
```python
text = "The Milky Way is a <mask> galaxy."
generator = pipeline("fill-mask", model=model, tokenizer=tokenizer)

generated_result = generator(text, top_k=5)
```

그리고 다른 방식으로는 직접 모델에 넣어서 생성을 할 수 있다. 

```python
inputs = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
mask_token_index = torch.where(inputs == tokenizer.mask_token_id)[1]

logits = model(inputs).logits
mask_token_logits = logits[0, mask_token_index, :]
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
```

그리고 다른 방식으로는 직접 모델에 넣어서 생성을 할 수 있다. CLM과 다른 부분은 `fill-mask`{:.python}를 사용한다는 것이고, 입력을 할때 비어있는 부분에 `<mask>`{:.python}를 넣어줘야 한다. 그리고 출력을 할때는 `top_k`{:.python}를 사용하여 상위 k개의 토큰을 출력한다.

```txt
----------------------------------------------------------------
[DEBUG] Fill-Mask by pipeline
{'score': 0.4018346965312958, 'token': 21300, 'token_str': ' spiral', 'sequence': 'The Milky Way is a spiral galaxy.'}
{'score': 0.1144062951207161, 'token': 2232, 'token_str': ' massive', 'sequence': 'The Milky Way is a massive galaxy.'}
{'score': 0.05110162869095802, 'token': 3065, 'token_str': ' giant', 'sequence': 'The Milky Way is a giant galaxy.'}
{'score': 0.0350143164396286, 'token': 30794, 'token_str': ' dwarf', 'sequence': 'The Milky Way is a dwarf galaxy.'}
{'score': 0.03226202353835106, 'token': 1123, 'token_str': ' gas', 'sequence': 'The Milky Way is a gas galaxy.'}
----------------------------------------------------------------
[DEBUG] Text generation step by step
tensor([[    0,   133, 36713,  4846,    16,    10, 50264, 22703,     4,     2]],
       device='cuda:0')
[DEBUG] Mask token index: tensor([6], device='cuda:0')
The Milky Way is a  spiral galaxy.
The Milky Way is a  massive galaxy.
The Milky Way is a  giant galaxy.
The Milky Way is a  dwarf galaxy.
The Milky Way is a  gas galaxy.
```

이 둘의 결과를 확인해 보면 위에서 보는 것과 같은 결과를 얻을 수 있다.

[실행 코드 전체](https://github.com/sehwanhong/LLM/blob/main/python_codes/transformer_04.py)
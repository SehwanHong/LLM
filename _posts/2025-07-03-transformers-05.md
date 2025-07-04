---
title: "Hugging Face Transformers Tutorial - 05"
date: 2025-07-04
layout: default   # 또는 page, default 등 테마에 따라
tags: [transformers, huggingface, transformers-tutorial, llm]
---

## 1. Introduction

이전에는 HuggingFace가 제공해 주는 방식으로 훈련을 해보았다. 하지만, 규모가 큰 방식으로 훈련을 하거나, 모델의 구조를 바꾸는 등, 더 복잡한 작업을 하는 경우에는 직접적으로 훈련을 하는 것이 더 중요하다. 이번 포스트에서는 직접적으로 훈련을 하는 방식을 알아보자.

## 2. pytorch를 사용하여 훈련하기

훈련을 할때 기존의 포스트와 비슷하게 훈련을 할 수 있다. 일단은 모델이나, 데이터셋을 직접적으로 변경하는 부분은 차후의 포스트에서 다루도록 하고, 이번에는 훈련을 하는 과정의 기본적인 부분에 대하여 알아보자.

일단 기존의 코드를 보면 다음과 같은 형식이다.

```python
model = AutoModelForMaskedLM.from_pretrained( ... 생략 ...)

training_args = TrainingArguments(
    ... 생략 ...
)

trainer = Trainer(
    ... 생략 ...
)

trainer.train()
```

HuggingFace에서 제공하는 훈련 방식은 위와 같이 매우 간단하다. 하지만, 더 높은 자유도를 위해서 pytorch를 사용하여 훈련을 하는 방식을 알아보자.

```python
train_dataloader = DataLoader(lm_dataset["train"], shuffle=True, collate_fn=data_collator, batch_size=8)
eval_dataloader = DataLoader(lm_dataset["test"], collate_fn=data_collator, batch_size=8)

for epoch in range(10):
    # 1. Train Loop
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 2. Evaluation Loop
    model.eval()
    total_eval_loss = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
```

위의 코드는 전체적인 훈련과정을 보여주고 있다. 일단은 천천히 하나씩 관련된 부분을 살펴보면서 그 부분에 대하여 알아보자.

### 2.1 DataLoader 사용하기

가장 먼저 데이터셋을 불러오는 부분을 살펴보자.

```python
train_dataloader = DataLoader(lm_dataset["train"], shuffle=True, collate_fn=data_collator, batch_size=8)
eval_dataloader = DataLoader(lm_dataset["test"], collate_fn=data_collator, batch_size=8)
```

여기서 중점적으로 볼 부분은, Trainer 객채가 사용되지 않기 때문에, 데이터셋을 불러오는 부분을 직접 처리해줘야한다. 데이터 셋을 불러오는 부분에 대하여 dataset에서 Train과 Eval에 사용할 데이터를 분리하고, pytorch에서 사용하는 `DataLoader`{:.python}를 사용하여 데이터셋을 불러오는 방식을 사용하고 있다. pytorch의 `DataLoader`{:.python}에는 `collate_fn`{:.python}에 `data_collator`{:.python}를 사용하여 데이터를 정리하는 함수를 사용하고 있다.

### 2.2 훈련 하기

```python
for epoch in range(10):
    # 1. Train Loop
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

기본적으로 pytroch의 훈련 방식을 따라간다. 기존의 포스트에서는 Trainer 객체가 모든 것들을 처리해줬고 어떤 방식으로 Epoch을 가지고 가는지 등의 정보들을 모델에게 알려줬지만, 이번에는 우리가 직접적으로 처리를 해줘야 한다.

그리고 여기서 Loss와 관련되어, Model에서 직접적으로 처리를 해줬지만, Model이 변경되고, 더 복잡한 모델을 사용하는 경우에는 함수를 직접적으로 처리해줘야 하는데, 차후의 포스트에서 다루어 보겠다.

[실행 코드 전체](https://github.com/sehwanhong/LLM/blob/main/python_codes/transformer_05.py)
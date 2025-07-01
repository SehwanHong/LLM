---
title: "Hugging Face Transformers Tutorial - 02"
date: 2025-07-01
layout: default   # 또는 page, default 등 테마에 따라
tags: [transformers, huggingface, transformers-tutorial]
---

## 1. Introduction

기본적인 훈련을 해보자. 재대로된 훈련을 하기에는, 혹은 연구를 하면서 훈련을 하기에는 기본적인 pytorch나 pytorch lightning을 사용하는 것이 좋다. 하지만 간단하게 훈련을 돌려보기에는 huggingface에서 제공하는 것을 사용하는 것이 간단하고 편리하다.

## 2. 데이터셋 불러오기

```python
from datasets import load_dataset

dataset = load_dataset("rotten_tomatoes")
```

Huggingface에서 제공하는 데이터셋을 활용하기 위해서는 datasets 라이브러리를 사용하면 된다. 위의 코드는 Rotten Tomatoes 데이터셋을 불러오는 코드이다. 이를 실행하면 다음과 같은 결과가 나온다.

```txt
DatasetDict({
    train: Dataset({
        features: ['text', 'label'],
        num_rows: 8530
    })
    validation: Dataset({
        features: ['text', 'label'],
        num_rows: 1066
    })
    test: Dataset({
        features: ['text', 'label'],
        num_rows: 1066
    })
})
```

이를 통해서 데이터셋을 확인할 수 있다. 이 데이터셋은 영화 리뷰와 그 리뷰의 레이블로 구성되어 있다. 레이블은 0과 1로 구성되어 있는데, 0은 부정적인 리뷰, 1은 긍정적인 리뷰를 의미한다.

## 3. 데이터셋 토큰화

모든 데이터들은 기본적인 text 형식으로 되어 있다. 그렇기 때문에 모델에 직접적으로 넣어주기 위해서는 tokenizer를 통해서 토큰화 되어야 한다.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

def tokenize_dataset(dataset):
    return tokenizer(dataset["text"], padding=True, truncation=True)
dataset = dataset.map(tokenize_dataset, batched=True)
```

dataset의 map 함수는 데이터셋의 각 행에 대해서 함수를 적용하는 함수이다. 각 dataset은 dictionary로 되어 있어, dictionary에서 dictionary로 만들어주는 함수가 있으면 이 함수를 각각의 행에 적용을 해주게 된다. 위의 코드는 데이터셋의 각 행에 대해서 tokenizer를 적용하는 코드이다. 이를 실행하면 다음과 같은 결과가 나온다.

```txt
[DEBUG] Tokenized dataset
DatasetDict({
    train: Dataset({
        features: ['text', 'label', 'input_ids', 'attention_mask'],
        num_rows: 8530
    })
    validation: Dataset({
        features: ['text', 'label', 'input_ids', 'attention_mask'],
        num_rows: 1066
    })
    test: Dataset({
        features: ['text', 'label', 'input_ids', 'attention_mask'],
        num_rows: 1066
    })
})
```

이를 통해서 데이터셋을 토큰화 할 수 있다.

## 4. 데이터 콜레이터 설정

데이터셋을 토큰화 하면 각 행에 대해서 토큰화된 값이 들어가게 된다. 하지만 이를 모델에 넣어주기 위해서는 각 행의 길이가 동일해야 한다. 이를 위해서는 데이터 콜레이터를 사용하면 된다.

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

```txt
DataCollatorWithPadding(tokenizer=DistilBertTokenizerFast(name_or_path='distilbert/dist    ilbert-base-uncased', vocab_size=30522, model_max_length=512, is_fast=True, padding_sid    e='right', truncation_side='right', special_tokens={'unk_token': '[UNK]', 'sep_token':     '[SEP]', 'pad_token': '[PAD]', 'cls_token': '[CLS]', 'mask_token': '[MASK]'}, clean_up_    tokenization_spaces=False, added_tokens_decoder={
        0: AddedToken("[PAD]", rstrip=False, lstrip=False, single_word=False, normalize    d=False, special=True),
        100: AddedToken("[UNK]", rstrip=False, lstrip=False, single_word=False, normali    zed=False, special=True),
        101: AddedToken("[CLS]", rstrip=False, lstrip=False, single_word=False, normali    zed=False, special=True),
        102: AddedToken("[SEP]", rstrip=False, lstrip=False, single_word=False, normali    zed=False, special=True),
        103: AddedToken("[MASK]", rstrip=False, lstrip=False, single_word=False, normal    ized=False, special=True),
}
), padding=True, max_length=None, pad_to_multiple_of=None, return_tensors='pt')
```

이를 통해서 데이터셋을 모델에 넣어주기 위해서 각 행의 길이가 동일해지게 된다.

## 5. 훈련 설정

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="distilbert-rotten-tomatoes",
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
```

TrainingArguments는 훈련 과정에서 사용되는 다양한 설정을 담고 있는 클래스이다. 위의 코드는 훈련 과정에서 사용되는 다양한 설정을 담고 있는 클래스이다. 이를 실행하면 다음과 같은 결과가 나온다.

```txt
[DEBUG] Training arguments
TrainingArguments(
_n_gpu=1,
accelerator_config={'split_batches': False, 'dispatch_batches': None, 'even_batches': T    rue, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs    ': None, 'use_configured_state': False},
adafactor=False,
adam_beta1=0.9,
adam_beta2=0.999,
adam_epsilon=1e-08,
... 중략 ...
)
```

## 6. 훈련 실행

```python
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
```

Trainer가 기본적인 훈련을 실행시켜준다.


[실행 코드 전체](https://github.com/sehwanhong/LLM/blob/main/python_codes/transformer_02.py)
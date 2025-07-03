---
title: "Hugging Face Transformers Tutorial - 03"
date: 2025-07-02
layout: default   # 또는 page, default 등 테마에 따라
tags: [transformers, huggingface, transformers-tutorial, llm]
---

## 1. Introduction

언어 모델을 훈련하는 방식은 크게 두 가지 방식이 있다. 하나는 Causal Language Model (CLM) 방식이고, 다른 하나는 Masked Language Model (MLM) 방식이다. 일단 이 두가지 방식을 간단하게 알아보자. 이 두가지 방식은 모델의 구조가 다르다.

### 1.1 Causal Language Model (CLM)

Causal Language Model (CLM) 방식은 문장을 넣어주면 그 다음 단어를 예측하는 방식이다. 예를 들어 문장을 넣어주면 그 다음 단어를 예측하는 방식이다.

```txt
"Fly me to the moon, and let me play among the "
```
라는 문장을 넣어주면 다음과 같은 단어를 예측하는 방식이다.

```txt
"Fly me to the moon, and let me play among the stars"
```

기본적으로 Causal Language Model (CLM) 모델은 Decoder-only 모델이다. 즉 transformer의 decoder 부분만 사용하는 모델이다.

### 1.2 Masked Language Model (MLM)

Masked Language Model (MLM) 방식은 토큰을 마스킹하고, 마스킹된 토큰을 예측하는 방식이다.

```txt
"Fly me to the moon, and let me [MASK] among the stars"
```

라는 문장을 넣어주면 다음과 같은 단어를 예측하는 방식이다.

```txt
"Fly me to the moon, and let me play among the stars"
```

기본적으로 Masked Language Model (MLM) 모델은 Encoder-only 모델이다. 즉 transformer의 encoder 부분만 사용하는 모델이다.

## 2. CLM을 훈련해 보자

이번에는 CLM을 훈련하는 방식을 알아보자. 먼저 데이터셋을 불러오자.

```python
from datasets import load_dataset

eli5 = load_dataset("eli5_category", split="train[:5000]")
```

이 함수는 train 데이터셋 에서 5000개의 데이터를 불러온다. 이때 데이터를 확인해 보면 다음과 같다.

```txt
[DEBUG] ELI5-Category Dataset load
{'q_id': '5nhe1w', 'title': 'How do we know than an earthquake is "long over due"', 'se    lftext': '', 'category': 'Physics', 'subreddit': 'explainlikeimfive', 'answers': {'a_id    ': ['dcbja9x'], 'text': ['I\'m not an expert at all, but this is what I\'ve learned from high-school geography- when two tectonic plates, like the north american plate and the pacific plate, slide against each other at a transform fault line, they don\'t do it     really really slowly, like plates which move apart or towards each other. The plates (or at least at the fault line) slowly build up force, until suddenly the plates can\'t take it anymore and they snap past each other, tens of metres at a time. A good metaphor is this- take chunk of polystyrene and break it in half. Put the halves back togehter, and push one forwards and one backwards, so they\'re sliding against each other. You\'ll notice it won\'t slide easily and instead it takes a certain amount of force to get each "plate" moving against the other, and when it does move, it moves in one big leap. Geologists can look at the data they have (how long it\'s been since the last earthquake, how quickly each plate is moving) and predict when the tectonic plates will release all of that built up force. Naturally they can\'t be 100% certain, so they\'d make an estimate, and when that estimate is over, they would call the earthquake "long over due". Edit: rip it got removed. Edit 2: yay'], 'score': [4], 'text_urls': [[]]}, 'title_urls': ['url'], 'selftext_urls': ['url']}
```

이렇게 얻은 데이터를 정리하는 함수를 만들어서 정리를 해보자. 가장 먼저 nested되어 있는 정보들을 정리하는 것이 먼저이다.

```python
eli5 = eli5.flatten()
```

이렇게 정리된 데이터를 확인해 보면 다음과 같다.

```txt
[DEBUG] ELI-5 Flatten
eli5 flatten = DatasetDict({
    train: Dataset({
        features: ['q_id', 'title', 'selftext', 'category', 'subreddit', 'answers.a_id', 'answers.text', 'answers.score', 'answers.text_urls', 'title_urls', 'selftext_urls'],
        num_rows: 4000
    })
    test: Dataset({
        features: ['q_id', 'title', 'selftext', 'category', 'subreddit', 'answers.a_id', 'answers.text', 'answers.score', 'answers.text_urls', 'title_urls', 'selftext_urls'],
        num_rows: 1000
    })
})
```

원래는 `eli5['train']['answers']['text']`{:.python} 와 같은 형태였지만, 이렇게 정리하면 `eli5['train']['answers.text']`{:.python} 와 같은 형태로 정리된다. 

이제는 정리된 정보들을 모두 tokenizer를 통해서 token화 해야한다. 이는 모델이 정보를 처리할때 string 형태로 처리하는 것이 아니라 실수 형태의 tensor를 통해서 정리를 해야한다는 것이다.

```python
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
def preprocess_function(examples): 
    return tokenizer([" ".join(x) for x in examples["answers.text"]])

tokenized_eli5 = eli5.map(
    preprocess_function,
    batched=True,
    num_proc=4,
    remove_columns=eli5["train"].column_names,
)
```

이렇게 토큰화된 정보를 확인해 보면 모두 숫자들로 토큰화 되어 있는 것으로 알 수 있다.

```txt
{'input_ids': [ ... 중략 ... ], 'attention_mask': [ ... 중략 ... ]}
```

이러한 정보들을 바로 모델에 넣으면 좋겠지만 그 전에 미리 처리를 해야 하는 일들이 있다.

가장 먼저 해야 할 일은 데이터를 통합하는 것이나 각각의 데이터 셋에 맞게 처리를 해야한다.

이 eli5 데이터셋에서는 question과 answer등 따로 떨어져 있는 정보들이 있다. 이를 하나의 문장으로 통합하는 것이 좋다.

```python
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
```

여기서 생각해 볼 점은 여기서 길이가 긴 데이터들은 모두 block_size 만큼 잘랐다는 것이다. 그런데 실질적으로 GPU가 충분한 메모리를 가지고 있다면 drop을 할 필요가 없다.

```python
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
```

그 다음은 data collator를 통해서 훈련을 하기 좋게 정리를 해주는 것이다. DataCollatorForLanguageModeling은 모델에 맞게 데이터를 정리해 주는 것이다. 이때 mlm은 Masked Language Model인데, 우리는 Casual Language Model을 훈련하고 있기 때문에 False로 설정한다.

이제 훈련을 해보자.

```python
model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2", torch_dtype="auto")

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

이럴 때 SLURM과 accelerator를 같이 사용할 때 중요한 것은 model을 불러올때 `device_map="auto"`{:.python} 와 같이 설정하면 안된다. 이는 SLURM과 accelerator의 충돌이 발생해서 오류가 자주 발생한다.

이렇게 되면 훈련이 자동으로 완료가 된다. 완료가 된 정보들을 두가지 방식으로 생성을 할 수 있다.

하나는 아래와 같은 방식으로 pipeline을 통해서 생성을 할 수 있다.
```python
prompt = "Somatic hypermutation allows the immune system to"
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
```

그리고 다른 방식으로는 직접 모델에 넣어서 생성을 할 수 있다. 
```python
inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
outputs = model.generate(inputs, max_new_tokens=50)
```

이 둘의 결과를 확인해 보면 아래와 같은 결과를 얻을 수 있다.

```txt
----------------------------------------------------------------
[DEBUG] Text generation by pipeline
[{'generated_text': 'Somatic hypermutation allows the immune system to react to the stimulus of stimulus in order to react to the stimulus of stimulus. Thus, the brain is capable of generating a response to the stimulus of stimulus by producing a signal to the target of the stimulus of stimulus. So the immune system responds to'}]
----------------------------------------------------------------
[DEBUG] Text generation step by step
[DEBUG] Inputs: tensor([[   50, 13730,  8718,    76,  7094,  3578,   262, 10900,  1080,   284]], device='cuda:0')
[DEBUG] Outputs: tensor([[   50, 13730,  8718,    76,  7094,  3578,   262, 10900,  1080,   284,  4886,   290, 18175,   262, 10900,  2882,    13,   770,   318,  1521,   262, 10900,  1080,   318,   523,  8564,   284,   262,  4931,   286,   257,  1728,  2099,   286, 10900,  1080,    13,   770,   318,  1521,   262, 10900,  1080,   318,   523,  8564,   284,   262,  4931,   286,   257,  1728,  2099,   286, 10900,  1080,    13,   770,   318,  1521]], device='cuda:0')
Somatic hypermutation allows the immune system to detect and suppress the immune response. This is why the immune system is so sensitive to the presence of a certain type of immune system. This is why the immune system is so sensitive to the presence of a certain type of immune system. This is why
```

`pipeline`{:.python}을 사용하면 더 좋은 결과를 생성할 수 있다. 이는 pipeline에는 여러가지 조건들을 자동으로 적용해 반복을 줄이고 더 자연스러운 문장을 생성하기 때문이다. 하지만 모델은 자동으로 적용이 되지 않아서 직접 설정해야 한다. 

[실행 코드 전체](https://github.com/sehwanhong/LLM/blob/main/python_codes/transformer_03.py)
---
title: "Hugging Face Transformers Tutorial - 01"
date: 2025-06-30
layout: default   # 또는 page, default 등 테마에 따라
tags: [transformers, huggingface, transformers-tutorial, llm]
---

## 1. Introduction

Huggingface의 Transformers 라이브러리는 다양한 트랜스포머 기반의 모델들을 쉽게 사용할 수 있도록 만들어 준다.

## 2. 기초적인 설치

가장 먼저 훈련에 사용하기 위한 기초적인 플랫폼을 설치해야 한다.

```bash
# bash
pip install torch
```

이러한 설치가 끝이 난 이후에는 기본적으로 다음과 같은 라이브러리의 설치를 통해서 transformer를 사용할 수 있게 된다.

```bash
# bash
pip install -r requirements.txt
```

```txt
# requirements.txt

torch
transformers
datasets
evaluate
accelerate
timm
```

## 3. 기본적인 구조 및 실행

일단 가장 먼저 어떤 형식으로 이루어져 있는지 확인을 해 보자.

Transfomer를 사용하기 위해서는 기본적으로 model과 tokenizer가 필요하다. 가장 먼저 model의 구조를 확인해보자.

### 3.1. Model 구조 확인

가장 먼저 모델을 불러오고 구조를 확인해 보자.

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2", torch_dtype="auto", device_map="auto")

print(model)
```

위의 파이선 코드를 실행하면 모델을 불러올 수 있다. print문을 통해서 모델의 구조를 파악할 수 있다.

```txt
# 모델 구조
GPT2LMHeadModel(
  (transformer): GPT2Model(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.1, inplace=False)
    (h): ModuleList(
      (0-5): 6 x GPT2Block(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPT2Attention(
          (c_attn): Conv1D(nf=2304, nx=768)
          (c_proj): Conv1D(nf=768, nx=768)
          (attn_dropout): Dropout(p=0.1, inplace=False)
          (resid_dropout): Dropout(p=0.1, inplace=False)
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPT2MLP(
          (c_fc): Conv1D(nf=3072, nx=768)
          (c_proj): Conv1D(nf=768, nx=3072)
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)
```

이를 살펴보면 embedding, GPT2Block, Linear의 모듈로 구성되어 있다. 

### 3.2. Tokenizer 구조 확인

Tokenizer는 LLM에서 모델의 입력으로 들어갈 수 있도록 토큰화 해주는 역할을 한다.

LLM에서 모델은 문자열을 입력받는다. 하지만 모델에 실질적으로 들어가는 값은 실수 백터열이 들어가는데, Tokenizer는 문자열을 실수 백터 행렬로 변경해 준다. 변경해 주는 것을 토큰화라고 한다.

Tokenizer를 불러오고 구조를 확인해 보자.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

print(tokenizer)
```

위의 파이선 코드를 실행하면 tokenizer의 구조를 파악할 수 있다.

```txt
GPT2TokenizerFast(name_or_path='distilbert/distilgpt2', vocab_size=50257, model_max_length=1024, is_fast=True, padding_side='right', trun    cation_side='right', special_tokens={'bos_token': '<|endoftext|>', 'eos_token': '<|endoftext|>', 'unk_token': '<|endoftext|>'}, clean_up_    tokenization_spaces=False, added_tokens_decoder={
        50256: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
}
)
```

최대 길이 가 1024이고 특수 토큰이 3개 있는 것을 알수 있다.

### 3.3 Tokenizer 실행

```python
prompt = "The secret to baking a good cake is "
model_inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

print(f"prompt : {prompt}")
print(f"model_inputs : {model_inputs}")
```

일단 위의 코드는 prompt를 입력받아서 tokenizer를 통해서 토큰화 한다. 토큰화된 값을 확인하면 다음과 같다.

```txt
prompt : The secret to baking a good cake is
model_inputs : {'input_ids': tensor([[  464,  3200,   284, 16871,   257,   922, 12187,   318,   220]], device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0')}
```

이를 통해서 모델에 입력으로 들어가는 값을 확인할 수 있다.

### 3.4 모델 실행

```python
generated_outputs = model.generate(**model_inputs, max_new_tokens=100)
decoded_outputs = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)

print(f"generated_outputs : {generated_outputs}")
print(f"decoded_outputs : {decoded_outputs}")
```

토큰화 된 입력값을 모델에 넣어주면 다음 문장을 생성해 준다. 이는 CasualLM 모델은 다음 단어들을 생성해 주는 모델이기 때문이다. 

결과값을 확인하면 다음과 같은 값을 확인할 수 있다.

```txt
generated_outputs : tensor([[  464,  3200,   284, 16871,   257,   922, 12187,   318,   220,  3711,   351,   257,  1310,  1643,   286,  8268,    13,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198,   198]], device='cuda:0')
decoded_outputs : ['The secret to baking a good cake is iced with a little bit of salt.\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n    \n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n']
```

생성된 token을 확인해 보면 문장을 생성을 마치고 난 다음에는 문장의 끝을 알리는 토큰이 생성이 되고 다음에는 계속해서 다음줄로 넘어가는 문장 들이 생성이된다.

[실행 코드 전체](https://github.com/sehwanhong/LLM/blob/main/python_codes/transformer_01.py)
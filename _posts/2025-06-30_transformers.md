---
title: "Hugging Face Transformers Tutorial - 01"
date: 2025-06-30
layout: default   # 또는 page, default 등 테마에 따라
---

## 1. Introduction

Huggingface의 Transformers 라이브러리는 다양한 트랜스포머 기반의 모델들을 쉽게 사용할 수 있도록 만들어 준다.

## 2. 기초적인 설치

가장 먼저 훈련에 사용하기 위한 기초적인 플랫폼을 설치해야 한다.

```bash
pip install torch
```

이러한 설치가 끝이 난 이후에는 기본적으로 다음과 같은 라이브러리의 설치를 통해서 transformer를 사용할 수 있게 된다.

```bash
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


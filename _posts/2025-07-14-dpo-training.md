---
title: "DPO Training with PyTorch Lightning"
date: 2025-07-14
layout: default   # 또는 page, default 등 테마에 따라
tags: [llm, dpo, fine-tuning, llm, transformers, pytorch-lightning, fabric, ]
---

## Introduction

이번 포스트에서는 DPO를 활용하여 데이터셋을 활용하여 훈련을 하는 방법을 알아보자. 일단 가장 먼저 torchrun을 사용하여 실행하였는데, 이때 네트워크 연결 문제상으로 문제가 발생하여 pytorch-lightning을 사용하여 실행하였다.

## 코드

```python
# 정책 모델 포워드 패스
policy_chosen_logits = policy_model(
    input_ids=batch["chosen_input_ids"],
    attention_mask=batch["chosen_attention_mask"]
).logits
policy_rejected_logits = policy_model(
    input_ids=batch["rejected_input_ids"],
    attention_mask=batch["rejected_attention_mask"]
).logits

policy_chosen_logps = get_log_probs(policy_chosen_logits, batch["chosen_labels"])
policy_rejected_logps = get_log_probs(policy_rejected_logits, batch["rejected_labels"])

# 참조 모델 포워드 패스 (no_grad)
with torch.no_grad():
    ref_chosen_logits = reference_model(
        input_ids=batch["chosen_input_ids"],
        attention_mask=batch["chosen_attention_mask"]
    ).logits
    ref_rejected_logits = reference_model(
        input_ids=batch["rejected_input_ids"],
        attention_mask=batch["rejected_attention_mask"]
    ).logits
    
    ref_chosen_logps = get_log_probs(ref_chosen_logits, batch["chosen_labels"])
    ref_rejected_logps = get_log_probs(ref_rejected_logits, batch["rejected_labels"])

# DPO Loss 계산
pi_logratios = policy_chosen_logps - policy_rejected_logps
ref_logratios = ref_chosen_logps - ref_rejected_logps

logits = pi_logratios - ref_logratios
loss = -F.logsigmoid(beta * logits).mean()
```

## 코드 설명

위 코드는 DPO(Direct Preference Optimization)의 핵심인 손실 함수를 계산하는 부분입니다. 각 단계는 다음과 같은 의미를 가집니다.

1.  **모델별 응답 확률 계산**:
    -   `policy_..._logps`: 현재 **학습 중인 정책 모델(Policy Model)**이 "선호하는 응답(`chosen`)"과 "거절된 응답(`rejected`)" 각각에 대해 전체 시퀀스의 로그 확률(log-probabilities)을 계산합니다. 이 값이 높을수록 모델이 해당 응답을 생성할 확률이 높다는 의미입니다.
    -   `ref_..._logps`: **학습되지 않고 고정된 참조 모델(Reference Model)**에 대해서도 동일하게 로그 확률을 계산합니다. 참조 모델은 정책 모델이 너무 많이 변경되지 않도록 기준점 역할을 합니다. `with torch.no_grad()` 구문은 참조 모델에 대해서는 그래디언트를 계산할 필요가 없음을 명시하여 불필요한 연산을 줄이고 메모리를 절약합니다.

2.  **로그 확률 비율 계산**:
    -   `pi_logratios`: 정책 모델이 선호 응답을 비선호 응답보다 얼마나 더 선호하는지를 나타내는 값입니다. (선호 응답 로그 확률 - 비선호 응답 로그 확률)
    -   `ref_logratios`: 참조 모델이 선호 응답을 비선호 응답보다 얼마나 더 선호하는지를 나타내는 값입니다.

3.  **최종 Loss 계산**:
    -   `logits`: 정책 모델의 선호도(`pi_logratios`)와 참조 모델의 선호도(`ref_logratios`)의 차이를 계산합니다. 이 값은 **"정책 모델이 참조 모델보다 얼마나 더 선호 응답을 좋아하는가"**를 정량적으로 나타내는 핵심 지표입니다.
    -   `loss = -F.logsigmoid(beta * logits).mean()`: DPO 논문에 정의된 손실 함수를 그대로 코드로 구현한 것입니다.
        -   `beta * logits`: 계산된 `logits` 값에 `beta` 하이퍼파라미터를 곱하여 선호도 차이의 영향력을 조절합니다.
        -   `F.logsigmoid(...)`: 위 결과에 로그-시그모이드 함수를 적용합니다. 이는 모델이 선호 응답을 더 선호하고 비선호 응답을 덜 선호하도록 만드는 방향으로 학습을 유도합니다.
        -   `- ... .mean()`: 최종적으로 음수를 취하고 배치 전체에 대한 평균을 계산하여 최종 손실 값을 얻습니다. 옵티마이저는 이 손실 값을 최소화하는 방향으로 정책 모델의 가중치를 업데이트합니다.

이 과정을 통해 모델은 참조 모델의 분포에서 너무 멀리 벗어나지 않으면서도, 인간이 선호하는 응답 방향으로 일관성 있게 학습하게 됩니다.

## 참고자료

[실행 코드](https://github.com/sehwanhong/LLM/blob/main/python_codes/dpo_training.py)
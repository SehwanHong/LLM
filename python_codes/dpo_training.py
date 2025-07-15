from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as L
from lightning.fabric.plugins.environments.slurm import SLURMEnvironment
from lightning.pytorch.loggers import WandbLogger
from lightning.fabric import Fabric
import wandb
import os
import argparse

def get_log_probs(logits, labels):
    """
    logits와 labels를 사용하여 각 시퀀스의 로그 확률을 계산합니다.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # CrossEntropyLoss를 사용하여 각 토큰의 로그 확률을 계산합니다.
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    log_probs = -loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # 시퀀스별로 로그 확률을 합산합니다.
    return log_probs.view(shift_logits.size(0), -1).sum(-1)

def collate_fn(batch, tokenizer):
    prompts = [item['question'] for item in batch]
    chosens = [item['chosen'] for item in batch]
    rejecteds = [item['rejected'] for item in batch]

    chosen_inputs = tokenizer([p + c for p, c in zip(prompts, chosens)], return_tensors="pt", padding=True, truncation=True)
    rejected_inputs = tokenizer([p + r for p, r in zip(prompts, rejecteds)], return_tensors="pt", padding=True, truncation=True)
    prompt_tokens = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    
    chosen_labels = chosen_inputs['input_ids'].clone()
    rejected_labels = rejected_inputs['input_ids'].clone()
    
    # 프롬프트 부분은 loss 계산에서 제외
    prompt_len = prompt_tokens['input_ids'].shape[1]
    chosen_labels[:, :prompt_len] = -100
    rejected_labels[:, :prompt_len] = -100
    
    return {
        "chosen_input_ids": chosen_inputs["input_ids"],
        "chosen_attention_mask": chosen_inputs["attention_mask"],
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_inputs["input_ids"],
        "rejected_attention_mask": rejected_inputs["attention_mask"],
        "rejected_labels": rejected_labels,
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Train SVM argparser")
    parser.add_argument('--run_name', default="run")
    return parser.parse_args()

def main():
    args = parse_args()
    # --- 1. Fabric 설정 ---
    logger = WandbLogger(project="dpo_fabric_training", name=args.run_name)
    slurm_env = SLURMEnvironment()
    fabric = Fabric(
        accelerator="auto",
        devices="auto", 
        strategy="auto", 
        loggers=logger,
        precision='bf16-mixed',
        plugins=[
            slurm_env,
        ],
    )
    fabric.launch()

    # --- 2. 하이퍼파라미터 설정 ---
    model_name = "distilbert/distilgpt2"
    dataset_name = "Intel/orca_dpo_pairs"
    output_dir = "dpo_training_intel_orca_fabric"
    learning_rate = 1e-4
    beta = 0.1
    num_train_epochs = 1
    batch_size = 16

    os.makedirs(output_dir, exist_ok=True)

    # --- 3. 모델 및 토크나이저 로드 ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    policy_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
    )
    
    peft_config = LoraConfig(
        target_modules=["c_attn"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    policy_model = get_peft_model(policy_model, peft_config)

    reference_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # --- 4. 데이터셋 준비 ---
    dataset = load_dataset(dataset_name, split="train")
    train_dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: collate_fn(x, tokenizer), shuffle=True)

    # --- 5. Fabric으로 모델, 옵티마이저, 데이터 로더 설정 ---
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)
    policy_model, optimizer, train_dataloader = fabric.setup(policy_model, optimizer, train_dataloader)
    reference_model = fabric.setup_module(reference_model)

    # --- 6. 학습 루프 ---
    policy_model.train()
    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # 수동으로 배치 데이터를 GPU로 이동시킵니다.
            batch = {k: v.to(fabric.device) for k, v in batch.items()}

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
            
            # 역전파 및 최적화
            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            
            if fabric.global_rank == 0 and (step % 10 == 0):
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")
                fabric.log_dict({"loss": loss.item()}, step=step)

    # --- 7. 모델 저장 ---
    if fabric.global_rank == 0:
        save_path = os.path.join(output_dir, "final_model")
        policy_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")

    if fabric.global_rank == 0:
        wandb.finish()

if __name__ == "__main__":
    main()
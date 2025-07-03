#! /bin/bash

## Job Settings
#SBATCH --job-name=transformers
#SBATCH --time=99:00:00
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH --comment="LLM STUDY - Transformers"

## Output Settings
#SBATCH --output=./logs/transformers_%A.log

## Container Settings
#SBATCH --container-image=/purestorage/AILAB/AI_1/shhong/enroot_images/pytorch_2_5_1.sqsh
#SBATCH --container-mounts=/purestorage/:/purestorage/,/purestorage/AILAB/AI_1/shhong/cache:/home/$USER/.cache
#SBATCH --no-container-mount-home
#SBATCH --container-writable
#SBATCH --container-workdir=/purestorage/AILAB/AI_1/shhong/LLM/python_codes

## Run
pip install -r requirements.txt

# ------------------------------------------------------------------
# [FIX] 단일 GPU/단일 프로세스 환경에서 WORLD_SIZE 등이 설정되지 않아
# TrainingArguments가 분산 초기화 중 오류를 일으키는 문제 해결.
# srun 자동 감지가 컨테이너 내부에서 불가능하므로 수동으로 지정.
# ------------------------------------------------------------------
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29500 # 임의의 비어있는 포트

python transformer_04.py

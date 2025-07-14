#! /bin/bash

## Job Settings
#SBATCH --job-name=dpo_training
#SBATCH --time=99:00:00
#SBATCH --partition=hopper
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64gb
#SBATCH --comment="LLM STUDY - DPO Training"

## Output Settings
#SBATCH --output=./logs/dpo_training_%A.log

## Container Settings
#SBATCH --container-image=/purestorage/AILAB/AI_1/shhong/enroot_images/pytorch_2_7_1_cuda12_6_cudnn9_dev.sqsh
#SBATCH --container-mounts=/purestorage/:/purestorage/,/purestorage/AILAB/AI_1/shhong/cache:/home/$USER/.cache
#SBATCH --no-container-mount-home
#SBATCH --container-writable
#SBATCH --container-workdir=/purestorage/AILAB/AI_1/shhong/LLM/python_codes

## Run
pip install -r requirements.txt

# Explicitly set rendezvous settings for torchrun
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

torchrun --nnodes=$SLURM_NNODES \
         --nproc_per_node=$SLURM_GPUS_ON_NODE \
         --node_rank=$SLURM_NODEID \
         --rdzv_backend=c10d \
         --rdzv_id=$SLURM_JOB_ID \
         --rdzv_endpoint="${MASTER_ADDR}:29500" \
         dpo_training.py
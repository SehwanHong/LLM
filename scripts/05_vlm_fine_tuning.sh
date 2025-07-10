#! /bin/bash

## Job Settings
#SBATCH --job-name=vlm_fine_tuning
#SBATCH --time=99:00:00
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb
#SBATCH --comment="LLM STUDY - VLM Fine Tuning"

## Output Settings
#SBATCH --output=./logs/vlm_fine_tuning_%A.log

## Container Settings
#SBATCH --container-image=/purestorage/AILAB/AI_1/shhong/enroot_images/pytorch_2_3_0.sqsh
#SBATCH --container-mounts=/purestorage/:/purestorage/,/purestorage/AILAB/AI_1/shhong/cache:/home/$USER/.cache
#SBATCH --no-container-mount-home
#SBATCH --container-writable
#SBATCH --container-workdir=/purestorage/AILAB/AI_1/shhong/LLM/python_codes

## Run
pip install --upgrade "torch==2.4.0" torchvision tensorboard pillow
 
# Install Hugging Face libraries
pip install  --upgrade "transformers==4.45.1" "datasets==3.0.1" "accelerate==0.34.2" "evaluate==0.4.3" "bitsandbytes==0.44.0" "trl==0.11.1" "peft==0.13.0" "qwen-vl-utils"
torchrun --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID --nproc_per_node=$SLURM_GPUS_ON_NODE vlm_fine_tuning.py

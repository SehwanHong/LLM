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
#SBATCH --container-image=/purestorage/AILAB/AI_1/shhong/enroot_images/pytorch_2_5_1.sqsh
#SBATCH --container-mounts=/purestorage/:/purestorage/,/purestorage/AILAB/AI_1/shhong/cache:/home/$USER/.cache
#SBATCH --no-container-mount-home
#SBATCH --container-writable
#SBATCH --container-workdir=/purestorage/AILAB/AI_1/shhong/LLM/python_codes

## Run
pip install -r requirements.txt
torchrun vlm_fine_tuning.py
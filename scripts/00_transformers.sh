#! /bin/bash

## Job Settings
#SBATCH --job-name=transformers
#SBATCH --time=99:00:00
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --comment="LLM STUDY - Transformers"

## Output Settings
#SBATCH --output=./logs/transformers_%A.log

## Container Settings
#SBATCH --container-image=/purestorage/AILAB/AI_1/shhong/enroot_images/pytorch_2_5_1.sqsh
#SBATCH --container-mounts=/purestorage/:/purestorage/
#SBATCH --container-mounts=/purestorage/project/shhong/cache:/home/$USER/.cache
#SBATCH --no-container-mounts-home
#SBATCH --unbuffered
#SBATCH --container-writable
#SBATCH --container-workdir=/purestorage/AILAB/AI_1/shhong/llm/

srun bash -c "
pip install -r requirements.txt;
python main.py;
"
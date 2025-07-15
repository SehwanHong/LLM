#!/bin/bash
#SBATCH --job-name=dpo_training
#SBATCH --time=99:00:00
#SBATCH --partition=hopper
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=64gb
#SBATCH --comment="LLM STUDY - DPO Training"
#SBATCH --output=./logs/dpo_training_%A.log

# srun으로 컨테이너 실행과 작업 명령을 한 번에 전달합니다.
srun --container-image=/purestorage/AILAB/AI_1/shhong/enroot_images/pytorch_2_7_1_cuda12_6_cudnn9_dev.sqsh \
     --container-mounts=/purestorage/:/purestorage/,/purestorage/AILAB/AI_1/shhong/cache:/home/$USER/.cache \
     --no-container-mount-home \
     --container-writable \
     --container-workdir=/purestorage/AILAB/AI_1/shhong/LLM/python_codes \
     bash -c "
     # 0번 프로세스만 패키지를 설치합니다.
     if [ \$SLURM_PROCID -eq 0 ]; then
        echo 'Rank 0: Installing packages...';
        pip install -r requirements.txt;
     fi

     # 모든 프로세스가 여기에 도달할 때까지 기다립니다 (배리어 동기화).
     scontrol barrier \$SLURM_JOB_ID ;

     echo \"Starting training on rank \$SLURM_PROCID...\";
     python dpo_training.py --run_name dpo_training_${SLURM_JOB_ID};
     echo \"Training finished on rank \$SLURM_PROCID.\";
     "
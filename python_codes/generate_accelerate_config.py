import os
import yaml
import argparse
from typing import Dict, Any

def generate_config_dict() -> Dict[str, Any]:
    """
    SLURM 환경 변수를 기반으로 Accelerate 설정 사전을 생성합니다.
    """
    # SLURM 환경 변수를 읽어옵니다. 만약 변수가 없다면 기본값(1)을 사용합니다.
    # .get() 메서드는 변수가 없을 때 None을 반환하므로, 'or'를 사용해 기본값을 설정합니다.
    num_machines = int(os.environ.get('SLURM_NNODES', 1))
    num_processes = int(os.environ.get('SLURM_NPROCS', 1))
    
    # MASTER_ADDR와 MASTER_PORT는 accelerate launch가 SLURM에서 자동으로 처리하는 경우가 많습니다.
    # 스크립트의 명확성을 위해 None으로 설정해둡니다.
    main_process_ip = os.environ.get('MASTER_ADDR', None)
    main_process_port = os.environ.get('MASTER_PORT', None)
    
    # YAML 파일로 만들 파이썬 딕셔너리 구조
    config = {
        'compute_environment': 'LOCAL_MACHINE',
        'distributed_type': 'SLURM',
        'num_machines': num_machines,
        'num_processes': num_processes,
        'gpu_ids': 'all',  # 'all'로 설정하면 accelerate가 자동으로 할당합니다.
        'machine_rank': 0, # accelerate launch가 이 값을 덮어씁니다.
        'main_process_ip': main_process_ip,
        'main_process_port': main_process_port,
        'main_training_function': 'main',
        'mixed_precision': 'no',  # 필요에 따라 'fp16', 'bf16'으로 변경
        'rdzv_backend': 'static',
        'same_network': True,
        'tpu_use_cluster': False,
        'tpu_use_sudo': False,
        'use_cpu': False,
    }
    
    return config

def main():
    """
    메인 실행 함수: 커맨드 라인 인자를 파싱하고 설정 파일을 생성합니다.
    """
    parser = argparse.ArgumentParser(
        description="Generate a Hugging Face Accelerate config file for SLURM."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="accelerate_config.yaml",
        help="Path to save the generated YAML config file."
    )
    args = parser.parse_args()

    # 설정 딕셔너리 생성
    config_data = generate_config_dict()
    
    # 딕셔너리를 YAML 파일로 저장
    try:
        with open(args.output_file, 'w') as f:
            # default_flow_style=False 옵션으로 가독성 좋은 블록 스타일로 저장
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Accelerate config successfully generated at: {args.output_file}")
        print(f"   - Num Machines (Nodes): {config_data['num_machines']}")
        print(f"   - Num Processes (Total GPUs): {config_data['num_processes']}")
        
    except Exception as e:
        print(f"❌ Error generating config file: {e}")

if __name__ == "__main__":
    main()
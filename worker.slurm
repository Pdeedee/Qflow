#!/bin/bash
# ============================================================
# 这是一个SLURM作业脚本的参考模板
# 实际运行时，qflow会根据config.yaml中的slurm配置自动生成脚本
# ============================================================

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --partition=compute
#SBATCH --time=72:00:00
#SBATCH --output=worker_%j.out
#SBATCH --error=worker_%j.err

# 加载环境模块（根据需要取消注释）
# module load vasp/6.3.0
# module load intel/2021

# 激活conda环境
source activate mattersim

# 设置工作目录
cd $SLURM_SUBMIT_DIR

# 运行worker
python -m qflow.worker

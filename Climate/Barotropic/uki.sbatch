#!/bin/bash

#SBATCH --time=12:00:00   # 优化后预计12小时内完成（原48小时）
#SBATCH --ntasks=1        # 【最关键修改1】多线程只需要1个任务，原32个是错误的
#SBATCH --cpus-per-task=64 # 【最关键修改2】申请节点全部64个CPU核心
#SBATCH --nodes=1         
#SBATCH -p C064M0256G 
#SBATCH -J "uki-experiment"
#SBATCH -o "uki-experiment-%j.out"

# 加载 Julia 模块（如果你的集群需要，取消注释）
# module load julia/1.10.5

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "JULIA_NUM_THREADS: $JULIA_NUM_THREADS"

julia --project=. --optimize=3 --check-bounds=no run_uki.jl |& tee run_uki.log
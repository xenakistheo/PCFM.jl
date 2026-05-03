#!/bin/bash
#SBATCH --job-name=pcfm_infer_benchmarks
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/infer_benchmarks_%j.out
#SBATCH --error=logs/infer_benchmarks_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "GPUs:        $CUDA_VISIBLE_DEVICES"
echo "Start:       $(date)"
echo "Project dir: $SLURM_SUBMIT_DIR"
echo "--------------------------------------"

echo "Running heat inference..."
julia --project=. examples/infer_without_reactant.jl \
    > logs/infer_heat.log 2>&1 \
    && echo "Heat: done" || echo "Heat: FAILED"

echo "Running Burgers inference..."
julia --project=. examples/infer_burgers_without_reactant.jl \
    > logs/infer_burgers.log 2>&1 \
    && echo "Burgers: done" || echo "Burgers: FAILED"

echo "Running RD inference..."
julia --project=. examples/infer_rd_without_reactant.jl \
    > logs/infer_rd.log 2>&1 \
    && echo "RD: done" || echo "RD: FAILED"

echo "--------------------------------------"
echo "Done: $(date)"
echo "Logs: logs/infer_heat.log, logs/infer_burgers.log, logs/infer_rd.log"

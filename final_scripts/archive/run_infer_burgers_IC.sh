#!/bin/bash
#SBATCH --job-name=pcfm_burgers_IC
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/burgers_IC_%j.out
#SBATCH --error=logs/burgers_IC_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs datasets/data

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "GPUs:        $CUDA_VISIBLE_DEVICES"
echo "Start:       $(date)"
echo "Project dir: $SLURM_SUBMIT_DIR"
echo "--------------------------------------"


echo "Running Burgers inference..."
julia --project=. examples/infer_burgers_IC.jl \
    > logs/burgers_IC_infer.log 2>&1 \
    && echo "Inference: done" || { echo "Inference: FAILED"; exit 1; }

echo "--------------------------------------"
echo "Done: $(date)"
echo "Logs: logs/burgers_IC_infer.log"
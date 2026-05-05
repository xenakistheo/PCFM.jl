#!/bin/bash
#SBATCH --job-name=pcfm_burgersIC
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/rd_%j.out
#SBATCH --error=logs/rd_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs datasets/data

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "GPUs:        $CUDA_VISIBLE_DEVICES"
echo "Start:       $(date)"
echo "Project dir: $SLURM_SUBMIT_DIR"
echo "--------------------------------------"


echo "Running Burgers IC inference..."
julia --project=. examples/infer_burgers_IC.jl \
    > logs/burgers_IC_infer.log 2>&1 \
    && echo "Inference: done" || { echo "Inference: FAILED"; exit 1; }

echo "--------------------------------------"
echo "Done: $(date)"
echo "Logs: logs/burgers_IC_infer.log"

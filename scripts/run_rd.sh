#!/bin/bash
#SBATCH --job-name=pcfm_rd
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

echo "[1/3] Generating RD data..."
julia --project=. datasets/generate_rd_data.jl \
    > logs/rd_data.log 2>&1 \
    && echo "Data generation: done" || { echo "Data generation: FAILED"; exit 1; }

echo "[2/3] Training FFM on RD..."
julia --project=. examples/train/train_rd.jl \
    > logs/rd_train.log 2>&1 \
    && echo "Training: done" || { echo "Training: FAILED"; exit 1; }

echo "[3/3] Running RD inference..."
julia --project=. examples/infer_rd.jl \
    > logs/rd_infer.log 2>&1 \
    && echo "Inference: done" || { echo "Inference: FAILED"; exit 1; }

echo "--------------------------------------"
echo "Done: $(date)"
echo "Logs: logs/rd_data.log, logs/rd_train.log, logs/rd_infer.log"

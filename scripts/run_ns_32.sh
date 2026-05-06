#!/bin/bash
#SBATCH --job-name=pcfm_ns_32
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ns_32_%j.out
#SBATCH --error=logs/ns_32_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs datasets/data examples/checkpoints

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "GPUs:        $CUDA_VISIBLE_DEVICES"
echo "Start:       $(date)"
echo "Project dir: $SLURM_SUBMIT_DIR"
echo "======================================"

echo "[1/2] Generating 32x32 NS data..."
julia --project=. datasets/generate_ns_data.jl
echo "Data generation done: $(date)"
echo "--------------------------------------"

echo "[2/2] Training NS model (32x32)..."
julia --project=. examples/train/train_ns.jl
echo "Training done: $(date)"
echo "======================================"
echo "All done: $(date)"

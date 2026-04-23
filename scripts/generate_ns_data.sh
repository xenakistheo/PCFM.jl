#!/bin/bash
#SBATCH --job-name=pcfm_ns_data
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ns_data_%j.out
#SBATCH --error=logs/ns_data_%j.err

set -euo pipefail

# ── project root ──────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

mkdir -p logs datasets/data

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "GPUs:        $CUDA_VISIBLE_DEVICES"
echo "Start:       $(date)"
echo "Project dir: $PROJECT_DIR"
echo "--------------------------------------"

# ── run ───────────────────────────────────────────────────────────────────────
julia --project=. examples/generate_ns_data.jl

echo "--------------------------------------"
echo "Done: $(date)"

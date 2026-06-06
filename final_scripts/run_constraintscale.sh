#!/bin/bash
#SBATCH --job-name=pcfm_burgers_IC
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/scale_burgers_IC_%j.out
#SBATCH --error=logs/scale_burgers_IC_%j.err

# Usage:   sbatch final_scripts/run_constraintscale.sh --constraint IC_Mass_Flux --k 5 --run 1
# Example: sbatch final_scripts/run_constraintscale.sh --constraint IC_Mass --k 5 --run 2
# Constraint options: IC  |  IC_Mass  |  IC_Mass_Flux

set -euo pipefail

# Parse flags
CONSTRAINT="IC_Mass_Flux"
K=5
RUN=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --constraint) CONSTRAINT="$2"; shift 2 ;;
        --k)          K="$2";          shift 2 ;;
        --run)        RUN="$2";        shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

SAMPLES_PATH="samples_burgers_IC_${CONSTRAINT}_k${K}${RUN:+_run$RUN}.jld2"
LOG="logs/scale_burgers_IC_${CONSTRAINT}_k${K}${RUN:+_run$RUN}.log"

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs datasets/data

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "GPUs:        $CUDA_VISIBLE_DEVICES"
echo "Start:       $(date)"
echo "Project dir: $SLURM_SUBMIT_DIR"
echo "Constraint:  $CONSTRAINT"
echo "K:           $K"
echo "Run:         ${RUN:-default}"
echo "Samples out: $SAMPLES_PATH"
echo "--------------------------------------"

echo "Running Burgers IC inference..."
julia --project=. examples/infer_burgers_IC.jl "$CONSTRAINT" "$K" "$SAMPLES_PATH" \
    > "$LOG" 2>&1 \
    && echo "Inference: done" || { echo "Inference: FAILED"; exit 1; }

echo "--------------------------------------"
echo "Done: $(date)"
echo "Logs: $LOG"

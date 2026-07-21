#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage: sbatch --job-name=pcfm_train_<problem> final_scripts/train_ffm.sh <problem>
# Problems: ns | rd | burgers | heat
# Example:  sbatch --job-name=pcfm_train_rd final_scripts/train_ffm.sh rd
#
# Note: burgers_IC (examples/BurgersIC) and heat2 (examples/HeatEq2) reuse the
# burgers/heat checkpoints trained here — they don't have their own train step.

set -euo pipefail

PROBLEM="${1:-}"

case "$PROBLEM" in
    ns)      SCRIPT="examples/NavierStokes/train_ffm_ns.jl"       ; LOG="logs/ns_train.log"      ;;
    rd)      SCRIPT="examples/ReactionDiffusion/train_ffm_rd.jl"  ; LOG="logs/rd_train.log"      ;;
    burgers) SCRIPT="examples/BurgersBC/train_ffm_burgers.jl"     ; LOG="logs/burgers_train.log" ;;
    heat)    SCRIPT="examples/HeatEq1/train_ffm_HeatEq.jl"        ; LOG="logs/heat_train.log"    ;;
    *)
        echo "Usage: sbatch --job-name=pcfm_train_<problem> $0 <problem>"
        echo "  problem: ns | rd | burgers | heat"
        exit 1
        ;;
esac

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs examples/checkpoints

echo "Job ID:      ${SLURM_JOB_ID:-none}"
echo "Node:        ${SLURMD_NODENAME:-$(hostname)}"
echo "GPUs:        ${CUDA_VISIBLE_DEVICES:-none}"
echo "Start:       $(date)"
echo "Project dir: $(pwd)"
echo "Problem:     $PROBLEM"
echo "--------------------------------------"

echo "Training FFM on $PROBLEM..."
julia --project=. "$SCRIPT" \
    > "$LOG" 2>&1 \
    && echo "Training: done" || { echo "Training: FAILED"; exit 1; }

echo "--------------------------------------"
echo "Done: $(date)"
echo "Log: $LOG"

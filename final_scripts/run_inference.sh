#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage: sbatch --job-name=pcfm_<problem> final_scripts/run_inference.sh <problem> [run]
# Problems: heat | heat2 | burgers_BC | burgers_IC | rd
# Example:  sbatch --job-name=pcfm_heat final_scripts/run_inference.sh heat 2

set -euo pipefail

PROBLEM="${1:-}"
RUN="${2:-}"

case "$PROBLEM" in
    heat)       SCRIPT="examples/infer_heat.jl"       ; LOG_BASE="logs/heat_infer"       ;;
    heat2)      SCRIPT="examples/infer_heat_2.jl"     ; LOG_BASE="logs/heat2_infer"      ;;
    burgers_BC) SCRIPT="examples/infer_burgers_BC.jl" ; LOG_BASE="logs/burgers_BC_infer" ;;
    burgers_IC) SCRIPT="examples/infer_burgers_IC.jl" ; LOG_BASE="logs/burgers_IC_infer" ;;
    rd)         SCRIPT="examples/infer_rd.jl"         ; LOG_BASE="logs/rd_infer"         ;;
    *)
        echo "Usage: sbatch --job-name=pcfm_<problem> $0 <problem> [run]"
        echo "  problem: heat | heat2 | burgers_BC | burgers_IC | rd"
        exit 1
        ;;
esac

LOG="${LOG_BASE}${RUN:+-$RUN}.log"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs datasets/data

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "GPUs:        $CUDA_VISIBLE_DEVICES"
echo "Start:       $(date)"
echo "Project dir: $SLURM_SUBMIT_DIR"
echo "Problem:     $PROBLEM"
echo "--------------------------------------"

echo "Running $PROBLEM inference..."
julia --project=. "$SCRIPT" \
    > "$LOG" 2>&1 \
    && echo "Inference: done" || { echo "Inference: FAILED"; exit 1; }

echo "--------------------------------------"
echo "Done: $(date)"
echo "Logs: $LOG"

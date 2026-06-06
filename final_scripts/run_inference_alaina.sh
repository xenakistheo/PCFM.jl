#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage: sbatch --job-name=pcfm_alaina_<problem> final_scripts/run_inference_alaina.sh <problem> [run]
# Problems: heat | heat2 | burgers_BC | burgers_IC | rd | ns
# Example:  sbatch --job-name=pcfm_alaina_heat final_scripts/run_inference_alaina.sh heat 2

set -euo pipefail

PROBLEM="${1:-}"
RUN="${2:-}"

case "$PROBLEM" in
    heat)       SCRIPT="examples/infer_heat_alaina.jl"       ; LOG_BASE="logs/heat_alaina_infer"       ; N_SAMPLES=32 ;;
    heat2)      SCRIPT="examples/infer_heat_2_alaina.jl"     ; LOG_BASE="logs/heat2_alaina_infer"      ; N_SAMPLES=32 ;;
    burgers_BC) SCRIPT="examples/infer_burgers_BC_alaina.jl" ; LOG_BASE="logs/burgers_BC_alaina_infer" ; N_SAMPLES=32 ;;
    burgers_IC) SCRIPT="examples/infer_burgers_IC_alaina.jl" ; LOG_BASE="logs/burgers_IC_alaina_infer" ; N_SAMPLES=32 ;;
    rd)         SCRIPT="examples/infer_rd_alaina.jl"         ; LOG_BASE="logs/rd_alaina_infer"         ; N_SAMPLES=32 ;;
    ns)         SCRIPT="examples/infer_ns_alaina.jl"         ; LOG_BASE="logs/ns_alaina_infer"         ; N_SAMPLES=4  ;;
    *)
        echo "Usage: sbatch --job-name=pcfm_alaina_<problem> $0 <problem> [run]"
        echo "  problem: heat | heat2 | burgers_BC | burgers_IC | rd | ns"
        exit 1
        ;;
esac

SAMPLES_PATH="samples_${PROBLEM}_alaina${RUN:+_run$RUN}.jld2"
LOG="${LOG_BASE}${RUN:+-$RUN}.log"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs datasets/data

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "GPUs:        $CUDA_VISIBLE_DEVICES"
echo "Start:       $(date)"
echo "Project dir: $SLURM_SUBMIT_DIR"
echo "Problem:     $PROBLEM"
echo "Samples out: $SAMPLES_PATH"
echo "--------------------------------------"

echo "Running $PROBLEM inference (Alaina solvers)..."
julia --project=. "$SCRIPT" "$N_SAMPLES" "$SAMPLES_PATH" \
    > "$LOG" 2>&1 \
    && echo "Inference: done" || { echo "Inference: FAILED"; exit 1; }

echo "--------------------------------------"
echo "Done: $(date)"
echo "Logs: $LOG"

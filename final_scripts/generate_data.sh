#!/bin/bash
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage: sbatch --job-name=pcfm_gen_<problem> final_scripts/generate_data.sh <problem>
# Problems: ns | rd | burgers | heat
# Example:  sbatch --job-name=pcfm_gen_rd final_scripts/generate_data.sh rd

set -euo pipefail

PROBLEM="${1:-}"

case "$PROBLEM" in
    ns)      SCRIPT="examples/NavierStokes/generate_data_ns.jl"        ; LOG="logs/ns_gen.log"      ;;
    rd)      SCRIPT="examples/ReactionDiffusion/generate_data_rd.jl"   ; LOG="logs/rd_gen.log"      ;;
    burgers) SCRIPT="examples/BurgersBC/generate_data_burgers.jl"      ; LOG="logs/burgers_gen.log" ;;
    heat)    SCRIPT="examples/HeatEq1/generate_data_heat.jl"           ; LOG="logs/heat_gen.log"    ;;
    *)
        echo "Usage: sbatch --job-name=pcfm_gen_<problem> $0 <problem>"
        echo "  problem: ns | rd | burgers | heat"
        exit 1
        ;;
esac

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs datasets/data

echo "Job ID:      ${SLURM_JOB_ID:-none}"
echo "Node:        ${SLURMD_NODENAME:-$(hostname)}"
echo "GPUs:        ${CUDA_VISIBLE_DEVICES:-none}"
echo "Start:       $(date)"
echo "Project dir: $(pwd)"
echo "Problem:     $PROBLEM"
echo "--------------------------------------"

echo "Generating $PROBLEM data..."
julia --project=. "$SCRIPT" \
    > "$LOG" 2>&1 \
    && echo "Data generation: done" || { echo "Data generation: FAILED"; exit 1; }

echo "--------------------------------------"
echo "Done: $(date)"
echo "Log: $LOG"

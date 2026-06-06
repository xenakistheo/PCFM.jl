#!/bin/bash
#SBATCH --job-name=pcfm_heat
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/heat_%j.out
#SBATCH --error=logs/heat_%j.err

# Usage: bash final_scripts/run_heat_nsamplesscales.sh --nsamples 128 --run 1
# Example: bash final_scripts/run_heat_nsamplesscales.sh --nsamples 64 --run 2

set -euo pipefail

# Parse flags
NSAMPLES=32
RUN=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nsamples) NSAMPLES="$2"; shift 2 ;;
        --run)      RUN="$2";      shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

SAMPLES_PATH="samples_heat_nsamples${NSAMPLES}${RUN:+_run$RUN}.jld2"
LOG="logs/heat_infer_nsamples${NSAMPLES}${RUN:+_run$RUN}.log"

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs datasets/data

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "GPUs:        $CUDA_VISIBLE_DEVICES"
echo "Start:       $(date)"
echo "Project dir: $SLURM_SUBMIT_DIR"
echo "N samples:   $NSAMPLES"
echo "Run:         ${RUN:-default}"
echo "Samples out: $SAMPLES_PATH"
echo "--------------------------------------"

echo "Running Heat inference..."
julia --project=. examples/infer_heat.jl "$NSAMPLES" "$SAMPLES_PATH" \
    > "$LOG" 2>&1 \
    && echo "Inference: done" || { echo "Inference: FAILED"; exit 1; }

echo "--------------------------------------"
echo "Done: $(date)"
echo "Logs: $LOG"

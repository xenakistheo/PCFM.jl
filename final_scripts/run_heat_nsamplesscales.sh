#!/bin/bash
#SBATCH --job-name=pcfm_heat
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/heat_%j.out
#SBATCH --error=logs/heat_%j.err

set -euo pipefail

# Parse --nsamples flag
NSAMPLES=32
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nsamples) NSAMPLES="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

cd "$SLURM_SUBMIT_DIR"

mkdir -p logs datasets/data

echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "GPUs:        $CUDA_VISIBLE_DEVICES"
echo "Start:       $(date)"
echo "Project dir: $SLURM_SUBMIT_DIR"
echo "N samples:   $NSAMPLES"
echo "--------------------------------------"

echo "Running Heat inference..."
julia --project=. examples/infer_heat.jl "$NSAMPLES" "samples_heat_nsamples${NSAMPLES}.jld2" \
    > logs/heat_infer_nsamples${NSAMPLES}.log 2>&1 \
    && echo "Inference: done" || { echo "Inference: FAILED"; exit 1; }

echo "--------------------------------------"
echo "Done: $(date)"
echo "Logs: logs/heat_infer_nsamples${NSAMPLES}.log"

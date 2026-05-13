#!/bin/bash
#SBATCH --job-name=heat-reactant
#SBATCH --output=logs/heat_reactant_%j.out
#SBATCH --error=logs/heat_reactant_%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

cd /orcd/home/002/demri186/PCFM.jl-reactant

mkdir -p logs plots

export JULIA_NUM_PRECOMPILE_TASKS=1
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "Running heat Reactant inference..."
julia --project=. examples/infer_heat_reactant.jl

echo "Plotting heat Reactant results..."
julia --project=. examples/plot/plot_heat_reactant.jl

echo "Done heat Reactant."
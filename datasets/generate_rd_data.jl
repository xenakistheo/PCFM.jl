"""
Generate 1D Reaction-Diffusion equation training and test datasets and save to HDF5.

Grid: Nx=128 (cell-centred), nt=100 snapshots, domain [0,1].
Parameters: rho=0.01, nu=0.005. IC is a random sum of sinusoids normalised to [0,1].
BC: random Neumann fluxes gL ∈ [0, 0.05], gR ∈ [-0.05, 0].

Output files (written to datasets/data/):
  - RD_neumann_train_nIC80_nBC80.h5   (train: 6400 samples)
  - RD_neumann_test_nIC30_nBC30.h5    (test:   900 samples)
"""

include(joinpath(@__DIR__, "..", "datasets", "generate_RD1d_data.jl"))

data_dir = joinpath(@__DIR__, "..", "datasets", "data")

println("Generating RD training set (80 ICs × 80 BCs = 6400 samples)...")
run_parallel(data_dir; N_ic=80, N_bc=80, seed=42, filename="RD_neumann_train")

println("Generating RD test set (30 ICs × 30 BCs = 900 samples)...")
run_parallel(data_dir; N_ic=30, N_bc=30, seed=0, filename="RD_neumann_test")

println("Done.")

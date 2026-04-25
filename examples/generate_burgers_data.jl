"""
Generate 1D Burgers equation training and test datasets and save to HDF5.

Grid: Nx=100, Nt=100 → (Nx+1, Nt+1) = (101, 101) points per sample.
Parameters: sigmoid IC location p_loc ∈ [0.2, 0.8], left BC u_bc ∈ [0, 1].

Output files (written to datasets/data/):
  - burgers_train_nIC80_nBC80.h5   (train: 6400 samples)
  - burgers_test_nIC30_nBC30.h5    (test:   900 samples)
"""

include(joinpath(@__DIR__, "..", "datasets", "generate_burgers1d_data.jl"))

data_dir = joinpath(@__DIR__, "..", "datasets", "data")

println("Generating Burgers training set (80 ICs × 80 BCs = 6400 samples)...")
generate_burgers_dataset(data_dir, 80, 80; seed=42, filename="burgers_train")

println("Generating Burgers test set (30 ICs × 30 BCs = 900 samples)...")
generate_burgers_dataset(data_dir, 30, 30; seed=0, filename="burgers_test")

println("Done.")

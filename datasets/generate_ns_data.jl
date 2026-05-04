"""
Generate 2D Navier-Stokes training and test datasets and save to HDF5.

Runs the pseudo-spectral Crank-Nicolson solver for all (nw × nf) combinations
of initial vorticity fields and sinusoidal forcings.

Warning: generation can take a long time on CPU. Run on a machine with many
cores — the solver batches pairs internally but is not parallelised across batches.

Output files (written to datasets/data/):
  - ns_nw100_nf100_s64_t50_mu0.001.h5   (train: 10,000 solutions)
  - ns_nw30_nf30_s64_t50_mu0.001.h5     (test:    900 solutions)
"""

include(joinpath(@__DIR__, "..", "datasets", "generate_ns_2d.jl"))

data_dir = joinpath(@__DIR__, "..", "datasets", "data")

# Training set: 100 initial conditions × 100 forcings = 10,000 solutions
println("Generating training set...")
navier_stokes(data_dir;
              nw=100, nf=100, s=64, T=49, steps=50,
              mu=1e-3, batch_size=1024, seed=42, delta=1e-4, use_gpu=true)

# Test set: 30 × 30 = 900 solutions
println("Generating test set...")
navier_stokes(data_dir;
              nw=30, nf=30, s=64, T=49, steps=50,
              mu=1e-3, batch_size=1024, seed=0, delta=1e-4)

println("Done.")

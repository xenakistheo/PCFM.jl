"""
Training script for Functional Flow Matching on the 2D Navier-Stokes equation
(vorticity form, pseudo-spectral Crank-Nicolson solver).

Data is generated once and saved to HDF5. Each sample is a vorticity field of
shape (s, s, t) = (64, 64, 50).

NOTE: The current FFM model expects (nx, nt, 1, batch) — 1D spatial inputs.
Model training is left as a TODO pending 2D FFM support. This script handles
data generation and loading only.
"""

using PCFM
using Plots
using Random

Random.seed!(1234)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
datasets_dir = joinpath(@__DIR__, "..", "datasets")
data_dir     = joinpath(datasets_dir, "data")
train_file   = joinpath(data_dir, "ns_nw100_nf100_s64_t50_mu0.001.h5")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
batch_size = 32
s          = 64    # spatial grid size (s × s)
nt         = 50    # number of recorded time snapshots

# ---------------------------------------------------------------------------
println("=" ^ 60)
println("2D Navier-Stokes — Functional Flow Matching")
println("=" ^ 60)

# 1. Generate dataset if not present
if !isfile(train_file)
    println("\n[1/2] Generating Navier-Stokes dataset (saved to $data_dir)...")
    println("  Warning: this can take a long time. Consider running on GPU.")
    include(joinpath(datasets_dir, "generate_ns_2d.jl"))
    navier_stokes(data_dir;
                  nw=100, nf=100, s=s, T=49, steps=nt,
                  mu=1e-3, batch_size=1024, seed=42, delta=1e-3)
    println("  Done.")
else
    println("\n[1/2] Dataset found: $train_file")
end

# 2. Load a batch and inspect
println("\n[2/2] Loading batch...")
u_data = load_ns_batch(train_file, batch_size)
println("  Data shape: $(size(u_data))  — (s, s, t, batch_size)")

# ---------------------------------------------------------------------------
# Visualise a few vorticity snapshots
# ---------------------------------------------------------------------------
arr = u_data
p_snaps = [Plots.heatmap(arr[:, :, 1, i],
               title = "Sample $i, t=0", xlabel = "x", ylabel = "y", c = :RdBu)
           for i in 1:min(4, batch_size)]
display(Plots.plot(p_snaps..., layout = (2, 2), size = (800, 600)))

println("""
TODO: extend FFM to support 2D spatial inputs (s × s × t) before training.
      The model currently expects (nx, nt, 1, batch).
""")

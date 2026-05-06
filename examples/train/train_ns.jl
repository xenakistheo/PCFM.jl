"""
Training script for Functional Flow Matching on the 2D Navier-Stokes equation
(vorticity form, pseudo-spectral Crank-Nicolson solver).

Data: vorticity fields of shape (s, s, t) = (32, 32, 50).
FFM input format after loading: (s, s, t, 1, batch).
"""

using PCFM
using Reactant, Lux
using JLD2, Functors
using Plots
using Random

Random.seed!(1234)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
data_dir    = joinpath(@__DIR__, "..", "..", "datasets", "data")
train_file  = joinpath(data_dir, "ns_nw100_nf100_s32_t50_mu0.001.h5")
weight_file = joinpath(@__DIR__, "..", "checkpoints", "ffm_ns_s32_checkpoint.jld2")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
batch_size    = 16
s             = 32    # spatial grid size (s × s)
nt            = 50    # number of recorded time snapshots
emb_channels  = 32
n_epochs      = 1000
force_retrain = false

# Fourier modes: (s_modes, s_modes, t_modes) — keep well below s/2 and nt/2
ns_modes = (8, 8, 12)

println("=" ^ 60)
println("2D Navier-Stokes — Functional Flow Matching")
println("=" ^ 60)

# 1. Check dataset
if !isfile(train_file)
    error("Training data not found at $train_file.\nRun datasets/generate_ns_data.jl first.")
end
println("\n[1/5] Dataset found: $train_file")

# 2. Load a batch
println("\n[2/5] Loading batch...")
u_data = load_ns_batch(train_file, batch_size)
println("  Data shape: $(size(u_data))  — (s, s, t, 1, batch_size)")

# 3. Create model
println("\n[3/5] Creating FFM model for 2D Navier-Stokes...")
ffm = FFM(
    spatial_size  = (s, s),
    nt            = nt,
    emb_channels  = emb_channels,
    hidden_channels = 64,
    proj_channels = 256,
    n_layers      = 4,
    modes         = ns_modes,
    device        = reactant_device()
)
println("  spatial_size = $(ffm.config[:spatial_size]),  modes = $(ffm.config[:modes])")
println("  in_channels  = $(1 + emb_channels + length((s, s)) + 1)  (u + time_emb + pos_x + pos_y + pos_t)")

# 4. Compile
println("\n[4/5] Compiling functions with Reactant...")
compiled_funcs = PCFM.compile_functions(ffm, batch_size)

# 5. Train or load checkpoint
losses = Float32[]
if isfile(weight_file) && !force_retrain
    println("\n[5/5] Loading checkpoint from: $weight_file")
    saved  = JLD2.load(weight_file)
    device = ffm.config[:device]
    ps     = saved["parameters"] |> device
    st     = saved["states"]     |> device
    losses = saved["losses"]
    println("  Loaded parameters, states, and loss history")
else
    println("\n[5/5] Training for $n_epochs epochs...")
    losses, tstate = train_ffm!(ffm, u_data; compiled_funcs, epochs = n_epochs, verbose = true)
    println("\nFinal loss: $(losses[end])")

    ps = fmap(x -> x isa AbstractArray ? Array(x) : x, tstate.parameters)
    st = fmap(x -> x isa AbstractArray ? Array(x) : x, tstate.states)
    mkpath(dirname(weight_file))
    JLD2.save(weight_file, "parameters", ps, "states", st, "losses", losses, "config", ffm.config)
    println("Checkpoint saved to: $weight_file")
end

# ---------------------------------------------------------------------------
# Plot training curve
# ---------------------------------------------------------------------------
p1 = plot(1:length(losses), losses;
    yscale  = :log10,
    xlabel  = "Epoch", ylabel = "Loss (log)",
    title   = "NS Training Loss",
    legend  = false, linewidth = 2)
display(p1)

println("\nDone!")

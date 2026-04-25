"""
Train an FFM model on the 1D Reaction-Diffusion equation and save a checkpoint.

Assumes the training dataset already exists. Run generate_rd_data.jl first if needed.

Saves checkpoint to: examples/checkpoints/ffm_rd_checkpoint.jld2
  Keys: "parameters", "states", "losses", "config"
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
data_dir    = joinpath(@__DIR__, "..", "datasets", "data")
train_file  = joinpath(data_dir, "RD_neumann_train_nIC80_nBC80.h5")
weight_file = joinpath(@__DIR__, "checkpoints", "ffm_rd_checkpoint.jld2")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
batch_size   = 32
nx           = 128      # Nx spatial points (cell-centred)
nt           = 100      # nt time snapshots
emb_channels = 32
n_epochs     = 1000

# ---------------------------------------------------------------------------
println("=" ^ 60)
println("Reaction-Diffusion Equation — Train FFM")
println("=" ^ 60)

# 1. Load training batch
if !isfile(train_file)
    error("Training data not found at $train_file.\nRun examples/generate_rd_data.jl first.")
end
println("\n[1/3] Loading training batch from $train_file ...")
u_data = load_rd_batch(train_file, batch_size)
println("  Data shape: $(size(u_data))  — (nx, nt, 1, batch_size)")

# 2. Create model
println("\n[2/3] Creating FFM model...")
ffm = FFM(
    nx = nx,
    nt = nt,
    emb_channels = emb_channels,
    hidden_channels = 64,
    proj_channels = 256,
    n_layers = 4,
    modes = (32, 32),
    device = reactant_device()
)
println("  Model created successfully")

# 3. Compile and train
println("\n[3/3] Compiling and training for $n_epochs epochs...")
compiled_funcs = PCFM.compile_functions(ffm, batch_size)
losses, tstate = train_ffm!(ffm, u_data; compiled_funcs, epochs = n_epochs, verbose = true)
println("\nFinal loss: $(losses[end])")

# Save checkpoint (parameters and states moved to CPU for portability)
ps = fmap(x -> x isa AbstractArray ? Array(x) : x, tstate.parameters)
st = fmap(x -> x isa AbstractArray ? Array(x) : x, tstate.states)
mkpath(dirname(weight_file))
JLD2.save(weight_file, "parameters", ps, "states", st, "losses", losses, "config", ffm.config)
println("Checkpoint saved to: $weight_file")

# ---------------------------------------------------------------------------
# Visualise training curve
# ---------------------------------------------------------------------------
p1 = Plots.plot(1:length(losses), losses,
    yscale = :log10,
    xlabel = "Epoch", ylabel = "Loss (log scale)",
    title = "Reaction-Diffusion — Training Loss", legend = false, linewidth = 2)
display(p1)

println("\nDone!")

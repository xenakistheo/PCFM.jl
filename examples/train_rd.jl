"""
Training script for Functional Flow Matching on the 1D Reaction-Diffusion equation.

Data is generated once with a Strang-split solver and saved to HDF5.
The FFM model is then trained on random batches drawn from that file.

Grid: Nx=128 (cell-centred), nt=100 snapshots, domain [0,1].
Parameters: rho=0.01, nu=0.005. IC is a random sum of sinusoids normalised to [0,1].
BC: random Neumann fluxes gL ∈ [0, 0.05], gR ∈ [-0.05, 0].
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
datasets_dir  = joinpath(@__DIR__, "..", "datasets")
data_dir      = joinpath(datasets_dir, "data")
train_file    = joinpath(data_dir, "RD_neumann_train_nIC80_nBC80.h5")
weight_file   = joinpath(@__DIR__, "checkpoints", "ffm_rd_checkpoint.jld2")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
batch_size    = 32
nx            = 128      # Nx spatial points (cell-centred)
nt            = 100      # nt time snapshots
emb_channels  = 32
n_epochs      = 1000
force_retrain = false

# ---------------------------------------------------------------------------
println("=" ^ 60)
println("Reaction-Diffusion Equation — Functional Flow Matching")
println("=" ^ 60)

# 1. Generate dataset if not present
if !isfile(train_file)
    println("\n[1/4] Generating Reaction-Diffusion dataset (saved to $data_dir)...")
    include(joinpath(datasets_dir, "generate_RD1d_data.jl"))
    run_parallel(data_dir; N_ic=80, N_bc=80, seed=42, filename="RD_neumann_train")
    run_parallel(data_dir; N_ic=30, N_bc=30, seed=0,  filename="RD_neumann_test")
    println("  Done.")
else
    println("\n[1/4] Dataset found: $train_file")
end

# 2. Load training batch
println("\n[2/4] Loading training batch...")
u_data = load_rd_batch(train_file, batch_size)
println("  Data shape: $(size(u_data))")

# 3. Create model
println("\n[3/4] Creating FFM model...")
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

# 4. Train or load checkpoint
if isfile(weight_file) && !force_retrain
    println("\n[4/4] Loading checkpoint from: $weight_file")
    saved = JLD2.load(weight_file)
    device = ffm.config[:device]
    ps = saved["parameters"] |> device
    st = saved["states"] |> device
    losses = saved["losses"]
    println("  Loaded trained parameters and states")
else
    println("\n[4/4] Compiling and training for $n_epochs epochs...")
    compiled_funcs = PCFM.compile_functions(ffm, batch_size)
    losses, tstate = train_ffm!(ffm, u_data; compiled_funcs, epochs = n_epochs, verbose = true)
    println("\nFinal loss: $(losses[end])")

    ps = fmap(x -> x isa AbstractArray ? Array(x) : x, tstate.parameters)
    st = fmap(x -> x isa AbstractArray ? Array(x) : x, tstate.states)

    println("Saving checkpoint to: $weight_file")
    mkpath(dirname(weight_file))
    JLD2.save(weight_file, "parameters", ps, "states", st, "losses", losses, "config", ffm.config)
end

# ---------------------------------------------------------------------------
# Visualise training curve
# ---------------------------------------------------------------------------
if !isempty(losses)
    p1 = Plots.plot(1:length(losses), losses,
        yscale = :log10,
        xlabel = "Epoch", ylabel = "Loss (log scale)",
        title = "Reaction-Diffusion — Training Loss", legend = false, linewidth = 2)
    display(p1)
end

arr_data = Array(u_data)
p_data = [Plots.heatmap(arr_data[:, :, 1, i],
              title = "Training Sample $i", xlabel = "Time", ylabel = "Space", c = :viridis)
          for i in 1:min(2, batch_size)]
display(Plots.plot(p_data..., layout = (1, length(p_data)), size = (800, 300)))

println("\nDone!")

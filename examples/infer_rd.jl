"""
Load a trained FFM checkpoint and generate Reaction-Diffusion equation samples.

Run train_rd.jl first to produce the checkpoint.
"""

using PCFM
using Reactant, Lux
using JLD2
using Plots
using Random

Random.seed!(42)

# ---------------------------------------------------------------------------
# Paths and config — must match what was used during training
# ---------------------------------------------------------------------------
weight_file = joinpath(@__DIR__, "checkpoints", "ffm_rd_checkpoint.jld2")

n_samples    = 8
n_steps      = 100   # Euler steps in the flow
nx           = 128
nt           = 100
emb_channels = 32

# ---------------------------------------------------------------------------
println("=" ^ 60)
println("Reaction-Diffusion Equation — Load checkpoint and sample")
println("=" ^ 60)

if !isfile(weight_file)
    error("Checkpoint not found at $weight_file.\nRun examples/train_rd.jl first.")
end

# 1. Reconstruct model (architecture must match training)
println("\n[1/3] Reconstructing model...")
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

# 2. Load parameters and states from checkpoint
println("\n[2/3] Loading checkpoint from: $weight_file")
saved  = JLD2.load(weight_file)
device = ffm.config[:device]

ps = saved["parameters"] |> device
st = saved["states"]     |> device

n_epochs_trained = length(saved["losses"])
final_loss       = saved["losses"][end]
println("  Loaded $n_epochs_trained training epochs")
println("  Final training loss: $final_loss")

tstate_inf = (parameters = ps, states = st)

# 3. Compile and sample
println("\n[3/3] Compiling and generating $n_samples samples...")
compiled_funcs = PCFM.compile_functions(ffm, n_samples)

@time samples = sample_ffm(ffm, tstate_inf, n_samples, n_steps;
    compiled_funcs = compiled_funcs, verbose = true)

println("  Samples shape: $(size(samples))  (nx, nt, 1, n_samples)")

# ---------------------------------------------------------------------------
# Visualise
# ---------------------------------------------------------------------------
arr = Array(samples)
plots = [Plots.heatmap(arr[:, :, 1, i],
             title = "Sample $i", xlabel = "Time", ylabel = "Space", c = :viridis)
         for i in 1:min(4, n_samples)]
display(Plots.plot(plots..., layout = (2, 2), size = (900, 600)))

println("\nDone!")

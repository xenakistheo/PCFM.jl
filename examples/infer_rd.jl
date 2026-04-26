"""
Load a trained FFM checkpoint and generate Reaction-Diffusion equation samples
using physics-constrained flow matching (PCFM).

Run train_rd.jl first to produce the checkpoint.
"""

using PCFM
using Reactant, Lux
using JLD2
using Plots
using Random
using ExaModels, MadNLP

Random.seed!(42)

# ---------------------------------------------------------------------------
# Paths and config — must match what was used during training
# ---------------------------------------------------------------------------
weight_file = joinpath(@__DIR__, "checkpoints", "ffm_rd_checkpoint.jld2")

n_samples    = 8
n_steps      = 100
nx           = 128
nt           = 100
emb_channels = 32

# ---------------------------------------------------------------------------
# Initial condition (fixed seed for reproducibility)
# ---------------------------------------------------------------------------
function generate_ic(xc; k_tot=3, num_choice_k=2)
    selected = rand(1:k_tot, num_choice_k)
    onehot = zeros(Int, k_tot)
    for j in selected
        onehot[j] += 1
    end
    kk = 2π .* (1:k_tot) .* onehot ./ (xc[end] - xc[1])
    amp = rand(k_tot, 1)
    phs = 2π .* rand(k_tot, 1)
    u = vec(sum(amp .* sin.(kk .* xc' .+ phs), dims=1))
    if rand() < 0.1
        u = abs.(u)
    end
    u .*= rand([-1, 1])
    if rand() < 0.1
        xL_m = rand() * 0.35 + 0.1
        xR_m = rand() * 0.35 + 0.55
        trns = 0.01
        mask = 0.5 .* (tanh.((xc .- xL_m) ./ trns) .- tanh.((xc .- xR_m) ./ trns))
        u .*= mask
    end
    u .-= minimum(u)
    if maximum(u) > 0
        u ./= maximum(u)
    end
    return u
end

x_rd = Float32.(range(0f0, 1f0, length=nx))
Random.seed!(0)
u0_fixed = Float32.(generate_ic(collect(x_rd)))
Random.seed!(42)

dx_rd = x_rd[2] - x_rd[1]
IC_func_rd = x -> u0_fixed[clamp(round(Int, (x - x_rd[1]) / dx_rd) + 1, 1, nx)]

# ---------------------------------------------------------------------------
println("=" ^ 60)
println("Reaction-Diffusion Equation — PCFM sampling")
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

@time samples = sample_pcfm(ffm, tstate_inf, n_samples, n_steps, rd_constraints!;
    domain = (x_start=0f0, x_end=1f0, t_start=0f0, t_end=1f0),
    IC_func = IC_func_rd,
    constraint_parameters = (rho=0.01f0, nu=0.005f0),
    compiled_funcs = compiled_funcs,
    verbose = true)

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

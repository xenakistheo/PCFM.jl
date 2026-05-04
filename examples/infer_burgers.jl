"""
Load a trained FFM checkpoint and generate Burgers equation samples
using physics-constrained flow matching (PCFM).

Run train_burgers.jl first to produce the checkpoint.
"""

using PCFM
using Reactant, Lux
using JLD2
using Plots
using Random
using CUDA, KernelAbstractions
using ExaModels, MadNLP, MadNLPGPU
using JuMP, Ipopt
using BenchmarkTools

Random.seed!(42)

backend = CUDABackend()

# ---------------------------------------------------------------------------
# Paths and config — must match what was used during training
# ---------------------------------------------------------------------------
weight_file = joinpath(@__DIR__, "checkpoints", "ffm_burgers_checkpoint.jld2")

n_samples    = 8
n_steps      = 100
nx           = 101
nt           = 101
emb_channels = 32

# ---------------------------------------------------------------------------
# Initial condition and boundary conditions
# ---------------------------------------------------------------------------
const p_loc = 0.5f0   # sigmoid centre, in [0.2, 0.8] as in training
const eps_ic = 0.02f0

IC_func_burgers = x -> 1.0f0 / (1.0f0 + exp((x - p_loc) / eps_ic))

# Left Dirichlet BC — one value per sample, drawn from U[0,1] (training distribution)
left_bc_vals = rand(Float32, n_samples)

# ---------------------------------------------------------------------------
println("=" ^ 60)
println("Burgers Equation — PCFM sampling")
println("=" ^ 60)

if !isfile(weight_file)
    error("Checkpoint not found at $weight_file.\nRun examples/train_burgers.jl first.")
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

const burgers_domain = (x_start=0f0, x_end=1f0, t_start=0f0, t_end=1f0)
const burgers_params = (left_bc = left_bc_vals,)

# ExaModels, MadNLP, GPU
@btime sample_pcfm($ffm, $tstate_inf, $n_samples, $n_steps, burgers_constraints_BC_Mass!;
    domain = burgers_domain,
    IC_func = $IC_func_burgers,
    constraint_parameters = burgers_params,
    backend = backend,
    compiled_funcs = $compiled_funcs,
    verbose = true,
    mode = "exa");

# ExaModels, MadNLP, CPU
@btime sample_pcfm($ffm, $tstate_inf, $n_samples, $n_steps, burgers_constraints_BC_Mass!;
    domain = burgers_domain,
    IC_func = $IC_func_burgers,
    constraint_parameters = burgers_params,
    backend = CPU(),
    compiled_funcs = $compiled_funcs,
    verbose = true,
    mode = "exa");

# JuMP, MadNLP
@btime sample_pcfm($ffm, $tstate_inf, $n_samples, $n_steps, burgers_constraints_BC_Mass!;
    domain = burgers_domain,
    IC_func = $IC_func_burgers,
    constraint_parameters = burgers_params,
    backend = CPU(),
    compiled_funcs = $compiled_funcs,
    verbose = true,
    mode = "jump",
    optimizer = MadNLP.Optimizer);

# JuMP, Ipopt
@btime sample_pcfm($ffm, $tstate_inf, $n_samples, $n_steps, burgers_constraints_BC_Mass!;
    domain = burgers_domain,
    IC_func = $IC_func_burgers,
    constraint_parameters = burgers_params,
    backend = CPU(),
    compiled_funcs = $compiled_funcs,
    verbose = true,
    mode = "jump",
    optimizer = Ipopt.Optimizer);

# # ---------------------------------------------------------------------------
# # Visualise last result
# # ---------------------------------------------------------------------------
# samples = sample_pcfm(ffm, tstate_inf, n_samples, n_steps, burgers_constraints_BC_Mass!;
#     domain = burgers_domain,
#     IC_func = IC_func_burgers,
#     constraint_parameters = burgers_params,
#     compiled_funcs = compiled_funcs,
#     verbose = false)

# arr = Array(samples)
# plots = [Plots.heatmap(arr[:, :, 1, i],
#              title = "Sample $i", xlabel = "Time", ylabel = "Space", c = :viridis)
#          for i in 1:min(4, n_samples)]
# display(Plots.plot(plots..., layout = (2, 2), size = (900, 600)))

# println("\nDone!")

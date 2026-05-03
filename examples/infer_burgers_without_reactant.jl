"""
Load a trained FFM checkpoint and benchmark Burgers equation samples
using physics-constrained flow matching (PCFM), without Reactant.

Run train_burgers.jl first to produce the checkpoint.
"""

using PCFM
using ExaModels, MadNLP, MadNLPGPU
using Lux
using CUDA
using cuDNN
using KernelAbstractions
using JLD2, Functors
using JuMP
using Ipopt
using BenchmarkTools
using Random

include(joinpath(@__DIR__, "..", "optimisation", "plotUtils.jl"))

backend = CUDABackend()
dev_gpu = cu
dev_cpu = cpu_device
device  = dev_gpu

Random.seed!(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
batch_size   = 32
nx           = 101
nt           = 101
emb_channels = 32

weight_file = joinpath(@__DIR__, "checkpoints", "ffm_burgers_checkpoint.jld2")

t_range = (0.0f0, 1.0f0)

# Grid
x_grid = range(0.0f0, 1.0f0; length=nx)
dx     = Float32(x_grid[2] - x_grid[1])
dt     = 1.0f0 / (nt - 1)

# Initial condition: viscous Burgers sigmoid
const p_loc  = 0.5f0
const eps_ic = 0.02f0
IC_func_burgers = x -> 1.0f0 / (1.0f0 + exp((x - p_loc) / eps_ic))
u0_ic = Float32.(IC_func_burgers.(x_grid))

# ---------------------------------------------------------------------------
println("=" ^ 60)
println("Burgers Equation — Functional Flow Matching")
println("=" ^ 60)

# Create model
println("\n[1/3] Creating FFM model...")
ffm = FFM(
    nx = nx,
    nt = nt,
    emb_channels = emb_channels,
    hidden_channels = 64,
    proj_channels = 256,
    n_layers = 4,
    modes = (32, 32),
    device = dev_gpu
)
println("  Model created successfully")

# Load checkpoint
println("\n[2/3] Loading checkpoint from: $weight_file")
saved = JLD2.load(weight_file)
device = cu
ps = saved["parameters"] |> device
st = saved["states"]     |> device
println("  Loaded trained parameters and states")

_, st = Lux.setup(Random.default_rng(), ffm.model)
ps = ps |> device
st = st |> device

println("\n[3/3] Generating samples...")
n_samples = 32
tstate_inf = (parameters = ps, states = st)

# Per-sample left BC drawn from training distribution U[0,1]
left_bc_vals = rand(Float32, n_samples)

const burgers_domain = (x_start=0f0, x_end=1f0, t_start=0f0, t_end=1f0)
const burgers_params = (left_bc=left_bc_vals,)

@show backend

starting_noise = randn(Float32, nx, nt, 1, n_samples)

# ExaModels, MadNLP, GPU
@info "ExaModels, MadNLP, GPU"
@btime samples_exa_gpu = sample_pcfm($ffm, (parameters=$ps, states=$st),
                   $n_samples, 100, burgers_constraints!;
                   domain = burgers_domain,
                   IC_func = IC_func_burgers,
                   constraint_parameters = burgers_params,
                   backend = backend,
                   verbose = false,
                   mode = "exa",
                   initial_vals = $starting_noise)

# ExaModels, MadNLP, CPU
@info "ExaModels, MadNLP, CPU"
@btime samples_exa_cpu = sample_pcfm($ffm, (parameters=$ps, states=$st),
                   $n_samples, 100, burgers_constraints!;
                   domain = burgers_domain,
                   IC_func = IC_func_burgers,
                   constraint_parameters = burgers_params,
                   backend = CPU(),
                   verbose = false,
                   mode = "exa",
                   initial_vals = $starting_noise)

# JuMP, MadNLP
@info "JuMP, MadNLP"
@btime samples_jump_madnlp = sample_pcfm($ffm, (parameters=$ps, states=$st),
                   $n_samples, 100, burgers_constraints!;
                   domain = burgers_domain,
                   IC_func = IC_func_burgers,
                   constraint_parameters = burgers_params,
                   backend = CPU(),
                   verbose = false,
                   mode = "jump",
                   optimizer = MadNLP.Optimizer,
                   initial_vals = $starting_noise)

# JuMP, Ipopt
@info "JuMP, Ipopt"
@btime sample_pcfm($ffm, (parameters=$ps, states=$st),
                   $n_samples, 100, burgers_constraints!;
                   domain = burgers_domain,
                   IC_func = IC_func_burgers,
                   constraint_parameters = burgers_params,
                   backend = CPU(),
                   verbose = false,
                   mode = "jump",
                   optimizer = Ipopt.Optimizer,
                   initial_vals = $starting_noise)

# FFM (unconstrained)
@info "FFM"
@btime samples_ffm = sample_ffm($ffm, (parameters=$ps, states=$st), $n_samples, 100;
    verbose = false,
    initial_vals = $starting_noise)

##################
# Plot solutions

X = x_grid
T = range(t_range[1], t_range[2]; length=nt)
K = 1

fig_samples = plot_sample(K,
    [samples_exa_gpu, samples_exa_cpu, samples_jump_madnlp, samples_ffm],
    ["ExaGPU", "ExaCPU", "JuMP", "FFM"])
save("burgers_samples.png", fig_samples)

function ic_violation(u, params)
    nx, nt = params[1], params[2]
    return [sum(abs(u[i, j] - u[i, 1]) for i in 1:nx) for j in 1:nt]
end

fig_constraint = plot_constraint_violation(K,
    [samples_exa_gpu, samples_exa_cpu, samples_jump_madnlp, samples_ffm],
    ic_violation,
    ["ExaGPU", "ExaCPU", "JuMP", "FFM"];
    constraint_params=(nx, nt, dx, dt))
save("burgers_constraint_violation.png", fig_constraint)

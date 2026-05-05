"""
Load a trained FFM checkpoint and benchmark Reaction-Diffusion equation samples
using physics-constrained flow matching (PCFM), without Reactant.

Run train_rd.jl first to produce the checkpoint.

Note: Script does not use Reactant
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

# include(joinpath(@__DIR__, "..", "optimisation", "plotUtils.jl"))

backend = CUDABackend()
dev_gpu = cu
dev_cpu = cpu_device
device  = dev_gpu

Random.seed!(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
batch_size   = 32
nx           = 64
nt           = 100
emb_channels = 32

weight_file = joinpath(@__DIR__, "checkpoints", "ffm_rd_checkpoint_nx64.jld2")

t_range = (0.0f0, 1.0f0)

# Grid
x_grid = range(0.0f0, 1.0f0; length=nx)
dx     = Float32(x_grid[2] - x_grid[1])
dt     = 1.0f0 / (nt - 1)

# Initial condition: random spectral IC (fixed seed for reproducibility)
function generate_ic(xc; k_tot=3, num_choice_k=2)
    selected = rand(1:k_tot, num_choice_k)
    onehot = zeros(Int, k_tot)
    for j in selected; onehot[j] += 1; end
    kk  = 2π .* (1:k_tot) .* onehot ./ (xc[end] - xc[1])
    amp = rand(k_tot, 1)
    phs = 2π .* rand(k_tot, 1)
    u   = vec(sum(amp .* sin.(kk .* xc' .+ phs), dims=1))
    if rand() < 0.1; u = abs.(u); end
    u .*= rand([-1, 1])
    if rand() < 0.1
        xL_m = rand() * 0.35 + 0.1
        xR_m = rand() * 0.35 + 0.55
        trns = 0.01
        mask = 0.5 .* (tanh.((xc .- xL_m) ./ trns) .- tanh.((xc .- xR_m) ./ trns))
        u .*= mask
    end
    u .-= minimum(u)
    if maximum(u) > 0; u ./= maximum(u); end
    return u
end

Random.seed!(0)
u0_fixed = Float32.(generate_ic(collect(x_grid)))
Random.seed!(42)

IC_func_rd = x -> u0_fixed[clamp(round(Int, (x - x_grid[1]) / dx) + 1, 1, nx)]

const rd_domain = (x_start=0f0, x_end=1f0, t_start=0f0, t_end=1f0)
const rd_params = (rho=0.01f0, nu=0.005f0)

# ---------------------------------------------------------------------------
println("=" ^ 60)
println("Reaction-Diffusion Equation — Functional Flow Matching")
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

@show backend

starting_noise = randn(Float32, nx, nt, 1, n_samples)


begin
    @info "ExaModels, MadNLP, GPU"
    @btime sample_pcfm($ffm, (parameters=$ps, states=$st),
                       $n_samples, 100, rd_constraints_2!;
                       domain = rd_domain,
                       IC_func = IC_func_rd,
                       constraint_parameters = rd_params,
                       backend = backend,
                       verbose = false,
                       mode = "exa",
                       initial_vals = $starting_noise)
    flush(stdout)

    @info "ExaModels, MadNLP, CPU"
    @btime sample_pcfm($ffm, (parameters=$ps, states=$st),
                       $n_samples, 100, rd_constraints_2!;
                       domain = rd_domain,
                       IC_func = IC_func_rd,
                       constraint_parameters = rd_params,
                       backend = CPU(),
                       verbose = false,
                       mode = "exa",
                       initial_vals = $starting_noise)
    flush(stdout)

    @info "JuMP, MadNLP"
    @btime sample_pcfm($ffm, (parameters=$ps, states=$st),
                       $n_samples, 100, rd_constraints_2!;
                       domain = rd_domain,
                       IC_func = IC_func_rd,
                       constraint_parameters = rd_params,
                       backend = CPU(),
                       verbose = false,
                       mode = "jump",
                       optimizer = MadNLP.Optimizer,
                       initial_vals = $starting_noise)
    flush(stdout)

    @info "JuMP, Ipopt"
    @btime sample_pcfm($ffm, (parameters=$ps, states=$st),
                       $n_samples, 100, rd_constraints_2!;
                       domain = rd_domain,
                       IC_func = IC_func_rd,
                       constraint_parameters = rd_params,
                       backend = CPU(),
                       verbose = false,
                       mode = "jump",
                       optimizer = Ipopt.Optimizer,
                       initial_vals = $starting_noise)
    flush(stdout)

    @info "FFM"
    @btime sample_ffm($ffm, (parameters=$ps, states=$st), $n_samples, 100;
        verbose = false,
        initial_vals = $starting_noise)
    flush(stdout)
end

# Samples
begin
    @info "ExaModels, MadNLP, GPU"
    samples_exa_gpu = sample_pcfm(ffm, (parameters=ps, states=st),
                       n_samples, 100, rd_constraints_2!;
                       domain = rd_domain,
                       IC_func = IC_func_rd,
                       constraint_parameters = rd_params,
                       backend = backend,
                       verbose = true,
                       mode = "exa",
                       initial_vals = starting_noise)

    @info "ExaModels, MadNLP, CPU"
    samples_exa_cpu = sample_pcfm(ffm, (parameters=ps, states=st),
                       n_samples, 100, rd_constraints_2!;
                       domain = rd_domain,
                       IC_func = IC_func_rd,
                       constraint_parameters = rd_params,
                       backend = CPU(),
                       verbose = true,
                       mode = "exa",
                       initial_vals = starting_noise)

    @info "JuMP, MadNLP"
    samples_jump_madnlp = sample_pcfm(ffm, (parameters=ps, states=st),
                       n_samples, 100, rd_constraints_2!;
                       domain = rd_domain,
                       IC_func = IC_func_rd,
                       constraint_parameters = rd_params,
                       backend = CPU(),
                       verbose = true,
                       mode = "jump",
                       optimizer = MadNLP.Optimizer,
                       initial_vals = starting_noise)

    @info "JuMP, Ipopt"
    samples_jump_Ipopt = sample_pcfm(ffm, (parameters=ps, states=st),
                       n_samples, 100, rd_constraints_2!;
                       domain = rd_domain,
                       IC_func = IC_func_rd,
                       constraint_parameters = rd_params,
                       backend = CPU(),
                       verbose = true,
                       mode = "jump",
                       optimizer = Ipopt.Optimizer,
                       initial_vals = starting_noise)

    @info "FFM"
    samples_ffm = sample_ffm(ffm, (parameters=ps, states=st), n_samples, 100;
        verbose = false,
        initial_vals = starting_noise)
end
samples_ffm = Array(samples_ffm)

##################
# Plot solutions

# X = x_grid
# T = range(t_range[1], t_range[2]; length=nt)
# K = 1

# fig_samples = plot_sample(K,
#     [samples_exa_cpu, samples_jump_madnlp, samples_ffm],
#     ["ExaCPU", "JuMP", "FFM"])
# save("rd_samples.png", fig_samples)

# function ic_violation(u, params)
#     nx, nt = params[1], params[2]
#     return [sum(abs(u[i, j] - u[i, 1]) for i in 1:nx) for j in 1:nt]
# end

# fig_constraint = plot_constraint_violation(K,
#     [samples_exa_cpu, samples_jump_madnlp, samples_ffm],
#     ic_violation,
#     ["ExaCPU", "JuMP", "FFM"];
#     constraint_params=(nx, nt, dx, dt))
# save("rd_constraint_violation.png", fig_constraint)

# Save samples
JLD2.save("samples_rd.jld2",
    "samples_exa_gpu",     samples_exa_gpu,
    "samples_exa_cpu",     samples_exa_cpu,
    "samples_jump_madnlp", samples_jump_madnlp,
    "samples_ffm",         samples_ffm,
    "u0_fixed",            u0_fixed,
    "rd_params",           rd_params)

# Load samples
# data = JLD2.load("samples_rd.jld2")
# samples_exa_gpu     = data["samples_exa_gpu"]
# samples_exa_cpu     = data["samples_exa_cpu"]
# samples_jump_madnlp = data["samples_jump_madnlp"]
# samples_ffm         = data["samples_ffm"]

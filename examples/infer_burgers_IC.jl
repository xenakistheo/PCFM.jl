"""
Load a trained FFM checkpoint and benchmark Burgers equation samples
using physics-constrained flow matching (PCFM), without Reactant.

Run train_burgers.jl first to produce the checkpoint.

This is one of two versions of burgers-inference. 
Constraints outlined by D.7 - IC/Mass/Flux Constraints

Note: Script does not use Reactant
"""
# @show "Show: starting script"
@info "Info: starting script"
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
using HDF5


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

# Initial condition: viscous Burgers sigmoid - NOT USED 
const p_loc  = 0.5f0
const eps_ic = 0.02f0
IC_func_burgers = x -> 1.0f0 / (1.0f0 + exp((x - p_loc) / eps_ic))
u0_ic = Float32.(IC_func_burgers.(x_grid)) # Not Used!!!!!

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
const burgers_ic_flux_params = (k=5, eps=1f-6)

@show backend

starting_noise = randn(Float32, nx, nt, 1, n_samples)

# Benchmarks
# begin
    # @info "ExaModels, MadNLP, GPU"
    # @btime sample_pcfm($ffm, (parameters=$ps, states=$st),
    #                    $n_samples, 100, burgers_constraints_IC_Mass_Flux!;
    #                    domain = burgers_domain,
    #                    IC_func = IC_func_burgers,
    #                    constraint_parameters = burgers_ic_flux_params,
    #                    backend = backend,
    #                    verbose = false,
    #                    mode = "exa",
    #                    initial_vals = $starting_noise)
    # flush(stdout)

    # @info "ExaModels, MadNLP, CPU"
    # @btime sample_pcfm($ffm, (parameters=$ps, states=$st),
    #                    $n_samples, 100, burgers_constraints_IC_Mass_Flux!;
    #                    domain = burgers_domain,
    #                    IC_func = IC_func_burgers,
    #                    constraint_parameters = burgers_ic_flux_params,
    #                    backend = CPU(),
    #                    verbose = true,
    #                    mode = "exa",
    #                    initial_vals = $starting_noise)
    # # flush(stdout)

    # @info "JuMP, MadNLP"
    # @btime sample_pcfm($ffm, (parameters=$ps, states=$st),
    #                    $n_samples, 100, burgers_constraints_IC_Mass_Flux!;
    #                    domain = burgers_domain,
    #                    IC_func = IC_func_burgers,
    #                    constraint_parameters = burgers_ic_flux_params,
    #                    backend = CPU(),
    #                    verbose = true,
    #                    mode = "jump",
    #                    optimizer = MadNLP.Optimizer,
    #                    initial_vals = $starting_noise)
    # # flush(stdout)

    # @info "JuMP, Ipopt"
    # @btime sample_pcfm($ffm, (parameters=$ps, states=$st),
    #                    $n_samples, 100, burgers_constraints_IC_Mass_Flux!;
    #                    domain = burgers_domain,
    #                    IC_func = IC_func_burgers,
    #                    constraint_parameters = burgers_ic_flux_params,
    #                    backend = CPU(),
    #                    verbose = true,
    #                    mode = "jump",
    #                    optimizer = Ipopt.Optimizer,
    #                    initial_vals = $starting_noise)
    # # flush(stdout)

    # @info "FFM"
    # @btime sample_ffm($ffm, (parameters=$ps, states=$st), $n_samples, 100;
    #     verbose = false,
    #     initial_vals = $starting_noise)
    # flush(stdout)
# end

# Samples
begin
    @info "ExaModels, MadNLP, GPU"
    @time samples_exa_gpu = sample_pcfm(ffm, (parameters=ps, states=st),
                       n_samples, 100, burgers_constraints_IC_Mass_Flux!;
                       domain = burgers_domain,
                       IC_func = IC_func_burgers,
                       constraint_parameters = burgers_ic_flux_params,
                       backend = backend,
                       verbose = true,
                       mode = "exa",
                       initial_vals = starting_noise)

    @info "ExaModels, MadNLP, CPU"
    @time samples_exa_cpu = sample_pcfm(ffm, (parameters=ps, states=st),
                       n_samples, 100, burgers_constraints_IC_Mass_Flux!;
                       domain = burgers_domain,
                       IC_func = IC_func_burgers,
                       constraint_parameters = burgers_ic_flux_params,
                       backend = CPU(),
                       verbose = true,
                       mode = "exa",
                       initial_vals = starting_noise)

    @info "JuMP, MadNLP"
    @time samples_jump_madnlp = sample_pcfm(ffm, (parameters=ps, states=st),
                       n_samples, 100, burgers_constraints_IC_Mass_Flux!;
                       domain = burgers_domain,
                       IC_func = IC_func_burgers,
                       constraint_parameters = burgers_ic_flux_params,
                       backend = CPU(),
                       verbose = true,
                       mode = "jump",
                       optimizer = MadNLP.Optimizer,
                       initial_vals = starting_noise)

    @info "JuMP, Ipopt"
    @time samples_jump_ipopt = sample_pcfm(ffm, (parameters=ps, states=st),
                       n_samples, 100, burgers_constraints_IC_Mass_Flux!;
                       domain = burgers_domain,
                       IC_func = IC_func_burgers,
                       constraint_parameters = burgers_ic_flux_params,
                       backend = CPU(),
                       verbose = true,
                       mode = "jump",
                       optimizer = Ipopt.Optimizer,
                       initial_vals = starting_noise)

    @info "FFM"
    @time samples_ffm = sample_ffm(ffm, (parameters=ps, states=st), n_samples, 100;
        verbose = false,
        initial_vals = starting_noise)
end
samples_ffm = Array(samples_ffm)

##################
# Load reference solutions from the test dataset (ground truth)

test_data_file = joinpath(@__DIR__, "..", "datasets", "data", "burgers_test_nIC30_nBC30.h5")
ref_samples = zeros(Float32, nx, nt, 1, n_samples)

h5open(test_data_file, "r") do f
    p_locs = read(f["ic"])                          # (N_ic,) — sigmoid p_loc values
    nt_h5, nx_h5, N_bc, N_ic = size(f["u"])         # Julia reversed dims

    # Find the stored IC closest to p_loc = 0.5 (the fixed inference IC)
    _, i_ic = findmin(abs.(p_locs .- 0.5f0))
    @info "Reference IC: p_loc = $(round(p_locs[i_ic]; digits=4)) (index $i_ic / $N_ic)"

    n_load = min(n_samples, N_bc)
    for i_bc in 1:n_load
        arr = Float32.(f["u"][:, :, i_bc, i_ic])    # (nt+1, nx+1) in Julia HDF5 ordering
        ref_samples[:, :, 1, i_bc] = permutedims(arr, (2, 1))  # → (nx+1, nt+1)
    end
end

##################
# Plot solutions

# X = x_grid
# T = range(t_range[1], t_range[2]; length=nt)
# K = 1

# fig_samples = plot_sample(K,
#     [ref_samples, samples_exa_gpu, samples_exa_cpu, samples_jump_madnlp, samples_ffm],
#     ["Reference", "ExaGPU", "ExaCPU", "JuMP", "FFM"])
# save("burgers_samples.png", fig_samples)

# function ic_violation(u, params)
#     nx, nt = params[1], params[2]
#     return [sum(abs(u[i, j] - u[i, 1]) for i in 1:nx) for j in 1:nt]
# end

# fig_constraint = plot_constraint_violation(K,
#     [samples_exa_gpu, samples_exa_cpu, samples_jump_madnlp, samples_ffm],
#     ic_violation,
#     ["ExaGPU", "ExaCPU", "JuMP", "FFM"];
#     constraint_params=(nx, nt, dx, dt))
# save("burgers_constraint_violation.png", fig_constraint)

# Save samples
JLD2.save("samples_burgers_IC.jld2",
    "ref_samples",         ref_samples,
    "samples_exa_gpu",     samples_exa_gpu,
    "samples_exa_cpu",     samples_exa_cpu,
    "samples_jump_madnlp", samples_jump_madnlp,
    "samples_ffm",         samples_ffm)

# Load samples
# data = JLD2.load("samples_burgers_IC.jld2")
# ref_samples         = data["ref_samples"]
# samples_exa_gpu     = data["samples_exa_gpu"]
# samples_exa_cpu     = data["samples_exa_cpu"]
# samples_jump_madnlp = data["samples_jump_madnlp"]
# samples_ffm         = data["samples_ffm"]

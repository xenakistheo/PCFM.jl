"""
Load a trained FFM checkpoint and benchmark Burgers equation samples
using physics-constrained flow matching (PCFM), without Reactant.

Run train_burgers.jl first to produce the checkpoint.

This is one of two versions of burgers-inference. 
Constraints outlined by D.6 

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
using HDF5

Random.seed!(42)

backend = CUDABackend()
dev_gpu = cu
dev_cpu = cpu_device
device  = dev_gpu


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
nx           = 101
nt           = 101
emb_channels = 32
n_samples    = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 32

SAMPLES_PATH = length(ARGS) >= 2 ? ARGS[2] : "samples_burgers_BC.jld2"
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
u_0_ic = Float32.(IC_func_burgers.(x_grid))

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


ps = saved["parameters"] |> device
_, st = Lux.setup(Random.default_rng(), ffm.model)
st = st |> device
println("  Loaded trained parameters and states")


tstate_inf = (parameters = ps, states = st)

# Per-sample left BC drawn from training distribution U[0,1]
left_bc_vals = rand(Float32, n_samples)

const burgers_domain = (x_start=0f0, x_end=1f0, t_start=0f0, t_end=1f0)
const burgers_params = (left_bc=left_bc_vals,)


# ---------------------------------------------------------------------------
# Constraint data  (u_L defaults to IC value at x=0 ≈ 1.0) for LBFGS and IPNewton solvers (not needed for ExaModels or JuMP)
# ---------------------------------------------------------------------------
constraint_data = make_constraint_data(u_0_ic, nx, nt, n_samples; dx=dx)


@show backend


println("\n[3/3] Generating samples...")
starting_noise = randn(Float32, nx, nt, 1, n_samples);

# Samples
@info "ExaModels, MadNLP, GPU"
@time samples_exa_gpu = sample_pcfm(ffm, (parameters=ps, states=st),
                    n_samples, 100, burgers_constraints_BC_Mass!;
                    domain = burgers_domain,
                    IC_func = IC_func_burgers,
                    constraint_parameters = burgers_params,
                    backend = backend,
                    verbose = false,
                    mode = "exa",
                    initial_vals = starting_noise)

@info "ExaModels, MadNLP, CPU"
@time samples_exa_cpu = sample_pcfm(ffm, (parameters=ps, states=st),
                    n_samples, 100, burgers_constraints_BC_Mass!;
                    domain = burgers_domain,
                    IC_func = IC_func_burgers,
                    constraint_parameters = burgers_params,
                    backend = CPU(),
                    verbose = false,
                    mode = "exa",
                    initial_vals = starting_noise)

@info "JuMP, MadNLP"
@time samples_jump_madnlp = sample_pcfm(ffm, (parameters=ps, states=st),
                    n_samples, 100, burgers_constraints_BC_Mass!;
                    domain = burgers_domain,
                    IC_func = IC_func_burgers,
                    constraint_parameters = burgers_params,
                    backend = CPU(),
                    verbose = false,
                    mode = "jump",
                    optimizer = MadNLP.Optimizer,
                    initial_vals = starting_noise)

@info "JuMP, Ipopt"
@time samples_jump_ipopt = sample_pcfm(ffm, (parameters=ps, states=st),
                    n_samples, 100, burgers_constraints_BC_Mass!;
                    domain = burgers_domain,
                    IC_func = IC_func_burgers,
                    constraint_parameters = burgers_params,
                    backend = CPU(),
                    verbose = false,
                    mode = "jump",
                    optimizer = Ipopt.Optimizer,
                    initial_vals = starting_noise)

@info "LBFGS"
@time samples_lbfgs = sample_pcfm(ffm, (parameters=ps, states=st),
                    n_samples, 100,
                    BurgersBCMassSolver(),
                    constraint_data;
                    verbose=true)

# @info "IPNewton"
# @time samples_ipnewton = sample_pcfm(ffm, (parameters=ps, states=st),
#                     n_samples, 100,
#                     BurgersBCMassIPSolver(),
#                     constraint_data;
#                     verbose=true)

# @info "FFM"
# samples_ffm = sample_ffm(ffm, (parameters=ps, states=st), n_samples, 100;
#     verbose = false,
#     initial_vals = starting_noise)

# samples_ffm = Array(samples_ffm)

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

# Save samples
JLD2.save(SAMPLES_PATH,
    "ref_samples",         ref_samples,
    "samples_exa_gpu",     samples_exa_gpu,
    "samples_exa_cpu",     samples_exa_cpu,
    "samples_jump_madnlp", samples_jump_madnlp,
    "samples_jump_ipopt",  samples_jump_ipopt,
    "samples_lbfgs",        samples_lbfgs
    # "samples_ipnewton",    samples_ipnewton,
    # "samples_ffm",         samples_ffm
)

# Load samples
# data = JLD2.load("samples_burgers_BC.jld2")
# ref_samples         = data["ref_samples"]
# samples_exa_gpu     = data["samples_exa_gpu"]
# samples_exa_cpu     = data["samples_exa_cpu"]
# samples_jump_madnlp = data["samples_jump_madnlp"]
# samples_jump_ipopt  = data["samples_jump_ipopt"]
# samples_lbfgs        = data["samples_lbfgs"]
# samples_ipnewton    = data["samples_ipnewton"]
# samples_ffm         = data["samples_ffm"]

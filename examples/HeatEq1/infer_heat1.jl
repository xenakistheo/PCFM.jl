"""
Example script for sampling from a Functional Flow Matching model
on the 1D heat (diffusion) equation.

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


backend = CUDABackend()
backend isa GPU

dev_gpu = cu
dev_cpu = cpu_device

device = dev_gpu

# Set random seed
using Random
Random.seed!(1234)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
nx           = 100          # Spatial resolution
nt           = 100          # Temporal resolution
emb_channels = 32
force_retrain = false

# Output path 
SAMPLES_PATH = length(ARGS) >= 2 ? ARGS[2] : "samples_heat_1.jld2"
n_samples    = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 32

# Checkpoint path
weight_file = joinpath(@__DIR__, "checkpoints", "ffm_heat_1_checkpoint.jld2")

# Data generation parameters
t_range    = (0.0f0, 1.0f0)

# Grid
x_grid = range(0.0f0, 2.0f0*Float32(π); length = nx)
dx     = Float32(x_grid[2] - x_grid[1])


# Constraint params (passed through to heat_constraints!)
constraint_params = (Nx=nx, Nt=nt, dx=dx)

# ---------------------------------------------------------------------------
println("=" ^ 60)
println("Heat Equation — Functional Flow Matching")
println("=" ^ 60)


# 2. Create model
println("\n[2/5] Creating FFM model...")
ffm = FFM(
    nx = nx,
    nt = nt,
    emb_channels = emb_channels,
    hidden_channels = 64,
    proj_channels = 256,
    n_layers = 4,
    modes = (32, 32),
    device = device
)
println("  Model created successfully")

# 3. Load checkpoint
println("\n[3/5] Loading checkpoint from: $weight_file")
saved = JLD2.load(weight_file)


# Re-init Lux states for inference and move ps/st to device
ps = saved["parameters"] |> device
_, st = Lux.setup(Random.default_rng(), ffm.model)
st = st |> device
println("  Loaded trained parameters and states")

# ---------------------------------------------------------------------------
# Build constraint data  (IC = sin(x + π/4), same as infer_heat.jl)
# ---------------------------------------------------------------------------
u_0_ic = Float32.(sin.(x_grid .+ Float32(π)/4))   # (nx,)
constraint_data = make_constraint_data(u_0_ic, nx, nt, n_samples; dx=dx)


# ---------------------------------------------------------------------------
# 5. Generate samples
# ---------------------------------------------------------------------------
println("\n[5/5] Generating samples...")

n_samples = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 32

starting_noise = randn(Float32, nx, nt, 1, n_samples);




@info "ExaModels, MadNLP, GPU"
@time samples_exa_gpu = sample_pcfm(ffm, (parameters = ps, states = st),
                n_samples, 100, heat_constraints!;
                backend=backend,
                verbose = false,
                mode="exa", 
                initial_vals=starting_noise);



@info "ExaModels, MadNLP, CPU"
@time samples_exa_cpu = sample_pcfm(ffm, (parameters = ps, states = st),
                n_samples, 100, heat_constraints!;
                backend=CPU(),
                verbose = false,
                mode="exa", 
                initial_vals=starting_noise);



@info "JuMP, MadNLP"
@time samples_jump_madnlp = sample_pcfm(ffm, (parameters = ps, states = st),
                n_samples, 100, heat_constraints!;
                backend=CPU(),
                verbose = false,
                mode="jump",
                optimizer=MadNLP.Optimizer, 
                initial_vals=starting_noise);


@info "JuMP, Ipopt"
@time samples_jump_ipopt = sample_pcfm(ffm, (parameters = ps, states = st),
                n_samples, 100, heat_constraints!;
                backend=CPU(),
                verbose = false,
                mode="jump",
                optimizer=Ipopt.Optimizer,
                initial_vals=starting_noise);

@info "LBFGS"
@time samples_lbfgs = sample_pcfm(ffm, (parameters=ps, states=st),
                    n_samples, 100,
                    PenaltyLBFGSMassProjectionSolver(),
                    constraint_data;
                    verbose=true);

@info "IPNewton"
@time samples_ipnewton = sample_pcfm(ffm, (parameters=ps, states=st),
                    n_samples, 100,
                    IPMassProjectionSolver(),
                    constraint_data;
                    verbose=true);

# @info "FFM"
# samples_ffm = sample_ffm(ffm, (parameters = ps, states = st), n_samples, 100; 
#     verbose = false,
#     initial_vals=starting_noise);

# samples_ffm = Array(samples_ffm)




##################

# Compute Analytic Solution 
X = x_grid
T = range(t_range[1], t_range[2]; length = nt)
u_exact = exp.(-3 .* T') .* sin.(X .+ π/4)   # (nx, nt), analytical solution ν=3
u_analytic = similar(samples_exa_cpu)
u_analytic[:,:, 1, 1] = u_exact
u_analytic




# Save samples
JLD2.save(SAMPLES_PATH,
    "samples_exa_gpu",    samples_exa_gpu,
    "samples_exa_cpu",    samples_exa_cpu,
    "samples_jump_madnlp", samples_jump_madnlp,
    "samples_jump_ipopt", samples_jump_ipopt,
    "samples_lbfgs",    samples_lbfgs,
    "samples_ipnewton", samples_ipnewton,
    "u_analytic",         u_analytic)

# Load samples
# data = JLD2.load("samples_heat_1.jld2")
# samples_exa_gpu     = data["samples_exa_gpu"]
# samples_exa_cpu     = data["samples_exa_cpu"]
# samples_jump_madnlp = data["samples_jump_madnlp"]
# samples_jump_ipopt = data["samples_jump_ipopt"]
# samples_lbfgs       = data["samples_lbfgs"]
# samples_ipnewton    = data["samples_ipnewton"]
# u_analytic          = data["u_analytic"]


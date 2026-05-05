"""
Example script for sampling from a Functional Flow Matching model
on the 1D heat (diffusion) equation.

Note: Script does not use Reactant
"""

#S

using PCFM

using ExaModels, MadNLP, MadNLPGPU
# using Plots
using Lux
using CUDA
using cuDNN
using KernelAbstractions
using JLD2, Functors
using JuMP
using Ipopt
using BenchmarkTools
#using Reactant

include(joinpath(@__DIR__, "..", "optimisation", "plotUtils.jl"))


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
batch_size   = 32
nx           = 100          # Spatial resolution
nt           = 100          # Temporal resolution
emb_channels = 32
n_epochs     = 1000
force_retrain = false

# Checkpoint path
weight_file = joinpath(@__DIR__, "checkpoints", "ffm_heat_checkpoint.jld2")

# Data generation parameters
visc_range = (1.0f0, 5.0f0)
phi_range  = (0.0f0, Float32(π))
t_range    = (0.0f0, 1.0f0)

# Grid
x_grid = range(0.0f0, 2.0f0*Float32(π); length = nx)
dx     = Float32(x_grid[2] - x_grid[1])
dt     = 1.0f0 / (nt - 1)

# Initial condition: u(x, 0) = sin(x + π/4)
u0_ic = Float32.(sin.(x_grid .+ π/4))

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
    device = dev_gpu
)
println("  Model created successfully")

# 3. Load checkpoint

println("\n[3/5] Loading checkpoint from: $weight_file")
saved = JLD2.load(weight_file)
# device = ffm.config[:device]
device = cu
ps = saved["parameters"] |> device
st = saved["states"] |> device
losses = Float32[]
# compiled_funcs = PCFM.compile_functions(ffm, batch_size)
println("  Loaded trained parameters and states")


# Re-init Lux states for inference and move ps/st to device
# device = ffm.config[:device]
_, st = Lux.setup(Random.default_rng(), ffm.model)
ps = ps |> device
st = st |> device

# ---------------------------------------------------------------------------
# 5. Generate samples
# ---------------------------------------------------------------------------
println("\n[5/5] Generating samples...")
n_samples = 32
# sample_compiled_funcs = (n_samples == batch_size) ? compiled_funcs : PCFM.compile_functions(ffm, n_samples)
tstate_inf = (parameters = ps, states = st)



########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################

########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################

########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################


########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################


starting_noise = randn(Float32, nx, nt, 1, n_samples) 

begin 
# ExaModels, MadNLP, GPU
@info "ExaModels, MadNLP, GPU"
@btime sample_pcfm(ffm, (parameters = $ps, states = $st),
                   $n_samples, 100, heat_constraints!;
                   backend=backend,
                   verbose = false,
                   mode="exa", 
                   initial_vals=$starting_noise);




# # ExaModels, MadNLP, CPU
@info "ExaModels, MadNLP, CPU"
@btime sample_pcfm(ffm, (parameters = $ps, states = $st),
                   $n_samples, 100, heat_constraints!;
                   backend=CPU(),
                   verbose = false,
                   mode="exa", 
                   initial_vals=$starting_noise);



# #JuMP, MadNLP
@info "JuMP, MadNLP"
@btime sample_pcfm(ffm, (parameters = $ps, states = $st),
                   $n_samples, 100, heat_constraints!;
                   backend=CPU(),
                   verbose = false,
                   mode="jump",
                   optimizer=MadNLP.Optimizer, 
                   initial_vals=$starting_noise);



# #JuMP, Ipopt
@info "JuMP, Ipopt"
@btime sample_pcfm($ffm, (parameters = $ps, states = $st),
                   $n_samples, 100, heat_constraints!;
                   backend=CPU(),
                   verbose = false,
                   mode="jump",
                   optimizer=Ipopt.Optimizer,
                   initial_vals=$starting_noise);

# FFM
@info "FFM"
@btime sample_ffm(ffm, (parameters = $ps, states = $st), $n_samples, 100; 
    verbose = false,
    initial_vals=$starting_noise);

end 



begin 
    # ExaModels, MadNLP, GPU
@info "ExaModels, MadNLP, GPU"
samples_exa_gpu = sample_pcfm(ffm, (parameters = ps, states = st),
                   n_samples, 100, heat_constraints!;
                   backend=backend,
                   verbose = false,
                   mode="exa", 
                   initial_vals=starting_noise);




# # ExaModels, MadNLP, CPU
@info "ExaModels, MadNLP, CPU"
samples_exa_cpu = sample_pcfm(ffm, (parameters = ps, states = st),
                   n_samples, 100, heat_constraints!;
                   backend=CPU(),
                   verbose = false,
                   mode="exa", 
                   initial_vals=starting_noise);



# #JuMP, MadNLP
@info "JuMP, MadNLP"
samples_jump_madnlp = sample_pcfm(ffm, (parameters = ps, states = st),
                   n_samples, 100, heat_constraints!;
                   backend=CPU(),
                   verbose = false,
                   mode="jump",
                   optimizer=MadNLP.Optimizer, 
                   initial_vals=starting_noise);

# FFM
@info "FFM"
samples_ffm = sample_ffm(ffm, (parameters = ps, states = st), n_samples, 100; 
    verbose = false,
    initial_vals=starting_noise);
end 
samples_ffm = Array(samples_ffm)
##################
# Plot solutions to verify correctness 

X = x_grid
T = range(t_range[1], t_range[2]; length = nt)

# Compute Analytic Solution 
u_exact = exp.(-3 .* T') .* sin.(X .+ π/4)   # (nx, nt), analytical solution ν=3
u_analytic = similar(samples_exa_cpu)
u_analytic[:,:, 1, 1] = u_exact
u_analytic



K = 1

fig_samples = plot_sample(K, [u_analytic, samples_exa_gpu, samples_jump_madnlp, samples_ffm, samples_exa_cpu],
    ["Analytic", "ExaGPU new", "JuMP", "FFM", "ExaCPU"])


save("samples_heat.png", fig_samples)

function mass_constraint(u, params)
    Nx, Nt = params
    return [sum((u[i, j] - u[i,1]) for i in 1:(Nx-1)) for j in 1:Nt]
end

fig_constraint = plot_constraint_violation(K, [u_analytic, samples_exa_gpu, samples_jump_madnlp, samples_ffm, samples_exa_cpu],
    mass_constraint,
    ["Analytic", "ExaGPU new", "JuMP", "FFM", "ExaCPU"]
    ; constraint_params=(nx, nt, dx, dt))
save("constraint_violation_heat.png", fig_constraint)

# Save samples
JLD2.save("samples_heat.jld2",
    "samples_exa_gpu",    samples_exa_gpu,
    "samples_exa_cpu",    samples_exa_cpu,
    "samples_jump_madnlp", samples_jump_madnlp,
    "samples_ffm",        samples_ffm,
    "u_analytic",         u_analytic)

# Load samples
# data = JLD2.load("samples_heat.jld2")
# samples_exa_gpu     = data["samples_exa_gpu"]
# samples_exa_cpu     = data["samples_exa_cpu"]
# samples_jump_madnlp = data["samples_jump_madnlp"]
# samples_ffm         = data["samples_ffm"]
# u_analytic          = data["u_analytic"]


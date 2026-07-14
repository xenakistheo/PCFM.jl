# Load a trained FFM checkpoint and benchmark 2D Navier-Stokes samples
# using physics-constrained flow matching (PCFM).
#
# Run train/train_ns.jl first to produce the checkpoint.
#
# Note: Script does not use Reactant
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
using Statistics
using LinearAlgebra


backend = CUDABackend()
dev_gpu = cu
dev_cpu = cpu_device
device  = dev_gpu

Random.seed!(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
s            = 16     # spatial grid size (s × s), must match checkpoint
nt           = 50     # time steps, must match checkpoint
emb_channels = 32

# Output path
SAMPLES_PATH = length(ARGS) >= 1 ? ARGS[1] : "samples_ns.jld2"

# Checkpoint path
weight_file = joinpath(@__DIR__, "checkpoints", "ffm_ns_s16_checkpoint.jld2")

# Grid: periodic domain [0, 1) × [0, 1)
x_grid = range(0.0f0, 1.0f0; length=s+1)[1:end-1]
y_grid = range(0.0f0, 1.0f0; length=s+1)[1:end-1]
dx     = Float32(x_grid[2] - x_grid[1])
dy     = Float32(y_grid[2] - y_grid[1])
dt     = 49.0f0 / (nt - 1)   # T=49 over nt snapshots

# Initial condition: simple sinusoidal vorticity (Kolmogorov-like)
IC_func_ns = (x, y) -> sin(2f0 * Float32(π) * Float32(y))

# ---------------------------------------------------------------------------
println("=" ^ 60)
println("2D Navier-Stokes — Functional Flow Matching")
println("=" ^ 60)

# Create model
println("\n[1/3] Creating FFM model...")
ffm = FFM(
    spatial_size    = (s, s),
    nt              = nt,
    emb_channels    = emb_channels,
    hidden_channels = 64,
    proj_channels   = 256,
    n_layers        = 4,
    modes           = (4, 4, 12),
    device          = dev_gpu
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
n_samples = 2 # Hard Code number of samples for benchmarking. 
tstate_inf = (parameters = ps, states = st)

const ns_domain = (x_start=0f0, x_end=1f0, y_start=0f0, y_end=1f0, t_start=0f0, t_end=1f0)

@show backend

starting_noise = randn(Float32, s, s, nt, 1, n_samples)


# Samples

@info "ExaModels, MadNLP, GPU"
@time samples_exa_gpu = sample_pcfm_2d(ffm, (parameters=ps, states=st),
                    n_samples, 100, ns_enstrophy_constraints!;
                    domain = ns_domain,
                    IC_func = IC_func_ns,
                    backend = backend,
                    verbose = true,
                    mode = "exa",
                    initial_vals = starting_noise, 
                    solver_tol=1e-5)

@info "ExaModels, MadNLP, CPU"
@time samples_exa_cpu = sample_pcfm_2d(ffm, (parameters=ps, states=st),
                    n_samples, 100, ns_enstrophy_constraints!;
                    domain = ns_domain,
                    IC_func = IC_func_ns,
                    backend = CPU(),
                    verbose = true,
                    mode = "exa",
                    initial_vals = starting_noise, 
                    solver_tol=1e-5)

@info "JuMP, MadNLP"
@time samples_jump_madnlp = sample_pcfm_2d(ffm, (parameters=ps, states=st),
                    n_samples, 100, ns_enstrophy_constraints!;
                    domain = ns_domain,
                    IC_func = IC_func_ns,
                    backend = CPU(),
                    verbose = true,
                    mode = "jump",
                    optimizer = MadNLP.Optimizer,
                    initial_vals = starting_noise)

@info "JuMP, Ipopt"
@time samples_jump_ipopt = sample_pcfm_2d(ffm, (parameters=ps, states=st),
                    n_samples, 100, ns_enstrophy_constraints!;
                    domain = ns_domain,
                    IC_func = IC_func_ns,
                    backend = CPU(),
                    verbose = true,
                    mode = "jump",
                    optimizer = Ipopt.Optimizer,
                    initial_vals = starting_noise)

s  = 16
nt = 50
x_grid = range(0.0f0, 1.0f0; length=s)
y_grid = range(0.0f0, 1.0f0; length=s)
dx = Float32(x_grid[2] - x_grid[1])
dy = Float32(y_grid[2] - y_grid[1])

# Imposed IC: u0[i,j] = IC_func_ns(x_i, y_j) = sin(2π y_j), same for all samples
u0 = Float32.([sin(2f0 * Float32(π) * y) for x in x_grid, y in y_grid])  # (s, s)

# Constraint error as in pcfm.pdf Appendix J:
#   CE(τ) = (1/N) Σₙ ‖R_τ(û⁽ⁿ⁾)‖₂
# with residuals mirroring ns_enstrophy_constraints! in src/constraints.jl:
#   IC:        R[i,j] = u[i,j,1,s] - u0[i,j]                        for all i,j
#   Mass:      R[t-1] = ∑_{i,j} u[i,j,t,s] - ∑_{i,j} u0            for t in 2:nt (no dx*dy, as imposed)
#   Enstrophy: R[t-1] = ∑_{i,j} u[i,j,t,s]²*dx*dy - ∑ u0²*dx*dy    for t in 2:nt
function ns_enstrophy_constraint_errors(samples, u0, nt, dx, dy)
    u = ndims(samples) == 5 ? dropdims(samples, dims=4) : samples  # (nx, ny, nt, n_s)
    n_s = size(u, 4)

    M0 = sum(u0)
    E0 = sum(u0 .^ 2) * dx * dy

    ic_ce        = 0.0
    mass_ce      = 0.0
    enstrophy_ce = 0.0

    for s in 1:n_s
        us = u[:, :, :, s]  # (nx, ny, nt)

        r_ic        = vec(us[:, :, 1] .- u0)
        r_mass      = [sum(us[:, :, t]) - M0 for t in 2:nt]
        r_enstrophy = [sum(us[:, :, t] .^ 2) * dx * dy - E0 for t in 2:nt]

        ic_ce        += norm(r_ic)
        mass_ce      += norm(r_mass)
        enstrophy_ce += norm(r_enstrophy)
    end

    return ic_ce / n_s, mass_ce / n_s, enstrophy_ce / n_s
end

solver_names = ["ExaGPU", "ExaCPU", "JuMP_MadNLP", "JuMP_Ipopt"]
all_samples  = [samples_exa_gpu, samples_exa_cpu, samples_jump_madnlp, samples_jump_ipopt]
# solver_names = ["ExaGPU"]
# all_samples  = [samples_exa_gpu]

ces       = [ns_enstrophy_constraint_errors(s, u0, nt, dx, dy) for s in all_samples]
ic_vals   = [c[1] for c in ces]
mass_vals = [c[2] for c in ces]
enst_vals = [c[3] for c in ces]

println("Constraint error (L2 norm of residual per sample, averaged over samples — pcfm.pdf App. J):")
println(rpad("Solver", 20), rpad("IC", 14), rpad("Mass (CL)", 14), "Enstrophy")
for (name, ic, m, e) in zip(solver_names, ic_vals, mass_vals, enst_vals)
    println(rpad(name, 20),
            rpad(round(ic; sigdigits=4), 14),
            rpad(round(m;  sigdigits=4), 14),
            round(e; sigdigits=4))
end

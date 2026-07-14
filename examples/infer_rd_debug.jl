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
using Statistics
using LinearAlgebra

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


@info "ExaModels, MadNLP, GPU"
@time samples_exa_gpu = sample_pcfm(ffm, (parameters=ps, states=st),
                    n_samples, 100, rd_constraints_2!;
                    domain = rd_domain,
                    IC_func = IC_func_rd,
                    constraint_parameters = rd_params,
                    backend = backend,
                    verbose = true,
                    mode = "exa",
                    initial_vals = starting_noise)

@info "ExaModels, MadNLP, CPU"
@time samples_exa_cpu = sample_pcfm(ffm, (parameters=ps, states=st),
                    n_samples, 100, rd_constraints_2!;
                    domain = rd_domain,
                    IC_func = IC_func_rd,
                    constraint_parameters = rd_params,
                    backend = CPU(),
                    verbose = true,
                    mode = "exa",
                    initial_vals = starting_noise)

@info "JuMP, MadNLP"
@time samples_jump_madnlp = sample_pcfm(ffm, (parameters=ps, states=st),
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
@time samples_jump_ipopt = sample_pcfm(ffm, (parameters=ps, states=st),
                    n_samples, 100, rd_constraints_2!;
                    domain = rd_domain,
                    IC_func = IC_func_rd,
                    constraint_parameters = rd_params,
                    backend = CPU(),
                    verbose = true,
                    mode = "jump",
                    optimizer = Ipopt.Optimizer,
                    initial_vals = starting_noise)

###########

nt = 100          # Temporal resolution
dt = 0.01f0       # = 1/n_steps: sample_pcfm passes dt = 1/n_steps (100 steps) to the constraints

# Constraint error as in pcfm.pdf Appendix J:
#   CE(τ) = (1/N) Σₙ ‖R_τ(û⁽ⁿ⁾)‖₂
# with residuals mirroring rd_constraints_2! in src/constraints.jl — the constraint
# actually imposed by examples/infer_rd.jl (trapezoidal reaction + 1st-order boundary flux):
#   IC:   R[i]   = u[i,1,s] - u0[i]                                     for i in 1:nx
#   Mass: R[t-1] = M[t]-M[t-1] - 0.5*dt*rho*(S[t]+S[t-1])
#                  - 0.5*dt*(F[t]+F[t-1])                               for t in 2:nt
# dx and u0 are inferred from sample nx: u0_fixed applies only to nx=64 samples
# (IC residual is 0 for other resolutions, where u0 falls back to the first time step).
function rd_constraint_errors(samples, u0_fixed, rho, nu, dt)
    u = ndims(samples) == 4 ? dropdims(samples, dims=3) : samples  # (nx_s, nt_s, n_samples)
    nx_s, nt_s, n_samples = size(u)
    dx_s = 1.0f0 / (nx_s - 1)

    # Use u0_fixed if it matches nx_s, otherwise fall back to first time step (IC residual = 0)
    has_u0 = length(u0_fixed) == nx_s

    ic_ce   = 0.0
    mass_ce = 0.0

    for s in 1:n_samples
        us = u[:, :, s]
        u0 = has_u0 ? u0_fixed : us[:, 1]

        r_ic = us[:, 1] .- u0

        # 1st-order boundary flux, as in rd_constraints_2!
        flux_left(t_)  = -nu * (us[2, t_]    - us[1, t_])      / dx_s
        flux_right(t_) = -nu * (us[nx_s, t_] - us[nx_s-1, t_]) / dx_s

        r_mass = zeros(nt_s - 1)
        for t in 2:nt_s
            mass_diff = (sum(us[:, t]) - sum(us[:, t-1])) * dx_s
            reaction  = 0.5 * dt * rho * (
                sum(us[i, t]   * (1 - us[i, t])   for i in 1:nx_s) * dx_s +
                sum(us[i, t-1] * (1 - us[i, t-1]) for i in 1:nx_s) * dx_s
            )
            diffusion = 0.5 * dt * (
                (flux_left(t)   - flux_right(t)) +
                (flux_left(t-1) - flux_right(t-1))
            )
            r_mass[t-1] = mass_diff - reaction - diffusion
        end

        ic_ce   += norm(r_ic)
        mass_ce += norm(r_mass)
    end

    return ic_ce / n_samples, mass_ce / n_samples
end

solver_names = ["exa_gpu", "exa_cpu", "jump_madnlp", "jump_ipopt"]
all_samples  = [samples_exa_gpu, samples_exa_cpu, samples_jump_madnlp, samples_jump_ipopt]

ces       = [rd_constraint_errors(s, u0_fixed, rd_params.rho, rd_params.nu, dt) for s in all_samples]
ic_vals   = [c[1] for c in ces]
mass_vals = [c[2] for c in ces]

println("Constraint error (L2 norm of residual per sample, averaged over samples — pcfm.pdf App. J):")
println("Note: IC error is 0 for nx=100 solvers (u0_fixed is nx=64 only)")
println(rpad("Solver", 22), rpad("IC", 14), "Mass (CL)")
for (name, ic, m) in zip(solver_names, ic_vals, mass_vals)
    println(rpad(name, 22),
            rpad(round(ic; sigdigits=4), 14),
            round(m; sigdigits=4))
end

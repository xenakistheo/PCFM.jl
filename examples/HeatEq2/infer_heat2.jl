"""
Example script for sampling from a Functional Flow Matching model
on the 1D heat (diffusion) equation using the constraints defined 
as "Heat 2" 

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

Random.seed!(1234)

backend = CUDABackend()

dev_gpu = cu
dev_cpu = cpu_device

device = dev_gpu



# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
batch_size   = 32
nx           = 100          # Spatial resolution
nt           = 100          # Temporal resolution
emb_channels = 32
n_samples    = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 32

# Output path 
SAMPLES_PATH = length(ARGS) >= 2 ? ARGS[2] : "samples_heat_2.jld2"

# Checkpoint path
weight_file = joinpath(@__DIR__, "checkpoints", "ffm_heat_checkpoint.jld2")

# Data generation parameters
t_range    = (0.0f0, 1.0f0)

# Grid
x_grid = range(0.0f0, 2.0f0*Float32(π); length = nx)
dx     = Float32(x_grid[2] - x_grid[1])
dt     = 1.0f0 / (nt - 1)
dt_physics = 1.0f0 / (nt - 1)
const kappa = 0.01f0   # heat diffusivity 
k = 5

constraint_params = (; kappa = kappa, k = k)

u_0_ic = Float32.(sin.(x_grid .+ Float32(π)/4))   # (nx,)

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
# Constraint data for LBFGS and IPNewton solvers (not needed for ExaModels or JuMP)
# ---------------------------------------------------------------------------
constraint_data = make_constraint_data(u_0_ic, nx, nt, n_samples;
                                        dx=dx, dt_physics=dt_physics)


# ---------------------------------------------------------------------------
# Define constraint function for ExaModels and JuMP
# ---------------------------------------------------------------------------
tstate_inf = (parameters = ps, states = st)


function heat_constraints_IC_Mass_PDE_Energy!(
    model::Model, u, u0, nt, n_samples, grid_points, grid_spacing, dt, params=(;)
)
    nx = grid_points[1]
    dx = grid_spacing[1]

    κ = get(params, :kappa, 0.01)
    k = get(params, :k, nt - 1)
    k_eff = min(k, nt - 1)

    # --------------------------------------------------
    # 1. Initial condition: u(x,0) = u_IC(x)
    # --------------------------------------------------
    @constraint(model, [i in 1:nx, s in 1:n_samples],
        u[i, 1, s] == u0[i, 1, 1, s]
    )

    # --------------------------------------------------
    # 2. Constant mass:
    #    ∫ u(x,t) dx = ∫ u(x,0) dx
    #    t = 1 is excluded: that row equals dx * (sum of the IC rows), and the
    #    linearly dependent Jacobian stalls MadNLP's dual convergence.
    # --------------------------------------------------
    @constraint(model, [t in 2:nt, s in 1:n_samples],
        sum(u[i, t, s] for i in 1:nx) * dx ==
        sum(u0[i, 1, 1, s] for i in 1:nx) * dx
    )

    # --------------------------------------------------
    # 3. Local heat equation residual:
    #    (u[i,t+1]-u[i,t])/dt = κ*(u[i+1,t]-2u[i,t]+u[i-1,t])/dx^2
    # --------------------------------------------------
    @constraint(model, [t in 1:k_eff, i in 2:nx-1, s in 1:n_samples],
        (u[i, t+1, s] - u[i, t, s]) / dt -
        κ * (u[i+1, t, s] - 2*u[i, t, s] + u[i-1, t, s]) / dx^2
        == 0.0
    )

    # --------------------------------------------------
    # 4. Energy decay: E[t+1] ≤ E[t] over the whole trajectory.
    #    The continuum law dE/dt = -2κ||u_x||² cannot be imposed as an
    #    equality (or a lower bound) together with (1)-(3): those leave a
    #    single free boundary DOF per constrained step, and the reachable
    #    energy minimum sits O(dt²) above the continuum value, so an
    #    equality is infeasible and a rate bound is active exactly where
    #    its gradient degenerates (LICQ failure -> spurious MadNLP
    #    infeasibility exits). Monotone decay is feasible, non-degenerate,
    #    and automatically strict on the k_eff steps where (3) holds.
    # --------------------------------------------------
    @NLconstraint(model, [t in 1:nt-1, s in 1:n_samples],
        sum(u[i, t+1, s]^2 for i in 1:nx) * dx
        - sum(u[i, t, s]^2 for i in 1:nx) * dx
        <= 0.0
    )

    return nothing
end

function heat_constraints_IC_Mass_PDE_Energy!(
    core::ExaCore, u_flat, u0_flat, nt, n_samples,
    grid_points, grid_spacing, dt, params=(;); backend=CPU()
)
    nx = grid_points[1]
    dx = grid_spacing[1]

    κ = get(params, :kappa, 0.01)
    k = get(params, :k, nt - 1)
    k_eff = min(k, nt - 1)

    idx(i, t, s) = i + (t-1)*nx + (s-1)*nx*nt

    u0_param = parameter(core, u0_flat)

    # --------------------------------------------------
    # 1. Initial condition: u(i,1,s) = u0(i,s)
    # --------------------------------------------------
    constraint(core,
        (
            u_flat[idx(i, 1, s)] - u0_param[i, s]
            for i in 1:nx, s in 1:n_samples
        );
        lcon = KernelAbstractions.adapt(backend, zeros(nx * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros(nx * n_samples))
    )

    # --------------------------------------------------
    # 2. Constant mass:
    #    ∑ u[i,t,s] dx = ∑ u0[i,s] dx
    #    t = 1 is excluded: that row equals dx * (sum of the IC rows), and the
    #    linearly dependent Jacobian stalls MadNLP's dual convergence.
    # --------------------------------------------------
    ts_pairs = [(t, s) for t in 2:nt for s in 1:n_samples]

    constraint(core,
        (
            sum(u_flat[idx(i, d[1], d[2])] for i in 1:nx) * dx
            - sum(u0_param[i, d[2]] for i in 1:nx) * dx
            for d in ts_pairs
        );
        lcon = KernelAbstractions.adapt(backend, zeros((nt-1) * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros((nt-1) * n_samples))
    )

    # --------------------------------------------------
    # 3. Local heat equation residual:
    #    (u[i,t+1]-u[i,t])/dt - κDxx(u[i,t]) = 0
    # --------------------------------------------------
    tis_pairs = [(t, i, s) for t in 1:k_eff for i in 2:nx-1 for s in 1:n_samples]

    constraint(core,
        (
            (u_flat[idx(d[2], d[1]+1, d[3])] - u_flat[idx(d[2], d[1], d[3])]) / dt
            - κ * (
                u_flat[idx(d[2]+1, d[1], d[3])]
                - 2*u_flat[idx(d[2], d[1], d[3])]
                + u_flat[idx(d[2]-1, d[1], d[3])]
            ) / dx^2
            for d in tis_pairs
        );
        lcon = KernelAbstractions.adapt(backend, zeros(k_eff * (nx-2) * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros(k_eff * (nx-2) * n_samples))
    )

    # --------------------------------------------------
    # 4. Energy decay: E[t+1] ≤ E[t] over the whole trajectory.
    #    The continuum law dE/dt = -2κ||u_x||² cannot be imposed as an
    #    equality (or a lower bound) together with (1)-(3): those leave a
    #    single free boundary DOF per constrained step, and the reachable
    #    energy minimum sits O(dt²) above the continuum value, so an
    #    equality is infeasible and a rate bound is active exactly where
    #    its gradient degenerates (LICQ failure -> spurious MadNLP
    #    infeasibility exits). Monotone decay is feasible, non-degenerate,
    #    and automatically strict on the k_eff steps where (3) holds.
    # --------------------------------------------------
    ts_pairs_energy = [(t, s) for t in 1:nt-1 for s in 1:n_samples]

    constraint(core,
        (
            sum(u_flat[idx(i, d[1]+1, d[2])]^2 for i in 1:nx) * dx
            - sum(u_flat[idx(i, d[1], d[2])]^2 for i in 1:nx) * dx
            for d in ts_pairs_energy
        );
        lcon = KernelAbstractions.adapt(backend, fill(-Inf, (nt-1) * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros((nt-1) * n_samples))
    )

    return nothing
end


# ---------------------------------------------------------------------------
# 5. Generate samples
# ---------------------------------------------------------------------------
println("\n[5/5] Generating samples...")
starting_noise = randn(Float32, nx, nt, 1, n_samples);



@info "ExaModels, MadNLP, GPU"
@time samples_exa_gpu = sample_pcfm(ffm, (parameters = ps, states = st),
                n_samples, 100, heat_constraints_IC_Mass_PDE_Energy!;
                backend=backend,
                verbose = true,
                mode="exa", 
                constraint_parameters = constraint_params,
                initial_vals=starting_noise);



@info "ExaModels, MadNLP, CPU"
@time samples_exa_cpu = sample_pcfm(ffm, (parameters = ps, states = st),
                n_samples, 100, heat_constraints_IC_Mass_PDE_Energy!;
                backend=CPU(),
                verbose = true,
                mode="exa", 
                constraint_parameters = constraint_params,
                initial_vals=starting_noise);


@info "Heat PDE+Energy LBFGS (kappa=$kappa)"
@time samples_lbfgs = sample_pcfm(ffm, (parameters=ps, states=st),
                    n_samples, 100,
                    HeatICPDEEnergySolver(kappa=kappa),
                    constraint_data;
                    verbose=true)

# @info "Heat PDE+Energy IPNewton (kappa=$kappa)"
# @time samples_ipnewton = sample_pcfm(ffm, (parameters=ps, states=st),
#                     n_samples, 100,
#                     HeatICPDEEnergyIPSolver(kappa=kappa),
#                     constraint_data;
#                     verbose=true)


# # #JuMP, MadNLP
# @info "JuMP, MadNLP"
# @time samples_jump_madnlp = sample_pcfm(ffm, (parameters = ps, states = st),
#                 n_samples, 100, heat_constraints_IC_Mass_PDE_Energy!;
#                 backend=CPU(),
#                 verbose = true,
#                 mode="jump",
#                 optimizer=MadNLP.Optimizer, 
#                 constraint_parameters = constraint_params,
#                 initial_vals=starting_noise);

# @info "JuMP, Ipopt"
# @time samples_jump_ipopt = sample_pcfm(ffm, (parameters = ps, states = st),
#                 n_samples, 100, heat_constraints_IC_Mass_PDE_Energy!;
#                 backend=CPU(),
#                 verbose = true,
#                 mode="jump",
#                 optimizer=Ipopt.Optimizer, 
#                 constraint_parameters = constraint_params,
#                 initial_vals=starting_noise);

# FFM
# @info "FFM"
# @time samples_ffm = sample_ffm(ffm, (parameters = ps, states = st), n_samples, 100; 
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
    "samples_lbfgs",    samples_lbfgs,
    # "samples_ipnewton", samples_ipnewton,
    # "samples_jump_madnlp", samples_jump_madnlp,
    # "samples_jump_ipopt", samples_jump_ipopt,
    # "samples_ffm",        samples_ffm,
    "u_analytic",         u_analytic)

@info "Samples saved to $SAMPLES_PATH"


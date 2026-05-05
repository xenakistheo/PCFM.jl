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


k = 5
@show k 
# Constraint params (passed through to heat_constraints!)
constraint_params = (; kappa = 0.01, k = k)
# params = (; kappa = 0.01, k = 20)
# params = (; kappa = 0.01, k = nt - 1)

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
    # --------------------------------------------------
    @constraint(model, [t in 1:nt, s in 1:n_samples],
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
    # 4. Energy dissipation:
    #    ||u^{t+1}||² - ||u^t||² + 2κdt ||u_x^t||² = 0
    # --------------------------------------------------
    @NLconstraint(model, [t in 1:k_eff, s in 1:n_samples],
        sum(u[i, t+1, s]^2 for i in 1:nx) * dx
        - sum(u[i, t, s]^2 for i in 1:nx) * dx
        + 2 * κ * dt *
          sum(((u[i+1, t, s] - u[i, t, s]) / dx)^2 for i in 1:nx-1) * dx
        == 0.0
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
    # --------------------------------------------------
    ts_pairs = [(t, s) for t in 1:nt for s in 1:n_samples]

    constraint(core,
        (
            sum(u_flat[idx(i, d[1], d[2])] for i in 1:nx) * dx
            - sum(u0_param[i, d[2]] for i in 1:nx) * dx
            for d in ts_pairs
        );
        lcon = KernelAbstractions.adapt(backend, zeros(nt * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros(nt * n_samples))
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
    # 4. Energy dissipation:
    #    ||u^{t+1}||² - ||u^t||² + 2κdt ||u_x^t||² = 0
    # --------------------------------------------------
    ts_pairs_energy = [(t, s) for t in 1:k_eff for s in 1:n_samples]

    constraint(core,
        (
            sum(u_flat[idx(i, d[1]+1, d[2])]^2 for i in 1:nx) * dx
            - sum(u_flat[idx(i, d[1], d[2])]^2 for i in 1:nx) * dx
            + 2 * κ * dt *
              sum(
                  ((u_flat[idx(i+1, d[1], d[2])] - u_flat[idx(i, d[1], d[2])]) / dx)^2
                  for i in 1:nx-1
              ) * dx
            for d in ts_pairs_energy
        );
        lcon = KernelAbstractions.adapt(backend, zeros(k_eff * n_samples)),
        ucon = KernelAbstractions.adapt(backend, zeros(k_eff * n_samples))
    )

    return nothing
end


########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################


starting_noise = randn(Float32, nx, nt, 1, n_samples);

begin 
    # ExaModels, MadNLP, GPU
    @info "ExaModels, MadNLP, GPU"
    @btime sample_pcfm(ffm, (parameters = $ps, states = $st),
                    $n_samples, 100, heat_constraints_IC_Mass_PDE_Energy!;
                    backend=backend,
                    verbose = true,
                    mode="exa", 
                    constraint_parameters = constraint_params,
                    initial_vals=$starting_noise);




    # # ExaModels, MadNLP, CPU
    @info "ExaModels, MadNLP, CPU"
    @btime sample_pcfm(ffm, (parameters = $ps, states = $st),
                    $n_samples, 100, heat_constraints_IC_Mass_PDE_Energy!;
                    backend=CPU(),
                    verbose = true,
                    mode="exa",
                    constraint_parameters = constraint_params,
                    initial_vals=$starting_noise);



    # #JuMP, MadNLP
    @info "JuMP, MadNLP"
    @btime sample_pcfm(ffm, (parameters = $ps, states = $st),
                    $n_samples, 100, heat_constraints_IC_Mass_PDE_Energy!;
                    backend=CPU(),
                    verbose = true,
                    mode="jump",
                    optimizer=MadNLP.Optimizer,
                    constraint_parameters = constraint_params,
                    initial_vals=$starting_noise);



    # #JuMP, Ipopt
    @info "JuMP, Ipopt"
    @btime sample_pcfm($ffm, (parameters = $ps, states = $st),
                    $n_samples, 100, heat_constraints_IC_Mass_PDE_Energy!;
                    backend=CPU(),
                    verbose = true,
                    mode="jump",
                    optimizer=Ipopt.Optimizer,
                    constraint_parameters = constraint_params,
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
                    n_samples, 100, heat_constraints_IC_Mass_PDE_Energy!;
                    backend=backend,
                    verbose = false,
                    mode="exa", 
                    constraint_parameters = constraint_params,
                    initial_vals=starting_noise);




    # # ExaModels, MadNLP, CPU
    @info "ExaModels, MadNLP, CPU"
    samples_exa_cpu = sample_pcfm(ffm, (parameters = ps, states = st),
                    n_samples, 100, heat_constraints_IC_Mass_PDE_Energy!;
                    backend=CPU(),
                    verbose = false,
                    mode="exa", 
                    constraint_parameters = constraint_params,
                    initial_vals=starting_noise);



    # #JuMP, MadNLP
    @info "JuMP, MadNLP"
    samples_jump_madnlp = sample_pcfm(ffm, (parameters = ps, states = st),
                    n_samples, 100, heat_constraints_IC_Mass_PDE_Energy!;
                    backend=CPU(),
                    verbose = false,
                    mode="jump",
                    optimizer=MadNLP.Optimizer, 
                    constraint_parameters = constraint_params,
                    initial_vals=starting_noise);

    # FFM
    @info "FFM"
    samples_ffm = sample_ffm(ffm, (parameters = ps, states = st), n_samples, 100; 
        verbose = false,
        initial_vals=starting_noise);
end 
samples_ffm = Array(samples_ffm)


##################

# Compute Analytic Solution 
X = x_grid
T = range(t_range[1], t_range[2]; length = nt)
u_exact = exp.(-3 .* T') .* sin.(X .+ π/4)   # (nx, nt), analytical solution ν=3
u_analytic = similar(samples_exa_cpu)
u_analytic[:,:, 1, 1] = u_exact
u_analytic




# Save samples
JLD2.save("samples_heat_2.jld2",
    "samples_exa_gpu",    samples_exa_gpu,
    "samples_exa_cpu",    samples_exa_cpu,
    "samples_jump_madnlp", samples_jump_madnlp,
    "samples_ffm",        samples_ffm,
    "u_analytic",         u_analytic)

# Load samples
# data = JLD2.load("samples_heat_2.jld2")
# samples_exa_gpu     = data["samples_exa_gpu"]
# samples_exa_cpu     = data["samples_exa_cpu"]
# samples_jump_madnlp = data["samples_jump_madnlp"]
# samples_ffm         = data["samples_ffm"]
# u_analytic          = data["u_analytic"]


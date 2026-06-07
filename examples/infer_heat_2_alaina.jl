# Inference script for the 1D heat equation (IC + Mass + PDE + Energy) using
# Alaina's projection-based sample_pcfm (solve_projection interface).
#
# Mirrors infer_heat_2.jl: same model, same IC, same save format.
# Solvers: HeatICPDEEnergySolver (LBFGS) + HeatICPDEEnergyIPSolver (IPNewton)
#
# Sequential projection at each step k → k+1:
#   (a) PDE interior residual (explicit Euler, κ=0.01)
#   (b) Mass conservation: ∫ u dx = M₀
#   (c) Energy dissipation: ‖u_{k+1}‖²dx = ‖u_k‖²dx - 2κ dt ∫(∂u_k/∂x)²dx
# Model inference runs on GPU; projection solvers run on CPU.

using PCFM
using Lux
using CUDA
using cuDNN
using JLD2, Functors
using Random

Random.seed!(1234)

# ---------------------------------------------------------------------------
# Configuration  (must match the checkpoint)
# ---------------------------------------------------------------------------
nx           = 100
nt           = 100
emb_channels = 32

SAMPLES_PATH = length(ARGS) >= 2 ? ARGS[2] : "samples_heat_2_alaina.jld2"
n_samples    = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 32

weight_file = joinpath(@__DIR__, "checkpoints", "ffm_heat_checkpoint.jld2")

t_range    = (0.0f0, 1.0f0)
x_grid     = range(0.0f0, 2.0f0 * Float32(π); length=nx)
dx         = Float32(x_grid[2] - x_grid[1])
dt_physics = 1.0f0 / (nt - 1)

const kappa = 0.01f0   # heat diffusivity (matches infer_heat_2.jl)

u_0_ic = Float32.(sin.(x_grid .+ Float32(π)/4))   # (nx,)

# ---------------------------------------------------------------------------
println("=" ^ 60)
println("Heat-2 (PDE+Energy) — Functional Flow Matching (Alaina solvers)")
println("=" ^ 60)

println("\n[1/3] Creating FFM model...")
ffm = FFM(
    nx              = nx,
    nt              = nt,
    emb_channels    = emb_channels,
    hidden_channels = 64,
    proj_channels   = 256,
    n_layers        = 4,
    modes           = (32, 32),
    device          = gpu_device()
)
println("  Model created successfully")

println("\n[2/3] Loading checkpoint from: $weight_file")
saved = JLD2.load(weight_file)
ps = saved["parameters"] |> cu
println("  Loaded trained parameters and states")

_, st = Lux.setup(Random.default_rng(), ffm.model)
st = st |> cu

# ---------------------------------------------------------------------------
# Constraint data
# ---------------------------------------------------------------------------
constraint_data = make_constraint_data(u_0_ic, nx, nt, n_samples;
                                        dx=dx, dt_physics=dt_physics)

println("\n[3/3] Generating samples...")


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------
begin
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
end

# ---------------------------------------------------------------------------
# Analytic solution (same as infer_heat.jl / infer_heat_2.jl)
# ---------------------------------------------------------------------------
T = range(t_range[1], t_range[2]; length=nt)
u_exact    = exp.(-3 .* T') .* sin.(x_grid .+ Float32(π)/4)   # (nx, nt)
u_analytic = zeros(Float32, nx, nt, 1, 1)
u_analytic[:, :, 1, 1] = u_exact

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
JLD2.save(SAMPLES_PATH,
    "samples_lbfgs",    samples_lbfgs,
    # "samples_ipnewton", samples_ipnewton,
    "u_analytic",       u_analytic)

@info "Samples saved to $SAMPLES_PATH"

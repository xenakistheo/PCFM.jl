# Inference script for the 1D heat equation using Alaina's projection-based
# sample_pcfm (solve_projection interface) instead of ExaModels/JuMP.
#
# Mirrors infer_heat.jl: same model, same IC, same save format.
# Model inference runs on GPU; projection solvers run on CPU.

using PCFM
using Lux
using CUDA
using cuDNN
using JLD2, Functors
using BenchmarkTools
using Random

Random.seed!(1234)

# ---------------------------------------------------------------------------
# Configuration  (must match the checkpoint)
# ---------------------------------------------------------------------------
nx           = 100
nt           = 100
emb_channels = 32

SAMPLES_PATH = length(ARGS) >= 2 ? ARGS[2] : "samples_heat_alaina.jld2"
n_samples    = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 32

weight_file = joinpath(@__DIR__, "checkpoints", "ffm_heat_checkpoint.jld2")

t_range = (0.0f0, 1.0f0)

x_grid = range(0.0f0, 2.0f0 * Float32(π); length=nx)
dx     = Float32(x_grid[2] - x_grid[1])
dt     = 1.0f0 / (nt - 1)

# ---------------------------------------------------------------------------
println("=" ^ 60)
println("Heat Equation — Functional Flow Matching (Alaina solvers)")
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
    device          = cpu_device   # projection solvers are CPU-only
)
println("  Model created successfully")

println("\n[2/3] Loading checkpoint from: $weight_file")
saved = JLD2.load(weight_file)
ps = saved["parameters"] |> cu
println("  Loaded trained parameters and states")

_, st = Lux.setup(Random.default_rng(), ffm.model)
st = st |> cu

# ---------------------------------------------------------------------------
# Build constraint data  (IC = sin(x + π/4), same as infer_heat.jl)
# ---------------------------------------------------------------------------
u_0_ic = Float32.(sin.(x_grid .+ Float32(π)/4))   # (nx,)
constraint_data = make_constraint_data(u_0_ic, nx, nt, n_samples; dx=dx)

println("\n[3/3] Generating samples...")

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
begin
    @info "IPNewton IC+Mass projection"
    display(@benchmark sample_pcfm($ffm.model, $ps, $st, $nx, $nt, $emb_channels,
                        $n_samples, 100,
                        IPMassProjectionSolver(),
                        $constraint_data;
                        device=cu, verbose=false))

    @info "LBFGS IC+Mass projection"
    display(@benchmark sample_pcfm($ffm.model, $ps, $st, $nx, $nt, $emb_channels,
                        $n_samples, 100,
                        PenaltyLBFGSMassProjectionSolver(),
                        $constraint_data;
                        device=cu, verbose=false))
end

# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------
begin
    @info "IPNewton IC+Mass projection"
    @time samples_ipnewton = sample_pcfm(ffm.model, ps, st, nx, nt, emb_channels,
                        n_samples, 100,
                        IPMassProjectionSolver(),
                        constraint_data;
                        device=cu, verbose=true)

    @info "LBFGS IC+Mass projection"
    @time samples_lbfgs = sample_pcfm(ffm.model, ps, st, nx, nt, emb_channels,
                        n_samples, 100,
                        PenaltyLBFGSMassProjectionSolver(),
                        constraint_data;
                        device=cu, verbose=true)
end

# ---------------------------------------------------------------------------
# Analytic solution (same as infer_heat.jl)
# ---------------------------------------------------------------------------
T = range(t_range[1], t_range[2]; length=nt)
u_exact    = exp.(-3 .* T') .* sin.(x_grid .+ Float32(π)/4)   # (nx, nt)
u_analytic = zeros(Float32, nx, nt, 1, 1)
u_analytic[:, :, 1, 1] = u_exact

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
JLD2.save(SAMPLES_PATH,
    "samples_ipnewton", samples_ipnewton,
    "samples_lbfgs",    samples_lbfgs,
    "u_analytic",       u_analytic)

@info "Samples saved to $SAMPLES_PATH"

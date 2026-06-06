# Inference script for the 1D Reaction-Diffusion equation using Alaina's
# projection-based sample_pcfm (solve_projection interface).
#
# Mirrors infer_rd.jl: same model, same IC, same save format.
# Solvers: RDSolver (LBFGS) + RDIPNewtonSolver (IPNewton)
#
# NOTE: Alaina's RD solvers track mass evolution using the logistic reaction
# source only (rho * u * (1-u)). The diffusion term (nu) is not included in
# the mass update — this is a simplification relative to rd_constraints_2!.
# Projection solvers run on CPU only.

using PCFM
using Lux
using CUDA
using cuDNN
using JLD2, Functors
using BenchmarkTools
using Random

Random.seed!(42)

# ---------------------------------------------------------------------------
# Configuration  (must match the checkpoint)
# ---------------------------------------------------------------------------
nx           = 64
nt           = 100
emb_channels = 32

SAMPLES_PATH = length(ARGS) >= 2 ? ARGS[2] : "samples_rd_alaina.jld2"
n_samples    = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 32

weight_file = joinpath(@__DIR__, "checkpoints", "ffm_rd_checkpoint_nx64.jld2")

t_range    = (0.0f0, 1.0f0)
x_grid     = range(0.0f0, 1.0f0; length=nx)
dx         = Float32(x_grid[2] - x_grid[1])
dt_physics = 1.0f0 / (nt - 1)

const rd_rho = 0.01f0   # logistic reaction rate (matches infer_rd.jl)

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

# ---------------------------------------------------------------------------
println("=" ^ 60)
println("Reaction-Diffusion — Functional Flow Matching (Alaina solvers)")
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
    device          = cpu_device
)
println("  Model created successfully")

println("\n[2/3] Loading checkpoint from: $weight_file")
saved = JLD2.load(weight_file)
ps = saved["parameters"]
st = saved["states"]
println("  Loaded trained parameters and states")

_, st = Lux.setup(Random.default_rng(), ffm.model)

# ---------------------------------------------------------------------------
# Constraint data
# ---------------------------------------------------------------------------
constraint_data = make_constraint_data(u0_fixed, nx, nt, n_samples;
                                        dx=dx, dt_physics=dt_physics)

println("\n[3/3] Generating samples...")

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
begin
    @info "RD LBFGS (rho=$rd_rho)"
    display(@benchmark sample_pcfm($ffm.model, $ps, $st, $nx, $nt, $emb_channels,
                        $n_samples, 100,
                        RDSolver(rho=$rd_rho),
                        $constraint_data;
                        verbose=false))

    @info "RD IPNewton (rho=$rd_rho)"
    display(@benchmark sample_pcfm($ffm.model, $ps, $st, $nx, $nt, $emb_channels,
                        $n_samples, 100,
                        RDIPNewtonSolver(rho=$rd_rho),
                        $constraint_data;
                        verbose=false))
end

# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------
begin
    @info "RD LBFGS (rho=$rd_rho)"
    @time samples_lbfgs = sample_pcfm(ffm.model, ps, st, nx, nt, emb_channels,
                        n_samples, 100,
                        RDSolver(rho=rd_rho),
                        constraint_data;
                        verbose=true)

    @info "RD IPNewton (rho=$rd_rho)"
    @time samples_ipnewton = sample_pcfm(ffm.model, ps, st, nx, nt, emb_channels,
                        n_samples, 100,
                        RDIPNewtonSolver(rho=rd_rho),
                        constraint_data;
                        verbose=true)
end

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
JLD2.save(SAMPLES_PATH,
    "samples_lbfgs",    samples_lbfgs,
    "samples_ipnewton", samples_ipnewton,
    "u0_fixed",         u0_fixed)

@info "Samples saved to $SAMPLES_PATH"

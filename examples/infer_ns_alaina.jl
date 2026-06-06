# Inference script for 2D Navier-Stokes (vorticity form) using Alaina's
# projection-based sample_pcfm (solve_projection interface).
#
# Mirrors infer_ns.jl: same model, same IC, same save format.
# Solvers: NSVorticityLBFGSSolver + NSVorticityIPNewtonSolver
# Constraint: IC + vorticity integral (W₀ = ∫∫ ω dx dy)
# Model inference runs on GPU; projection solvers run on CPU.

using PCFM
using Lux
using CUDA
using cuDNN
using JLD2, Functors
using BenchmarkTools
using Random
using HDF5

Random.seed!(42)

# ---------------------------------------------------------------------------
# Configuration  (must match the checkpoint)
# ---------------------------------------------------------------------------
s            = 16     # spatial grid size (s × s)
nt           = 50
emb_channels = 32

SAMPLES_PATH = length(ARGS) >= 2 ? ARGS[2] : "samples_ns_alaina.jld2"
n_samples    = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 4

weight_file = joinpath(@__DIR__, "checkpoints", "ffm_ns_s16_checkpoint.jld2")

# Grid: periodic domain [0, 1) × [0, 1)
x_grid = range(0.0f0, 1.0f0; length=s+1)[1:end-1]
y_grid = range(0.0f0, 1.0f0; length=s+1)[1:end-1]
dx     = Float32(x_grid[2] - x_grid[1])

# Initial condition: sinusoidal vorticity (Kolmogorov-like)
IC_func_ns = (x, y) -> sin(2f0 * Float32(π) * Float32(y))
u_0_ic_2d  = Float32.(IC_func_ns.(x_grid, y_grid'))   # (s, s)

# ---------------------------------------------------------------------------
println("=" ^ 60)
println("2D Navier-Stokes — Functional Flow Matching (Alaina solvers)")
println("=" ^ 60)

println("\n[1/3] Creating FFM model...")
ffm = FFM(
    spatial_size    = (s, s),
    nt              = nt,
    emb_channels    = emb_channels,
    hidden_channels = 64,
    proj_channels   = 256,
    n_layers        = 4,
    modes           = (4, 4, 12),
    device          = cpu_device
)
println("  Model created successfully")

println("\n[2/3] Loading checkpoint from: $weight_file")
saved = JLD2.load(weight_file)
ps = saved["parameters"] |> cu
println("  Loaded trained parameters and states")

_, st = Lux.setup(Random.default_rng(), ffm.model)
st = st |> cu

# ---------------------------------------------------------------------------
# Constraint data  (cdata.M0 = W₀, the conserved vorticity integral)
# ---------------------------------------------------------------------------
constraint_data = make_constraint_data(u_0_ic_2d, (s, s), nt, n_samples; dx=dx)

println("\n[3/3] Generating samples...")

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
begin
    @info "NS Vorticity LBFGS"
    display(@benchmark sample_pcfm($ffm.model, $ps, $st, ($s, $s), $nt, $emb_channels,
                        $n_samples, 100,
                        NSVorticityLBFGSSolver(),
                        $constraint_data;
                        device=cu, verbose=false))

    @info "NS Vorticity IPNewton"
    display(@benchmark sample_pcfm($ffm.model, $ps, $st, ($s, $s), $nt, $emb_channels,
                        $n_samples, 100,
                        NSVorticityIPNewtonSolver(),
                        $constraint_data;
                        device=cu, verbose=false))
end

# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------
begin
    @info "NS Vorticity LBFGS"
    @time samples_lbfgs = sample_pcfm(ffm.model, ps, st, (s, s), nt, emb_channels,
                        n_samples, 100,
                        NSVorticityLBFGSSolver(),
                        constraint_data;
                        device=cu, verbose=true)

    @info "NS Vorticity IPNewton"
    @time samples_ipnewton = sample_pcfm(ffm.model, ps, st, (s, s), nt, emb_channels,
                        n_samples, 100,
                        NSVorticityIPNewtonSolver(),
                        constraint_data;
                        device=cu, verbose=true)
end

# ---------------------------------------------------------------------------
# Reference solutions from test dataset
# ---------------------------------------------------------------------------
test_data_file = joinpath(@__DIR__, "..", "datasets", "data", "ns_nw30_nf30_s16_t50_mu0.001.h5")
ref_samples = zeros(Float32, s, s, nt, 1, n_samples)

if isfile(test_data_file)
    h5open(test_data_file, "r") do f
        u_all = read(f["u"])   # (nt, s, s, nf, nw) in Julia HDF5 ordering
        n_load = min(n_samples, size(u_all, 4) * size(u_all, 5))
        idx = 1
        for iw in 1:size(u_all, 5), if_ in 1:size(u_all, 4)
            idx > n_load && break
            ref_samples[:, :, :, 1, idx] = permutedims(u_all[:, :, :, if_, iw], (2, 3, 1))
            idx += 1
        end
    end
    @info "Reference solutions loaded from $test_data_file"
else
    @warn "Test data not found at $test_data_file"
end

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
JLD2.save(SAMPLES_PATH,
    "samples_lbfgs",    samples_lbfgs,
    "samples_ipnewton", samples_ipnewton,
    "ref_samples",      ref_samples)

@info "Samples saved to $SAMPLES_PATH"

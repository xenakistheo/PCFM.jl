"""
Load a trained FFM checkpoint and benchmark 2D Navier-Stokes samples
using physics-constrained flow matching (PCFM), without Reactant.

Run train_ns.jl first to produce the checkpoint.
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
using FFTW
using CairoMakie

include(joinpath(@__DIR__, "..", "datasets", "random_fields.jl"))

backend = CUDABackend()
dev_gpu = cu
dev_cpu = cpu_device
device  = dev_gpu

Random.seed!(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# NS NLP size = s*s*nt*n_samples; keep n_samples small for exa/jump modes
n_samples    = 4
s            = 64   # spatial grid size (s × s)
nt           = 50
emb_channels = 32

weight_file    = joinpath(@__DIR__, "checkpoints", "ffm_ns_checkpoint.jld2")
test_data_file = joinpath(@__DIR__, "..", "datasets", "data", "ns_nw30_nf30_s64_t50_mu0.001.h5")

# Initial condition: 2D Gaussian random vorticity field (fixed seed)
Random.seed!(0)
grf   = GaussianRF(2, s; alpha=2, tau=3)
w0_ic = Float32.(sample(grf, 1)[:, :, 1])   # (s, s)
Random.seed!(42)

const ns_domain = (x_start=0f0, x_end=2f0π, y_start=0f0, y_end=2f0π, t_start=0f0, t_end=1f0)

# ---------------------------------------------------------------------------
println("=" ^ 60)
println("2D Navier-Stokes — Functional Flow Matching")
println("=" ^ 60)

# 1. Create model
println("\n[1/3] Creating FFM model...")
ffm = FFM(
    spatial_size    = (s, s),
    nt              = nt,
    emb_channels    = emb_channels,
    hidden_channels = 64,
    proj_channels   = 256,
    n_layers        = 4,
    modes           = (12, 12, 16),
    device          = dev_gpu
)
println("  Model created successfully")

# 2. Load checkpoint
println("\n[2/3] Loading checkpoint from: $weight_file")
saved = JLD2.load(weight_file)
ps = saved["parameters"] |> dev_gpu
st = saved["states"]     |> dev_gpu
_, st = Lux.setup(Random.default_rng(), ffm.model)
ps = ps |> dev_gpu
st = st |> dev_gpu
println("  Loaded trained parameters and states")

println("\n[3/3] Generating samples...")
tstate_inf = (parameters = ps, states = st)

@show backend

starting_noise = randn(Float32, s, s, nt, 1, n_samples)

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
begin
    # GPU Exa omitted: NS NLP (s*s*nt*n_samples ≈ 800K+ vars) exceeds GPU
    # kernel parameter memory limits.

    @info "Analytical projection (CPU)"
    @btime sample_pcfm($ffm, (parameters=$ps, states=$st),
                       $n_samples, 100;
                       domain    = ns_domain,
                       IC_field  = $w0_ic,
                       backend   = CPU(),
                       verbose   = false,
                       mode      = "analytical",
                       proj!     = ns_proj!,
                       initial_vals = $starting_noise)
    flush(stdout)

    @info "ExaModels, MadNLP, CPU"
    @btime sample_pcfm($ffm, (parameters=$ps, states=$st),
                       $n_samples, 100, ns_constraints!;
                       domain    = ns_domain,
                       IC_field  = $w0_ic,
                       backend   = CPU(),
                       verbose   = false,
                       mode      = "exa",
                       initial_vals = $starting_noise)
    flush(stdout)

    @info "FFM (unconstrained)"
    @btime sample_ffm($ffm, (parameters=$ps, states=$st), $n_samples, 100;
        verbose      = false,
        initial_vals = $starting_noise)
    flush(stdout)
end

# ---------------------------------------------------------------------------
# Generate samples
# ---------------------------------------------------------------------------
begin
    @info "Analytical projection (CPU)"
    samples_analytical = sample_pcfm(ffm, (parameters=ps, states=st),
                       n_samples, 100;
                       domain    = ns_domain,
                       IC_field  = w0_ic,
                       backend   = CPU(),
                       verbose   = false,
                       mode      = "analytical",
                       proj!     = ns_proj!,
                       initial_vals = starting_noise)

    @info "ExaModels, MadNLP, CPU"
    samples_exa_cpu = sample_pcfm(ffm, (parameters=ps, states=st),
                       n_samples, 100, ns_constraints!;
                       domain    = ns_domain,
                       IC_field  = w0_ic,
                       backend   = CPU(),
                       verbose   = false,
                       mode      = "exa",
                       initial_vals = starting_noise)

    @info "FFM (unconstrained)"
    samples_ffm = sample_ffm(ffm, (parameters=ps, states=st), n_samples, 100;
        verbose      = false,
        initial_vals = starting_noise)
end
samples_ffm = Array(samples_ffm)

# ---------------------------------------------------------------------------
# Load reference solutions from the NS test dataset
# ---------------------------------------------------------------------------
ref_samples = zeros(Float32, s, s, nt, 1, n_samples)
if isfile(test_data_file)
    batch = load_ns_batch(test_data_file, n_samples)   # (s, s, nt, 1, n_samples)
    ref_samples .= batch
    @info "Loaded reference samples from $test_data_file"
else
    @warn "Test data not found at $test_data_file; reference column will be zeros"
end

# ---------------------------------------------------------------------------
# Plot vorticity snapshots at selected time steps
# ---------------------------------------------------------------------------
k      = 1
t_idxs = [1, nt÷4, nt÷2, 3*nt÷4, nt]

methods = [
    ("Reference",    ref_samples),
    ("Analytical",   samples_analytical),
    ("Exa (CPU)",    samples_exa_cpu),
    ("FFM",          samples_ffm),
]

fig = Figure(size = (300 * length(t_idxs), 320 * length(methods)))
for (row, (label, data)) in enumerate(methods)
    Label(fig[row, 0], label; rotation = π/2, tellheight = false)
    for (col, tidx) in enumerate(t_idxs)
        ax = Axis(fig[row, col]; title = "t = $tidx", aspect = 1)
        heatmap!(ax, data[:, :, tidx, 1, k]; colormap = :RdBu)
        hidedecorations!(ax)
    end
end
Label(fig[0, :], "NS vorticity — sample $k"; fontsize = 18, font = :bold)
save("ns_samples.png", fig)
@info "Saved ns_samples.png"

# ---------------------------------------------------------------------------
# Mass conservation violation vs time
# ---------------------------------------------------------------------------
function mass_violation(u, sample_idx=1)
    M0 = sum(u[:, :, 1, 1, sample_idx])
    return [abs(sum(u[:, :, t, 1, sample_idx]) - M0) for t in axes(u, 3)]
end

fig2 = Figure(size = (800, 400))
ax   = Axis(fig2[1, 1];
    xlabel = "Time step", ylabel = "|ΔMass|",
    title  = "Mass conservation violation (sample $k)")
lines!(ax, 1:nt, mass_violation(samples_analytical); label = "Analytical")
lines!(ax, 1:nt, mass_violation(samples_exa_cpu);    label = "Exa (CPU)")
lines!(ax, 1:nt, mass_violation(samples_ffm);        label = "FFM")
if isfile(test_data_file)
    lines!(ax, 1:nt, mass_violation(ref_samples); label = "Reference", linestyle = :dash)
end
axislegend(ax; position = :lt)
save("ns_mass_violation.png", fig2)
@info "Saved ns_mass_violation.png"

# ---------------------------------------------------------------------------
# Save samples
# ---------------------------------------------------------------------------
JLD2.save("samples_ns.jld2",
    "ref_samples",       ref_samples,
    "samples_analytical", samples_analytical,
    "samples_exa_cpu",   samples_exa_cpu,
    "samples_ffm",       samples_ffm)
@info "Saved samples_ns.jld2"

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
                    initial_vals = starting_noise)

@info "ExaModels, MadNLP, CPU"
@time samples_exa_cpu = sample_pcfm_2d(ffm, (parameters=ps, states=st),
                    n_samples, 100, ns_enstrophy_constraints!;
                    domain = ns_domain,
                    IC_func = IC_func_ns,
                    backend = CPU(),
                    verbose = true,
                    mode = "exa",
                    initial_vals = starting_noise)

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

##################
# Load reference solutions from test dataset
test_data_file = joinpath(@__DIR__, "..", "datasets", "data", "ns_nw30_nf30_s16_t50_mu0.001.h5")
ref_samples = zeros(Float32, s, s, nt, 1, n_samples)

if isfile(test_data_file)
    h5open(test_data_file, "r") do f
        # HDF5 layout (Julia): u is (steps, s, s, nf, nw)
        u_all = read(f["u"])                            # (nt, s, s, nf, nw)
        n_load = min(n_samples, size(u_all, 4) * size(u_all, 5))
        idx = 1
        for iw in 1:size(u_all, 5), if_ in 1:size(u_all, 4)
            idx > n_load && break
            ref_samples[:, :, :, 1, idx] = permutedims(u_all[:, :, :, if_, iw], (2, 3, 1))  # (s, s, nt)
            idx += 1
        end
    end
    @info "Reference solutions loaded from $test_data_file"
else
    @warn "Test data not found at $test_data_file — saving samples without reference"
end

##################
# Save samples
JLD2.save(SAMPLES_PATH,
    "ref_samples",     ref_samples,
    "samples_exa_gpu", samples_exa_gpu,
    "samples_exa_cpu", samples_exa_cpu,
    "samples_jump_madnlp", samples_jump_madnlp,
    "samples_jump_ipopt", samples_jump_ipopt)

@info "Samples saved to $SAMPLES_PATH"

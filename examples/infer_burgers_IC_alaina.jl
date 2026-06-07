# Inference script for the 1D Burgers equation (IC + Mass + Godunov flux regime)
# using Alaina's projection-based sample_pcfm (solve_projection interface).
#
# Mirrors infer_burgers_IC.jl: same model, same IC, same save format.
# Solvers: BurgersICFluxSolver (LBFGS) + BurgersICFluxIPSolver (IPNewton)
# Model inference runs on GPU; projection solvers run on CPU.

using PCFM
using Lux
using CUDA
using cuDNN
using JLD2, Functors
using Random
using HDF5

Random.seed!(42)

# ---------------------------------------------------------------------------
# Configuration  (must match the checkpoint)
# ---------------------------------------------------------------------------
nx           = 101
nt           = 101
emb_channels = 32

SAMPLES_PATH = length(ARGS) >= 2 ? ARGS[2] : "samples_burgers_IC_alaina.jld2"
n_samples    = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 32

weight_file = joinpath(@__DIR__, "checkpoints", "ffm_burgers_checkpoint.jld2")

t_range    = (0.0f0, 1.0f0)
x_grid     = range(0.0f0, 1.0f0; length=nx)
dx         = Float32(x_grid[2] - x_grid[1])
dt_physics = 1.0f0 / (nt - 1)

# Initial condition: viscous Burgers sigmoid
const p_loc  = 0.5f0
const eps_ic = 0.02f0
IC_func_burgers = x -> 1.0f0 / (1.0f0 + exp((x - p_loc) / eps_ic))
u_0_ic = Float32.(IC_func_burgers.(x_grid))

# ---------------------------------------------------------------------------
println("=" ^ 60)
println("Burgers IC — Functional Flow Matching (Alaina solvers)")
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
SAMPLES_PATH = "samples_burgers_IC_only_alaina.jld2"
for iter in 1:4

    @info "BurgersIC only IPNewton"
    @time samples_ic_only_ip = sample_pcfm(ffm, (parameters=ps, states=st),
                        n_samples, 100,
                        BurgersICIPSolver(),
                        constraint_data;
                        verbose=true)

    @info "BurgersIC only LBFGS"
    @time samples_ic_only = sample_pcfm(ffm, (parameters=ps, states=st),
                        n_samples, 100,
                        BurgersICSolver(),
                        constraint_data;
                        verbose=true)

    # @info "BurgersICFlux LBFGS"
    # @time samples_lbfgs = sample_pcfm(ffm, (parameters=ps, states=st),
    #                     n_samples, 100,
    #                     BurgersICFluxSolver(),
    #                     constraint_data;
    #                     verbose=true)

    # @info "BurgersICFlux IPNewton"
    # @time samples_ipnewton = sample_pcfm(ffm, (parameters=ps, states=st),
    #                     n_samples, 100,
    #                     BurgersICFluxIPSolver(),
    #                     constraint_data;
    #                     verbose=true)
end

begin
    @info "BurgersIC only IPNewton"
    @time samples_ic_only_ip = sample_pcfm(ffm, (parameters=ps, states=st),
                        n_samples, 100,
                        BurgersICIPSolver(),
                        constraint_data;
                        verbose=true)


    @info "BurgersIC only LBFGS"
    @time samples_ic_only = sample_pcfm(ffm, (parameters=ps, states=st),
                        n_samples, 100,
                        BurgersICSolver(),
                        constraint_data;
                        verbose=true)


    # @info "BurgersICFlux LBFGS"
    # @time samples_lbfgs = sample_pcfm(ffm, (parameters=ps, states=st),
    #                     n_samples, 100,
    #                     BurgersICFluxSolver(),
    #                     constraint_data;
    #                     verbose=true)

    # @info "BurgersICFlux IPNewton"
    # @time samples_ipnewton = sample_pcfm(ffm, (parameters=ps, states=st),
    #                     n_samples, 100,
    #                     BurgersICFluxIPSolver(),
    #                     constraint_data;
    #                     verbose=true)
end

# ---------------------------------------------------------------------------
# Reference solutions
# ---------------------------------------------------------------------------
test_data_file = joinpath(@__DIR__, "..", "datasets", "data", "burgers_test_nIC30_nBC30.h5")
ref_samples = zeros(Float32, nx, nt, 1, n_samples)

h5open(test_data_file, "r") do f
    p_locs = read(f["ic"])
    nt_h5, nx_h5, N_bc, N_ic = size(f["u"])
    _, i_ic = findmin(abs.(p_locs .- 0.5f0))
    @info "Reference IC: p_loc = $(round(p_locs[i_ic]; digits=4)) (index $i_ic / $N_ic)"
    n_load = min(n_samples, N_bc)
    for i_bc in 1:n_load
        arr = Float32.(f["u"][:, :, i_bc, i_ic])
        ref_samples[:, :, 1, i_bc] = permutedims(arr, (2, 1))
    end
end

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
JLD2.save(SAMPLES_PATH,
    "samples_ic_only",    samples_ic_only,
    "samples_ic_only_ip", samples_ic_only_ip,
    # "samples_lbfgs",      samples_lbfgs,
    # "samples_ipnewton",   samples_ipnewton,
    "ref_samples",        ref_samples)

@info "Samples saved to $SAMPLES_PATH"

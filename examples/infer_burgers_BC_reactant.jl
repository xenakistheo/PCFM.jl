using PCFM
using Lux
using CUDA
using cuDNN
using KernelAbstractions
using JLD2
using Functors
using Random

backend = CPU()
dev_gpu = cu
device = dev_gpu

Random.seed!(42)

println("=" ^ 60)
println("Burgers BC + Mass — Reactant PCFM Inference")
println("=" ^ 60)

nx = 101
nt = 101
emb_channels = 32

weight_file = joinpath(
    @__DIR__,
    "checkpoints",
    "checkpoints",
    "ffm_burgers_checkpoint.jld2",
)

if !isfile(weight_file)
    weight_file = joinpath(@__DIR__, "checkpoints", "ffm_burgers_checkpoint.jld2")
end

domain = (
    x_start = 0.0f0,
    x_end = 1.0f0,
    t_start = 0.0f0,
    t_end = 1.0f0,
)

x_grid = range(domain.x_start, domain.x_end; length = nx)
dx = Float32(x_grid[2] - x_grid[1])

const p_loc = 0.5f0
const eps_ic = 0.02f0
IC_func_burgers = x -> 1.0f0 / (1.0f0 + exp((x - p_loc) / eps_ic))

println("\n[1/4] Creating FFM model...")

ffm = FFM(
    nx = nx,
    nt = nt,
    emb_channels = emb_channels,
    hidden_channels = 64,
    proj_channels = 256,
    n_layers = 4,
    modes = (32, 32),
    device = device,
)

println("Model created successfully")

println("\n[2/4] Loading checkpoint from: $weight_file")

saved = JLD2.load(weight_file)

ps = saved["parameters"] |> device
st = saved["states"] |> device

_, st = Lux.setup(Random.default_rng(), ffm.model)

ps = ps |> device
st = st |> device

tstate_inf = (
    parameters = ps,
    states = st,
)

println("Loaded trained parameters and states")

println("\n[3/4] Generating Reactant samples...")

n_samples = 32
n_steps = 100

left_bc_vals = rand(Float32, n_samples)
burgers_params = (
    left_bc = left_bc_vals,
)

starting_noise = randn(Float32, nx, nt, 1, n_samples)

@info "Reactant matrix-free SQP sampling"

samples_reactant = sample_pcfm(
    ffm,
    tstate_inf,
    n_samples,
    n_steps,
    burgers_constraints_BC_Mass!;
    domain = domain,
    IC_func = IC_func_burgers,
    constraint_parameters = burgers_params,
    backend = CPU(),
    verbose = true,
    mode = "reactant",
    initial_vals = starting_noise,
    reactant_maxiter = 4,
    reactant_reg = 1f-7,
    reactant_cg_iters = 30,
)

samples_reactant = Array(samples_reactant)

println("\n[4/4] Saving output and checking constraints...")

JLD2.save(
    "samples_burgers_BC_reactant.jld2",
    "samples_reactant", samples_reactant,
    "starting_noise", starting_noise,
    "x_grid", collect(x_grid),
    "left_bc_vals", left_bc_vals,
    "constraint_params", burgers_params,
)

left_bc_violation = [
    samples_reactant[1, t, 1, s] - left_bc_vals[s]
    for t in 1:nt, s in 1:n_samples
]

neumann_violation = [
    samples_reactant[nx, t, 1, s] - samples_reactant[nx - 1, t, 1, s]
    for t in 1:nt, s in 1:n_samples
]

dt_sample = 1.0f0 / Float32(n_steps)

mass_violation = [
    (
        sum(samples_reactant[:, t, 1, s]) * dx -
        sum(samples_reactant[:, t - 1, 1, s]) * dx +
        0.5f0 * dt_sample * (
            0.5f0 * samples_reactant[nx, t, 1, s]^2 -
            0.5f0 * samples_reactant[1, t, 1, s]^2 +
            0.5f0 * samples_reactant[nx, t - 1, 1, s]^2 -
            0.5f0 * samples_reactant[1, t - 1, 1, s]^2
        )
    )
    for t in 2:nt, s in 1:n_samples
]

println("Reactant Burgers BC max left BC violation: ", maximum(abs.(left_bc_violation)))
println("Reactant Burgers BC mean left BC violation: ", sum(abs.(left_bc_violation)) / length(left_bc_violation))

println("Reactant Burgers BC max Neumann violation: ", maximum(abs.(neumann_violation)))
println("Reactant Burgers BC mean Neumann violation: ", sum(abs.(neumann_violation)) / length(neumann_violation))

println("Reactant Burgers BC max mass violation: ", maximum(abs.(mass_violation)))
println("Reactant Burgers BC mean mass violation: ", sum(abs.(mass_violation)) / length(mass_violation))

println("Saved samples_burgers_BC_reactant.jld2")
println("Done.")
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
println("Burgers IC + Mass + Flux — Reactant PCFM Inference")
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

burgers_ic_flux_params = (
    k = 5,
    eps = 1f-6,
)

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

starting_noise = randn(Float32, nx, nt, 1, n_samples)

@info "Reactant matrix-free SQP sampling"

samples_reactant = sample_pcfm(
    ffm,
    tstate_inf,
    n_samples,
    n_steps,
    PCFM.burgers_constraints_IC_Mass_Flux!;
    domain = domain,
    IC_func = IC_func_burgers,
    constraint_parameters = burgers_ic_flux_params,
    backend = CPU(),
    verbose = true,
    mode = "reactant",
    initial_vals = starting_noise,
    reactant_maxiter = 8,
    reactant_reg = 1f-7,
    reactant_cg_iters = 50,
)

samples_reactant = Array(samples_reactant)

println("\n[4/4] Saving output and checking constraints...")

u0 = Float32.(IC_func_burgers.(x_grid))

ic_violation = [
    samples_reactant[i, 1, 1, s] - u0[i]
    for i in 1:nx, s in 1:n_samples
]

mass0 = sum(u0) * dx

mass_violation = [
    sum(samples_reactant[:, t, 1, s]) * dx - mass0
    for t in 1:nt, s in 1:n_samples
]

k_eff = min(burgers_ic_flux_params.k, nt - 1)
eps_smooth = burgers_ic_flux_params.eps
lambda_cfl = Float32(1.0f0 / n_steps) / dx

smooth_pos_check(x, eps_smooth) = 0.5f0 * (x + sqrt(x^2 + eps_smooth^2))
smooth_neg_check(x, eps_smooth) = 0.5f0 * (x - sqrt(x^2 + eps_smooth^2))

flux_violation = [
    begin
        F_i =
            0.5f0 * smooth_pos_check(samples_reactant[i, t, 1, s], eps_smooth)^2 +
            0.5f0 * smooth_neg_check(samples_reactant[i + 1, t, 1, s], eps_smooth)^2

        F_im1 =
            0.5f0 * smooth_pos_check(samples_reactant[i - 1, t, 1, s], eps_smooth)^2 +
            0.5f0 * smooth_neg_check(samples_reactant[i, t, 1, s], eps_smooth)^2

        samples_reactant[i, t + 1, 1, s] -
        samples_reactant[i, t, 1, s] +
        lambda_cfl * (F_i - F_im1)
    end
    for t in 1:k_eff, i in 2:nx-1, s in 1:n_samples
]

JLD2.save(
    "samples_burgers_IC_reactant.jld2",
    "samples_reactant", samples_reactant,
    "starting_noise", starting_noise,
    "x_grid", collect(x_grid),
    "constraint_params", burgers_ic_flux_params,
    "ic_violation", ic_violation,
    "mass_violation", mass_violation,
    "flux_violation", flux_violation,
)

println("Reactant Burgers IC max IC violation: ", maximum(abs.(ic_violation)))
println("Reactant Burgers IC mean IC violation: ", sum(abs.(ic_violation)) / length(ic_violation))

println("Reactant Burgers IC max mass violation: ", maximum(abs.(mass_violation)))
println("Reactant Burgers IC mean mass violation: ", sum(abs.(mass_violation)) / length(mass_violation))

println("Reactant Burgers IC max flux violation: ", maximum(abs.(flux_violation)))
println("Reactant Burgers IC mean flux violation: ", sum(abs.(flux_violation)) / length(flux_violation))

println("Saved samples_burgers_IC_reactant.jld2")
println("Done.")
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
println("Reaction-Diffusion — Reactant PCFM Inference")
println("=" ^ 60)

nx = 128
nt = 100
emb_channels = 32

weight_file = joinpath(
    @__DIR__,
    "checkpoints",
    "checkpoints",
    "ffm_rd_checkpoint.jld2",
)

if !isfile(weight_file)
    weight_file = joinpath(@__DIR__, "checkpoints", "ffm_rd_checkpoint.jld2")
end

domain = (
    x_start = 0.0f0,
    x_end = 1.0f0,
    t_start = 0.0f0,
    t_end = 1.0f0,
)

x_grid = range(domain.x_start, domain.x_end; length = nx)
dx = Float32(x_grid[2] - x_grid[1])

function generate_ic(xc; k_tot = 3, num_choice_k = 2)
    selected = rand(1:k_tot, num_choice_k)
    onehot = zeros(Int, k_tot)

    for j in selected
        onehot[j] += 1
    end

    kk = 2f0 * Float32(pi) .* Float32.(1:k_tot) .* Float32.(onehot) ./ Float32(xc[end] - xc[1])
    amp = rand(Float32, k_tot, 1)
    phs = 2f0 * Float32(pi) .* rand(Float32, k_tot, 1)

    u = vec(sum(amp .* sin.(kk .* Float32.(xc') .+ phs), dims = 1))

    if rand() < 0.1
        u = abs.(u)
    end

    u .*= rand(Float32[-1f0, 1f0])

    if rand() < 0.1
        xL_m = rand(Float32) * 0.35f0 + 0.1f0
        xR_m = rand(Float32) * 0.35f0 + 0.55f0
        trns = 0.01f0
        mask = 0.5f0 .* (
            tanh.((Float32.(xc) .- xL_m) ./ trns) .-
            tanh.((Float32.(xc) .- xR_m) ./ trns)
        )
        u .*= mask
    end

    u .-= minimum(u)

    if maximum(u) > 0
        u ./= maximum(u)
    end

    return u
end

Random.seed!(0)
u0_fixed = Float32.(generate_ic(collect(x_grid)))
Random.seed!(42)

IC_func_rd = x -> u0_fixed[clamp(round(Int, (x - x_grid[1]) / dx) + 1, 1, nx)]

rd_params = (
    rho = 0.01f0,
    nu = 0.005f0,
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
    rd_constraints!;
    domain = domain,
    IC_func = IC_func_rd,
    constraint_parameters = rd_params,
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

JLD2.save(
    "samples_rd_reactant.jld2",
    "samples_reactant", samples_reactant,
    "starting_noise", starting_noise,
    "x_grid", collect(x_grid),
    "u0_fixed", u0_fixed,
    "constraint_params", rd_params,
)

u0 = Float32.(IC_func_rd.(x_grid))

ic_violation = [
    samples_reactant[i, 1, 1, s] - u0[i]
    for i in 1:nx, s in 1:n_samples
]

rho = rd_params.rho
nu = rd_params.nu
dt_sample = 1.0f0 / Float32(n_steps)

left_deriv(u, t, s) =
    (-25f0 * u[1, t, 1, s] +
     48f0 * u[2, t, 1, s] -
     36f0 * u[3, t, 1, s] +
     16f0 * u[4, t, 1, s] -
      3f0 * u[5, t, 1, s]) / (12f0 * dx)

right_deriv(u, t, s) =
    (25f0 * u[nx, t, 1, s] -
     48f0 * u[nx - 1, t, 1, s] +
     36f0 * u[nx - 2, t, 1, s] -
     16f0 * u[nx - 3, t, 1, s] +
      3f0 * u[nx - 4, t, 1, s]) / (12f0 * dx)

mass_violation = [
    begin
        M_t = sum(samples_reactant[:, t, 1, s]) * dx
        M_prev = sum(samples_reactant[:, t - 1, 1, s]) * dx

        S_t = sum(samples_reactant[i, t, 1, s] * (1f0 - samples_reactant[i, t, 1, s]) for i in 1:nx) * dx
        S_prev = sum(samples_reactant[i, t - 1, 1, s] * (1f0 - samples_reactant[i, t - 1, 1, s]) for i in 1:nx) * dx

        flux_t =
            -nu * left_deriv(samples_reactant, t, s) +
             nu * right_deriv(samples_reactant, t, s)

        flux_prev =
            -nu * left_deriv(samples_reactant, t - 1, s) +
             nu * right_deriv(samples_reactant, t - 1, s)

        M_t -
        M_prev -
        0.5f0 * dt_sample * rho * (S_t + S_prev) -
        0.5f0 * dt_sample * (flux_t + flux_prev)
    end
    for t in 2:nt, s in 1:n_samples
]

println("Reactant RD max IC violation: ", maximum(abs.(ic_violation)))
println("Reactant RD mean IC violation: ", sum(abs.(ic_violation)) / length(ic_violation))

println("Reactant RD max mass violation: ", maximum(abs.(mass_violation)))
println("Reactant RD mean mass violation: ", sum(abs.(mass_violation)) / length(mass_violation))

println("Saved samples_rd_reactant.jld2")
println("Done.")
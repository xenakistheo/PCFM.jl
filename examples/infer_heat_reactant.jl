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
dev_cpu = cpu_device()
device = dev_gpu

Random.seed!(1234)

println("=" ^ 60)
println("Heat Equation — Reactant PCFM Inference")
println("=" ^ 60)

batch_size = 32
nx = 100
nt = 100
emb_channels = 32

weight_file = joinpath(
    @__DIR__,
    "checkpoints",
    "checkpoints",
    "ffm_diffusion_checkpoint.jld2",
)

domain = (
    x_start = 0.0f0,
    x_end = 2.0f0 * Float32(pi),
    t_start = 0.0f0,
    t_end = 1.0f0,
)

IC_func = x -> sin(x + Float32(pi / 4))

x_grid = range(domain.x_start, domain.x_end; length = nx)
dx = Float32(x_grid[2] - x_grid[1])
dt_grid = Float32((domain.t_end - domain.t_start) / (nt - 1))

constraint_params = (
    Nx = nx,
    Nt = nt,
    dx = dx,
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
    heat_constraints!;
    backend = CPU(),
    verbose = true,
    mode = "reactant",
    initial_vals = starting_noise,
    domain = domain,
    IC_func = IC_func,
    constraint_parameters = constraint_params,
    reactant_maxiter = 1,
    reactant_reg = 1f-6,
    reactant_cg_iters = 3,
)

samples_reactant = Array(samples_reactant)

println("\n[4/4] Computing analytic solution and saving output...")

T = range(domain.t_start, domain.t_end; length = nt)

u_exact = zeros(Float32, nx, nt)

for j in 1:nt
    u_exact[:, j] .= exp(-3.0f0 * Float32(T[j])) .* Float32.(sin.(x_grid .+ Float32(pi / 4)))
end

u_analytic = zeros(Float32, nx, nt, 1, n_samples)

for s in 1:n_samples
    u_analytic[:, :, 1, s] .= u_exact
end

JLD2.save(
    "samples_heat_reactant.jld2",
    "samples_reactant", samples_reactant,
    "u_analytic", u_analytic,
    "starting_noise", starting_noise,
    "x_grid", collect(x_grid),
    "t_grid", collect(T),
    "constraint_params", constraint_params,
)

println("Saved samples_heat_reactant.jld2")

u0 = Float32.(IC_func.(x_grid))
mass0 = sum(u0[1:nx-1])

mass_violation = [
    sum(samples_reactant[1:nx-1, t, 1, s]) - mass0
    for t in 2:nt, s in 1:n_samples
]

println("Reactant heat max mass violation: ", maximum(abs.(mass_violation)))
println("Reactant heat mean mass violation: ", sum(abs.(mass_violation)) / length(mass_violation))

ic_violation = [
    samples_reactant[i, 1, 1, s] - u0[i]
    for i in 1:nx, s in 1:n_samples
]

println("Reactant heat max IC violation: ", maximum(abs.(ic_violation)))
println("Reactant heat mean IC violation: ", sum(abs.(ic_violation)) / length(ic_violation))

println("Done.")
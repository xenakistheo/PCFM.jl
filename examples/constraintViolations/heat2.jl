
using JLD2
using CairoMakie
include(joinpath(@__DIR__, "..", "..", "utils", "plotUtils.jl"))


# Load samples
data_path = joinpath(@__DIR__, "..", "..", "datasets", "samples", "samples_heat_2.jld2")
data_path2 = joinpath(@__DIR__, "..", "..", "datasets", "samples", "alaina_results_heat_1.jld2")
data = JLD2.load(data_path)
data2 = JLD2.load(data_path2)

results = data2["results"]


#TODO: Claude - Benchmark these!
samples_LBFGS = results[7].samples  # (nx, nt, 1, n_samples)
samples_IPNewton = results[9].samples  # (nx, nt, 1, n_samples)
samples_exa_gpu     = data["samples_exa_gpu"]
samples_exa_cpu     = data["samples_exa_cpu"]
samples_jump_madnlp = data["samples_jump_madnlp"]
samples_ffm         = data["samples_ffm"]
u_analytic          = data["u_analytic"]








batch_size   = 32
nx           = 100          # Spatial resolution
nt           = 100          # Temporal resolution
emb_channels = 32
t_range    = (0.0f0, 1.0f0)

x_grid = range(0.0f0, 2.0f0*Float32(π); length = nx)
dx     = Float32(x_grid[2] - x_grid[1])
dt     = 1.0f0 / (nt - 1)

# Initial condition: u(x, 0) = sin(x + π/4)
u0_ic = Float32.(sin.(x_grid .+ π/4))

X = x_grid
T = range(t_range[1], t_range[2]; length = nt)


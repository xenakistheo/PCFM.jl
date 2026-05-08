
using JLD2
using CairoMakie
include(joinpath(@__DIR__, "..", "..", "utils", "plotUtils.jl"))


# Load samples
data_path = joinpath(@__DIR__, "..", "..", "datasets", "samples", "samples_burgers_BC.jld2")
data_path2 = joinpath(@__DIR__, "..", "..", "datasets", "samples", "alaina_results_burgers_BC.jld2")
data = JLD2.load(data_path)
data2 = JLD2.load(data_path2)
results = data2["results"]

#TODO: Claude - Benchmark these!
samples_LBFGS = results[3].samples  # (nx, nt, 1, n_samples)  
samples_exa_gpu     = data["samples_exa_gpu"][1:end-1, 1:end-1, :, :]  # (nx, nt, 1, n_samples) - remove last row/col which are BCs
samples_exa_cpu     = data["samples_exa_cpu"][1:end-1, 1:end-1, :, :]
samples_jump_madnlp = data["samples_jump_madnlp"][1:end-1, 1:end-1, :, :]



nx = 100
nt = 100
x_range = (0.0f0, 1.0f0)
t_range = (0.0f0, 1.0f0)
dt = 1.0f0 / (nt - 1)
dx = 1.0f0 / (nx - 1)

X = range(x_range[1], x_range[2]; length = nx)
T = range(t_range[1], t_range[2]; length = nt)



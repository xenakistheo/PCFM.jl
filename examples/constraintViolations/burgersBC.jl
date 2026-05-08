
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
samples_LBFGS = results[1].samples  # (nx, nt, 1, n_samples)  
samples_IPNewton = results[2].samples  # (nx, nt, 1, n_samples)  
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




# Left Dirichlet BC: u[1, t] should be constant (== u[1, 1]) for all t
function left_bc_violation(u, params)
    nx, nt = params[1], params[2]
    return [abs(u[1, j] - u[1, 1]) for j in 1:nt]
end



# Neumann BC at right: u[nx, t] - u[nx-1, t] should be 0
function neumann_bc_violation(u, params)
    nx, nt = params[1], params[2]
    return [u[nx, j] - u[nx-1, j] for j in 1:nt]
end




# Mass evolution: ∑u[:,t]*dx - ∑u[:,t-1]*dx + 0.5*dt*(flux[t] + flux[t-1]) == 0
# Returns per-step residual for t in 2:nt (padded with 0 at t=1 for consistent length)
function mass_violation(u, params)
    nx, nt, dx, dt = params
    flux(t) = 0.5f0 * u[nx, t]^2 - 0.5f0 * u[1, t]^2
    residuals = Float32[0.0f0]
    for t in 2:nt
        r = sum(u[i, t] for i in 1:nx) * dx -
            sum(u[i, t-1] for i in 1:nx) * dx +
            0.5f0 * dt * (flux(t) + flux(t-1))
        push!(residuals, r)
    end
    return residuals
end


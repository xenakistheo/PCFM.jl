
using JLD2
using CairoMakie
using Statistics
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

# Violations for burgers_constraints_BC_Mass!:
#   1. Dirichlet: u[1,t,s] == left_bc[s]  (inferred as u[1,1,s])  for all t
#   2. Neumann:   u[nx,t,s] == u[nx-1,t,s]                        for all t
#   3. Mass:      ∑u[:,t,s]*dx - ∑u[:,t-1,s]*dx + 0.5*dt*(flux[t]+flux[t-1]) == 0  for t in 2:nt
# Returns simple mean of the three mean-absolute violations, averaged over samples.
function burgersBC_constraint_violations(samples, nx, nt, dx, dt)
    u = ndims(samples) == 4 ? dropdims(samples, dims=3) : samples  # (nx, nt, n_samples)
    n_samples = size(u, 3)

    dirichlet_total = 0.0
    neumann_total   = 0.0
    mass_total      = 0.0

    for s in 1:n_samples
        us = u[:, :, s]
        left_bc = us[1, 1]

        # 1. Dirichlet
        dirichlet_total += mean(abs.(us[1, :] .- left_bc))

        # 2. Neumann
        neumann_total += mean(abs.(us[nx, :] .- us[nx-1, :]))

        # 3. Mass evolution
        flux(t) = 0.5 * us[nx, t]^2 - 0.5 * us[1, t]^2
        mass_total += mean(abs(
            sum(us[:, t]) * dx - sum(us[:, t-1]) * dx
            + 0.5 * dt * (flux(t) + flux(t-1))
        ) for t in 2:nt)
    end

    dirichlet_viol = dirichlet_total / n_samples
    neumann_viol   = neumann_total   / n_samples
    mass_viol      = mass_total      / n_samples

    return (dirichlet_viol + neumann_viol + mass_viol) / 3
end

solver_names = ["LBFGS", "IPNewton", "exa_gpu", "exa_cpu", "jump_madnlp"]
all_samples  = [samples_LBFGS, samples_IPNewton, samples_exa_gpu, samples_exa_cpu, samples_jump_madnlp]

println("Constraint violations (mean absolute, averaged over samples):")
println(rpad("Solver", 20), "Violation")
for (name, samples) in zip(solver_names, all_samples)
    viol = burgersBC_constraint_violations(samples, nx, nt, dx, dt)
    println(rpad(name, 20), round(viol; sigdigits=4))
end


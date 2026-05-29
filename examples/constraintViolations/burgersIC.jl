
using JLD2
using CairoMakie
using Statistics
include(joinpath(@__DIR__, "..", "..", "utils", "plotUtils.jl"))


# Load samples
data_path = joinpath(@__DIR__, "..", "..", "datasets", "samples", "samples_burgers_IC.jld2")
data_path2 = joinpath(@__DIR__, "..", "..", "datasets", "samples", "alaina_results_burgers_BC.jld2")
data = JLD2.load(data_path)
data2 = JLD2.load(data_path2)
results = data2["results"]

#TODO: Claude - Benchmark these!
samples_LBFGS = results[3].samples  # (nx, nt, 1, n_samples)  
samples_exa_gpu = data["samples_exa_gpu"][1:end-1, 1:end-1, :, :]  # (nx, nt, 1, n_samples) - remove last row/col which are BCs
samples_exa_cpu = data["samples_exa_cpu"][1:end-1, 1:end-1, :, :]
u0_per_sample   = data["ref_samples"][1:end-1, 1, 1, :]  # (nx, n_samples) — IC from reference first time step



nx = 100
nt = 100
x_range = (0.0f0, 1.0f0)
t_range = (0.0f0, 1.0f0)
dt = 1.0f0 / (nt - 1)
dx = 1.0f0 / (nx - 1)

X = range(x_range[1], x_range[2]; length = nx)
T = range(t_range[1], t_range[2]; length = nt)

smooth_pos(x, eps) = 0.5 * (x + sqrt(x^2 + eps^2))
smooth_neg(x, eps) = 0.5 * (x - sqrt(x^2 + eps^2))

# Violations for burgers_constraints_IC_Mass_Flux!:
#   1. IC:      u[i,1,s] == u0[i,s]                              for all i
#   2. Mass:    sum(u[:,t,s])*dx == sum(u0[:,s])*dx              for all t
#   3. Godunov: u[i,t+1,s] - u[i,t,s] + λ*(F[i,t,s]-F[i-1,t,s]) == 0
#              for t in 1:k_eff, i in 2:nx-1
# Returns simple mean of the three mean-absolute violations, averaged over samples.
function burgersIC_constraint_violations(samples, u0_per_sample, nx, nt, dx, dt; k=5, eps=1e-6)
    u = ndims(samples) == 4 ? dropdims(samples, dims=3) : samples  # (nx, nt, n_samples)
    n_samples = size(u, 3)
    λ = dt / dx
    k_eff = min(k, nt - 1)

    ic_total      = 0.0
    mass_total    = 0.0
    godunov_total = 0.0

    for s in 1:n_samples
        us = u[:, :, s]
        u0 = u0_per_sample[:, s]
        M0 = sum(u0) * dx

        # 1. IC
        ic_total += mean(abs.(us[:, 1] .- u0))

        # 2. Mass (all t)
        for t in 1:nt
            mass_total += abs(sum(us[:, t]) * dx - M0)
        end

        # 3. Godunov flux steps (interior points, t in 1:k_eff)
        for t in 1:k_eff, i in 2:nx-1
            F_i   = 0.5 * smooth_pos(us[i,   t], eps)^2 + 0.5 * smooth_neg(us[i+1, t], eps)^2
            F_im1 = 0.5 * smooth_pos(us[i-1, t], eps)^2 + 0.5 * smooth_neg(us[i,   t], eps)^2
            godunov_total += abs(us[i, t+1] - us[i, t] + λ * (F_i - F_im1))
        end
    end

    ic_viol      = ic_total      / n_samples
    mass_viol    = mass_total    / (n_samples * nt)
    godunov_viol = godunov_total / (n_samples * k_eff * (nx - 2))

    return (ic_viol + mass_viol + godunov_viol) / 3
end

solver_names = ["LBFGS", "exa_gpu", "exa_cpu"]
all_samples  = [samples_LBFGS, samples_exa_gpu, samples_exa_cpu]

println("Constraint violations (mean absolute, averaged over samples):")
println(rpad("Solver", 20), "Violation")
for (name, samples) in zip(solver_names, all_samples)
    viol = burgersIC_constraint_violations(samples, u0_per_sample, nx, nt, dx, dt)
    println(rpad(name, 20), round(viol; sigdigits=4))
end

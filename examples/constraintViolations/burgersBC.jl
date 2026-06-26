using JLD2
using CairoMakie
using Statistics
include(joinpath(@__DIR__, "..", "..", "utils", "plotUtils.jl"))

# Load samples — trim ghost points, drop singleton dim 3 → (nx, nt, n_samples)
function load_raw(d, key)
    s = d[key][1:end-1, 1:end-1, :, :]   # (nx, nt, 1, n_samples)
    return dropdims(s, dims=3)             # (nx, nt, n_samples)
end

data_path  = joinpath(@__DIR__, "..", "..", "final_samples", "samples_burgers_BC.jld2")
data_path2 = joinpath(@__DIR__, "..", "..", "final_samples", "samples_burgers_BC_alaina.jld2")
data  = JLD2.load(data_path)
data2 = JLD2.load(data_path2)

analytic        = load_raw(data,  "ref_samples")
samples_exa_gpu = load_raw(data,  "samples_exa_gpu")
samples_exa_cpu = load_raw(data,  "samples_exa_cpu")
samples_madnlp  = load_raw(data,  "samples_jump_madnlp")
samples_ipopt   = load_raw(data,  "samples_jump_ipopt")
samples_lbfgs   = load_raw(data2, "samples_lbfgs")

nx = 100
nt = 100
dt = 1.0f0 / (nt - 1)
dx = 1.0f0 / (nx - 1)

X = range(0.0f0, 1.0f0; length=nx)
T = range(0.0f0, 1.0f0; length=nt)

# Violations for burgers_constraints_BC_Mass!:
#   1. Dirichlet: u[1,t,s] == u[1,1,s]          for all t
#   2. Neumann:   u[nx,t,s] == u[nx-1,t,s]       for all t
#   3. Mass:      ∑u[:,t]*dx - ∑u[:,t-1]*dx + 0.5*dt*(flux[t]+flux[t-1]) == 0  for t >= 2
# Returns mean-absolute violation for each constraint, averaged over samples.
function burgersBC_constraint_violations(samples, nx, nt, dx, dt)
    u = ndims(samples) == 4 ? dropdims(samples, dims=3) : samples  # (nx, nt, n_samples)
    n_samples = size(u, 3)

    dirichlet_total = 0.0
    neumann_total   = 0.0
    mass_total      = 0.0
    max_viol        = 0.0

    for s in 1:n_samples
        us = u[:, :, s]
        left_bc = us[1, 1]

        for t in 1:nt
            max_viol = max(max_viol, abs(us[1, t] - left_bc))
            max_viol = max(max_viol, abs(us[nx, t] - us[nx-1, t]))
        end
        dirichlet_total += mean(abs.(us[1, :] .- left_bc))
        neumann_total   += mean(abs.(us[nx, :] .- us[nx-1, :]))

        flux(t) = 0.5 * us[nx, t]^2 - 0.5 * us[1, t]^2
        for t in 2:nt
            r = abs(sum(us[:, t]) * dx - sum(us[:, t-1]) * dx
                    + 0.5 * dt * (flux(t) + flux(t-1)))
            max_viol = max(max_viol, r)
        end
        mass_total += mean(
            abs(sum(us[:, t]) * dx - sum(us[:, t-1]) * dx
                + 0.5 * dt * (flux(t) + flux(t-1)))
            for t in 2:nt)
    end

    dirichlet_viol = dirichlet_total / n_samples
    neumann_viol   = neumann_total   / n_samples
    mass_viol      = mass_total      / n_samples

    return dirichlet_viol, neumann_viol, mass_viol, max_viol
end

solver_names = ["Reference", "LBFGS", "IPOPT", "ExaGPU", "ExaCPU", "MADNLP"]
all_samples  = [analytic, samples_lbfgs, samples_ipopt, samples_exa_gpu, samples_exa_cpu, samples_madnlp]

viols          = [burgersBC_constraint_violations(s, nx, nt, dx, dt) for s in all_samples]
dirichlet_vals = [v[1] for v in viols]
neumann_vals   = [v[2] for v in viols]
mass_vals      = [v[3] for v in viols]
max_vals       = [v[4] for v in viols]

# Combined score: each constraint normalized by its mean across solvers so scale
# differences don't bias the result, then averaged into a single number.
combined = (dirichlet_vals ./ mean(dirichlet_vals)
          .+ neumann_vals  ./ mean(neumann_vals)
          .+ mass_vals     ./ mean(mass_vals)) ./ 3

println("Constraint violations (mean absolute, averaged over samples):")
println(rpad("Solver", 20), rpad("Dirichlet", 14), rpad("Neumann", 14), rpad("Mass", 14), "Combined")
for (name, d, n, m, c) in zip(solver_names, dirichlet_vals, neumann_vals, mass_vals, combined)
    println(rpad(name, 20),
            rpad(round(d; sigdigits=4), 14),
            rpad(round(n; sigdigits=4), 14),
            rpad(round(m; sigdigits=4), 14),
            round(c; sigdigits=4))
end

println("Max violations:")
for (name, max_viol) in zip(solver_names, max_vals)
    println(rpad(name, 20), round(max_viol; sigdigits=4))
end

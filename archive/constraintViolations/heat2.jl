using JLD2
using CairoMakie
using Statistics
include(joinpath(@__DIR__, "..", "..", "utils", "plotUtils.jl"))


# Load samples
data_path = joinpath(@__DIR__, "..", "..", "final_samples", "samples_heat.jld2")
data_path2 = joinpath(@__DIR__, "..", "..", "final_samples", "samples_heat2_alaina_run1.jld2")
data = JLD2.load(data_path)
data2 = JLD2.load(data_path2)

samples_exa_gpu = dropdims(mean(data["samples_exa_gpu"], dims=(3,4)), dims=(3,4))  
samples_exa_cpu = dropdims(mean(data["samples_exa_cpu"], dims=(3,4)), dims=(3,4)) 
samples_jump_madnlp = dropdims(mean(data["samples_jump_madnlp"], dims=(3,4)), dims=(3,4))
samples_jump_ipopt = dropdims(mean(data["samples_jump_ipopt"], dims=(3,4)), dims=(3,4))
samples_lbfgs = dropdims(mean(data2["samples_lbfgs"], dims=(3,4)), dims=(3,4))
# samples_ipnewton = dropdims(mean(data2["samples_ipnewton"], dims=(3,4)), dims=(3,4))

analytic = data["u_analytic"][:,:,1,1]








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

κ     = 0.01f0
k_eff = nt - 1  # default k = nt-1

# Violations for heat_constraints_IC_Mass_PDE_Energy!:
#   1. IC:     u[i,1,s] == u0_ic[i]
#   2. Mass:   sum(u[:,t,s])*dx == sum(u0_ic)*dx  for all t
#   3. PDE:    (u[i,t+1,s]-u[i,t,s])/dt - κ*(u[i+1,t,s]-2u[i,t,s]+u[i-1,t,s])/dx^2 == 0
#              for t in 1:k_eff, i in 2:nx-1
#   4. Energy: ||u^{t+1}||²dx - ||u^t||²dx + 2κdt||u_x^t||²dx == 0  for t in 1:k_eff
# Returns simple mean of the four mean-absolute violations, averaged over samples.
function heat2_constraint_violations(samples, u0_ic, nx, nt, dx, dt, κ, k_eff)
    u = ndims(samples) == 4 ? dropdims(samples, dims=3) : samples  # (nx, nt, n_samples)
    n_samples = size(u, 3)
    M0 = sum(u0_ic) * dx

    ic_total     = 0.0
    mass_total   = 0.0
    pde_total    = 0.0
    energy_total = 0.0

    max_viol = 0.0

    for s in 1:n_samples
        us = u[:, :, s]

        # 1. IC
        ic_total += mean(abs.(us[:, 1] .- u0_ic))
        for i in 1:nx
            max_viol = max(max_viol, abs(us[i, 1] - u0_ic[i]))
        end

        # 2. Mass (all t)
        for t in 1:nt
            mass_total += abs(sum(us[:, t]) * dx - M0)
            max_viol = max(max_viol, abs(sum(us[:, t]) * dx - M0))
        end

        # 3. PDE residual (interior points, t in 1:k_eff)
        for t in 1:k_eff, i in 2:nx-1
            pde_total += abs(
                (us[i, t+1] - us[i, t]) / dt
                - κ * (us[i+1, t] - 2*us[i, t] + us[i-1, t]) / dx^2
            )
            max_viol = max(max_viol, abs(
                (us[i, t+1] - us[i, t]) / dt
                - κ * (us[i+1, t] - 2*us[i, t] + us[i-1, t]) / dx^2
            ))
            max_viol = max(max_viol, abs(
                (us[i, t+1] - us[i, t]) / dt
                - κ * (us[i+1, t] - 2*us[i, t] + us[i-1, t]) / dx^2
            ))
        end

        # 4. Energy dissipation (t in 1:k_eff)
        for t in 1:k_eff
            ux_sq = sum(((us[i+1, t] - us[i, t]) / dx)^2 for i in 1:nx-1) * dx
            energy_total += abs(
                sum(us[:, t+1].^2) * dx
                - sum(us[:, t].^2) * dx
                + 2 * κ * dt * ux_sq
            )
            max_viol = max(max_viol, abs(
                sum(us[:, t+1].^2) * dx
                - sum(us[:, t].^2) * dx
                + 2 * κ * dt * ux_sq
            ))
        end
    end

    ic_viol     = ic_total     / n_samples
    mass_viol   = mass_total   / (n_samples * nt)
    pde_viol    = pde_total    / (n_samples * k_eff * (nx - 2))
    energy_viol = energy_total / (n_samples * k_eff)

    return (ic_viol + mass_viol + pde_viol + energy_viol) / 4, max_viol
end

solver_names = ["LBFGS", "exa_gpu", "exa_cpu", "jump_ipopt", "jump_madnlp"]
all_samples  = [samples_lbfgs, samples_exa_gpu, samples_exa_cpu, samples_jump_ipopt, samples_jump_madnlp]

println("Constraint violations (mean absolute, averaged over samples):")
println(rpad("Solver", 20), "Violation")
for (name, samples) in zip(solver_names, all_samples)
    viol, max_viol = heat2_constraint_violations(samples, u0_ic, nx, nt, dx, dt, κ, k_eff)
    println(rpad(name, 20), round(viol; sigdigits=4), " (max: ", round(max_viol; sigdigits=4), ")")
end


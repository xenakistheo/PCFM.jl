using JLD2
using Statistics
using LinearAlgebra

# Load samples — keep the sample dimension: (nx, nt, 1, n_samples) → (nx, nt, n_samples)
load_samples(d, key) = dropdims(d[key], dims=3)

data_path = joinpath(@__DIR__, "..", "..", "final_samples", "samples_heat_2_soltol_e5.jld2")
data_path = joinpath(@__DIR__, "..", "..", "final_samples_old", "samples_heat_2.jld2")
data_path2 = joinpath(@__DIR__, "..", "..", "final_samples_old", "samples_heat2_alaina_run1.jld2")
data = JLD2.load(data_path)
data2 = JLD2.load(data_path2)

samples_exa_gpu = load_samples(data,  "samples_exa_gpu")
samples_exa_cpu = load_samples(data,  "samples_exa_cpu")
samples_lbfgs   = load_samples(data2, "samples_lbfgs")

nx = 100          # Spatial resolution
nt = 100          # Temporal resolution

x_grid = range(0.0f0, 2.0f0*Float32(π); length = nx)
dx     = Float32(x_grid[2] - x_grid[1])
dt     = 0.01f0   # = 1/n_steps: sample_pcfm passes dt = 1/n_steps (100 steps) to the constraints

# Initial condition: u(x, 0) = sin(x + π/4)
u0_ic = Float32.(sin.(x_grid .+ π/4))

κ     = 0.01f0
k_eff = 5  # infer_heat_2.jl ran with constraint_params k = 5

# Constraint error as in pcfm.pdf Appendix J:
#   CE(τ) = (1/N) Σₙ ‖R_τ(û⁽ⁿ⁾)‖₂
# with residuals mirroring heat_constraints_IC_Mass_PDE_Energy! in src/constraints.jl:
#   IC:     R[i]   = u[i,1,s] - u0_ic[i]                              for i in 1:nx
#   Mass:   R[t]   = ∑u[:,t,s]*dx - ∑u0_ic*dx                        for t in 1:nt
#   PDE:    R[t,i] = (u[i,t+1,s]-u[i,t,s])/dt
#                    - κ*(u[i+1,t,s]-2u[i,t,s]+u[i-1,t,s])/dx²        for t in 1:k_eff, i in 2:nx-1
#   Energy: R[t]   = ||u^{t+1}||²dx - ||u^t||²dx + 2κdt||u_x^t||²dx  for t in 1:k_eff
function heat2_constraint_errors(samples, u0_ic, nx, nt, dx, dt, κ, k_eff)
    u = ndims(samples) == 4 ? dropdims(samples, dims=3) : samples  # (nx, nt, n_samples)
    n_samples = size(u, 3)
    M0 = sum(u0_ic) * dx

    ic_ce     = 0.0
    mass_ce   = 0.0
    pde_ce    = 0.0
    energy_ce = 0.0

    for s in 1:n_samples
        us = u[:, :, s]

        r_ic   = us[:, 1] .- u0_ic
        r_mass = [sum(us[:, t]) * dx - M0 for t in 1:nt]
        r_pde  = [((us[i, t+1] - us[i, t]) / dt -
                   κ * (us[i+1, t] - 2*us[i, t] + us[i-1, t]) / dx^2)
                  for t in 1:k_eff for i in 2:nx-1]
        r_energy = [(sum(us[:, t+1].^2) * dx
                     - sum(us[:, t].^2) * dx
                     + 2 * κ * dt * sum(((us[i+1, t] - us[i, t]) / dx)^2 for i in 1:nx-1) * dx)
                    for t in 1:k_eff]

        ic_ce     += norm(r_ic)
        mass_ce   += norm(r_mass)
        pde_ce    += norm(r_pde)
        energy_ce += norm(r_energy)
    end

    return ic_ce / n_samples, mass_ce / n_samples, pde_ce / n_samples, energy_ce / n_samples
end

solver_names = ["LBFGS", "exa_gpu", "exa_cpu"]
all_samples  = [samples_lbfgs, samples_exa_gpu, samples_exa_cpu]

ces         = [heat2_constraint_errors(s, u0_ic, nx, nt, dx, dt, κ, k_eff) for s in all_samples]
ic_vals     = [c[1] for c in ces]
mass_vals   = [c[2] for c in ces]
pde_vals    = [c[3] for c in ces]
energy_vals = [c[4] for c in ces]

println("Constraint error (L2 norm of residual per sample, averaged over samples — pcfm.pdf App. J):")
println(rpad("Solver", 20), rpad("IC", 14), rpad("Mass (CL)", 14), rpad("PDE", 14), "Energy")
for (name, ic, m, p, e) in zip(solver_names, ic_vals, mass_vals, pde_vals, energy_vals)
    println(rpad(name, 20),
            rpad(round(ic; sigdigits=4), 14),
            rpad(round(m;  sigdigits=4), 14),
            rpad(round(p;  sigdigits=4), 14),
            round(e; sigdigits=4))
end

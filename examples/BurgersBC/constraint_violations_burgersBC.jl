using JLD2
using Statistics
using LinearAlgebra

# Load samples — keep the FULL grid (constraints were imposed on all 101×101 points)
load_raw(d, key) = dropdims(d[key], dims=3)   # (nx, nt, n_samples)

data_path  = joinpath(@__DIR__, "..", "..", "final_samples", "samples_burgers_BC_solvertol_e7_x.jld2")
data  = JLD2.load(data_path)

analytic        = load_raw(data,  "ref_samples")
samples_exa_gpu = load_raw(data,  "samples_exa_gpu")
samples_exa_cpu = load_raw(data,  "samples_exa_cpu")
samples_madnlp  = load_raw(data,  "samples_jump_madnlp")
samples_ipopt   = load_raw(data,  "samples_jump_ipopt")
samples_lbfgs   = load_raw(data, "samples_lbfgs")

nx = 101
nt = 101
dt = 0.01f0             # = 1/n_steps: sample_pcfm passes dt = 1/n_steps (100 steps) to the constraints
dx = 1.0f0 / (nx - 1)   # x_grid = range(0, 1; length=101)

# Constraint error as in pcfm.pdf Appendix J:
#   CE(τ) = (1/N) Σₙ ‖R_τ(û⁽ⁿ⁾)‖₂
# with residuals mirroring burgers_constraints_BC_Mass! in src/constraints.jl:
#   Dirichlet: R[t]   = u[1,t,s] - left_bc[s]                          for t in 1:nt
#   Neumann:   R[t]   = u[nx,t,s] - u[nx-1,t,s]                        for t in 1:nt
#   Mass:      R[t-1] = ∑u[:,t]*dx - ∑u[:,t-1]*dx
#                       + 0.5*dt*(flux[t]+flux[t-1])                   for t in 2:nt
# BC is the paper's constraint type: L2 norm of the concatenated [Dirichlet; Neumann] residual.
function burgersBC_constraint_errors(samples, nx, nt, dx, dt)
    u = ndims(samples) == 4 ? dropdims(samples, dims=3) : samples  # (nx, nt, n_samples)
    n_samples = size(u, 3)

    dirichlet_ce = 0.0
    neumann_ce   = 0.0
    bc_ce        = 0.0
    mass_ce      = 0.0

    for s in 1:n_samples
        us = u[:, :, s]
        left_bc = us[1, 1]

        r_dirichlet = us[1, :] .- left_bc
        r_neumann   = us[nx, :] .- us[nx-1, :]

        flux(t) = 0.5 * us[nx, t]^2 - 0.5 * us[1, t]^2
        r_mass = [(sum(us[:, t]) * dx - sum(us[:, t-1]) * dx
                   + 0.5 * dt * (flux(t) + flux(t-1)))
                  for t in 2:nt]

        dirichlet_ce += norm(r_dirichlet)
        neumann_ce   += norm(r_neumann)
        bc_ce        += norm(vcat(r_dirichlet, r_neumann))
        mass_ce      += norm(r_mass)
    end

    return dirichlet_ce / n_samples, neumann_ce / n_samples, bc_ce / n_samples, mass_ce / n_samples
end

solver_names = ["Reference", "LBFGS", "IPOPT", "ExaGPU", "ExaCPU", "MADNLP"]
all_samples  = [analytic, samples_lbfgs, samples_ipopt, samples_exa_gpu, samples_exa_cpu, samples_madnlp]

ces            = [burgersBC_constraint_errors(s, nx, nt, dx, dt) for s in all_samples]
dirichlet_vals = [c[1] for c in ces]
neumann_vals   = [c[2] for c in ces]
bc_vals        = [c[3] for c in ces]
mass_vals      = [c[4] for c in ces]

println("Constraint error (L2 norm of residual per sample, averaged over samples — pcfm.pdf App. J):")
println(rpad("Solver", 20), rpad("Dirichlet", 14), rpad("Neumann", 14), rpad("BC (both)", 14), "Mass (CL)")
for (name, d, n, b, m) in zip(solver_names, dirichlet_vals, neumann_vals, bc_vals, mass_vals)
    println(rpad(name, 20),
            rpad(round(d; sigdigits=4), 14),
            rpad(round(n; sigdigits=4), 14),
            rpad(round(b; sigdigits=4), 14),
            round(m; sigdigits=4))
end

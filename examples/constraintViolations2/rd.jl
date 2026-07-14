using JLD2
using Statistics
using LinearAlgebra

# Load samples
data_path  = joinpath(@__DIR__, "..", "..", "final_samples", "samples_rd_soltol_e5.jld2")
data_path2 = joinpath(@__DIR__, "..", "..", "final_samples_old", "samples_rd_alaina_run3.jld2")

data = JLD2.load(data_path)
data2 = JLD2.load(data_path2)

u0_fixed  = data["u0_fixed"]
rd_params = data["rd_params"]

samples_exa_gpu     = data["samples_exa_gpu"]
samples_exa_cpu     = data["samples_exa_cpu"]
samples_jump_madnlp = data["samples_jump_madnlp"]
samples_jump_ipopt  = data["samples_jump_ipopt"]
samples_lbfgs       = data2["samples_lbfgs"]
samples_ipnewton    = data2["samples_ipnewton"]

nt = 100          # Temporal resolution
dt = 0.01f0       # = 1/n_steps: sample_pcfm passes dt = 1/n_steps (100 steps) to the constraints

# Constraint error as in pcfm.pdf Appendix J:
#   CE(τ) = (1/N) Σₙ ‖R_τ(û⁽ⁿ⁾)‖₂
# with residuals mirroring rd_constraints_2! in src/constraints.jl — the constraint
# actually imposed by examples/infer_rd.jl (trapezoidal reaction + 1st-order boundary flux):
#   IC:   R[i]   = u[i,1,s] - u0[i]                                     for i in 1:nx
#   Mass: R[t-1] = M[t]-M[t-1] - 0.5*dt*rho*(S[t]+S[t-1])
#                  - 0.5*dt*(F[t]+F[t-1])                               for t in 2:nt
# dx and u0 are inferred from sample nx: u0_fixed applies only to nx=64 samples
# (IC residual is 0 for other resolutions, where u0 falls back to the first time step).
function rd_constraint_errors(samples, u0_fixed, rho, nu, dt)
    u = ndims(samples) == 4 ? dropdims(samples, dims=3) : samples  # (nx_s, nt_s, n_samples)
    nx_s, nt_s, n_samples = size(u)
    dx_s = 1.0f0 / (nx_s - 1)

    # Use u0_fixed if it matches nx_s, otherwise fall back to first time step (IC residual = 0)
    has_u0 = length(u0_fixed) == nx_s

    ic_ce   = 0.0
    mass_ce = 0.0

    for s in 1:n_samples
        us = u[:, :, s]
        u0 = has_u0 ? u0_fixed : us[:, 1]

        r_ic = us[:, 1] .- u0

        # 1st-order boundary flux, as in rd_constraints_2!
        flux_left(t_)  = -nu * (us[2, t_]    - us[1, t_])      / dx_s
        flux_right(t_) = -nu * (us[nx_s, t_] - us[nx_s-1, t_]) / dx_s

        r_mass = zeros(nt_s - 1)
        for t in 2:nt_s
            mass_diff = (sum(us[:, t]) - sum(us[:, t-1])) * dx_s
            reaction  = 0.5 * dt * rho * (
                sum(us[i, t]   * (1 - us[i, t])   for i in 1:nx_s) * dx_s +
                sum(us[i, t-1] * (1 - us[i, t-1]) for i in 1:nx_s) * dx_s
            )
            diffusion = 0.5 * dt * (
                (flux_left(t)   - flux_right(t)) +
                (flux_left(t-1) - flux_right(t-1))
            )
            r_mass[t-1] = mass_diff - reaction - diffusion
        end

        ic_ce   += norm(r_ic)
        mass_ce += norm(r_mass)
    end

    return ic_ce / n_samples, mass_ce / n_samples
end

solver_names = ["LBFGS (nx=100)", "IPNewton (nx=100)", "exa_gpu", "exa_cpu", "jump_madnlp", "jump_ipopt"]
all_samples  = [samples_lbfgs, samples_ipnewton, samples_exa_gpu, samples_exa_cpu, samples_jump_madnlp, samples_jump_ipopt]

ces       = [rd_constraint_errors(s, u0_fixed, rd_params.rho, rd_params.nu, dt) for s in all_samples]
ic_vals   = [c[1] for c in ces]
mass_vals = [c[2] for c in ces]

println("Constraint error (L2 norm of residual per sample, averaged over samples — pcfm.pdf App. J):")
println("Note: IC error is 0 for nx=100 solvers (u0_fixed is nx=64 only)")
println(rpad("Solver", 22), rpad("IC", 14), "Mass (CL)")
for (name, ic, m) in zip(solver_names, ic_vals, mass_vals)
    println(rpad(name, 22),
            rpad(round(ic; sigdigits=4), 14),
            round(m; sigdigits=4))
end

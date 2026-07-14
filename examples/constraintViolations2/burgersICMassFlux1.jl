using JLD2
using Statistics
using LinearAlgebra

# Load samples — keep the FULL grid (constraints were imposed on all 101×101 points)
load_raw(d, key) = dropdims(d[key], dims=3)   # (nx, nt, n_samples)

data_path = joinpath(@__DIR__, "..", "..", "final_samples_old", "samples_burgers_IC_Flux_1.jld2")
data = JLD2.load(data_path)

analytic        = load_raw(data, "ref_samples")
samples_exa_gpu = load_raw(data, "samples_exa_gpu")
samples_exa_cpu = load_raw(data, "samples_exa_cpu")

nx = 101
nt = 101
dt = 0.01f0             # = 1/n_steps: sample_pcfm passes dt = 1/n_steps (100 steps) to the constraints
dx = 1.0f0 / (nx - 1)   # x_grid = range(0, 1; length=101)

# Imposed IC (same for all samples): sample_pcfm sets u0 = IC_func.(x_grid),
# the sigmoid from infer_burgers_IC.jl with p_loc = 0.5, eps = 0.02
x_grid = range(0.0f0, 1.0f0; length=nx)
p_loc  = 0.5f0
eps_ic = 0.02f0
u0_ic  = Float32.(1.0f0 ./ (1.0f0 .+ exp.((x_grid .- p_loc) ./ eps_ic)))  # (nx,)

smooth_pos(x, eps) = 0.5 * (x + sqrt(x^2 + eps^2))
smooth_neg(x, eps) = 0.5 * (x - sqrt(x^2 + eps^2))

# Constraint error as in pcfm.pdf Appendix J:
#   CE(τ) = (1/N) Σₙ ‖R_τ(û⁽ⁿ⁾)‖₂
# with residuals mirroring burgers_constraints_IC_Mass_Flux! in src/constraints.jl (k=1):
#   IC:      R[i]   = u[i,1,s] - u0_ic[i]                                   for i in 1:nx
#   Mass:    R[t]   = ∑u[:,t,s]*dx - ∑u0_ic*dx                            for t in 1:nt
#   Godunov: R[t,i] = u[i,t+1,s] - u[i,t,s] + λ*(F[i,t,s]-F[i-1,t,s])     for t in 1:k_eff, i in 2:nx-1
function burgersICMassFlux_constraint_errors(samples, u0_ic, nx, nt, dx, dt; k=1, eps=1e-6)
    u = ndims(samples) == 4 ? dropdims(samples, dims=3) : samples  # (nx, nt, n_samples)
    n_samples = size(u, 3)
    λ = dt / dx
    k_eff = min(k, nt - 1)

    ic_ce      = 0.0
    mass_ce    = 0.0
    godunov_ce = 0.0

    for s in 1:n_samples
        us = u[:, :, s]
        u0 = u0_ic
        M0 = sum(u0) * dx

        r_ic   = us[:, 1] .- u0
        r_mass = [sum(us[:, t]) * dx - M0 for t in 1:nt]

        r_godunov = zeros(k_eff * (nx - 2))
        idx = 1
        for t in 1:k_eff, i in 2:nx-1
            F_i   = 0.5 * smooth_pos(us[i,   t], eps)^2 + 0.5 * smooth_neg(us[i+1, t], eps)^2
            F_im1 = 0.5 * smooth_pos(us[i-1, t], eps)^2 + 0.5 * smooth_neg(us[i,   t], eps)^2
            r_godunov[idx] = us[i, t+1] - us[i, t] + λ * (F_i - F_im1)
            idx += 1
        end

        ic_ce      += norm(r_ic)
        mass_ce    += norm(r_mass)
        godunov_ce += norm(r_godunov)
    end

    return ic_ce / n_samples, mass_ce / n_samples, godunov_ce / n_samples
end

solver_names = ["Reference", "ExaGPU", "ExaCPU"]
all_samples  = [analytic, samples_exa_gpu, samples_exa_cpu]

ces          = [burgersICMassFlux_constraint_errors(s, u0_ic, nx, nt, dx, dt; k=1) for s in all_samples]
ic_vals      = [c[1] for c in ces]
mass_vals    = [c[2] for c in ces]
godunov_vals = [c[3] for c in ces]

println("Constraint error (L2 norm of residual per sample, averaged over samples — pcfm.pdf App. J):")
println(rpad("Solver", 20), rpad("IC", 14), rpad("Mass (CL)", 14), "Godunov(k=1)")
for (name, ic, m, g) in zip(solver_names, ic_vals, mass_vals, godunov_vals)
    println(rpad(name, 20),
            rpad(round(ic; sigdigits=4), 14),
            rpad(round(m;  sigdigits=4), 14),
            round(g; sigdigits=4))
end

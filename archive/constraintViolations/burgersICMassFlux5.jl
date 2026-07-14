using JLD2
using Statistics



# Load samples
data_path  = joinpath(@__DIR__, "..", "..", "final_samples", "samples_burgers_IC.jld2")
data = JLD2.load(data_path)

function load_raw(d, key)
    s = d[key][1:end-1, 1:end-1, :, :]   # (nx, nt, 1, n_samples)
    return dropdims(s, dims=3)             # (nx, nt, n_samples)
end

analytic        = load_raw(data, "ref_samples")
samples_exa_gpu = load_raw(data, "samples_exa_gpu")
samples_exa_cpu = load_raw(data, "samples_exa_cpu")

nx = 100
nt = 100
dt = 1.0f0 / (nt - 1)
dx = 1.0f0 / (nx - 1)

u0_per_sample = analytic[:, 1, :]  # (nx, n_samples)

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
    max_viol      = 0.0

    for s in 1:n_samples
        us = u[:, :, s]
        u0 = u0_per_sample[:, s]
        M0 = sum(u0) * dx

        # 1. IC
        ic_total += mean(abs.(us[:, 1] .- u0))
        for i in 1:nx
            max_viol = max(max_viol, abs(us[i, 1] - u0[i]))
        end

        # 2. Mass (all t)
        for t in 1:nt
            r = abs(sum(us[:, t]) * dx - M0)
            mass_total += r
            max_viol = max(max_viol, r)
        end

        # 3. Godunov flux steps (interior points, t in 1:k_eff)
        for t in 1:k_eff, i in 2:nx-1
            F_i   = 0.5 * smooth_pos(us[i,   t], eps)^2 + 0.5 * smooth_neg(us[i+1, t], eps)^2
            F_im1 = 0.5 * smooth_pos(us[i-1, t], eps)^2 + 0.5 * smooth_neg(us[i,   t], eps)^2
            r = abs(us[i, t+1] - us[i, t] + λ * (F_i - F_im1))
            godunov_total += r
            max_viol = max(max_viol, r)
        end
    end

    ic_viol      = ic_total      / n_samples
    mass_viol    = mass_total    / (n_samples * nt)
    godunov_viol = godunov_total / (n_samples * k_eff * (nx - 2))

    return ic_viol, mass_viol, godunov_viol, max_viol
end

solver_names = ["Reference", "ExaGPU", "ExaCPU"]
all_samples  = [analytic, samples_exa_gpu, samples_exa_cpu]

viols        = [burgersIC_constraint_violations(s, u0_per_sample, nx, nt, dx, dt) for s in all_samples]
ic_vals      = [v[1] for v in viols]
mass_vals    = [v[2] for v in viols]
godunov_vals = [v[3] for v in viols]
max_vals     = [v[4] for v in viols]

combined = (ic_vals      ./ mean(ic_vals)
          .+ mass_vals    ./ mean(mass_vals)
          .+ godunov_vals ./ mean(godunov_vals)) ./ 3

println("Constraint violations (mean absolute, averaged over samples):")
println(rpad("Solver", 20), rpad("IC", 14), rpad("Mass", 14), rpad("Godunov(k=5)", 16), "Combined")
for (name, ic, m, g, c) in zip(solver_names, ic_vals, mass_vals, godunov_vals, combined)
    println(rpad(name, 20),
            rpad(round(ic; sigdigits=4), 14),
            rpad(round(m;  sigdigits=4), 14),
            rpad(round(g;  sigdigits=4), 16),
            round(c; sigdigits=4))
end

println("Max violations:")
for (name, max_viol) in zip(solver_names, max_vals)
    println(rpad(name, 20), round(max_viol; sigdigits=4))
end

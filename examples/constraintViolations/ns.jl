
using JLD2
using Statistics

# Load samples — shape (nx, ny, nt, 1, n_samples); keep all dims for time-resolved violation checks
data_path  = joinpath(@__DIR__, "..", "..", "final_samples", "samples_ns.jld2")
data_path2 = joinpath(@__DIR__, "..", "..", "final_samples", "samples_ns_alaina.jld2")
data  = JLD2.load(data_path)
data2 = JLD2.load(data_path2)

samples_exa_gpu     = data["samples_exa_gpu"]
samples_exa_cpu     = data["samples_exa_cpu"]
samples_jump_madnlp = data["samples_jump_madnlp"]
samples_jump_ipopt  = data["samples_jump_ipopt"]
samples_lbfgs       = data2["samples_lbfgs"]
reference           = data["ref_samples"]

s  = 16
nt = 50
x_grid = range(0.0f0, 1.0f0; length=s+1)[1:end-1]
y_grid = range(0.0f0, 1.0f0; length=s+1)[1:end-1]
dx = Float32(x_grid[2] - x_grid[1])
dy = Float32(y_grid[2] - y_grid[1])

u0 = Float32.(sin.(2f0 * Float32(π) .* y_grid'))  # (s, s), same IC for all samples

# Violations for ns_enstrophy_constraints!:
#   1. IC:         u[i,j,1,s] == u0[i,j]                                 for all i,j,s
#   2. Mass:       ∑_{i,j} u[i,j,t,s]*dx*dy == ∑_{i,j} u0*dx*dy         for t >= 2
#   3. Enstrophy:  ∑_{i,j} u[i,j,t,s]²*dx*dy == ∑_{i,j} u0²*dx*dy       for t >= 2
# Each metric is the mean absolute violation, averaged over samples.
function ns_enstrophy_constraint_violations(samples, u0, nt, dx, dy)
    u = ndims(samples) == 5 ? dropdims(samples, dims=4) : samples  # (nx, ny, nt, n_s)
    nx, ny, _, n_s = size(u)

    M0 = sum(u0) * dx * dy
    E0 = sum(u0 .^ 2) * dx * dy

    ic_total        = 0.0
    mass_total      = 0.0
    enstrophy_total = 0.0
    max_viol        = 0.0

    for s in 1:n_s
        us = u[:, :, :, s]  # (nx, ny, nt)

        ic_total += mean(abs.(us[:, :, 1] .- u0))

        for t in 2:nt
            mass_total      += abs(sum(us[:, :, t]) * dx * dy - M0)
            enstrophy_total += abs(sum(us[:, :, t] .^ 2) * dx * dy - E0)
            max_viol = max(max_viol, abs(sum(us[:, :, t]) * dx * dy - M0))
            max_viol = max(max_viol, abs(sum(us[:, :, t] .^ 2) * dx * dy - E0))
        end
    end

    ic_viol        = ic_total        / n_s
    mass_viol      = mass_total      / (n_s * (nt - 1))
    enstrophy_viol = enstrophy_total / (n_s * (nt - 1))

    return ic_viol, mass_viol, enstrophy_viol, max_viol
end

solver_names = ["Reference", "ExaGPU", "ExaCPU", "MADNLP", "IPOPT", "LBFGS"]
all_samples  = [reference, samples_exa_gpu, samples_exa_cpu, samples_jump_madnlp, samples_jump_ipopt, samples_lbfgs]

viols     = [ns_enstrophy_constraint_violations(s, u0, nt, dx, dy) for s in all_samples]
ic_vals   = [v[1] for v in viols]
mass_vals = [v[2] for v in viols]
enst_vals = [v[3] for v in viols]
max_vals  = [v[4] for v in viols]

# Combined score: each constraint normalized by its mean across solvers so scale
# differences don't bias the result, then averaged into a single number.
combined = (ic_vals    ./ mean(ic_vals)
          .+ mass_vals ./ mean(mass_vals)
          .+ enst_vals ./ mean(enst_vals)) ./ 3

println("Constraint violations (mean absolute, averaged over samples):")
println(rpad("Solver", 20), rpad("IC", 14), rpad("Mass", 14), rpad("Enstrophy", 14), "Combined")
for (name, ic, m, e, c) in zip(solver_names, ic_vals, mass_vals, enst_vals, combined)
    println(rpad(name, 20),
            rpad(round(ic; sigdigits=4), 14),
            rpad(round(m;  sigdigits=4), 14),
            rpad(round(e;  sigdigits=4), 14),
            round(c; sigdigits=4))
end

println("Max violations:")
for (name, max_viol) in zip(solver_names, max_vals)
    println(rpad(name, 20), round(max_viol; sigdigits=4))
end

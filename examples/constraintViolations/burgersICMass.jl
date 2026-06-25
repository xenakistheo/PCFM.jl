using JLD2
using Statistics

function load_raw(d, key)
    s = d[key][1:end-1, 1:end-1, :, :]   # (nx, nt, 1, n_samples)
    return dropdims(s, dims=3)             # (nx, nt, n_samples)
end

data_path = joinpath(@__DIR__, "..", "..", "final_samples", "samples_burgers_IC_Mass.jld2")
data = JLD2.load(data_path)

analytic        = load_raw(data, "ref_samples")
samples_exa_gpu = load_raw(data, "samples_exa_gpu")
samples_exa_cpu = load_raw(data, "samples_exa_cpu")
samples_madnlp  = load_raw(data, "samples_jump_madnlp")

nx = 100
nt = 100
dt = 1.0f0 / (nt - 1)
dx = 1.0f0 / (nx - 1)

u0_per_sample = analytic[:, 1, :]  # (nx, n_samples)

# Constraints:
#   1. IC:   u[:,1,s] == u0[:,s]
#   2. Mass: sum(u[:,t,s])*dx == sum(u0[:,s])*dx  for all t
function burgersICMass_constraint_violations(samples, u0_per_sample, nx, nt, dx, dt)
    u = ndims(samples) == 4 ? dropdims(samples, dims=3) : samples  # (nx, nt, n_samples)
    n_samples = size(u, 3)

    ic_total   = 0.0
    mass_total = 0.0

    for s in 1:n_samples
        us = u[:, :, s]
        u0 = u0_per_sample[:, s]
        M0 = sum(u0) * dx

        ic_total += mean(abs.(us[:, 1] .- u0))

        for t in 1:nt
            mass_total += abs(sum(us[:, t]) * dx - M0)
        end
    end

    ic_viol   = ic_total   / n_samples
    mass_viol = mass_total / (n_samples * nt)

    return ic_viol, mass_viol
end

solver_names = ["Reference", "ExaGPU", "ExaCPU", "MADNLP"]
all_samples  = [analytic, samples_exa_gpu, samples_exa_cpu, samples_madnlp]

viols     = [burgersICMass_constraint_violations(s, u0_per_sample, nx, nt, dx, dt) for s in all_samples]
ic_vals   = [v[1] for v in viols]
mass_vals = [v[2] for v in viols]

combined = (ic_vals ./ mean(ic_vals) .+ mass_vals ./ mean(mass_vals)) ./ 2

println("Constraint violations (mean absolute, averaged over samples):")
println(rpad("Solver", 20), rpad("IC", 14), rpad("Mass", 14), "Combined")
for (name, ic, m, c) in zip(solver_names, ic_vals, mass_vals, combined)
    println(rpad(name, 20),
            rpad(round(ic; sigdigits=4), 14),
            rpad(round(m;  sigdigits=4), 14),
            round(c; sigdigits=4))
end

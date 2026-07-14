using JLD2
using Statistics

function load_raw(d, key)
    s = d[key][1:end-1, 1:end-1, :, :]   # (nx, nt, 1, n_samples)
    return dropdims(s, dims=3)             # (nx, nt, n_samples)
end

data_path  = joinpath(@__DIR__, "..", "..", "final_samples", "samples_burgers_IC_only.jld2")
data_path2 = joinpath(@__DIR__, "..", "..", "final_samples", "samples_burgers_IC_only_alaina.jld2")
data  = JLD2.load(data_path)
data2 = JLD2.load(data_path2)

analytic         = load_raw(data,  "ref_samples")
samples_exa_gpu  = load_raw(data,  "samples_exa_gpu")
samples_exa_cpu  = load_raw(data,  "samples_exa_cpu")
samples_madnlp   = load_raw(data,  "samples_jump_madnlp")
samples_ipopt    = load_raw(data,  "samples_jump_ipopt")
samples_lbfgs    = load_raw(data2, "samples_ic_only")
samples_ipnewton = load_raw(data2, "samples_ic_only_ip")

nx = 100
nt = 100
dt = 1.0f0 / (nt - 1)
dx = 1.0f0 / (nx - 1)

# u0 per sample: first time step of the reference solution → (nx, n_samples)
u0_per_sample = analytic[:, 1, :]

# Constraint: u[:,1,s] == u0[:,s]
function burgersIC_constraint_violations(samples, u0_per_sample, nx, nt, dx, dt)
    u = ndims(samples) == 4 ? dropdims(samples, dims=3) : samples  # (nx, nt, n_samples)
    n_samples = size(u, 3)

    ic_total = 0.0
    max_viol = 0.0
    for s in 1:n_samples
        ic_total += mean(abs.(u[:, 1, s] .- u0_per_sample[:, s]))
        for i in 1:nx
            max_viol = max(max_viol, abs(u[i, 1, s] - u0_per_sample[i, s]))
        end
    end

    return ic_total / n_samples, max_viol
end

solver_names = ["Reference", "LBFGS", "IPNewton", "IPOPT", "ExaGPU", "ExaCPU", "MADNLP"]
all_samples  = [analytic, samples_lbfgs, samples_ipnewton, samples_ipopt,
                samples_exa_gpu, samples_exa_cpu, samples_madnlp]

viols    = [burgersIC_constraint_violations(s, u0_per_sample, nx, nt, dx, dt) for s in all_samples]
ic_vals  = [v[1] for v in viols]
max_vals = [v[2] for v in viols]

println("Constraint violations (mean absolute, averaged over samples):")
println(rpad("Solver", 20), "IC")
for (name, ic) in zip(solver_names, ic_vals)
    println(rpad(name, 20), round(ic; sigdigits=4))
end

println("Max violations:")
for (name, max_viol) in zip(solver_names, max_vals)
    println(rpad(name, 20), round(max_viol; sigdigits=4))
end

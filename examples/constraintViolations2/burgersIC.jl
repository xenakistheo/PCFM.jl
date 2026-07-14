using JLD2
using Statistics
using LinearAlgebra

# Load samples — keep the FULL grid (constraints were imposed on all 101×101 points)
load_raw(d, key) = dropdims(d[key], dims=3)   # (nx, nt, n_samples)

data_path  = joinpath(@__DIR__, "..", "..", "final_samples_old", "samples_burgers_IC_only.jld2")
data_path2 = joinpath(@__DIR__, "..", "..", "final_samples_old", "samples_burgers_IC_only_alaina.jld2")
data  = JLD2.load(data_path)
data2 = JLD2.load(data_path2)

analytic         = load_raw(data,  "ref_samples")
samples_exa_gpu  = load_raw(data,  "samples_exa_gpu")
samples_exa_cpu  = load_raw(data,  "samples_exa_cpu")
samples_madnlp   = load_raw(data,  "samples_jump_madnlp")
samples_ipopt    = load_raw(data,  "samples_jump_ipopt")
samples_lbfgs    = load_raw(data2, "samples_ic_only")
samples_ipnewton = load_raw(data2, "samples_ic_only_ip")

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

# Constraint error as in pcfm.pdf Appendix J:
#   CE(τ) = (1/N) Σₙ ‖R_τ(û⁽ⁿ⁾)‖₂
# with residuals mirroring burgers_constraints_IC! in src/constraints.jl:
#   IC: R[i] = u[i,1,s] - u0_ic[i]  for i in 1:nx
function burgersIC_constraint_errors(samples, u0_ic, nx, nt, dx, dt)
    u = ndims(samples) == 4 ? dropdims(samples, dims=3) : samples  # (nx, nt, n_samples)
    n_samples = size(u, 3)

    ic_ce = 0.0
    for s in 1:n_samples
        r_ic = u[:, 1, s] .- u0_ic
        ic_ce += norm(r_ic)
    end

    return ic_ce / n_samples
end

solver_names = ["Reference", "LBFGS", "IPNewton", "IPOPT", "ExaGPU", "ExaCPU", "MADNLP"]
all_samples  = [analytic, samples_lbfgs, samples_ipnewton, samples_ipopt,
                samples_exa_gpu, samples_exa_cpu, samples_madnlp]

ic_vals = [burgersIC_constraint_errors(s, u0_ic, nx, nt, dx, dt) for s in all_samples]

println("Constraint error (L2 norm of residual per sample, averaged over samples — pcfm.pdf App. J):")
println(rpad("Solver", 20), "IC")
for (name, ic) in zip(solver_names, ic_vals)
    println(rpad(name, 20), round(ic; sigdigits=4))
end

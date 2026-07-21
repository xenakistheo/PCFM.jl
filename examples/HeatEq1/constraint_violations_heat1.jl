
using JLD2
using Statistics
using LinearAlgebra

# Load samples — keep the sample dimension: (nx, nt, 1, n_samples) → (nx, nt, n_samples)
load_samples(d, key) = dropdims(d[key], dims=3)

data_path = joinpath(@__DIR__, "..", "..", "final_samples", "samples_heat_soltol_e5.jld2")
data = JLD2.load(data_path)


samples_exa_gpu     = load_samples(data,  "samples_exa_gpu")
samples_exa_cpu     = load_samples(data,  "samples_exa_cpu")
samples_jump_madnlp = load_samples(data,  "samples_jump_madnlp")
samples_jump_ipopt  = load_samples(data,  "samples_jump_ipopt")
samples_lbfgs       = load_samples(data, "samples_lbfgs")
samples_ipnewton    = load_samples(data, "samples_ipnewton")

nx = 100          # Spatial resolution
nt = 100          # Temporal resolution

x_grid = range(0.0f0, 2.0f0*Float32(π); length = nx)
dx     = Float32(x_grid[2] - x_grid[1])
dt     = 1.0f0 / (nt - 1)

# Initial condition: u(x, 0) = sin(x + π/4)
u0_ic = Float32.(sin.(x_grid .+ π/4))

# Constraint error as in pcfm.pdf Appendix J:
#   CE(τ) = (1/N) Σₙ ‖R_τ(û⁽ⁿ⁾)‖₂
# with residuals mirroring heat_constraints! in src/constraints.jl:
#   IC:   R_IC[i]   = u[i,1,s] - u0_ic[i]                          for i in 1:nx
#   Mass: R_CL[t-1] = ∑_{i=1}^{nx-1} u[i,t,s] - ∑_{i=1}^{nx-1} u0  for t in 2:nt
function heat_constraint_errors(samples, u0_ic, nx, nt)
    u = ndims(samples) == 4 ? dropdims(samples, dims=3) : samples  # (nx, nt, n_samples)
    n_samples = size(u, 3)
    m0 = sum(u0_ic[1:nx-1])

    ic_ce   = 0.0
    mass_ce = 0.0

    for s in 1:n_samples
        r_ic   = [u[i, 1, s] - u0_ic[i] for i in 1:nx]
        r_mass = [sum(u[1:nx-1, t, s]) - m0 for t in 2:nt]
        ic_ce   += norm(r_ic)
        mass_ce += norm(r_mass)
    end

    return ic_ce / n_samples, mass_ce / n_samples
end

solver_names = ["LBFGS", "IPNewton", "exa_gpu", "exa_cpu", "jump_madnlp", "jump_ipopt"]
all_samples  = [samples_lbfgs, samples_ipnewton, samples_exa_gpu, samples_exa_cpu, samples_jump_madnlp, samples_jump_ipopt]

ces       = [heat_constraint_errors(s, u0_ic, nx, nt) for s in all_samples]
ic_vals   = [c[1] for c in ces]
mass_vals = [c[2] for c in ces]

println("Constraint error (L2 norm of residual per sample, averaged over samples — pcfm.pdf App. J):")
println(rpad("Solver", 20), rpad("IC", 14), "Mass (CL)")
for (name, ic, m) in zip(solver_names, ic_vals, mass_vals)
    println(rpad(name, 20),
            rpad(round(ic; sigdigits=4), 14),
            round(m; sigdigits=4))
end

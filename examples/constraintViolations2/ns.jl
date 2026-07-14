
using JLD2
using Statistics
using LinearAlgebra

# Load samples — shape (nx, ny, nt, 1, n_samples); keep all dims for per-sample residuals
data_path  = joinpath(@__DIR__, "..", "..", "final_samples", "samples_ns_soltol_e5.jld2")
data_path2 = joinpath(@__DIR__, "..", "..", "final_samples_old", "samples_ns_alaina.jld2")
data  = JLD2.load(data_path)
data2 = JLD2.load(data_path2)

samples_exa_gpu     = data["samples_exa_gpu"]
samples_exa_cpu     = data["samples_exa_cpu"]
samples_jump_madnlp = data["samples_jump_madnlp"]
samples_jump_ipopt  = data["samples_jump_ipopt"]
samples_lbfgs       = data2["samples_lbfgs"]

# Grid as used inside sample_pcfm_2d: range(domain.x_start, domain.x_end, length=nx)
# with domain (0,1) — i.e. INCLUDING the endpoint, spacing 1/(s-1) (not the periodic 1/s grid).
s  = 16
nt = 50
x_grid = range(0.0f0, 1.0f0; length=s)
y_grid = range(0.0f0, 1.0f0; length=s)
dx = Float32(x_grid[2] - x_grid[1])
dy = Float32(y_grid[2] - y_grid[1])

# Imposed IC: u0[i,j] = IC_func_ns(x_i, y_j) = sin(2π y_j), same for all samples
u0 = Float32.([sin(2f0 * Float32(π) * y) for x in x_grid, y in y_grid])  # (s, s)

# Constraint error as in pcfm.pdf Appendix J:
#   CE(τ) = (1/N) Σₙ ‖R_τ(û⁽ⁿ⁾)‖₂
# with residuals mirroring ns_enstrophy_constraints! in src/constraints.jl:
#   IC:        R[i,j] = u[i,j,1,s] - u0[i,j]                        for all i,j
#   Mass:      R[t-1] = ∑_{i,j} u[i,j,t,s] - ∑_{i,j} u0            for t in 2:nt (no dx*dy, as imposed)
#   Enstrophy: R[t-1] = ∑_{i,j} u[i,j,t,s]²*dx*dy - ∑ u0²*dx*dy    for t in 2:nt
function ns_enstrophy_constraint_errors(samples, u0, nt, dx, dy)
    u = ndims(samples) == 5 ? dropdims(samples, dims=4) : samples  # (nx, ny, nt, n_s)
    n_s = size(u, 4)

    M0 = sum(u0)
    E0 = sum(u0 .^ 2) * dx * dy

    ic_ce        = 0.0
    mass_ce      = 0.0
    enstrophy_ce = 0.0

    for s in 1:n_s
        us = u[:, :, :, s]  # (nx, ny, nt)

        r_ic        = vec(us[:, :, 1] .- u0)
        r_mass      = [sum(us[:, :, t]) - M0 for t in 2:nt]
        r_enstrophy = [sum(us[:, :, t] .^ 2) * dx * dy - E0 for t in 2:nt]

        ic_ce        += norm(r_ic)
        mass_ce      += norm(r_mass)
        enstrophy_ce += norm(r_enstrophy)
    end

    return ic_ce / n_s, mass_ce / n_s, enstrophy_ce / n_s
end

solver_names = ["ExaGPU", "ExaCPU", "MADNLP", "IPOPT", "LBFGS"]
all_samples  = [samples_exa_gpu, samples_exa_cpu, samples_jump_madnlp, samples_jump_ipopt, samples_lbfgs]

ces       = [ns_enstrophy_constraint_errors(s, u0, nt, dx, dy) for s in all_samples]
ic_vals   = [c[1] for c in ces]
mass_vals = [c[2] for c in ces]
enst_vals = [c[3] for c in ces]

println("Constraint error (L2 norm of residual per sample, averaged over samples — pcfm.pdf App. J):")
println(rpad("Solver", 20), rpad("IC", 14), rpad("Mass (CL)", 14), "Enstrophy")
for (name, ic, m, e) in zip(solver_names, ic_vals, mass_vals, enst_vals)
    println(rpad(name, 20),
            rpad(round(ic; sigdigits=4), 14),
            rpad(round(m;  sigdigits=4), 14),
            round(e; sigdigits=4))
end

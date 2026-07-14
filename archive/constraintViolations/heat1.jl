
using JLD2
using CairoMakie
using Statistics
# include(joinpath(@__DIR__, "..", "..", "utils", "plotUtils.jl"))


# Load samples
data_path = joinpath(@__DIR__, "..", "..", "final_samples", "samples_heat_solvertol_e5.jld2")
data_path2 = joinpath(@__DIR__, "..", "..", "final_samples_old", "samples_heat_alaina_run4.jld2")
data = JLD2.load(data_path)
data2 = JLD2.load(data_path2)



samples_exa_gpu = dropdims(mean(data["samples_exa_gpu"], dims=(3,4)), dims=(3,4))  
samples_exa_cpu = dropdims(mean(data["samples_exa_cpu"], dims=(3,4)), dims=(3,4)) 
samples_jump_madnlp = dropdims(mean(data["samples_jump_madnlp"], dims=(3,4)), dims=(3,4))
samples_jump_ipopt = dropdims(mean(data["samples_jump_ipopt"], dims=(3,4)), dims=(3,4))
samples_lbfgs = dropdims(mean(data2["samples_lbfgs"], dims=(3,4)), dims=(3,4))
samples_ipnewton = dropdims(mean(data2["samples_ipnewton"], dims=(3,4)), dims=(3,4))

analytic = data["u_analytic"][:,:,1,1]








batch_size   = 32
nx           = 100          # Spatial resolution
nt           = 100          # Temporal resolution
emb_channels = 32
t_range    = (0.0f0, 1.0f0)

x_grid = range(0.0f0, 2.0f0*Float32(π); length = nx)
dx     = Float32(x_grid[2] - x_grid[1])
dt     = 1.0f0 / (nt - 1)

# Initial condition: u(x, 0) = sin(x + π/4)
u0_ic = Float32.(sin.(x_grid .+ π/4))

X = x_grid
T = range(t_range[1], t_range[2]; length = nt)


function mass_constraint(u, params)
    Nx, Nt = params
    return [sum((u[i, j] - u[i,1]) for i in 1:(Nx-1)) for j in 1:Nt]
end

function ic_violation(u, params)
    nx, nt = params[1], params[2]
    return [sum(abs(u[j, i] - u[1, i]) for i in 1:nx) for j in 1:nt]
end

# Returns (ic_viol, mass_viol): mean absolute violation per constraint, averaged over samples.
# Handles both (nx, nt, n_samples) and (nx, nt, 1, n_samples) shapes.
function heat_constraint_violations(samples, u0_ic, nx, nt)
    u = ndims(samples) == 4 ? dropdims(samples, dims=3) : samples  # (nx, nt, n_samples)
    n_samples = size(u, 3)
    m0 = sum(u0_ic[1:nx-1])

    ic_total   = 0.0
    mass_total = 0.
    max_viol = 0.0

    for s in 1:n_samples
        # IC: mean |u[i,1,s] - u0_ic[i]| over i
        ic_total += mean(abs.(u[:, 1, s] .- u0_ic))
        
        for i in 1:nx
            max_viol = max(max_viol, abs(u[i, 1, s] - u0_ic[i]))
        end

        # Mass: mean |sum(u[1:nx-1,t,s]) - m0| over t in 2:nt
        for t in 2:nt
            mass_total += abs(sum(u[1:nx-1, t, s]) - m0)
            max_viol = max(max_viol, abs(sum(u[1:nx-1, t, s]) - m0))

        end
    end

    ic_viol   = ic_total / n_samples
    mass_viol = mass_total / (n_samples * (nt - 1))
    return (ic_viol + mass_viol) / 2, max_viol
end

solver_names = ["LBFGS", "IPNewton", "exa_gpu", "exa_cpu", "jump_madnlp", "jump_ipopt"]
all_samples  = [samples_lbfgs, samples_ipnewton, samples_exa_gpu, samples_exa_cpu, samples_jump_madnlp, samples_jump_ipopt]

println("Constraint violations (mean absolute, averaged over samples):")
println(rpad("Solver", 20), "Violation")
for (name, samples) in zip(solver_names, all_samples)
    viol, max_viol  = heat_constraint_violations(samples, u0_ic, nx, nt)
    println(rpad(name, 20), round(viol; sigdigits=4), " (max: ", round(max_viol; sigdigits=4), ")")
end

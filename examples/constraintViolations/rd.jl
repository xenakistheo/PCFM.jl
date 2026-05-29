
using JLD2
using CairoMakie
using HDF5
using Statistics


###
# Load samples
data_path = joinpath(@__DIR__, "..", "..", "datasets", "samples", "samples_rd.jld2")
data_path2 = joinpath(@__DIR__, "..", "..", "datasets", "samples", "alaina_results_rd.jld2")

data = JLD2.load(data_path)
data2 = JLD2.load(data_path2)
results = data2["results"]

#TODO: Claude - Benchmark these!
samples_LBFGS = results[1].samples  # (nx, nt, 1, n_samples)
samples_IPNewton = results[2].samples  # (nx, nt, 1, n_samples)
samples_exa_gpu     = data["samples_exa_gpu"]
samples_exa_cpu     = data["samples_exa_cpu"]
samples_jump_madnlp = data["samples_jump_madnlp"]
samples_jump_ipopt = data["samples_jump_ipopt"]
u0_fixed            = data["u0_fixed"]
rd_params           = data["rd_params"]


batch_size   = 32
nx           = 64          # Spatial resolution
nt           = 100          # Temporal resolution
emb_channels = 32
t_range    = (0.0f0, 1.0f0)

x_grid = range(0.0f0, 1.0f0; length = nx)
dx     = Float32(x_grid[2] - x_grid[1])
dt     = 1.0f0 / (nt - 1)

# Initial condition: u(x, 0) = sin(x + π/4)
# u0_ic = Float32.(sin.(x_grid .+ π/4))

X = x_grid
T = range(t_range[1], t_range[2]; length = nt)




# Constraint 1: IC — returns pointwise error over x (length nx)
# params = (u0_fixed, nx)
function ic_violation_rd(u, params)
    u0, nx = params
    return [abs(u[i, 1] - u0[i]) for i in 1:nx]
end

# Constraint 2: mass evolution — returns residual of the integral PDE constraint over t (length nt-1)
# mirrors rd_constraints_2! exactly: trapezoidal reaction + boundary diffusion flux
# params = (nx, nt, dx, dt, rho, nu)
function mass_evolution_violation_rd(u, params)
    nx, nt, dx, dt, rho, nu = params
    residuals = zeros(Float32, nt - 1)
    for t in 2:nt
        mass_diff = (sum(u[:, t]) - sum(u[:, t-1])) * dx
        reaction  = 0.5f0*dt*rho * (
            sum(u[i, t  ] * (1 - u[i, t  ]) for i in 1:nx) * dx +
            sum(u[i, t-1] * (1 - u[i, t-1]) for i in 1:nx) * dx
        )
        flux_t   = nu * ((u[nx, t  ] - u[nx-1, t  ]) - (u[2, t  ] - u[1, t  ])) / dx
        flux_tm1 = nu * ((u[nx, t-1] - u[nx-1, t-1]) - (u[2, t-1] - u[1, t-1])) / dx
        diffusion = 0.5f0*dt * (flux_t + flux_tm1)
        residuals[t-1] = mass_diff - reaction - diffusion
    end
    return residuals
end


mass_params = (nx, nt, dx, dt, rd_params.rho, rd_params.nu)

# Violations for rd_constraints!:
#   1. IC:   u[i,1,s] == u0[i]  for all i
#            (u0_fixed used when sample nx==64; first time step used otherwise — IC viol = 0)
#   2. Mass: sum(u[:,t,s])*dx - sum(u[:,t-1,s])*dx
#            - 0.5*dt*rho*(reaction_t + reaction_{t-1})
#            - 0.5*dt*(flux4_left_t - flux4_right_t + flux4_left_{t-1} - flux4_right_{t-1}) == 0
#            using 4th-order one-sided boundary flux as in rd_constraints!
# dx and u0 are inferred from sample nx: u0_fixed applies only to nx=64 samples.
function rd_constraint_violations(samples, u0_fixed, rho, nu, dt)
    u = ndims(samples) == 4 ? dropdims(samples, dims=3) : samples  # (nx_s, nt_s, n_samples)
    nx_s, nt_s, n_samples = size(u)
    dx_s = 1.0f0 / (nx_s - 1)

    # Use u0_fixed if it matches nx_s, otherwise fall back to first time step (IC viol = 0)
    has_u0 = length(u0_fixed) == nx_s

    ic_total   = 0.0
    mass_total = 0.0

    for s in 1:n_samples
        us = u[:, :, s]
        u0 = has_u0 ? u0_fixed : us[:, 1]

        # 1. IC
        ic_total += mean(abs.(us[:, 1] .- u0))

        # 2. Mass evolution with 4th-order boundary flux (mirrors rd_constraints!)
        for t in 2:nt_s
            mass_diff = (sum(us[:, t]) - sum(us[:, t-1])) * dx_s
            reaction  = 0.5 * dt * rho * (
                sum(us[i, t]   * (1 - us[i, t])   for i in 1:nx_s) * dx_s +
                sum(us[i, t-1] * (1 - us[i, t-1]) for i in 1:nx_s) * dx_s
            )
            function flux4_left(t_)
                -nu * (-25*us[1,t_] + 48*us[2,t_] - 36*us[3,t_] + 16*us[4,t_] - 3*us[5,t_]) / (12*dx_s)
            end
            function flux4_right(t_)
                -nu * (25*us[nx_s,t_] - 48*us[nx_s-1,t_] + 36*us[nx_s-2,t_] - 16*us[nx_s-3,t_] + 3*us[nx_s-4,t_]) / (12*dx_s)
            end
            diffusion = 0.5 * dt * (
                (flux4_left(t)   - flux4_right(t)) +
                (flux4_left(t-1) - flux4_right(t-1))
            )
            mass_total += abs(mass_diff - reaction - diffusion)
        end
    end

    ic_viol   = ic_total   / n_samples
    mass_viol = mass_total / (n_samples * (nt_s - 1))
    return (ic_viol + mass_viol) / 2
end

solver_names = ["LBFGS (nx=100)", "IPNewton (nx=100)", "exa_gpu", "exa_cpu", "jump_madnlp", "jump_ipopt", "ffm"]
all_samples  = [samples_LBFGS, samples_IPNewton, samples_exa_gpu, samples_exa_cpu, samples_jump_madnlp, samples_jump_ipopt, samples_ffm]

println("Constraint violations (mean absolute, averaged over samples):")
println("Note: IC violation is 0 for nx=100 solvers (u0_fixed is nx=64 only)")
println(rpad("Solver", 22), "Violation")
for (name, samples) in zip(solver_names, all_samples)
    viol = rd_constraint_violations(samples, u0_fixed, rd_params.rho, rd_params.nu, dt)
    println(rpad(name, 22), round(viol; sigdigits=4))
end

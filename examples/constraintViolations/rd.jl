
using JLD2
using CairoMakie
using HDF5


###
# Load samples
data_path = joinpath(@__DIR__, "..", "..", "datasets", "samples", "samples_rd.jld2")
data_path2 = joinpath(@__DIR__, "..", "..", "datasets", "samples", "alaina_results_rd.jld2")

data = JLD2.load(data_path)
data2 = JLD2.load(data_path2)


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

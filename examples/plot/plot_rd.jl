
using JLD2
using CairoMakie
include(joinpath(@__DIR__, "..", "..", "utils", "plotUtils.jl"))




###
# Load Test Set
using HDF5

test_data_path = joinpath(@__DIR__, "..", "..", "datasets", "data", "RD_neumann_test_nIC30_nBC30_nx64.h5")
u_reference, ref_ic, ref_bc = h5open(test_data_path, "r") do f
    permutedims(read(f["u"]), (2, 1, 3, 4)),  # (nx=64, nt=100, n_bc=30, n_ic=30)
    read(f["ic"]),                             # (nx=64, n_ic=30)
    read(f["bc"])                              # (2,     n_bc=30)
end



###
# Load samples
data_path = joinpath(@__DIR__, "..", "..", "datasets", "samples", "samples_rd.jld2")
data = JLD2.load(data_path)

data

samples_exa_gpu     = data["samples_exa_gpu"]
samples_exa_cpu     = data["samples_exa_cpu"]
samples_jump_madnlp = data["samples_jump_madnlp"]
samples_jump_ipopt = data["samples_jump_ipopt"]
samples_ffm         = data["samples_ffm"]
u0_fixed            = data["u0_fixed"]
rd_params           = data["rd_params"]
# u_reference

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



##### Simulated solution
K = 3

fig_samples = plot_sample(K, [u_reference, samples_exa_gpu, samples_exa_cpu, samples_jump_madnlp, samples_jump_ipopt, samples_ffm],
    ["Reference", "Exa - MadNLP (GPU)", "Exa - MadNLP", "JuMP - MadNLP", "JuMP - Ipopt", "FFM"]; suptitle="Heat Eq. Samples")

# save("plots/samples_rd.png", fig_samples)


# ##### Deviation from Analytic
# fig_samples_deviation = plot_sample(K, [samples_exa_gpu .- u_analytic, samples_jump_madnlp .- u_analytic, samples_ffm .- u_analytic],
#     ["ExaModels", "JuMP", "FFM"]; suptitle="Deviation from analytic solution - Heat Eq.")

# # save("plots/samples_heat_deviation.png", fig_samples)



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

fig_constraint_ic = plot_constraint_violation(K,
    [samples_exa_gpu, samples_exa_cpu, samples_jump_madnlp, samples_jump_ipopt, samples_ffm],
    ic_violation_rd,
    ["Exa - MadNLP (GPU)", "Exa - MadNLP", "JuMP - MadNLP", "JuMP - Ipopt", "FFM"];
    constraint_params=(u0_fixed, nx),
    suptitle="IC Constraint Violation - RD")

# save("plots/ic_constraint_rd.png", fig_constraint_ic)

fig_constraint_mass = plot_constraint_violation(K,
    [samples_exa_gpu, samples_exa_cpu, samples_jump_madnlp, samples_jump_ipopt, samples_ffm],
    mass_evolution_violation_rd,
    ["Exa - MadNLP (GPU)", "Exa - MadNLP", "JuMP - MadNLP", "JuMP - Ipopt", "FFM"];
    constraint_params=(nx, nt, dx, dt, rd_params.rho, rd_params.nu),
    suptitle="Mass Evolution Constraint Violation - RD")

# save("plots/mass_constraint_rd.png", fig_constraint_mass)

samples_exa_gpu # nx, nt, 1, n_samples = 64, 100, 1, 32
# Constraint over time 
mass_evolution_violation_rd(samples_exa_gpu[:, :, 1, 1], (nx, nt, dx, dt, rd_params.rho, rd_params.nu))

using JLD2
using CairoMakie
include(joinpath(@__DIR__, "..", "..", "utils", "plotUtils.jl"))


# Load samples
data_path = joinpath(@__DIR__, "..", "..", "datasets", "samples", "samples_burgers.jld2")
data = JLD2.load(data_path)

samples_exa_gpu     = data["samples_exa_gpu"]
samples_exa_cpu     = data["samples_exa_cpu"]
samples_jump_madnlp = data["samples_jump_madnlp"]
samples_ffm         = data["samples_ffm"]
ref_samples         = data["ref_samples"]

nx = 101
nt = 101
x_range = (0.0f0, 1.0f0)
t_range = (0.0f0, 1.0f0)
dt = 1.0f0 / (nt - 1)
dx = 1.0f0 / (nx - 1)

X = range(x_range[1], x_range[2]; length = nx)
T = range(t_range[1], t_range[2]; length = nt)


##### Simulated solution
K = 1

fig_samples = plot_sample(K, [ref_samples, samples_exa_gpu, samples_jump_madnlp, samples_ffm],
    ["Reference", "ExaModels", "JuMP", "FFM"]; suptitle="Burgers Eq. Samples")

save("plots/samples_burgers.png", fig_samples)


##### Deviation from Reference
fig_samples_deviation = plot_sample(K, [samples_exa_gpu .- ref_samples, samples_exa_cpu .- ref_samples, samples_jump_madnlp .- ref_samples, samples_ffm .- ref_samples],
    ["ExaModels GPU", "ExaModels CPU", "JuMP", "FFM"]; suptitle="Deviation from reference - Burgers Eq.")

save("plots/samples_burgers_deviation.png", fig_samples_deviation)


# Left Dirichlet BC: u[1, t] should be constant (== u[1, 1]) for all t
function left_bc_violation(u, params)
    nx, nt = params[1], params[2]
    return [abs(u[1, j] - u[1, 1]) for j in 1:nt]
end

fig_left_bc = plot_constraint_violation(K,
    [ref_samples, samples_exa_cpu, samples_jump_madnlp, samples_ffm],
    left_bc_violation,
    ["Reference", "ExaCPU", "JuMP", "FFM"];
    constraint_params=(nx, nt, dx, dt),
    suptitle="Left Dirichlet BC Violation - Burgers Eq.")

save("plots/left_bc_violation_burgers.png", fig_left_bc)


# Neumann BC at right: u[nx, t] - u[nx-1, t] should be 0
function neumann_bc_violation(u, params)
    nx, nt = params[1], params[2]
    return [u[nx, j] - u[nx-1, j] for j in 1:nt]
end

fig_neumann_bc = plot_constraint_violation(K,
    [ref_samples, samples_exa_cpu, samples_jump_madnlp, samples_ffm],
    neumann_bc_violation,
    ["Reference", "ExaModels", "JuMP", "FFM"];
    constraint_params=(nx, nt, dx, dt),
    suptitle="Right Neumann BC Violation - Burgers Eq.")

save("plots/neumann_bc_violation_burgers.png", fig_neumann_bc)


# Mass evolution: ∑u[:,t]*dx - ∑u[:,t-1]*dx + 0.5*dt*(flux[t] + flux[t-1]) == 0
# Returns per-step residual for t in 2:nt (padded with 0 at t=1 for consistent length)
function mass_violation(u, params)
    nx, nt, dx, dt = params
    flux(t) = 0.5f0 * u[nx, t]^2 - 0.5f0 * u[1, t]^2
    residuals = Float32[0.0f0]
    for t in 2:nt
        r = sum(u[i, t] for i in 1:nx) * dx -
            sum(u[i, t-1] for i in 1:nx) * dx +
            0.5f0 * dt * (flux(t) + flux(t-1))
        push!(residuals, r)
    end
    return residuals
end

fig_mass = plot_constraint_violation(K,
    [ref_samples, samples_exa_cpu, samples_jump_madnlp, samples_ffm],
    mass_violation,
    ["Reference", "ExaModels", "JuMP", "FFM"];
    constraint_params=(nx, nt, dx, dt),
    suptitle="Mass Evolution Violation - Burgers Eq.")

save("plots/mass_violation_burgers.png", fig_mass)

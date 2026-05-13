using JLD2
using CairoMakie

include(joinpath(@__DIR__, "..", "..", "utils", "plotUtils.jl"))

mkpath("plots")

reactant_data_path = joinpath(@__DIR__, "..", "..", "samples_rd_reactant.jld2")
if !isfile(reactant_data_path)
    reactant_data_path = joinpath(@__DIR__, "..", "..", "datasets", "samples", "samples_rd_reactant.jld2")
end

reactant_data = JLD2.load(reactant_data_path)

samples_reactant = reactant_data["samples_reactant"]

ref_samples = nothing
reference_data_path = joinpath(@__DIR__, "..", "..", "datasets", "samples", "samples_rd.jld2")
if isfile(reference_data_path)
    reference_data = JLD2.load(reference_data_path)
    if haskey(reference_data, "ref_samples")
        ref_samples = reference_data["ref_samples"]
    end
end

nx = size(samples_reactant, 1)
nt = size(samples_reactant, 2)
batch_size = size(samples_reactant, 4)

x_range = (0.0f0, 1.0f0)
t_range = (0.0f0, 1.0f0)

dt = 1.0f0 / Float32(nt - 1)
dx = 1.0f0 / Float32(nx - 1)

X = haskey(reactant_data, "x_grid") ? reactant_data["x_grid"] : range(x_range[1], x_range[2]; length = nx)
T = range(t_range[1], t_range[2]; length = nt)

params = haskey(reactant_data, "constraint_params") ? reactant_data["constraint_params"] : (rho = 0.01f0, nu = 0.005f0)

rho = get(params, :rho, 0.01f0)
nu = get(params, :nu, 0.005f0)

if haskey(reactant_data, "u0_fixed")
    u0 = Float32.(reactant_data["u0_fixed"])
else
    u0 = Float32.(samples_reactant[:, 1, 1, 1])
end

K = 1

if ref_samples !== nothing
    fig_samples = plot_sample(
        K,
        [ref_samples, samples_reactant],
        ["Reference", "Reactant"];
        suptitle = "Reaction-Diffusion Samples - Reactant",
    )

    save("plots/samples_rd_reactant.png", fig_samples)

    fig_samples_deviation = plot_sample(
        K,
        [samples_reactant .- ref_samples],
        ["Reactant"];
        suptitle = "Deviation from reference - Reaction-Diffusion Reactant",
    )

    save("plots/samples_rd_deviation_reactant.png", fig_samples_deviation)
else
    fig_samples = plot_sample(
        K,
        [samples_reactant],
        ["Reactant"];
        suptitle = "Reaction-Diffusion Samples - Reactant",
    )

    save("plots/samples_rd_reactant.png", fig_samples)
end

function ic_violation(u, params)
    nx, nt, dx, dt, u0 = params
    return [sum(abs(u[i, 1] - u0[i]) for i in 1:nx) for _ in 1:nt]
end

if ref_samples !== nothing
    fig_ic = plot_constraint_violation(
        K,
        [ref_samples, samples_reactant],
        ic_violation,
        ["Reference", "Reactant"];
        constraint_params = (nx, nt, dx, dt, u0),
        suptitle = "IC Violation - Reaction-Diffusion Reactant",
    )
else
    fig_ic = plot_constraint_violation(
        K,
        [samples_reactant],
        ic_violation,
        ["Reactant"];
        constraint_params = (nx, nt, dx, dt, u0),
        suptitle = "IC Violation - Reaction-Diffusion Reactant",
    )
end

save("plots/ic_violation_rd_reactant.png", fig_ic)

function left_deriv(u, t, params)
    nx, nt, dx, dt, rho, nu = params

    return (
        -25f0 * u[1, t] +
         48f0 * u[2, t] -
         36f0 * u[3, t] +
         16f0 * u[4, t] -
          3f0 * u[5, t]
    ) / (12f0 * dx)
end

function right_deriv(u, t, params)
    nx, nt, dx, dt, rho, nu = params

    return (
         25f0 * u[nx, t] -
         48f0 * u[nx - 1, t] +
         36f0 * u[nx - 2, t] -
         16f0 * u[nx - 3, t] +
          3f0 * u[nx - 4, t]
    ) / (12f0 * dx)
end

function mass_violation(u, params)
    nx, nt, dx, dt, rho, nu = params

    residuals = Float32[0.0f0]

    for t in 2:nt
        M_t = sum(u[i, t] for i in 1:nx) * dx
        M_prev = sum(u[i, t - 1] for i in 1:nx) * dx

        S_t = sum(u[i, t] * (1f0 - u[i, t]) for i in 1:nx) * dx
        S_prev = sum(u[i, t - 1] * (1f0 - u[i, t - 1]) for i in 1:nx) * dx

        flux_t =
            -nu * left_deriv(u, t, params) +
             nu * right_deriv(u, t, params)

        flux_prev =
            -nu * left_deriv(u, t - 1, params) +
             nu * right_deriv(u, t - 1, params)

        r =
            M_t -
            M_prev -
            0.5f0 * dt * rho * (S_t + S_prev) -
            0.5f0 * dt * (flux_t + flux_prev)

        push!(residuals, r)
    end

    return residuals
end

if ref_samples !== nothing
    fig_mass = plot_constraint_violation(
        K,
        [ref_samples, samples_reactant],
        mass_violation,
        ["Reference", "Reactant"];
        constraint_params = (nx, nt, dx, dt, rho, nu),
        suptitle = "Mass Evolution Violation - Reaction-Diffusion Reactant",
    )
else
    fig_mass = plot_constraint_violation(
        K,
        [samples_reactant],
        mass_violation,
        ["Reactant"];
        constraint_params = (nx, nt, dx, dt, rho, nu),
        suptitle = "Mass Evolution Violation - Reaction-Diffusion Reactant",
    )
end

save("plots/mass_violation_rd_reactant.png", fig_mass)

ic_violation_reactant = [
    ic_violation(samples_reactant[:, :, 1, s], (nx, nt, dx, dt, u0))
    for s in 1:batch_size
]

mass_violation_reactant = [
    mass_violation(samples_reactant[:, :, 1, s], (nx, nt, dx, dt, rho, nu))
    for s in 1:batch_size
]

JLD2.save(
    "plots/violations_rd_reactant.jld2",
    "ic_violation_reactant", ic_violation_reactant,
    "mass_violation_reactant", mass_violation_reactant,
    "x_grid", collect(X),
    "t_grid", collect(T),
)

println("Saved plots:")
println("plots/samples_rd_reactant.png")
if ref_samples !== nothing
    println("plots/samples_rd_deviation_reactant.png")
end
println("plots/ic_violation_rd_reactant.png")
println("plots/mass_violation_rd_reactant.png")

println("Saved raw violations:")
println("plots/violations_rd_reactant.jld2")

ic_flat = reduce(vcat, ic_violation_reactant)
mass_flat = reduce(vcat, mass_violation_reactant)

println("Reactant max IC violation: ", maximum(abs.(ic_flat)))
println("Reactant mean IC violation: ", sum(abs.(ic_flat)) / length(ic_flat))

println("Reactant max mass violation: ", maximum(abs.(mass_flat)))
println("Reactant mean mass violation: ", sum(abs.(mass_flat)) / length(mass_flat))
using JLD2
using CairoMakie

include(joinpath(@__DIR__, "..", "..", "utils", "plotUtils.jl"))

mkpath("plots")

reactant_data_path = joinpath(@__DIR__, "..", "..", "samples_burgers_BC_reactant.jld2")
if !isfile(reactant_data_path)
    reactant_data_path = joinpath(@__DIR__, "..", "..", "datasets", "samples", "samples_burgers_BC_reactant.jld2")
end

reactant_data = JLD2.load(reactant_data_path)

samples_reactant = reactant_data["samples_reactant"]

ref_samples = nothing
reference_data_path = joinpath(@__DIR__, "..", "..", "datasets", "samples", "samples_burgers_BC.jld2")
if isfile(reference_data_path)
    reference_data = JLD2.load(reference_data_path)
    if haskey(reference_data, "ref_samples")
        ref_samples = reference_data["ref_samples"]
    end
end

nx = size(samples_reactant, 1)
nt = size(samples_reactant, 2)

x_range = (0.0f0, 1.0f0)
t_range = (0.0f0, 1.0f0)
dt = 1.0f0 / Float32(nt - 1)
dx = 1.0f0 / Float32(nx - 1)

X = haskey(reactant_data, "x_grid") ? reactant_data["x_grid"] : range(x_range[1], x_range[2]; length = nx)
T = range(t_range[1], t_range[2]; length = nt)

K = 1

if ref_samples !== nothing
    fig_samples = plot_sample(
        K,
        [ref_samples, samples_reactant],
        ["Reference", "Reactant"];
        suptitle = "Burgers Eq. Samples - Reactant",
    )

    save("plots/samples_burgers_BC_reactant.png", fig_samples)

    fig_samples_deviation = plot_sample(
        K,
        [samples_reactant .- ref_samples],
        ["Reactant"];
        suptitle = "Deviation from reference - Burgers Eq. Reactant",
    )

    save("plots/samples_burgers_BC_deviation_reactant.png", fig_samples_deviation)
else
    fig_samples = plot_sample(
        K,
        [samples_reactant],
        ["Reactant"];
        suptitle = "Burgers Eq. Samples - Reactant",
    )

    save("plots/samples_burgers_BC_reactant.png", fig_samples)
end

function left_bc_violation(u, params)
    nx, nt = params[1], params[2]
    return [abs(u[1, j] - u[1, 1]) for j in 1:nt]
end

if ref_samples !== nothing
    fig_left_bc = plot_constraint_violation(
        K,
        [ref_samples, samples_reactant],
        left_bc_violation,
        ["Reference", "Reactant"];
        constraint_params = (nx, nt, dx, dt),
        suptitle = "Left Dirichlet BC Violation - Burgers Eq. Reactant",
    )
else
    fig_left_bc = plot_constraint_violation(
        K,
        [samples_reactant],
        left_bc_violation,
        ["Reactant"];
        constraint_params = (nx, nt, dx, dt),
        suptitle = "Left Dirichlet BC Violation - Burgers Eq. Reactant",
    )
end

save("plots/left_bc_violation_burgers_BC_reactant.png", fig_left_bc)

function neumann_bc_violation(u, params)
    nx, nt = params[1], params[2]
    return [u[nx, j] - u[nx - 1, j] for j in 1:nt]
end

if ref_samples !== nothing
    fig_neumann_bc = plot_constraint_violation(
        K,
        [ref_samples, samples_reactant],
        neumann_bc_violation,
        ["Reference", "Reactant"];
        constraint_params = (nx, nt, dx, dt),
        suptitle = "Right Neumann BC Violation - Burgers Eq. Reactant",
    )
else
    fig_neumann_bc = plot_constraint_violation(
        K,
        [samples_reactant],
        neumann_bc_violation,
        ["Reactant"];
        constraint_params = (nx, nt, dx, dt),
        suptitle = "Right Neumann BC Violation - Burgers Eq. Reactant",
    )
end

save("plots/neumann_bc_violation_burgers_BC_reactant.png", fig_neumann_bc)

function mass_violation(u, params)
    nx, nt, dx, dt = params
    flux(t) = 0.5f0 * u[nx, t]^2 - 0.5f0 * u[1, t]^2
    residuals = Float32[0.0f0]
    for t in 2:nt
        r = sum(u[i, t] for i in 1:nx) * dx -
            sum(u[i, t - 1] for i in 1:nx) * dx +
            0.5f0 * dt * (flux(t) + flux(t - 1))
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
        constraint_params = (nx, nt, dx, dt),
        suptitle = "Mass Evolution Violation - Burgers Eq. Reactant",
    )
else
    fig_mass = plot_constraint_violation(
        K,
        [samples_reactant],
        mass_violation,
        ["Reactant"];
        constraint_params = (nx, nt, dx, dt),
        suptitle = "Mass Evolution Violation - Burgers Eq. Reactant",
    )
end

save("plots/mass_violation_burgers_BC_reactant.png", fig_mass)

batch_size = size(samples_reactant, 4)

left_bc_violation_reactant = [
    left_bc_violation(samples_reactant[:, :, 1, s], (nx, nt, dx, dt))
    for s in 1:batch_size
]

neumann_bc_violation_reactant = [
    neumann_bc_violation(samples_reactant[:, :, 1, s], (nx, nt, dx, dt))
    for s in 1:batch_size
]

mass_violation_reactant = [
    mass_violation(samples_reactant[:, :, 1, s], (nx, nt, dx, dt))
    for s in 1:batch_size
]

JLD2.save(
    "plots/violations_burgers_BC_reactant.jld2",
    "left_bc_violation_reactant", left_bc_violation_reactant,
    "neumann_bc_violation_reactant", neumann_bc_violation_reactant,
    "mass_violation_reactant", mass_violation_reactant,
    "x_grid", collect(X),
    "t_grid", collect(T),
)

println("Saved plots:")
println("plots/samples_burgers_BC_reactant.png")
if ref_samples !== nothing
    println("plots/samples_burgers_BC_deviation_reactant.png")
end
println("plots/left_bc_violation_burgers_BC_reactant.png")
println("plots/neumann_bc_violation_burgers_BC_reactant.png")
println("plots/mass_violation_burgers_BC_reactant.png")

println("Saved raw violations:")
println("plots/violations_burgers_BC_reactant.jld2")

left_flat = reduce(vcat, left_bc_violation_reactant)
neumann_flat = reduce(vcat, neumann_bc_violation_reactant)
mass_flat = reduce(vcat, mass_violation_reactant)

println("Reactant max left BC violation: ", maximum(abs.(left_flat)))
println("Reactant mean left BC violation: ", sum(abs.(left_flat)) / length(left_flat))

println("Reactant max Neumann BC violation: ", maximum(abs.(neumann_flat)))
println("Reactant mean Neumann BC violation: ", sum(abs.(neumann_flat)) / length(neumann_flat))

println("Reactant max mass violation: ", maximum(abs.(mass_flat)))
println("Reactant mean mass violation: ", sum(abs.(mass_flat)) / length(mass_flat))
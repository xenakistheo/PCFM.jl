using JLD2
using CairoMakie

include(joinpath(@__DIR__, "..", "..", "utils", "plotUtils.jl"))

mkpath("plots")

data_path = joinpath(@__DIR__, "..", "..", "samples_heat_reactant.jld2")

if !isfile(data_path)
    data_path = joinpath(@__DIR__, "..", "..", "datasets", "samples", "samples_heat_reactant.jld2")
end

data = JLD2.load(data_path)

samples_reactant = data["samples_reactant"]
u_analytic = data["u_analytic"]

batch_size = size(samples_reactant, 4)
nx = size(samples_reactant, 1)
nt = size(samples_reactant, 2)
emb_channels = 32

t_range = (0.0f0, 1.0f0)

x_grid = haskey(data, "x_grid") ? data["x_grid"] : range(0.0f0, 2.0f0 * Float32(π); length = nx)
dx = Float32(x_grid[2] - x_grid[1])
dt = 1.0f0 / Float32(nt - 1)

u0_ic = Float32.(sin.(x_grid .+ Float32(π / 4)))

X = x_grid
T = haskey(data, "t_grid") ? data["t_grid"] : range(t_range[1], t_range[2]; length = nt)

K = 1

fig_samples = plot_sample(
    K,
    [u_analytic, samples_reactant],
    ["Analytic", "Reactant"];
    suptitle = "Heat Eq. Samples — Reactant",
)

save("plots/samples_heat_reactant.png", fig_samples)

fig_samples_deviation = plot_sample(
    K,
    [samples_reactant .- u_analytic],
    ["Reactant"];
    suptitle = "Deviation from analytic solution - Heat Eq. Reactant",
)

save("plots/samples_heat_deviation_reactant.png", fig_samples_deviation)

function mass_constraint(u, params)
    Nx, Nt = params[1], params[2]
    return [sum((u[i, j] - u[i, 1]) for i in 1:(Nx - 1)) for j in 1:Nt]
end

fig_constraint_mass = plot_constraint_violation(
    K,
    [samples_reactant],
    mass_constraint,
    ["Reactant"];
    constraint_params = (nx, nt, dx, dt),
    suptitle = "Mass Constraint Violation - Heat Eq. Reactant",
)

save("plots/mass_constraint_violation_heat_reactant.png", fig_constraint_mass)

function ic_violation(u, params)
    Nx, Nt = params[1], params[2]
    return [sum(abs(u[i, 1] - u0_ic[i]) for i in 1:Nx) for _ in 1:Nt]
end

fig_constraint_ic = plot_constraint_violation(
    K,
    [u_analytic, samples_reactant],
    ic_violation,
    ["Analytic", "Reactant"];
    constraint_params = (nx, nt, dx, dt),
    suptitle = "IC Constraint Violation - Heat Eq. Reactant",
)

save("plots/IC_constraint_violation_heat_reactant.png", fig_constraint_ic)

mass_violation_reactant = [
    mass_constraint(samples_reactant[:, :, 1, s], (nx, nt, dx, dt))
    for s in 1:batch_size
]

ic_violation_reactant = [
    ic_violation(samples_reactant[:, :, 1, s], (nx, nt, dx, dt))
    for s in 1:batch_size
]

JLD2.save(
    "plots/violations_heat_reactant.jld2",
    "mass_violation_reactant", mass_violation_reactant,
    "ic_violation_reactant", ic_violation_reactant,
    "x_grid", collect(x_grid),
    "t_grid", collect(T),
)

println("Saved plots:")
println("plots/samples_heat_reactant.png")
println("plots/samples_heat_deviation_reactant.png")
println("plots/mass_constraint_violation_heat_reactant.png")
println("plots/IC_constraint_violation_heat_reactant.png")
println("Saved raw violations:")
println("plots/violations_heat_reactant.jld2")

println("Reactant max mass violation: ", maximum(abs.(reduce(vcat, mass_violation_reactant))))
println("Reactant mean mass violation: ", sum(abs.(reduce(vcat, mass_violation_reactant))) / length(reduce(vcat, mass_violation_reactant)))

println("Reactant max IC violation: ", maximum(abs.(reduce(vcat, ic_violation_reactant))))
println("Reactant mean IC violation: ", sum(abs.(reduce(vcat, ic_violation_reactant))) / length(reduce(vcat, ic_violation_reactant)))
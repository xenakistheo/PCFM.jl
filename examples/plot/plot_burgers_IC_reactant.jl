using JLD2
using CairoMakie

include(joinpath(@__DIR__, "..", "..", "utils", "plotUtils.jl"))

mkpath("plots")

reactant_data_path = joinpath(@__DIR__, "..", "..", "samples_burgers_IC_reactant.jld2")
if !isfile(reactant_data_path)
    reactant_data_path = joinpath(@__DIR__, "..", "..", "datasets", "samples", "samples_burgers_IC_reactant.jld2")
end

reactant_data = JLD2.load(reactant_data_path)

samples_reactant = reactant_data["samples_reactant"]

ref_samples = nothing
reference_data_path = joinpath(@__DIR__, "..", "..", "datasets", "samples", "samples_burgers_IC.jld2")
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

params = haskey(reactant_data, "constraint_params") ? reactant_data["constraint_params"] : (k = 5, eps = 1f-6)

k = get(params, :k, 5)
eps = get(params, :eps, 1f-6)
k_eff = min(k, nt - 1)
λ = dt / dx

p_loc = 0.5f0
eps_ic = 0.02f0
IC_func_burgers = x -> 1.0f0 / (1.0f0 + exp((x - p_loc) / eps_ic))
u0 = Float32.(IC_func_burgers.(X))

K = 1

if ref_samples !== nothing
    fig_samples = plot_sample(
        K,
        [ref_samples, samples_reactant],
        ["Reference", "Reactant"];
        suptitle = "Burgers IC + Mass + Flux Samples - Reactant",
    )

    save("plots/samples_burgers_IC_reactant.png", fig_samples)

    fig_samples_deviation = plot_sample(
        K,
        [samples_reactant .- ref_samples],
        ["Reactant"];
        suptitle = "Deviation from reference - Burgers IC + Mass + Flux Reactant",
    )

    save("plots/samples_burgers_IC_deviation_reactant.png", fig_samples_deviation)
else
    fig_samples = plot_sample(
        K,
        [samples_reactant],
        ["Reactant"];
        suptitle = "Burgers IC + Mass + Flux Samples - Reactant",
    )

    save("plots/samples_burgers_IC_reactant.png", fig_samples)
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
        suptitle = "IC Violation - Burgers Eq. Reactant",
    )
else
    fig_ic = plot_constraint_violation(
        K,
        [samples_reactant],
        ic_violation,
        ["Reactant"];
        constraint_params = (nx, nt, dx, dt, u0),
        suptitle = "IC Violation - Burgers Eq. Reactant",
    )
end

save("plots/ic_violation_burgers_IC_reactant.png", fig_ic)

function mass_violation(u, params)
    nx, nt, dx, dt, u0 = params
    mass0 = sum(u0[i] for i in 1:nx) * dx
    return [sum(u[i, t] for i in 1:nx) * dx - mass0 for t in 1:nt]
end

if ref_samples !== nothing
    fig_mass = plot_constraint_violation(
        K,
        [ref_samples, samples_reactant],
        mass_violation,
        ["Reference", "Reactant"];
        constraint_params = (nx, nt, dx, dt, u0),
        suptitle = "Constant Mass Violation - Burgers Eq. Reactant",
    )
else
    fig_mass = plot_constraint_violation(
        K,
        [samples_reactant],
        mass_violation,
        ["Reactant"];
        constraint_params = (nx, nt, dx, dt, u0),
        suptitle = "Constant Mass Violation - Burgers Eq. Reactant",
    )
end

save("plots/mass_violation_burgers_IC_reactant.png", fig_mass)

smooth_pos_check(x, eps) = 0.5f0 * (x + sqrt(x^2 + eps^2))
smooth_neg_check(x, eps) = 0.5f0 * (x - sqrt(x^2 + eps^2))

function flux_violation(u, params)
    nx, nt, dx, dt, k_eff, eps = params
    λ = dt / dx

    residuals = Float32[0.0f0]

    for t in 1:k_eff
        for i in 2:nx-1
            F_i =
                0.5f0 * smooth_pos_check(u[i, t], eps)^2 +
                0.5f0 * smooth_neg_check(u[i + 1, t], eps)^2

            F_im1 =
                0.5f0 * smooth_pos_check(u[i - 1, t], eps)^2 +
                0.5f0 * smooth_neg_check(u[i, t], eps)^2

            r = u[i, t + 1] - u[i, t] + λ * (F_i - F_im1)

            push!(residuals, r)
        end
    end

    return residuals
end

if ref_samples !== nothing
    fig_flux = plot_constraint_violation(
        K,
        [ref_samples, samples_reactant],
        flux_violation,
        ["Reference", "Reactant"];
        constraint_params = (nx, nt, dx, dt, k_eff, eps),
        suptitle = "Local Flux Update Violation - Burgers Eq. Reactant",
    )
else
    fig_flux = plot_constraint_violation(
        K,
        [samples_reactant],
        flux_violation,
        ["Reactant"];
        constraint_params = (nx, nt, dx, dt, k_eff, eps),
        suptitle = "Local Flux Update Violation - Burgers Eq. Reactant",
    )
end

save("plots/flux_violation_burgers_IC_reactant.png", fig_flux)

ic_violation_reactant = [
    ic_violation(samples_reactant[:, :, 1, s], (nx, nt, dx, dt, u0))
    for s in 1:batch_size
]

mass_violation_reactant = [
    mass_violation(samples_reactant[:, :, 1, s], (nx, nt, dx, dt, u0))
    for s in 1:batch_size
]

flux_violation_reactant = [
    flux_violation(samples_reactant[:, :, 1, s], (nx, nt, dx, dt, k_eff, eps))
    for s in 1:batch_size
]

JLD2.save(
    "plots/violations_burgers_IC_reactant.jld2",
    "ic_violation_reactant", ic_violation_reactant,
    "mass_violation_reactant", mass_violation_reactant,
    "flux_violation_reactant", flux_violation_reactant,
    "x_grid", collect(X),
    "t_grid", collect(T),
)

println("Saved plots:")
println("plots/samples_burgers_IC_reactant.png")
if ref_samples !== nothing
    println("plots/samples_burgers_IC_deviation_reactant.png")
end
println("plots/ic_violation_burgers_IC_reactant.png")
println("plots/mass_violation_burgers_IC_reactant.png")
println("plots/flux_violation_burgers_IC_reactant.png")

println("Saved raw violations:")
println("plots/violations_burgers_IC_reactant.jld2")

ic_flat = reduce(vcat, ic_violation_reactant)
mass_flat = reduce(vcat, mass_violation_reactant)
flux_flat = reduce(vcat, flux_violation_reactant)

println("Reactant max IC violation: ", maximum(abs.(ic_flat)))
println("Reactant mean IC violation: ", sum(abs.(ic_flat)) / length(ic_flat))

println("Reactant max mass violation: ", maximum(abs.(mass_flat)))
println("Reactant mean mass violation: ", sum(abs.(mass_flat)) / length(mass_flat))

println("Reactant max flux violation: ", maximum(abs.(flux_flat)))
println("Reactant mean flux violation: ", sum(abs.(flux_flat)) / length(flux_flat))
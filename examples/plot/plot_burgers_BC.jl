
using JLD2
using CairoMakie
include(joinpath(@__DIR__, "..", "..", "utils", "plotUtils.jl"))


# Load samples
data_path = joinpath(@__DIR__, "..", "..", "datasets", "samples", "samples_burgers_BC.jld2")
data_path2 = joinpath(@__DIR__, "..", "..", "datasets", "samples", "alaina_results_burgers_BC.jld2")
data = JLD2.load(data_path)
data2 = JLD2.load(data_path2)
results = data2["results"]

samples_LBFGS = results[1].samples  # (nx, nt, 1, n_samples)  
samples_IPNewton = results[2].samples  # (nx, nt, 1, n_samples)  
samples_exa_gpu     = data["samples_exa_gpu"][1:end-1, 1:end-1, :, :]  # (nx, nt, 1, n_samples) - remove last row/col which are BCs
samples_exa_cpu     = data["samples_exa_cpu"][1:end-1, 1:end-1, :, :]
samples_jump_madnlp = data["samples_jump_madnlp"][1:end-1, 1:end-1, :, :]
# samples_ffm         = data["samples_ffm"]
# ref_samples         = data["ref_samples"]

nx = 100
nt = 100
x_range = (0.0f0, 1.0f0)
t_range = (0.0f0, 1.0f0)
dt = 1.0f0 / (nt - 1)
dx = 1.0f0 / (nx - 1)

X = range(x_range[1], x_range[2]; length = nx)
T = range(t_range[1], t_range[2]; length = nt)


##### Simulated solution
K = 1

fig_samples = plot_sample(K, [samples_exa_gpu, samples_exa_cpu, samples_jump_madnlp, samples_IPNewton, samples_LBFGS],
    ["ExaModels GPU", "ExaModels CPU", "JuMP", "IPNewton", "LBFGS"]; suptitle="Burgers Eq. Samples")

# save("plots/samples_burgers.png", fig_samples)


##### Deviation from Reference
# fig_samples_deviation = plot_sample(K, [samples_exa_gpu .- ref_samples, samples_exa_cpu .- ref_samples, samples_jump_madnlp .- ref_samples, samples_ffm .- ref_samples],
#     ["ExaModels GPU", "ExaModels CPU", "JuMP", "FFM"]; suptitle="Deviation from reference - Burgers Eq.")

# save("plots/samples_burgers_deviation.png", fig_samples_deviation)


# Left Dirichlet BC: u[1, t] should be constant (== u[1, 1]) for all t
function left_bc_violation(u, params)
    nx, nt = params[1], params[2]
    return [abs(u[1, j] - u[1, 1]) for j in 1:nt]
end

# fig_left_bc = plot_constraint_violation(K,
#     [samples_exa_gpu, samples_exa_cpu, samples_jump_madnlp, samples_IPNewton, samples_LBFGS],
#     left_bc_violation,
#     ["ExaModels GPU", "ExaModels CPU", "JuMP", "IPNewton", "LBFGS"];
#     constraint_params=(nx, nt, dx, dt),
#     suptitle="Left Dirichlet BC Violation - Burgers Eq.")

# save("plots/left_bc_violation_burgers.png", fig_left_bc)


# Neumann BC at right: u[nx, t] - u[nx-1, t] should be 0
function neumann_bc_violation(u, params)
    nx, nt = params[1], params[2]
    return [u[nx, j] - u[nx-1, j] for j in 1:nt]
end

# fig_neumann_bc = plot_constraint_violation(K,
#     [ref_samples, samples_exa_cpu, samples_jump_madnlp, samples_ffm],
#     neumann_bc_violation,
#     ["Reference", "ExaModels", "JuMP", "FFM"];
#     constraint_params=(nx, nt, dx, dt),
#     suptitle="Right Neumann BC Violation - Burgers Eq.")

# save("plots/neumann_bc_violation_burgers.png", fig_neumann_bc)


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

all_samples   = [samples_exa_gpu, samples_exa_cpu, samples_jump_madnlp, samples_IPNewton, samples_LBFGS]
method_labels = ["ExaModels GPU", "ExaModels CPU", "JuMP", "IPNewton", "LBFGS"]
grid_pos      = [(1,1), (1,2), (1,3), (2,1), (2,2)]

fig_mass = Figure(size = (1200, 600))
Label(fig_mass[0, :], "Mass Evolution Violation — Burgers Eq.", fontsize = 20, font = :bold)

for (k, (sol, label, pos)) in enumerate(zip(all_samples, method_labels, grid_pos))
    ax = Axis(fig_mass[pos...]; title = label, xlabel = "Time step", ylabel = "Residual")
    lines!(ax, mass_violation(sol[:, :, 1, K], (nx, nt, dx, dt)))
    hlines!(ax, [0.0f0]; color = :gray, linestyle = :dash, linewidth = 1)
end

save("examples/plot/finalPlots/mass_violation_burgers_grid.png", fig_mass)

function plot_constraint_violation(frame, solutions, H, titles; constraint_params=nothing, suptitle=nothing)
    f = Figure(size = (2400, 600))
    N = size(solutions)[1]
    @assert N == length(titles)

    axes = []
    for i in 1:N
        ax = Axis(f[1, i], 
                title = titles[i],
                xlabel = "Time", 
                ylabel = "X")
        push!(axes, ax)
    end 

    for i in 1:N
        lines!(axes[i], H(solutions[i][:,:,1,frame], constraint_params))
    end

    if !isnothing(suptitle)
        Label(f[0, :], suptitle, fontsize=20, font=:bold)
    end
    return f
end
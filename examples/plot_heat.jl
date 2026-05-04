
using JLD2
using CairoMakie
include(joinpath(@__DIR__, "..", "utils", "plotUtils.jl"))


# Load samples
data = JLD2.load("datasets/samples/samples_heat.jld2")
samples_exa_gpu     = data["samples_exa_gpu"]
samples_exa_cpu     = data["samples_exa_cpu"]
samples_jump_madnlp = data["samples_jump_madnlp"]
samples_ffm         = data["samples_ffm"]
u_analytic          = data["u_analytic"]

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



##### Simulated solution
K = 1

fig_samples = plot_sample(K, [u_analytic, samples_exa_gpu, samples_jump_madnlp, samples_ffm],
    ["Analytic", "ExaModels", "JuMP", "FFM"]; suptitle="Heat Eq. Samples")

# save("plots/samples_heat.png", fig_samples)


##### Deviation from Analytic
fig_samples_deviation = plot_sample(K, [samples_exa_gpu .- u_analytic, samples_jump_madnlp .- u_analytic, samples_ffm .- u_analytic],
    ["ExaModels", "JuMP", "FFM"]; suptitle="Deviation from analytic solution - Heat Eq.")

# save("plots/samples_heat_deviation.png", fig_samples)



function mass_constraint(u, params)
    Nx, Nt = params
    return [sum((u[i, j] - u[i,1]) for i in 1:(Nx-1)) for j in 1:Nt]
end

fig_constraint_mass = plot_constraint_violation(K, [samples_exa_gpu, samples_jump_madnlp, samples_ffm],
    mass_constraint,
    ["ExaModels", "JuMP", "FFM"]
    ; constraint_params=(nx, nt, dx, dt),
    suptitle="Mass Constraint Violation - Heat Eq.")

# save("plots/mass_constraint_violation_heat.png", fig_constraint_mass)


#TODO
# Need to change plotting function. Instead of plotting deviation over time, 
# for initial value constraints we want to print deviation over x. 

function ic_violation(u, params)
    nx, nt = params[1], params[2]
    return [sum(abs(u[j, i] - u[1, i]) for i in 1:nx) for j in 1:nt]
end

fig_constraint_ic = plot_constraint_violation(K,
    [u_analytic, samples_exa_gpu, samples_exa_cpu, samples_jump_madnlp, samples_ffm],
    ic_violation,
    ["analytic", "ExaGPU", "ExaCPU", "JuMP", "FFM"];
    constraint_params=(nx, nt, dx, dt),
    suptitle = "IC Constrain Violation - Heat Eq.")

save("plots/IC_constraint_violation_heat.png", fig_constraint_ic)


#TODO 
# Calculate some metrics that show quantitatively how much constrains are being violated - as is being done in PCFM paper. 

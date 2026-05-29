
using JLD2
using CairoMakie
include(joinpath(@__DIR__, "..", "..", "utils", "plotUtils.jl"))


# Load samples
data_path = joinpath(@__DIR__, "..", "..", "datasets", "samples", "samples_heat.jld2")
data_path2 = joinpath(@__DIR__, "..", "..", "datasets", "samples", "alaina_results_heat_1.jld2")
data = JLD2.load(data_path)
data2 = JLD2.load(data_path2)

results = data2["results"]
results[10]
samples_LBFGS = results[8].samples  # (nx, nt, 1, n_samples)
samples_IPNewton = results[10].samples  # (nx, nt, 1, n_samples)

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




function plot_sample(frame, solutions, titles; suptitle=nothing)
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
        heatmap!(axes[i], T, X, solutions[i][:,:,1,frame]', colormap = :viridis)
    end 


    if !isnothing(suptitle)
        Label(f[0, :], suptitle, fontsize=20, font=:bold)
    end
    
    return f
end 

##### Simulated solution
K = 1

let sols   = [u_analytic, samples_exa_gpu, samples_exa_cpu, samples_jump_madnlp, samples_LBFGS, samples_IPNewton],
    labels = ["Analytic", "ExaModels - GPU", "ExaModels - CPU", "JuMP", "LBFGS", "IPNewton"],
    grid   = [(1,1), (1,2), (1,3), (2,1), (2,2), (2,3)]
    global fig_samples = Figure(size = (1200, 700))
    for (sol, label, pos) in zip(sols, labels, grid)
        ax = Axis(fig_samples[pos...]; title = label, titlesize = 24,
                  xticksvisible = false, xticklabelsvisible = false,
                  yticksvisible = false, yticklabelsvisible = false)
        heatmap!(ax, T, X, sol[:, :, 1, K]', colormap = :viridis)
    end
    resize_to_layout!(fig_samples)
end

save("examples/plot/finalPlots/samples_heat_grid.png", fig_samples)


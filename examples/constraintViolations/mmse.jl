using JLD2
using Statistics

"""
Metric of variability of samples. Compare the mean
across the generated samples to 
the mean across the true sample. 
"""

data_path = joinpath(@__DIR__, "..", "..", "datasets", "samples")
typeof(data_path)

heat_data = JLD2.load(joinpath(data_path, "samples_heat.jld2"))

heat_data

samples_exa_gpu     = heat_data["samples_exa_gpu"][:,:,1,:]
samples_exa_cpu     = heat_data["samples_exa_cpu"][:,:,1,:]
samples_jump_madnlp = heat_data["samples_jump_madnlp"][:,:,1,:]


analytic_solution = heat_data["u_analytic"][:,:,1,:]

mean_arr = dropdims(mean(samples_exa_gpu, dims=3), dims=3)  # 100x100
std_arr  = dropdims(std(samples_exa_gpu, dims=3), dims=3)   # 100x100

std_analytic = dropdims(std(analytic_solution, dims=3), dims=3)  # 100x100

analytic_solution[:,:,2]
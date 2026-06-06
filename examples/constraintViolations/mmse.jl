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

heat_data["results"]

samples_exa_gpu     = heat_data["samples_exa_gpu"]
samples_exa_cpu     = data["samples_exa_cpu"]
samples_jump_madnlp = data["samples_jump_madnlp"]


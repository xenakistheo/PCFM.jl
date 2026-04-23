module PCFM

using NeuralOperators
using Lux
using Random
using Optimisers
using Reactant
using MadNLP
using JuMP
using HDF5

# Make training API origin explicit for downstream includes.
const Training = Lux.Training

# Include submodules
include("./data.jl")
include("./model.jl")
include("./training.jl")
include("./sampling.jl")
include("./constraints.jl")

# Export main functions
export FFM
export prepare_input, interpolate_flow
export train_ffm!, sample_ffm, sample_pcfm
# Data
export generate_diffusion_data
export load_burgers_batch, load_rd_batch, load_ns_batch
# Constraints
export heat_constraints!, rd_constraints!, burgers_constraints!, ns_constraints!

end # module PCFM
